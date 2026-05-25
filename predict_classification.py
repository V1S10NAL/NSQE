"""
predict_classification.py
Cross-evaluate classification models on measured datasets.
加载多训练集、多随机种子的分类模型，对真实实测数据集进行交叉评估，并计算相关性。
"""
import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import collections
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tools

# ================= Environment / 环境配置 =================
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--MP_type', type=str, default=['PC', 'PE', 'PET', 'PP', 'PS', 'PVC', 'PMMA', 'PTFE'])
args, _ = parser.parse_known_args()

# ================= Paths / 路径配置 =================
BASE_RUN_DIR = r'./run_classification'
TEST_DATA_DIR = r'.\measured_dataset'
METRICS_EXCEL = r'.\generative model assessment metrics.xlsx'

#  (measured_dataset_1 - measured_dataset_8)
test_path_list = [os.path.join(TEST_DATA_DIR, f'measured_dataset_{i}.mat') for i in range(1, 9)]

categories = args.MP_type
num_classes = len(categories)
category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

def get_stars(p):
    """Significance stars / 显著性星号"""
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'ns'

if __name__ == '__main__':

    train_dataset_dirs = []
    for d in os.listdir(BASE_RUN_DIR):
        if os.path.isdir(os.path.join(BASE_RUN_DIR, d)):

            if 'dataset_baseline' in d or 'simulated_dataset_' in d:
                train_dataset_dirs.append(d)

    print(f">>> Found {len(train_dataset_dirs)} training directories to evaluate.")

    # 1. Preload measured test data / 预加载实测测试集入内存
    preloaded_test_data = {}
    print("\n>>> Preloading measured test datasets...")
    for path in test_path_list:
        if not os.path.exists(path):
            print(f"  [Warning] Not found: {path}")
            continue

        test_name = os.path.splitext(os.path.basename(path))[0]
        spectra_mat = tools.load_data(path)
        X_test, y_test = [], []

        for var_name in categories:
            if var_name in spectra_mat:
                data = np.array(spectra_mat[var_name])
                label = np.zeros(num_classes)
                label[category_to_idx[var_name]] = 1
                X_test.append(np.squeeze(data))
                y_test.append(np.tile(label, (data.shape[0], 1)))

        if X_test:
            X_test_arr = np.expand_dims(np.vstack(X_test), axis=-1)
            y_classes = np.argmax(np.vstack(y_test), axis=1)
            preloaded_test_data[test_name] = {'X': X_test_arr, 'y': y_classes}
            print(f"  - Loaded: {test_name} (Samples: {X_test_arr.shape[0]})")

    cross_eval_results = collections.defaultdict(lambda: collections.defaultdict(lambda: {'acc': [], 'pre': [], 'rec': [], 'f1': []}))
    cross_eval_cms = collections.defaultdict(lambda: collections.defaultdict(list))
    total_models = 0

    # 2. Cross-evaluate models / 交叉评估模型
    print("\n>>> Starting cross-evaluation...")
    for train_dir_name in train_dataset_dirs:
        train_dir_path = os.path.join(BASE_RUN_DIR, train_dir_name)
        seed_folders = [d for d in os.listdir(train_dir_path) if 'seed' in d]

        for seed_folder in seed_folders:
            checkpoint_dir = os.path.join(train_dir_path, seed_folder, 'checkpoint')
            if not os.path.exists(checkpoint_dir): continue

            h5_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
            if not h5_files: continue

            model_path = os.path.join(checkpoint_dir, h5_files[0])
            net_name = seed_folder.split('_seed')[0]

            print(f"\n  >> Evaluating Model: [{train_dir_name} | {seed_folder}]")

            custom_objects = tools.load_custom_objects(net_name) if hasattr(tools, 'load_custom_objects') else {}
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            except Exception as e:
                print(f"     [Error] Failed to load model {model_path}: {e}")
                continue

            model.trainable = False
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            model.compile()

            total_models += 1
            for test_name, test_data in preloaded_test_data.items():
                y_pred_proba = model.predict(test_data['X'], batch_size=args.batch_size, verbose=0)
                y_pred_classes = np.argmax(y_pred_proba, axis=1)

                acc = accuracy_score(test_data['y'], y_pred_classes)
                pre = precision_score(test_data['y'], y_pred_classes, average='macro', zero_division=0)
                rec = recall_score(test_data['y'], y_pred_classes, average='macro', zero_division=0)
                f1  = f1_score(test_data['y'], y_pred_classes, average='macro', zero_division=0)
                cm = confusion_matrix(test_data['y'], y_pred_classes, labels=range(num_classes))

                cross_eval_results[train_dir_name][test_name]['acc'].append(acc)
                cross_eval_results[train_dir_name][test_name]['pre'].append(pre)
                cross_eval_results[train_dir_name][test_name]['rec'].append(rec)
                cross_eval_results[train_dir_name][test_name]['f1'].append(f1)
                cross_eval_cms[train_dir_name][test_name].append(cm)

                print(f"     - [Test: {test_name}] Acc: {acc:.4f} | Pre: {pre:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

            tf.keras.backend.clear_session()
        print(f"\n  -> {train_dir_name} all seeds evaluated.")

    # 3. Export results / 数据汇总与导出 Excel
    combined_list, separated_list = [], []
    test_names = list(preloaded_test_data.keys())

    for train_data in train_dataset_dirs:
        if train_data not in cross_eval_results: continue
        for test_data in test_names:
            metrics = cross_eval_results[train_data][test_data]
            if not metrics['acc']: continue

            avg_acc, std_acc = np.mean(metrics['acc']), np.std(metrics['acc'])
            avg_pre, std_pre = np.mean(metrics['pre']), np.std(metrics['pre'])
            avg_rec, std_rec = np.mean(metrics['rec']), np.std(metrics['rec'])
            avg_f1, std_f1 = np.mean(metrics['f1']), np.std(metrics['f1'])

            combined_list.append({
                'Train_Model': train_data, 'Test_Dataset': test_data,
                'Accuracy': f"{avg_acc:.4f} ± {std_acc:.4f}",
                'Precision': f"{avg_pre:.4f} ± {std_pre:.4f}",
                'Recall': f"{avg_rec:.4f} ± {std_rec:.4f}",
                'F1_Score': f"{avg_f1:.4f} ± {std_f1:.4f}"
            })
            separated_list.append({
                'Train_Model': train_data, 'Test_Dataset': test_data,
                'Acc_Mean': avg_acc, 'Acc_Std': std_acc, 'F1_Mean': avg_f1, 'F1_Std': std_f1
            })

    output_excel_path = os.path.join(BASE_RUN_DIR, f'Measured_CrossEval_{time.strftime("%Y%m%d_%H%M")}.xlsx')
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        pd.DataFrame(combined_list).to_excel(writer, sheet_name='Combined_Metrics', index=False)
        pd.DataFrame(separated_list).to_excel(writer, sheet_name='Separated_Metrics', index=False)
    print(f"\n[Success] Metrics exported to: {output_excel_path}")

    # 4. Plot average confusion matrix / 计算并绘制平均混淆矩阵
    cm_output_dir = os.path.join(BASE_RUN_DIR, f'CM_Measured_Average_{time.strftime("%Y%m%d_%H%M")}')
    os.makedirs(cm_output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    for train_data in train_dataset_dirs:
        if train_data not in cross_eval_cms: continue
        for test_data in test_names:
            cms = cross_eval_cms[train_data][test_data]
            if not cms: continue

            avg_cm = np.mean(cms, axis=0)
            avg_cm_norm = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis] # Normalized CM

            plt.figure(figsize=(8, 6))
            sns.heatmap(avg_cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
            plt.title(f'Avg CM (Norm)\nTrain: {train_data} | Test: {test_data}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(cm_output_dir, f'CM_{train_data}_vs_{test_data}.png'), dpi=200)
            plt.close()

    # 5. Calculate SRCC/PLCC / 计算生成指标与分类性能的相关性
    if os.path.exists(METRICS_EXCEL):
        print("\n>>> Calculating SRCC and PLCC correlation...")
        try:
            df_metrics = pd.read_excel(METRICS_EXCEL)
            if 'data set' not in df_metrics.columns:
                df_metrics.rename(columns={df_metrics.columns[0]: 'data set'}, inplace=True)
            df_metrics.set_index('data set', inplace=True)

            perf_data = []
            for train_data in train_dataset_dirs:

                match = re.search(r'_(\d+)$', train_data)
                if not match:
                    continue
                t_id = int(match.group(1))

                row_data = {'data set': t_id}
                for test_data in test_names:
                    if cross_eval_results.get(train_data, {}).get(test_data, {}).get('acc'):
                        row_data[f'Acc_{test_data}'] = np.mean(cross_eval_results[train_data][test_data]['acc'])
                        row_data[f'F1_{test_data}'] = np.mean(cross_eval_results[train_data][test_data]['f1'])
                perf_data.append(row_data)

            df_merged = df_metrics.join(pd.DataFrame(perf_data).set_index('data set'), how='inner')
            corr_results = []

            for test_data in test_names:
                acc_col, f1_col = f'Acc_{test_data}', f'F1_{test_data}'
                if acc_col not in df_merged.columns: continue

                for metric in df_metrics.columns:
                    valid_data = df_merged[[metric, acc_col, f1_col]].dropna()
                    if len(valid_data) < 2: continue

                    srcc_acc, p_s_acc = spearmanr(valid_data[metric], valid_data[acc_col])
                    plcc_acc, p_p_acc = pearsonr(valid_data[metric], valid_data[acc_col])

                    corr_results.append({
                        'Measured_Dataset': test_data, 'Generative_Metric': metric,
                        'SRCC_Acc': f"{srcc_acc:.4f}{get_stars(p_s_acc)}",
                        'PLCC_Acc': f"{plcc_acc:.4f}{get_stars(p_p_acc)}"
                    })

            corr_path = os.path.join(BASE_RUN_DIR, f'Measured_Correlation_Analysis_{time.strftime("%Y%m%d_%H%M")}.xlsx')
            pd.DataFrame(corr_results).to_excel(corr_path, index=False)
            print(f"[Success] Correlation analysis exported to: {corr_path}")
        except Exception as e:
            print(f"[Error] Failed to compute correlation: {e}")

    print("\n>>> All tasks completed successfully!")
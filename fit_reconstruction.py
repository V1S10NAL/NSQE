"""
fit_reconstruction.py
In extra simulated dataset batch evaluate models on extra datasets and compute grouped statistical metrics (Mean ± Std).
在额外模拟数据集 批量验证重建模型并统计分组指标均值与标准差。
"""
import os
import argparse
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import glob
from datetime import datetime
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

# ================= Parameters / 参数配置 =================
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--net_list', type=str, default=['ResUNet'])
args, _ = parser.parse_known_args()

MODEL_DIR = r'./run_reconstruction'
EVAL_DATA_FILE = r'.\simulated_dataset\dataset_extra_reconstruction.mat'
OUTPUT_DIR = r'./data_evaluation_stats'
HISTORICAL_EXCEL_PATH = r'./data_evaluation_stats\Batch_Evaluation_Results.xlsx'

os.makedirs(OUTPUT_DIR, exist_ok=True)



# ================= Main Execution / 主执行逻辑 =================
if __name__ == '__main__':

    # 1. Load evaluation dataset once / 仅加载一次验证集
    print(f">>> Loading external evaluation dataset from: {EVAL_DATA_FILE}")
    data_dict = tools.load_data(EVAL_DATA_FILE)
    X_extra = data_dict['all_simulate_spectra']
    y_extra = data_dict['all_original_spectra']
    X_extra_3d = np.expand_dims(X_extra, axis=-1)

    # 2. Search and group models / 搜索并对模型进行分组
    model_list = []
    for net in args.net_list:
        pattern = os.path.join(MODEL_DIR, "Simulated_*", f"{net}_seed*", "checkpoint", f"{net}_model*")
        model_list.extend(glob.glob(pattern))

    models_by_category = defaultdict(list)
    for model_path in model_list:
        category = os.path.normpath(model_path).split(os.sep)[-4]
        models_by_category[category].append(model_path)

    print(f">>> Search complete! Found {len(model_list)} models across {len(models_by_category)} categories.")
    results_list = []

    # 3. Evaluate models by category / 按类别依次评估模型
    for category, paths in models_by_category.items():
        print(f"\n{'=' * 65}\nProcessing Category [{category}] with {len(paths)} models\n{'=' * 65}")

        for model_path in paths:
            run_name = os.path.normpath(model_path).split(os.sep)[-3]
            net_name = run_name.split('_seed')[0]

            print(f"\n  >> Evaluating model: {run_name}")
            custom_objects = tools.load_custom_objects(net_name) if hasattr(tools, 'load_custom_objects') else {}

            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            except Exception as e:
                print(f"     [Error] Failed to load model: {e}")
                continue

            # Lock BN layers / 固定BN层状态
            model.trainable = False
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            model.compile()

            # Prediction / 模型预测
            start_time = time.time()
            y_extra_pred = np.squeeze(model.predict(X_extra_3d, batch_size=args.batch_size, verbose=0))
            print(f"     - Prediction time: {time.time() - start_time:.2f} s")

            # Metrics Calculation / 计算并记录指标
            metrics = tools.evaluate_all_metrics(y_extra, y_extra_pred)
            metrics['Category'] = category
            metrics['Model_Run_Name'] = run_name
            results_list.append(metrics)

            print(f"     - Metrics -> R2: {metrics['R2']:.8f} | MSE: {metrics['MSE']:.8f} | SSIM: {metrics['SSIM']:.8f} | PSNR: {metrics['PSNR']:.8f}")

            # Clear graph to prevent OOM / 清理计算图释放显存
            tf.keras.backend.clear_session()

    # 4. Statistical Data Aggregation & Export / 数据统计与导出
    print("\n>>> All evaluations complete! Computing statistics and exporting...")

    if not results_list:
        print("[Error] No results to export.")
        exit()

    df_raw = pd.DataFrame(results_list)

    # Reorder columns / 调整列展示顺序
    metadata_cols = ['Category', 'Model_Run_Name']
    metrics_cols = ['R2', 'MSE', 'RMSE', 'MAE', 'CS', 'Log-Cosh', 'SSIM', 'PSNR', 'SNR', 'SAM']
    df_raw = df_raw[metadata_cols + metrics_cols]

    # Calculate Mean and Std / 分组计算均值与标准差
    grouped = df_raw.groupby('Category')[metrics_cols]
    df_mean = grouped.mean()
    df_std = grouped.std().fillna(0)

    # Format as "Mean ± Std" / 格式化汇总表
    df_summary = pd.DataFrame(index=df_mean.index)
    for col in metrics_cols:
        df_summary[col] = df_mean[col].map('{:.8g}'.format) + " ± " + df_std[col].map('{:.8g}'.format)

    print("\n=== Category Metrics Summary (Mean ± Std) ===")
    print(df_summary.to_string())

    # Export to Excel / 导出多Sheet Excel
    output_excel_path = os.path.join(OUTPUT_DIR, f"Batch_Evaluation_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary (Mean ± Std)')
        df_mean.to_excel(writer, sheet_name='Mean Only')
        df_raw.to_excel(writer, sheet_name='Raw Details', index=False)

    print(f"\n>>> Statistics successfully exported to: {output_excel_path}")
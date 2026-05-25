"""
train_classification.py - Multi-dataset and Multi-seed Training for Classification
"""
import time
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import os
import sys
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split
from scipy.io import savemat
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tools

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# ================= Configuration / 全局配置 =================
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=1024, help='spectral length')
parser.add_argument('--num_outputs', type=int, default=1, help='num_outputs')
parser.add_argument('--train_data_ratio', type=float, default=0.8, help='training set ratio')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--net_list', type=str, default=['SE_ResNet_classification'], help='net_list')
parser.add_argument('--num_substances', type=int, default=8, help='num_substances')
parser.add_argument('--MP_type', type=str, default=['PC', 'PE', 'PET', 'PP', 'PS', 'PVC', 'PMMA', 'PTFE'], help='microplastics types')
args, unknown = parser.parse_known_args()

# List of random seeds / 随机种子列表
seed_list = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035] #[2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]


# Simulated Datasets Paths (.mat format) from Contents.txt / 根据 Contents.txt 定义数据集路径
path_data_train_list = [
    rf'.\simulated_dataset\simulated_dataset_{i}.mat' for i in range(13, 25)
]
# ==============================================================

data_vars = args.MP_type
categories = args.MP_type
num_classes = len(categories)
category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

base_output_dir = './run_classification'
os.makedirs(base_output_dir, exist_ok=True)

for ds_idx, path_data_train in enumerate(path_data_train_list):
    dataset_name = os.path.splitext(os.path.basename(path_data_train))[0]
    ds_base_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(ds_base_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f">>> Processing dataset [{ds_idx + 1}/{len(path_data_train_list)}]: {dataset_name}")
    print("=" * 80)

    if not os.path.exists(path_data_train):
        print(f"[Error] Dataset file not found: {path_data_train}, skipping.")
        continue

    start_time = time.perf_counter()
    spectra_mat = tools.load_data(path_data_train)
    print(f"[{dataset_name}] Loading time: {time.perf_counter() - start_time:.4f}s")

    for current_seed in seed_list:
        print("\n" + "-" * 60)
        print(f"--- Current Seed: {current_seed} | Dataset: {dataset_name} ---")
        print("-" * 60)

        tools.set_seed(current_seed)

        data_dict = {'X': {m: [] for m in data_vars}, 'y': {m: [] for m in data_vars}}
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        num_spectra = 0

        # Determine raw data key based on availability
        spectrum_key = 'test_data' if 'test_data' in spectra_mat else 'raw_spectra'

        for var_name in data_vars:
            # Note: For classification, we assume labels exist as variables in .mat
            if var_name not in spectra_mat:
                continue

            data = np.array(spectra_mat[var_name][:300])
            label = np.zeros(num_classes)
            label[category_to_idx[var_name]] = 1

            data_dict['X'][var_name] = data
            data_dict['y'][var_name] = np.tile(label, (data.shape[0], 1))
            num_spectra += data.shape[0]

            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                np.squeeze(data_dict['X'][var_name]),
                np.squeeze(data_dict['y'][var_name]), test_size=0.3, random_state=current_seed)
            X_v, X_te, y_v, y_te = train_test_split(
                X_tmp, y_tmp, test_size=1/3, random_state=current_seed)

            X_train.append(X_tr); y_train.append(y_tr)
            X_val.append(X_v); y_val.append(y_v)
            X_test.append(X_te); y_test.append(y_te)

        X_train = np.expand_dims(np.vstack(X_train), axis=-1); y_train = np.vstack(y_train)
        X_val = np.expand_dims(np.vstack(X_val), axis=-1); y_val = np.vstack(y_val)
        X_test = np.expand_dims(np.vstack(X_test), axis=-1); y_test = np.vstack(y_test)
        X_full = np.vstack((X_train, X_val, X_test))
        y_full = np.vstack((y_train, y_val, y_test))

        args.num_spectra = num_spectra

        for net in args.net_list:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_path = os.path.join(ds_base_dir, f'{net}_seed{current_seed}_{current_time}')
            result_path = os.path.join(run_path, 'result')
            checkpoint_path = os.path.join(run_path, 'checkpoint')
            os.makedirs(result_path, exist_ok=True)
            os.makedirs(checkpoint_path, exist_ok=True)

            print(f"> Model: {net} | Output: {run_path}")

            start_time = time.perf_counter()
            model, custom_objects = tools.built_model(model_type=net, args=args)

            print_lr_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(f" Epoch {epoch + 1}: LR = {model.optimizer.lr(model.optimizer.iterations).numpy():.6f}"))

            history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks=[print_lr_callback])

            print(f"> Running time: {time.perf_counter() - start_time:.2f}s")

            model.save(os.path.join(checkpoint_path, f'{net}_model.h5'))

            df = pd.DataFrame({
                'Epoch': range(1, len(history.history['loss']) + 1),
                'Train Loss': history.history['loss'],
                'Validation Loss': history.history['val_loss'],
                'Accuracy': history.history['categorical_accuracy'],
                'Precision': history.history['precision'],
                'Recall': history.history['recall'],
                'Val_Accuracy': history.history['val_categorical_accuracy'],
                'Val_Precision': history.history['val_precision'],
                'Val_Recall': history.history['val_recall']
            })
            df.to_excel(os.path.join(result_path, "history.xlsx"), index=False)

            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)

            # Generate Report
            with open(os.path.join(result_path, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\nClassification Metrics Report\n" + "=" * 60 + "\n\n")
                f.write(f"Test Set Accuracy: {accuracy_score(y_test_classes, y_pred_classes):.4f}\n")
                f.write("Detailed Test Report:\n")
                f.write(classification_report(y_test_classes, y_pred_classes, digits=4, target_names=categories, zero_division=0) + "\n")
                f.write("Confusion Matrix:\n")
                f.write(str(confusion_matrix(y_test_classes, y_pred_classes)) + "\n")

            fpr, tpr, roc_auc = tools.plot_roc_curves(y_test, y_pred, categories,
                                                      os.path.join(result_path, "roc_test.png"))
            tools.dict_to_xlsx(fpr, os.path.join(result_path, 'fpr_test.xlsx'))
            tools.dict_to_xlsx(tpr, os.path.join(result_path, 'tpr_test.xlsx'))
            tools.dict_to_xlsx(roc_auc, os.path.join(result_path, 'roc_test.xlsx'))

            savemat(os.path.join(result_path, f"{current_time}_classification_results.mat"), {
                'simulate_raw': np.squeeze(X_full),
                'simulate_labels': y_full,
                'simulate_predictions': y_pred
            }, oned_as='column')

            tf.keras.backend.clear_session()
            print(">>> GPU resources released, preparing for next iteration...\n")

print("\n>>> All dataset and seed iterations have completed successfully!")
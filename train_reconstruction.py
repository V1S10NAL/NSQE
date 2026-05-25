"""
train_reconstruction.py - Multi-dataset and Multi-seed Training for Spectral Reconstruction
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
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
parser.add_argument('--train_data_ratio', type=float, default=0.8, help='training set ratio')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035], help='list of global seeds to run') #[2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
parser.add_argument('--net_list', type=str, default=['ResUNet'], help='net_list')
parser.add_argument('--video', type=bool, default=False, help='generate video')
parser.add_argument('--num_spectra', type=int, default=15000, help="number of spectra (will be updated dynamically)")
args = parser.parse_args()

# Simulated Datasets Paths / 训练集与额外验证集路径配置
datasets_list = []
#
for i in range(1, 13):
    datasets_list.append({
        'dataset_name': f'Simulated_{i}',
        'train_path': rf'.\simulated_dataset\simulated_dataset_{i}.mat',
        'extra_path': r'.\simulated_dataset\dataset_extra_reconstruction.mat'
    })
# ==============================================================

for ds_info in datasets_list:
    ds_name = ds_info['dataset_name']
    train_path = ds_info['train_path']
    extra_path = ds_info['extra_path']

    print("\n" + "#" * 80)
    print(f"🚀 STARTING DATASET TASK: {ds_name}")
    print("#" * 80)

    if not os.path.exists(train_path) or not os.path.exists(extra_path):
        print(f"[Error] Required dataset files not found. Skipping {ds_name}.")
        continue

    start_time = time.perf_counter()

    # --- Load Training Data ---
    spectra_train = tools.load_data(train_path)
    simulated_spectra = spectra_train.get('all_simulate_spectra', spectra_train.get('raw_spectra'))
    original_spectra = spectra_train.get('all_original_spectra', simulated_spectra)

    # --- Load Extra Evaluation Data ---
    spectra_extra = tools.load_data(extra_path)
    extra_simulated = spectra_extra.get('all_simulate_spectra')
    extra_original = spectra_extra.get('all_original_spectra')

    if extra_simulated is None or extra_original is None:
        print("[Error] Variables 'simulate_spectra_input' or 'target' not found in extra_path. Skipping.")
        continue

    print(f"Loading complete. Time: {time.perf_counter() - start_time:.4f}s")

    args.num_spectra = simulated_spectra.shape[0]

    X = np.expand_dims(simulated_spectra, axis=-1)
    y = np.expand_dims(original_spectra, axis=-1)
    w = 1.0 + np.square(np.square(y)) * 100

    X_extra = np.expand_dims(extra_simulated, axis=-1)
    y_extra = np.expand_dims(extra_original, axis=-1)

    ds_base_dir = f'./run_reconstruction/{ds_name}'
    os.makedirs(ds_base_dir, exist_ok=True)

    for net in args.net_list:
        print(f"\n{'='*60}\nEvaluating Model: {net} | Dataset: {ds_name}\n{'='*60}")
        metrics_summary = {'train': [], 'val': [], 'test': [], 'extra': []}
        global_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        for run_idx, seed in enumerate(args.seeds):
            print(f"\n--- Run {run_idx+1}/{len(args.seeds)} | Seed: {seed} | Net: {net} ---")

            tools.set_seed(seed)
            X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(X, y, w, test_size=0.3, random_state=seed, shuffle=True)
            X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(X_temp, y_temp, w_temp, test_size=1 / 3, random_state=seed, shuffle=True)
            run_path = os.path.join(ds_base_dir, f'{net}_seed{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            result_path = os.path.join(run_path, 'result')
            checkpoint_path = os.path.join(run_path, 'checkpoint')
            os.makedirs(result_path, exist_ok=True)
            os.makedirs(checkpoint_path, exist_ok=True)

            start_time = time.perf_counter()
            model, custom_objects = tools.built_model(model_type=net, args=args)

            if run_idx == 0:
                print('\nModel FLOPs Info:')
                x_dummy = tf.constant(np.random.randn(1, args.num_features, 1))
                print(f"{tools.get_flops(model, [x_dummy]):.2f} MFLOPs (Single data)")
                # print(f"{tools.calculate_flops(model, batch_size=args.batch_size):.4f} GFLOPs (Batch data)\n")

            print_lr_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(f" Epoch {epoch + 1}: LR = {model.optimizer.lr(model.optimizer.iterations).numpy():.6f}"))
            video_cb = tools.SaveAndVideoCallback(X_train[0:1], np.linspace(132, 4051, args.num_features), run_path, args.video)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=args.batch_size,
                sample_weight=w_train,
                epochs=args.epochs,
                verbose=1,
                callbacks=[print_lr_callback, video_cb]
            )

            print(f"Run {run_idx+1} training time: {time.perf_counter() - start_time:.4f}s")
            model.save(os.path.join(checkpoint_path, f'{net}_model.h5'))

            pd.DataFrame(history.history).to_excel(os.path.join(result_path, "history.xlsx"), index=False)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            y_extra_pred = model.predict(X_extra)

            # Flatten to evaluate
            train_metrics = tools.calculate_metrics(y_train.reshape(y_train.shape[0], -1), y_train_pred.reshape(y_train_pred.shape[0], -1))
            val_metrics = tools.calculate_metrics(y_val.reshape(y_val.shape[0], -1), y_val_pred.reshape(y_val_pred.shape[0], -1))
            test_metrics = tools.calculate_metrics(y_test.reshape(y_test.shape[0], -1), y_test_pred.reshape(y_test_pred.shape[0], -1))
            extra_metrics = tools.calculate_metrics(y_extra.reshape(y_extra.shape[0], -1), y_extra_pred.reshape(y_extra_pred.shape[0], -1))

            metrics_summary['train'].append(train_metrics); metrics_summary['val'].append(val_metrics)
            metrics_summary['test'].append(test_metrics); metrics_summary['extra'].append(extra_metrics)

            with open(os.path.join(result_path, "metrics.txt"), "w") as file:
                file.write(f"Test Set Metrics: R²: {test_metrics[0]:.6f}, MSE: {test_metrics[1]:.6f}, CS: {test_metrics[5]:.6f}\n")
                file.write(f"Extra Set Metrics: R²: {extra_metrics[0]:.6f}, MSE: {extra_metrics[1]:.6f}, CS: {extra_metrics[5]:.6f}\n")

            tf.keras.backend.clear_session()

        print(f"\n{'-'*60}\nSummary for {net} on Dataset '{ds_name}' over {len(args.seeds)} seeds\n{'-'*60}")
        avg_test = np.mean(metrics_summary['test'], axis=0); std_test = np.std(metrics_summary['test'], axis=0)
        avg_extra = np.mean(metrics_summary['extra'], axis=0); std_extra = np.std(metrics_summary['extra'], axis=0)

        metric_names = ['R²', 'MSE', 'RMSE', 'MAE', 'MAPE', 'CS', 'logcosh']
        print(f"AVERAGE Test Set Metrics ({ds_name}):")
        for i in range(len(metric_names)): print(f"  {metric_names[i]}: {avg_test[i]:.6f} (±{std_test[i]:.6f})")

        with open(os.path.join(ds_base_dir, f'{net}_Global_Summary_{global_time}.txt'), "w") as f:
            f.write(f"[Average Test Set Metrics]\nR²: {avg_test[0]:.8f} ± {std_test[0]:.8f}\nMSE: {avg_test[1]:.8f} ± {std_test[1]:.8f}\nCS: {avg_test[5]:.8f} ± {std_test[5]:.8f}\n\n")
            f.write(f"[Average Extra Set Metrics]\nR²: {avg_extra[0]:.8f} ± {std_extra[0]:.8f}\nMSE: {avg_extra[1]:.8f} ± {std_extra[1]:.8f}\nCS: {avg_extra[5]:.8f} ± {std_extra[5]:.8f}\n")

print("\n>>> ALL DATASETS EVALUATED SUCCESSFULLY!")
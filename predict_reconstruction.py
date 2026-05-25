"""
predict_reconstruction.py
Apply trained model on measured data, plot 8-subplot comparison, and save results.
应用训练好的模型对实测数据进行去噪，绘制 8 子图对比并导出预测结果。
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import savemat
import tools

# Environment Setup / 硬件环境配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Paths Configuration / 路径配置
MODEL_PATH = r'E:\NSQE\run_reconstruction\Simulated_1\ResUNet_seed2026_20260321_233854\checkpoint\ResUNet_model.h5'
MEASURED_DATA_PATH = r'E:\NSQE\measured_dataset\measured_dataset_1.mat'
OUTPUT_DIR = r'./data_processed_reconstruction'
MATERIALS = ['PC', 'PE', 'PET', 'PP', 'PS', 'PVC', 'PMMA', 'PTFE']

def parse_identifiers(model_path, data_path):
    """Generate unique ID from paths / 自动提取路径信息生成唯一标识符"""
    path_parts = os.path.normpath(model_path).split(os.sep)
    train_ds = next((p for p in path_parts if "Simulated_" in p), "Simulated_X")
    model_name, seed_str = "UnknownModel", "seedX"
    for p in path_parts:
        if "_seed" in p:
            model_name, seed_str = p.split('_seed')[0], "seed" + p.split('_seed')[1].split('_')[0]
            break
    measured_ds = os.path.splitext(os.path.basename(data_path))[0]
    return f"{model_name}_{train_ds}_{seed_str}_{measured_ds}"

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_prefix = parse_identifiers(MODEL_PATH, MEASURED_DATA_PATH)

    # Load Model and Data / 载入模型与数据
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=tools.load_custom_objects(output_prefix.split('_')[0]), compile=False)
    data_dict = tools.load_data(MEASURED_DATA_PATH)

    raw_spectra = data_dict['raw_spectra']
    wavenumber = np.squeeze(data_dict['wavenumber'])
    export_dict = {'raw_spectra': raw_spectra, 'wavenumber': wavenumber}

    # Plot Initialization / 初始化画布
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle(f"Denoising Evaluation: {output_prefix}", fontsize=16, fontweight='bold', y=0.98)

    all_predicted_spectra = []

    # Independent Processing per Material / 独立提取并处理每种物质
    for idx, mat_name in enumerate(MATERIALS):
        ax = axes.flatten()[idx]
        processed_var_name = f"{mat_name}_processed"

        if mat_name in data_dict:
            # Directly read the 300x1024 data matrix / 直接读取物质对应的数据矩阵
            cat_spectra = np.array(data_dict[mat_name])
            export_dict[mat_name] = cat_spectra

            if cat_spectra.shape[0] > 0:
                # Batch Inference / 批量推理去噪
                cat_preds = np.squeeze(model.predict(np.expand_dims(cat_spectra, axis=-1), batch_size=32, verbose=0))
                # Ensure 2D shape / 确保为二维矩阵
                cat_preds = np.expand_dims(cat_preds, axis=0) if cat_preds.ndim == 1 else cat_preds

                export_dict[processed_var_name] = cat_preds
                all_predicted_spectra.append(cat_preds)

                # Plot the first spectrum / 绘制该类别的第一条光谱对比
                ax.plot(wavenumber, cat_spectra[0], label='Noisy Raw', color='gray', alpha=0.7)
                ax.plot(wavenumber, cat_preds[0], label='Denoised', color='crimson')
                ax.set_title(f"{mat_name} (n={cat_spectra.shape[0]})", fontweight='bold')
                ax.legend(loc='upper right', fontsize='small')

    # Combine all predictions / 按顺序垂直拼接所有预测结果以匹配 2400x1024
    if all_predicted_spectra:
        export_dict['predicted_spectra'] = np.vstack(all_predicted_spectra)

    # Export & Display / 保存与弹窗展示
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_comparison.png"), dpi=300)
    savemat(os.path.join(OUTPUT_DIR, f"{output_prefix}.mat"), export_dict, oned_as='column')

    print(f"\n>>> Results successfully saved to: {OUTPUT_DIR}")
    plt.show()
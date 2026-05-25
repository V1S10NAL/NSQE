"""
fit_kid.py - 1D Kernel Inception Distance (KID) Evaluation
"""
import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import polynomial_kernel
import tensorflow as tf
import tools

# Environment Configuration
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# ================= Configuration / 全局配置 =================
# Feature extraction method ('PCA' or 'ResNet') / 特征提取方法 ('PCA' 或 'ResNet')
FEATURE_METHOD = 'ResNet'

# Absolute path to the pre-trained model / 预训练模型的绝对路径
PRETRAINED_MODEL_PATH = r'./run/ResNet_classifier/checkpoint/ResNet_classifier.h5'

# Target layer name for feature extraction / 提取特征的目标层名称
TARGET_LAYER_NAME = 'dense_9'

# Data directories / 数据目录
REAL_DATA_DIR = r'.\measured_dataset'
SIMULATED_DATA_DIR = r'.\simulated_dataset'
# ==============================================================

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

def calculate_kid(features_real, features_sim, degree=3):
    gamma = 1.0 / features_real.shape[1]
    K_XX = polynomial_kernel(features_real, features_real, degree=degree, gamma=gamma, coef0=1)
    K_YY = polynomial_kernel(features_sim, features_sim, degree=degree, gamma=gamma, coef0=1)
    K_XY = polynomial_kernel(features_real, features_sim, degree=degree, gamma=gamma, coef0=1)

    m, n = features_real.shape[0], features_sim.shape[0]

    # print(f"features_real: {m}")
    # print(f"features_sim: {n}")
    term_XX = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    term_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    term_XY = np.sum(K_XY) / (m * n)

    kid = term_XX + term_YY - 2 * term_XY
    return kid

if __name__ == '__main__':
    set_seed()
    print(f"\n>>> Current feature extraction method: [{FEATURE_METHOD}]\n")

    # 1. Load Authentic Data Base
    real_parts = tools.load_and_merge_real_data(REAL_DATA_DIR)
    real_base_data = real_parts[0].astype(np.float32)
    print(f">>> Authentic data loaded, shape: {real_base_data.shape}")

    real_sample = real_base_data
    print(f">>> Authentic data sampled, shape: {real_sample.shape}")

    # 2. Feature Extractor Setup
    if FEATURE_METHOD == 'PCA':
        pca = PCA(n_components=64, random_state=2026).fit(real_sample)
        feat_real = pca.transform(real_sample)

    elif FEATURE_METHOD == 'ResNet':
        print(f">>> Loading pre-trained model: {PRETRAINED_MODEL_PATH}")
        custom_objects = tools.load_custom_objects('SE_ResNet_classification')
        full_model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH, custom_objects=custom_objects, compile=False)

        resnet_model = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.get_layer(TARGET_LAYER_NAME).output,
            name='Pretrained_Feature_Extractor'
        )
        resnet_model.trainable = False
        print(f">>> Successfully extracted layer [{TARGET_LAYER_NAME}] as feature space (Shape: {resnet_model.output_shape})")

        real_sample_3d = np.expand_dims(real_sample, axis=-1)
        feat_real = resnet_model.predict(real_sample_3d, batch_size=256, verbose=0)

    # 3. Define Simulated Datasets Paths
    datasets_paths = {}
    for i in range(1, 13):
        datasets_paths[f'Simulated_{i}'] = os.path.join(SIMULATED_DATA_DIR, f'simulated_dataset_{i}.npy')
    for i in range(13, 25):
        datasets_paths[f'Simulated_{i}'] = os.path.join(SIMULATED_DATA_DIR, f'simulated_dataset_{i}.mat')

    print("\n" + "=" * 50)
    print("=== KID Evaluation Results ===")
    print("=" * 50)

    for cat, path in datasets_paths.items():
        if os.path.exists(path):
            data_dict = tools.load_data(path)

            spectra = data_dict.get('test_data', data_dict.get('raw_spectra')).astype(np.float32)

            if FEATURE_METHOD == 'PCA':
                feat_sim = pca.transform(spectra)
            elif FEATURE_METHOD == 'ResNet':
                spectra_3d = np.expand_dims(spectra, axis=-1)
                feat_sim = resnet_model.predict(spectra_3d, batch_size=256, verbose=0)

            score = calculate_kid(feat_real, feat_sim)
            print(f"[{cat}] KID Score: {score:.16f}")
        else:
            print(f"[Warning] File not found: {path}")
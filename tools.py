"""
tools.py - Core dependencies and utilities
"""
import os
import random
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import SE_ResUNet
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import logging
import math
from itertools import cycle
import matplotlib.patches as mpatches
from fontTools.unicodedata import block
from matplotlib import patches, legend
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from tensorflow.keras import layers
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_curve, auc, confusion_matrix, \
    classification_report
import scipy.io
import numpy as np
import seaborn as sns
import scipy.special
from scipy.optimize import fsolve
from scipy.ndimage import convolve1d
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
from scipy import interp
from sklearn.metrics import accuracy_score
from itertools import combinations
import gc
import os
import cv2
from scipy.special import gamma
from scipy.ndimage import convolve1d
import tools  # 必须导入 tools 以使用 compute_1d_gradcam
import scipy.special as sp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from scipy.optimize import fmin
from scipy.stats import weibull_min
from scipy.ndimage import uniform_filter1d
import shap
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


class SaveAndVideoCallback(tf.keras.callbacks.Callback):
    """保存每个epoch的预测图并生成视频（可选开关） / Save prediction images per epoch and generate video (optional)"""

    def __init__(self, X_data, wavenumber, output_dir, generate_video=True, fps=15, sample_idx=0):
        """
        Args:
            X_data (np.ndarray): 输入数据 / Input data
            wavenumber (np.ndarray): 拉曼位移波长数组 / Raman shift wavenumber array
            output_dir (str): 输出目录路径 / Output directory path
            generate_video (bool): 是否生成图像和视频，默认True / Whether to generate images and video, default True
            fps (int): 视频帧率，默认15 / Video frames per second, default 15
            sample_idx (int): 选择可视化的样本索引，默认0 / Sample index to visualize, default 0
        """
        super().__init__()
        self.generate_video = generate_video
        if not self.generate_video:
            return  # 不初始化相关参数

        self.X_data = X_data
        self.wavenumber = wavenumber
        self.output_dir = output_dir
        self.fps = fps
        self.frame_dir = os.path.join(output_dir, "frames and video")
        os.makedirs(self.frame_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if not self.generate_video:
            return
        # 原图像生成逻辑
        y_pred = self.model.predict(self.X_data, verbose=0).reshape(-1)
        plt.figure(figsize=(16, 9))
        plt.plot(self.wavenumber, y_pred, linewidth=3)
        plt.title(f"Epoch {epoch + 1}")
        plt.xlabel('Raman shift')
        plt.ylabel('Intensity')
        frame_path = os.path.join(self.frame_dir, f"epoch_{epoch + 1:04d}.png")
        plt.savefig(frame_path, bbox_inches="tight", dpi=600)
        plt.close()


    def on_train_end(self, logs=None):
        if not self.generate_video:
            return
        # 原视频生成逻辑
        images = sorted(
            [f for f in os.listdir(self.frame_dir) if f.endswith(".png")],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        if not images:
            print("[Callback] video error")
            return
        first_frame = cv2.imread(os.path.join(self.frame_dir, images[0]))
        height, width, _ = first_frame.shape
        video_path = os.path.join(self.output_dir, "prediction_evolution.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        for img in images:
            video.write(cv2.imread(os.path.join(self.frame_dir, img)))
        video.release()
        print(f"video at: {video_path}")

def set_seed(seed=2026):
    """Set global random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

def get_flops(model, model_inputs=None, batch_size=1) -> float:
    """
    计算 tf.keras.Model 的 FLOPS [MFLOPs]
    """
    if hasattr(model, 'generator'):
        model = model.generator

    if not isinstance(model, (tf.keras.models.Sequential, tf.keras.models.Model)):
        raise ValueError(
            "Calculating FLOPS is only supported for `tf.keras.Model` and `tf.keras.Sequential` instances.")

    # 如果没有手动传入输入，自动从模型提取
    if model_inputs is None:
        model_inputs = model.inputs

    # 1. 动态拦截并屏蔽 TF 底层烦人的 Deprecation Warning
    tf_logger = tf.get_logger()
    original_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    try:
        # 获取输入规范 (TensorSpec)
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in model_inputs
        ]

        # 转换为 Concrete Function 并冻结计算图
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

        # 2. 【核心修复】：开辟独立的沙盒 Graph 环境进行静态图分析
        # 这完美避开了 TF2 动态图环境的冲突，且无需使用危险的 reset_default_graph()
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()

            # 配置 Profiler，强制设为静默输出 (避免在控制台打印一长串内部信息)
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts['output'] = 'none'

            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )

            if flops is None:
                return 0.0

            return flops.total_float_ops / 1e6

    finally:
        # 3. 无论计算是否成功，严格恢复原来的日志打印级别
        tf_logger.setLevel(original_level)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

def load_and_merge_real_data(base_path):
    """
    Read 8 sub-files, merge specified variables, and split into 5 parts
    读取 8 个子文件，合并指定变量，并切分为 5 份
    """
    materials = ['PC', 'PE', 'PET', 'PP', 'PS', 'PVC', 'PMMA', 'PTFE']
    file_prefix = 'measured_dataset_'
    data_parts = [[] for _ in range(5)]

    print("=" * 60)
    print(">>> Step 1: Merging and splitting authentic data from 8 source files...")
    print("=" * 60)

    total_samples_count = 0
    for mat in materials:
        var_name = f'{mat}'
        mat_raw_list = []
        for i in range(1, 9):
            file_path = os.path.join(base_path, f'{file_prefix}{i}.mat')
            if not os.path.exists(file_path):
                continue
            try:
                data = sio.loadmat(file_path)
                if var_name in data:
                    mat_raw_list.append(data[var_name])
            except Exception as e:
                print(f"[Error] Reading {file_path}: {e}")

        if not mat_raw_list:
            continue

        full_mat_data = np.concatenate(mat_raw_list, axis=0)
        total_samples_count += full_mat_data.shape[0]

        for idx in range(5):
            split_data = full_mat_data[idx::5]
            data_parts[idx].append(split_data)

    final_parts = [np.concatenate(p, axis=0) for p in data_parts]
    print("-" * 60)
    print(f">>> Data preparation complete. Total samples: {total_samples_count}")
    print("-" * 60)
    return final_parts

def load_data(path):
    """Load .mat or .npy file"""
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.npy':
        data = np.load(path, allow_pickle=True)
        return {'test_data': data}
    elif ext == '.mat':
        mat_data = sio.loadmat(path)
        numpy_data = {}
        for key, value in mat_data.items():
            if not key.startswith('__'):
                numpy_data[key] = value.toarray() if 'scipy.sparse' in str(type(value)) else np.array(value)
        return numpy_data
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def load_custom_objects(model_type):
    """Load custom layers and loss functions"""
    if model_type in ['ResUNet', 'SE_ResNet_classification']:
        return {
            'se_block': SE_ResUNet.se_block,
            'residual_block': SE_ResUNet.se_residual_block,
            'encoder_block': SE_ResUNet.se_encoder_block,
            'decoder_block': SE_ResUNet.se_decoder_block
        }
    return None

def built_model(model_type, args):
    """Build model based on type"""
    se_custom = {
        'se_block': SE_ResUNet.se_block,
        'residual_block': SE_ResUNet.se_residual_block,
        'encoder_block': SE_ResUNet.se_encoder_block,
        'decoder_block': SE_ResUNet.se_decoder_block
    }
    model_map = {
        'ResUNet': (SE_ResUNet.ResUNet, se_custom),
        'SE_ResNet_classification': (SE_ResUNet.seResNet_classification, se_custom),
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    constructor, custom_objects = model_map[model_type]
    model = constructor(args.num_features, 1, args)
    return model, custom_objects

class LogCoshError(tf.keras.metrics.Metric):
    """Custom LogCosh Metric"""
    def __init__(self, name="log_cosh_error", **kwargs):
        super(LogCoshError, self).__init__(name=name, **kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = y_pred - y_true
        log_cosh_val = tf.math.log((tf.exp(error) + tf.exp(-error)) / 2)
        loss = tf.reduce_sum(log_cosh_val)
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=log_cosh_val.dtype)
        self.total.assign_add(loss)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total / self.count

def calculate_metrics(y_true, y_pred):
    """Calculate reconstruction metrics"""
    if y_true.ndim == 3:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

    r2_scores = [r2_score(y_true[i], y_pred[i]) for i in range(len(y_true))]
    mean_r2 = np.mean(r2_scores)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    dot_products = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)
    denom = norm_true * norm_pred
    safe_denom = np.where(denom == 0, 1.0, denom)
    cos_sims = dot_products / safe_denom

    zero_mask = (denom == 0)
    if np.any(zero_mask):
        both_zero = (norm_true == 0) & (norm_pred == 0)
        cos_sims[both_zero] = 1.0
        one_zero = zero_mask & (~both_zero)
        cos_sims[one_zero] = 0.0
    mean_cos_sim = np.mean(cos_sims)

    def stable_log_cosh(x):
        x = np.abs(x)
        return np.where(x < 50, np.log(np.cosh(x)), x - np.log(2.0))
    logcosh = np.mean(stable_log_cosh(y_pred - y_true))

    return mean_r2, mse, rmse, mae, mape, mean_cos_sim, logcosh

# ================= Metrics Functions / 评估指标函数 =================
def calculate_base_metrics(y_true, y_pred):
    """Calculate R2, MSE, RMSE, MAE, CS, Log-Cosh / 计算基础回归指标"""
    if y_true.ndim == 3:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

    r2_scores = [r2_score(y_true[i], y_pred[i]) for i in range(len(y_true))]
    mean_r2 = np.mean(r2_scores)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    dot_products = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)
    denom = norm_true * norm_pred
    safe_denom = np.where(denom == 0, 1.0, denom)
    cos_sims = dot_products / safe_denom

    zero_mask = (denom == 0)
    if np.any(zero_mask):
        both_zero = (norm_true == 0) & (norm_pred == 0)
        cos_sims[both_zero] = 1.0
        one_zero = zero_mask & (~both_zero)
        cos_sims[one_zero] = 0.0
    mean_cos_sim = np.mean(cos_sims)

    delta = y_pred - y_true
    def stable_log_cosh(x):
        x = np.abs(x)
        return np.where(x < 50, np.log(np.cosh(x)), x - np.log(2.0))
    logcosh = np.mean(stable_log_cosh(delta))

    return mean_r2, mse, rmse, mae, mean_cos_sim, logcosh

def sam(x_0, x):
    """Spectral Angle Mapper / 光谱角映射"""
    n = x.shape[0]
    sam_all = np.zeros(n)
    for i in range(n):
        target, reference = x[i, :], x_0[i, :]
        denominator = np.sqrt(np.sum(target ** 2)) * np.sqrt(np.sum(reference ** 2))
        if denominator == 0:
            sam_all[i] = np.nan
        else:
            cos_theta = np.clip(np.sum(target * reference) / denominator, -1.0, 1.0)
            sam_all[i] = np.arccos(cos_theta)
    return np.nanmean(sam_all)

def snr(x_0, x):
    """Signal-to-Noise Ratio / 信噪比"""
    n = x.shape[0]
    snr_all = np.zeros(n)
    for i in range(n):
        denoised_signal, original_signal = x[i, :], x_0[i, :]
        noise = original_signal - denoised_signal
        signal_power, noise_power = np.sum(denoised_signal ** 2), np.sum(noise ** 2)

        if noise_power == 0:
            snr_all[i] = np.nan if signal_power == 0 else np.inf
        else:
            ratio = signal_power / noise_power
            snr_all[i] = -np.inf if ratio == 0 else 10 * np.log10(ratio)
    return np.nanmean(snr_all)

def ssim_1d(x, y, win_size=3):
    """1D Structural Similarity Index / 一维结构相似度"""
    n_samples, _ = x.shape
    data_ranges = np.max(np.maximum(x, y), axis=1) - np.min(np.minimum(x, y), axis=1)
    C1, C2 = (0.01 * data_ranges) ** 2, (0.03 * data_ranges) ** 2
    ssim_all = np.zeros(n_samples)

    for i in range(n_samples):
        xi, yi, c1, c2 = x[i], y[i], C1[i], C2[i]
        mu_x = uniform_filter1d(xi, size=win_size, mode='nearest')
        mu_y = uniform_filter1d(yi, size=win_size, mode='nearest')
        sigma_x_sq = uniform_filter1d(xi ** 2, win_size, mode='nearest') - mu_x ** 2
        sigma_y_sq = uniform_filter1d(yi ** 2, win_size, mode='nearest') - mu_y ** 2
        sigma_xy = uniform_filter1d(xi * yi, win_size, mode='nearest') - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-10
        ssim_all[i] = np.mean(numerator / denominator)
    return np.mean(ssim_all)

def psnr_1d(x_0, x):
    """Peak Signal-to-Noise Ratio / 峰值信噪比"""
    mse_array = np.mean((x_0 - x) ** 2, axis=1)
    data_ranges = np.max(x_0, axis=1) - np.min(x_0, axis=1)
    psnr_array = 10 * np.log10((data_ranges ** 2) / (mse_array + 1e-10))
    return np.nanmean(psnr_array)

def evaluate_all_metrics(y_true, y_pred):
    """Evaluate all required metrics / 计算所有所需指标"""
    if y_true.ndim == 3:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

    r2, mse, rmse, mae, cs, logcosh = calculate_base_metrics(y_true, y_pred)
    sam_val = sam(y_true, y_pred)
    snr_val = snr(y_true, y_pred)
    ssim_val = ssim_1d(y_pred, y_true)
    psnr_val = psnr_1d(y_true, y_pred)

    return {
        'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'CS': cs,
        'Log-Cosh': logcosh, 'SSIM': ssim_val, 'PSNR': psnr_val,
        'SNR': snr_val, 'SAM': sam_val
    }
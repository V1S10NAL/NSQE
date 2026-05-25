"""
Natural Spectra Quality Evaluator (NSQE).
A method for evaluating the authenticity of simulated Raman spectra.
It assesses authenticity by calculating the Mahalanobis distance between the spectrum dataset under evaluation and a reference multivariate Gaussian model.
This method aims to objectively evaluate the authenticity of simulated datasets
by quantifying the statistical distance between the statistical distributions of the tested spectra and those of authentic spectra.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.special as sp
from scipy.stats import weibull_min
from scipy.ndimage import convolve1d
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

def compute_dispersion_index(vec):
    """
    Args:
        vec (np.ndarray): 输入的一维特征向量 / Input 1D feature vector
    Returns:
        float: 计算得到的离散指数 / Calculated Dispersion Index
    """
    vec = np.abs(vec)
    mu = np.mean(vec)
    sigma = np.std(vec)
    if mu < 1e-6:
        return 0.0
    return sigma / mu


def extract_dispersion_features(signal):
    """
    Args:
        signal (np.ndarray): 一维光谱信号数据 / 1D spectral signal data
    Returns:
        list: 包含原始信号和梯度离散指数的特征列表 / List of dispersion features containing raw signal and gradient indices
    """
    di_raw = compute_dispersion_index(signal)
    grad = np.diff(signal)
    di_grad = compute_dispersion_index(grad)
    return [di_raw, di_grad]


def extract_log_gabor_features(mscn, n_scales=3):
    """
    Args:
        mscn (np.ndarray): 经MSCN处理后的归一化系数向量 / Normalized coefficient vector after MSCN processing
        n_scales (int, optional): Log-Gabor滤波器的尺度数量，可选参数，默认值为3 / Number of scales for Log-Gabor filter, optional parameter, default is 3
    Returns:
        list: 包含多个尺度下形状和尺度参数的特征列表 / List of features containing shape and scale parameters across multiple scales
    """
    N = len(mscn)
    f = np.fft.fftfreq(N)
    f[0] = 1e-8
    features = []
    center_freqs = np.geomspace(0.05, 0.4, n_scales)
    sigma_f = 0.5
    spectrum = np.fft.fft(mscn)
    for w0 in center_freqs:
        mask = f > 0
        log_term = np.log(np.abs(f[mask]) / w0)
        filt = np.zeros(N)
        filt[mask] = np.exp(- (log_term ** 2) / (2 * (np.log(sigma_f) ** 2)))
        response_f = spectrum * filt
        response = np.real(np.fft.ifft(response_f))
        alpha, sigma_sq = estimate_ggd_params(response)
        features.extend([alpha, sigma_sq])
    return features


def extract_gradient_features(signal):
    """
    Args:
        signal (np.ndarray): 一维光谱信号数据 / 1D spectral signal data
    Returns:
        list: 包含二阶梯度Weibull分布的形状和尺度参数列表 / List containing shape and scale parameters of the second-order gradient Weibull distribution
    """
    grad = np.abs(np.diff(signal, n=2))
    shape, scale = estimate_weibull_param(grad)
    return [shape, scale]


def compute_mscn(signal, kernel_size=7):
    """
    Args:
        signal (np.ndarray): 一维光谱信号数据 / 1D spectral signal data
        kernel_size (int, optional): 高斯加权窗口的大小，可选参数，默认值为7 / Size of the Gaussian weighted window, optional parameter, default is 7
    Returns:
        np.ndarray: 计算得到的一维MSCN系数向量 / Calculated 1D MSCN coefficient vector
    """
    signal = signal.astype(np.float64)
    sigma = 7.0 / 6.0
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
    gauss /= np.sum(gauss)
    mu = convolve1d(signal, gauss, mode='reflect')
    mu_sq = mu * mu
    sigma_sq = convolve1d(signal * signal, gauss, mode='reflect') - mu_sq
    sigma_local = np.sqrt(np.maximum(sigma_sq, 0))
    mscn = (signal - mu) / (sigma_local + 1.0)
    return mscn


def estimate_ggd_params(vec):
    """
    Args:
        vec (np.ndarray): 待拟合的特征分布向量 / Feature distribution vector to be fitted
    Returns:
        tuple: 广义高斯分布的(形状参数, 尺度参数) / (Shape parameter, Scale parameter) of the Generalized Gaussian Distribution
    """
    gam = np.arange(0.2, 10.0, 0.001)
    r_gam = (sp.gamma(1.0 / gam) * sp.gamma(3.0 / gam)) / (sp.gamma(2.0 / gam) ** 2)
    sigma_sq = np.mean(vec ** 2)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / (E ** 2 + 1e-8)
    diff = np.abs(rho - r_gam)
    alpha = gam[np.argmin(diff)]
    sigma = np.sqrt(sigma_sq)
    beta = sigma * np.sqrt(sp.gamma(1.0 / alpha) / sp.gamma(3.0 / alpha))
    return alpha, beta


def estimate_aggd_params(vec):
    """
    Args:
        vec (np.ndarray): 待拟合的相邻系数乘积向量 / Pairwise product vector of adjacent coefficients to be fitted
    Returns:
        tuple: 非对称广义高斯分布的(均值, 左尺度参数, 右尺度参数, 形状参数) / (Mean, Left scale parameter, Right scale parameter, Shape parameter) of the Asymmetric Generalized Gaussian Distribution
    """
    gam, _ = estimate_ggd_params(np.abs(vec))
    left = vec[vec < 0]
    right = vec[vec >= 0]
    sigma_l_sq = np.mean(left ** 2) if len(left) > 0 else 0
    sigma_r_sq = np.mean(right ** 2) if len(right) > 0 else 0
    sigma_l = np.sqrt(sigma_l_sq)
    sigma_r = np.sqrt(sigma_r_sq)
    ratio = np.sqrt(sp.gamma(1.0 / gam) / sp.gamma(3.0 / gam))
    beta_l = sigma_l * ratio
    beta_r = sigma_r * ratio
    eta = (beta_r - beta_l) * sp.gamma(2.0 / gam) / sp.gamma(1.0 / gam)
    return eta, beta_l, beta_r, gam


def estimate_weibull_param(vec):
    """
    Args:
        vec (np.ndarray): 待拟合的非负特征向量 / Non-negative feature vector to be fitted
    Returns:
        tuple: Weibull分布的(形状参数, 尺度参数) / (Shape parameter, Scale parameter) of the Weibull Distribution
    """
    data = vec[vec > 0]
    if len(data) < 2:
        return 1.0, 1.0
    shape, loc, scale = weibull_min.fit(data, floc=0)
    return shape, scale


def extract_enriched_features(signal, config=None):
    """
    Args:
        signal (np.ndarray): 输入的单条一维光谱数据 / Input single 1D spectral data
        config (dict, optional): 控制各个特征模块开启状态的配置字典，可选参数 / Configuration dictionary to toggle individual feature modules, optional parameter
    Returns:
        np.ndarray: 提取并拼接好的统计特征向量 / Extracted and concatenated statistical feature vector
    """
    if config is None:
        config = {'mscn': True, 'mscn_pair': True, 'gradient': False, 'log_gabor': False, 'dispersion': False}
    feat_vec = []
    mscn = compute_mscn(signal)
    if config.get('mscn', True):
        alpha, sigma_sq = estimate_ggd_params(mscn)
        feat_vec.extend([alpha, sigma_sq])
    if config.get('mscn_pair', True):
        pair = mscn[:-1] * mscn[1:]
        aggd_params = estimate_aggd_params(pair)
        feat_vec.extend(aggd_params)
    if config.get('gradient', False):
        grad_params = extract_gradient_features(signal)
        feat_vec.extend(grad_params)
    if config.get('log_gabor', False):
        lg_params = extract_log_gabor_features(mscn, n_scales=3)
        feat_vec.extend(lg_params)
    if config.get('dispersion', False):
        di_params = extract_dispersion_features(signal)
        feat_vec.extend(di_params)
    return np.array(feat_vec)


def _extract_feature_wrapper(args):
    """
    Args:
        args (tuple): 包含(单条光谱数据, 特征配置字典)的元组 / Tuple containing (single spectral data, feature configuration dictionary)
    Returns:
        np.ndarray: 提取好的统计特征向量 / Extracted statistical feature vector
    """
    spectrum, config = args
    return extract_enriched_features(spectrum.flatten(), config)


class SpectralNSQE:
    def __init__(self, feature_config=None):
        """
        Args:
            feature_config (dict, optional): 初始化NSQE评估器时指定启用的特征配置，可选参数 / Feature configuration to enable specific modules when initializing the NSQE evaluator, optional parameter
        """
        self.mu_pris = None
        self.cov_pris = None
        default_config = {
            'mscn': True,
            'mscn_pair': True,
            'gradient': False,
            'log_gabor': False,
            'dispersion': False
        }
        if feature_config is None:
            self.feature_config = default_config
        elif isinstance(feature_config, dict):
            self.feature_config = default_config.copy()
            self.feature_config.update(feature_config)
        else:
            self.feature_config = default_config

    def fit(self, baseline_data):
        """
        Args:
            baseline_data (np.ndarray): 用于构建自然光谱MVG基准的真实数据矩阵，形状应为 (样本数, 光谱长度) / Real data matrix for building the natural spectrum MVG baseline, expected shape (n_samples, spectral_length)
        Returns:
            self: 训练完毕的评估器实例 / Fitted evaluator instance
        """
        if baseline_data.ndim == 1:
            baseline_data = baseline_data.reshape(1, -1)
        feats_list = []
        for i in range(baseline_data.shape[0]):
            f = extract_enriched_features(baseline_data[i], self.feature_config)
            feats_list.append(f)
        feats = np.array(feats_list)
        feats = np.nan_to_num(feats)
        self.mu_pris = np.mean(feats, axis=0)
        self.cov_pris = np.cov(feats, rowvar=False)
        return self

    def predict(self, test_data, max_workers=None):
        """
        Args:
            test_data (np.ndarray): 待评估的模拟/失真光谱数据矩阵，形状应为 (样本数, 光谱长度) / Simulated/distorted spectral data matrix to be evaluated, expected shape (n_samples, spectral_length)
            max_workers (int, optional): 多进程加速的工作线程数，不指定时将自动根据CPU计算，可选参数 / Number of worker threads for multiprocessing acceleration; computed automatically based on CPU if unspecified, optional parameter
        Returns:
            np.ndarray: 计算得到的每个测试样本的NSQE得分数组 / Array of computed NSQE scores for each test sample
        """
        if self.mu_pris is None:
            raise ValueError("Model not fitted")
        if test_data.ndim == 1:
            test_data = test_data.reshape(1, -1)
        n_samples = test_data.shape[0]
        if max_workers is None:
            total_cores = multiprocessing.cpu_count()
            limit_workers = max(1, int(total_cores * 0.3))
            max_workers = min(limit_workers, 6)
        task_args = ((test_data[i], self.feature_config) for i in range(n_samples))
        chunk_size = max(1, n_samples // (max_workers * 4))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_extract_feature_wrapper, task_args, chunksize=chunk_size)
            features_list = list(results)
        features = np.array(features_list)
        features = np.nan_to_num(features)
        diff = features - self.mu_pris
        inv_cov = np.linalg.pinv(self.cov_pris)
        temp = np.dot(diff, inv_cov)
        dist_sq = np.sum(temp * diff, axis=1)
        scores = np.sqrt(np.abs(dist_sq))
        return scores


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 1. Set the paths for the saved NPY files
    baseline_npy_path = r'.\measured_dataset\baseline_data.npy'
    test_npy_path = r'.\simulated_dataset\simulated_dataset_1.npy'

    # 2. Select the advanced feature configuration to build A-NSQE
    current_config = {
        'mscn': True,
        'mscn_pair': True,
        'gradient': True,     # Enable gradient features (Optional)
        'log_gabor': True,    # Enable frequency domain features (Optional)
        'dispersion': True    # Enable dispersion features (Optional)
    }

    # 3. Check if files exist, load data, and compute
    if os.path.exists(baseline_npy_path) and os.path.exists(test_npy_path):

        # Load real baseline data from NPY
        print(f"\nLoading baseline data from {baseline_npy_path}...")
        baseline_data = np.load(baseline_npy_path)

        # Load test data to be evaluated from NPY
        print(f"\nLoading test data from {test_npy_path}...")
        test_data = np.load(test_npy_path)

        # Data loading is complete. Start timer for MVG model building.
        t_mvg_start = time.perf_counter()

        # Build and fit the A-NSQE evaluator instance
        nsqe_evaluator = SpectralNSQE(feature_config=current_config)
        nsqe_evaluator.fit(baseline_data)

        # Stop MVG model building timer
        t_mvg_end = time.perf_counter()
        mvg_build_time = t_mvg_end - t_mvg_start
        print(f"\nA-NSQE Baseline Model successfully fitted.")
        print(f"\n-> Time taken to build MVG model: {mvg_build_time:.4f} seconds")
        print("-" * 50)
        # ========================================================

        # Execute batch prediction evaluation
        print("\nCalculating NSQE scores (this may take some time)...")

        # Start timer for calculating NSQE scores.
        t_nsqe_start = time.perf_counter()

        scores = nsqe_evaluator.predict(test_data)

        # Stop NSQE score calculation timer
        t_nsqe_end = time.perf_counter()
        nsqe_calc_time = t_nsqe_end - t_nsqe_start
        # ========================================================

        print(f"Evaluated {len(scores)} samples.")
        print(f"Mean A-NSQE score: {np.mean(scores):.4f}")
        print(f"-> Time taken to calculate NSQE scores: {nsqe_calc_time:.4f} seconds")
        print("-" * 50)

    else:
        print("Error: baseline_data.npy or test_data.npy not found. Please ensure the files are in the current directory.")

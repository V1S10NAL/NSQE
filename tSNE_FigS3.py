import os
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import tools

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def fspecial_gaussian_1d(window_size, sigma):
    half_size = window_size // 2
    x = np.linspace(-half_size, half_size, window_size)
    gauss = np.exp(-x**2 / (2 * sigma**2))
    return gauss / gauss.sum()

def compute_1d_mscn(spectrum, window_size=7):
    spectrum = spectrum.astype(np.float64)
    sigma = 7.0 / 6.0
    window = fspecial_gaussian_1d(window_size, sigma)
    pad_len = window_size // 2
    padded = np.pad(spectrum, pad_len, mode='edge')
    mu = np.convolve(padded, window, mode='valid')
    mu_sq = mu * mu
    sigma_sq = np.convolve(padded ** 2, window, mode='valid') - mu_sq
    sigma_local = np.sqrt(np.maximum(sigma_sq, 0))
    mscn = (spectrum - mu) / (sigma_local + 1.0)
    return mscn, mu, sigma_local

def compute_mscn_matrix(X, window_size=7):
    X_mscn = np.zeros_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        X_mscn[i], _, _ = compute_1d_mscn(X[i], window_size)
    return X_mscn

def _extract_feature_wrapper(args):
    spectrum, config = args
    return tools.extract_enriched_features(spectrum.flatten(), config)

def plot_confidence_ellipse(x, y, ax, confidence=0.95, **kwargs):
    if len(x) < 2 or len(y) < 2:
        return
    data = np.vstack((x, y)).T
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    cov += np.eye(2) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    chisquare_val = chi2.ppf(confidence, 2)
    width, height = 2 * np.sqrt(chisquare_val * eigenvalues)
    ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    base_path = r'.\measured_dataset'
    materials = ['PC', 'PE', 'PET', 'PP', 'PS', 'PVC', 'PMMA', 'PTFE']
    file_prefix = 'measured_dataset_'

    X_raw_list = []
    y_polymer_list = []
    first_spectra_dict = {}

    for cond_idx in range(1, 9):
        file_path = os.path.join(base_path, f'{file_prefix}{cond_idx}.mat')
        if not os.path.exists(file_path):
            continue

        try:
            data_mat = sio.loadmat(file_path)
            for poly_idx, mat_name in enumerate(materials):
                var_name = f'{mat_name}'
                if var_name in data_mat:
                    data = data_mat[var_name]

                    if cond_idx == 4 and mat_name not in first_spectra_dict:
                        first_spectra_dict[mat_name] = data[0].copy()

                    np.random.seed(2026)
                    select_idx = np.random.choice(data.shape[0], min(2400, data.shape[0]), replace=False)
                    selected_data = data

                    X_raw_list.append(selected_data)
                    y_polymer_list.append(np.full(selected_data.shape[0], poly_idx))
        except Exception as e:
            print(f" {file_path} error: {e}")

    X_raw = np.vstack(X_raw_list)
    y_polymer = np.concatenate(y_polymer_list)
    print(f"sample: {X_raw.shape[0]}")


    X_mscn = compute_mscn_matrix(X_raw, window_size=7)

    feature_config_full = {
            'mscn': True,
            'mscn_pair': True,
            'gradient': True,
            'log_gabor': True,
            'dispersion': True
        }


    n_samples = X_raw.shape[0]
    max_workers = min(max(1, int(multiprocessing.cpu_count() * 0.7)), 12)
    task_args = ((X_raw[i], feature_config_full) for i in range(n_samples))
    chunk_size = max(1, n_samples // (max_workers * 4))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        features_list = list(executor.map(_extract_feature_wrapper, task_args, chunksize=chunk_size))

    X_feat = np.nan_to_num(np.array(features_list))

    tsne_model = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=2026, n_jobs=-1)

    X_raw_tsne = tsne_model.fit_transform(StandardScaler().fit_transform(X_raw))
    X_mscn_tsne = tsne_model.fit_transform(StandardScaler().fit_transform(X_mscn))
    X_feat_tsne = tsne_model.fit_transform(StandardScaler().fit_transform(X_feat))

    fig_tsne, axes_tsne = plt.subplots(2, 2, figsize=(6, 6), facecolor='white')
    axes_tsne = axes_tsne.flatten()

    custom_colors = ['#95003E', '#EB6046', '#F8D61D', '#57B1AB', '#73A7D3', '#2A7BBF', '#584A99', '#292247']
    cmap_polymer = mcolors.ListedColormap(custom_colors)

    plot_point_size = 1
    plot_alpha = 0.5
    scatter_edge_alpha = 0.8
    scatter_edge_linewidth = 0.3
    ellipse_alpha = 0.2
    ellipse_linewidth = 0.5

    for idx, mat_name in enumerate(materials):
        mask = (y_polymer == idx)
        x_pts, y_pts = X_raw_tsne[mask, 0], X_raw_tsne[mask, 1]
        color = cmap_polymer(idx)
        face_color = (color[0], color[1], color[2], plot_alpha)
        edge_color = (color[0], color[1], color[2], scatter_edge_alpha)

        axes_tsne[0].scatter(x_pts, y_pts, c=[face_color], edgecolors=[edge_color], linewidths=scatter_edge_linewidth, label=mat_name, s=plot_point_size)
        plot_confidence_ellipse(x_pts, y_pts, axes_tsne[0], confidence=0.95, edgecolor=color, facecolor=color, alpha=ellipse_alpha, linewidth=ellipse_linewidth, zorder=0)

    axes_tsne[0].text(0.04, 0.96, "(a)", transform=axes_tsne[0].transAxes, fontweight='bold', fontsize=8, va='top', ha='left')
    axes_tsne[0].set_xlabel("t-SNE Dimension 1")
    axes_tsne[0].set_ylabel("t-SNE Dimension 2")
    axes_tsne[0].set_box_aspect(1)

    for idx, mat_name in enumerate(materials):
        mask = (y_polymer == idx)
        x_pts, y_pts = X_mscn_tsne[mask, 0], X_mscn_tsne[mask, 1]
        color = cmap_polymer(idx)
        face_color = (color[0], color[1], color[2], plot_alpha)
        edge_color = (color[0], color[1], color[2], scatter_edge_alpha)

        axes_tsne[1].scatter(x_pts, y_pts, c=[face_color], edgecolors=[edge_color], linewidths=scatter_edge_linewidth, label=mat_name, s=plot_point_size)
        plot_confidence_ellipse(x_pts, y_pts, axes_tsne[1], confidence=0.95, edgecolor=color, facecolor=color, alpha=ellipse_alpha, linewidth=ellipse_linewidth, zorder=0)

    axes_tsne[1].text(0.04, 0.96, "(b)", transform=axes_tsne[1].transAxes, fontweight='bold', fontsize=8, va='top', ha='left')
    axes_tsne[1].set_xlabel("t-SNE Dimension 1")
    axes_tsne[1].set_ylabel("t-SNE Dimension 2")
    axes_tsne[1].set_box_aspect(1)

    for idx, mat_name in enumerate(materials):
        mask = (y_polymer == idx)
        x_pts, y_pts = X_feat_tsne[mask, 0], X_feat_tsne[mask, 1]
        color = cmap_polymer(idx)
        face_color = (color[0], color[1], color[2], plot_alpha)
        edge_color = (color[0], color[1], color[2], scatter_edge_alpha)

        axes_tsne[2].scatter(x_pts, y_pts, c=[face_color], edgecolors=[edge_color], linewidths=scatter_edge_linewidth, label=mat_name, s=plot_point_size)
        plot_confidence_ellipse(x_pts, y_pts, axes_tsne[2], confidence=0.95, edgecolor=color, facecolor=color, alpha=ellipse_alpha, linewidth=ellipse_linewidth, zorder=0)

    axes_tsne[2].text(0.04, 0.96, "(c)", transform=axes_tsne[2].transAxes, fontweight='bold', fontsize=8, va='top', ha='left')
    axes_tsne[2].set_xlabel("t-SNE Dimension 1")
    axes_tsne[2].set_ylabel("t-SNE Dimension 2")
    axes_tsne[2].set_box_aspect(1)

    handles, labels = axes_tsne[0].get_legend_handles_labels()
    axes_tsne[3].axis('off')
    axes_tsne[3].legend(handles, labels, loc='center', frameon=False, fontsize=8, ncol=2, markerscale=3.0)

    plt.tight_layout(w_pad=1.5, h_pad=1.5)
    tsne_filename_base = 'tSNE_FigS3'
    plt.savefig(f'{tsne_filename_base}.png', dpi=1200, bbox_inches='tight', facecolor='w', transparent=False)
    plt.show()

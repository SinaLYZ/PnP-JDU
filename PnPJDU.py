#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunIndianPines experiments as a Python script instead of an ipynb notebook.

What this script does:
1. Loads Indian Pines data, endmembers, and abundance ground truth once.
2. Runs the experiment for all 6 kernels.
3. For each kernel, runs multiple independent Gaussian noise realizations.
4. Runs PnP-JDU, FCLS, UCLS, NNLS, and PnP.
5. Calculates RMSE, PSNR, SSIM, SAD, ERGAS, and abundance RMSE.
6. Saves all outputs under one output folder.
7. For each Kernel/seed folder, saves MAT files, RGBs, and abundance figures in separate subfolders.

Default output files:
    Output/all_runs_metrics.csv
    Output/summary_mean_std_metrics.csv
    Output/summary_mean_std_metrics_paper_format.csv
    Output/kernel_blur_diagnostics.csv
    Output/Kernel*/seed_*/...
      IndianPines_Kernel*_seed_XX_input.mat
    Output/kernel_blur_diagnostics.csv

Run example:
    python IndianPines_6Kernels_10NoiseRuns.py

Optional examples:
    python IndianPines_6Kernels_10NoiseRuns.py --num_runs 10
    python IndianPines_6Kernels_10NoiseRuns.py --methods FCLS UCLS NNLS
    python IndianPines_6Kernels_10NoiseRuns.py --save_figures
    python IndianPines_6Kernels_10NoiseRuns.py --save_figures --figure_methods PnP-JDU --figure_seed 0

Convolution mode:
    --conv_mode scico   : default; use the original SCICO CircularConvolve for Python results.
                          This is the correct mode for reproducing your original PUK metrics.
    --conv_mode matlab  : use MATLAB-style EigConvMat FFT convolution.
                          Use this only when you specifically want MATLAB ADMSolver-compatible data.
"""

from __future__ import annotations
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # important when running as .py on a server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import loadmat, savemat
from skimage.metrics import structural_similarity as ssim

import jax
import jax.numpy as jnp

import scico.random
import scico.numpy as snp
from scico import linop, loss, functional, plot
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

from spectral import get_rgb
import spectral.io.envi as envi

# These are imported here because they are used by the linear unmixing methods.
from pysptools.abundance_maps.amaps import FCLS, UCLS, NNLS


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

KERNEL_SPECS = [
    {"kernel_id": "Kernel1", "kernel_name": "Gaussian_A", "file": "kernel_1.mat", "noise_sigma": 0.01},
    {"kernel_id": "Kernel2", "kernel_name": "Gaussian_B", "file": "kernel_2.mat", "noise_sigma": 0.01},
    {"kernel_id": "Kernel3", "kernel_name": "Gaussian_C", "file": "kernel_3.mat", "noise_sigma": 0.03},
    # Circle is set to 0.01 directly, matching the paper's scenario.
    # This avoids the old notebook's double scaling: np.random.normal(scale=0.01) * (20/500).
    {"kernel_id": "Kernel4", "kernel_name": "Circle", "file": "kernel_4.mat", "noise_sigma": 0.01},
    {"kernel_id": "Kernel5", "kernel_name": "Motion", "file": "kernel_5.mat", "noise_sigma": 0.01},
    {"kernel_id": "Kernel6", "kernel_name": "Square", "file": "kernel_6.mat", "noise_sigma": 0.01},
]

DEFAULT_METHODS = ["PnP-JDU", "FCLS", "UCLS", "NNLS", "PnP"]

# Fixed RGB visualization settings.
# These match your separate RGB.py style:
# one fixed channel-wise RGB scale from the ground truth, applied to all methods.
RGB_P_LOW = 1
RGB_P_HIGH = 99


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def rmse_hsi(x_true: np.ndarray, x_est: np.ndarray) -> float:
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    return float(np.sqrt(np.mean((x_est - x_true) ** 2)))


def psnr_hsi(x_true: np.ndarray, x_est: np.ndarray, max_val: float = 1.0) -> float:
    """Global PSNR for normalized data in [0, 1]."""
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    mse = np.mean((x_est - x_true) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def ssim_hsi(x_true: np.ndarray, x_est: np.ndarray, data_range: float = 1.0) -> Tuple[float, np.ndarray]:
    """
    Average SSIM over all spectral bands, matching the paper's definition:
    calculate SSIM for each 2D band, then average over bands.
    """
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    if x_true.shape != x_est.shape:
        raise ValueError(f"SSIM shape mismatch: {x_true.shape} vs {x_est.shape}")

    n_bands = x_true.shape[-1]
    values = []
    for b in range(n_bands):
        values.append(
            ssim(
                x_true[:, :, b],
                x_est[:, :, b],
                data_range=data_range,
            )
        )
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), values


def sad_hsi(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """Mean spectral angle distance over all pixels, in radians."""
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)

    l_bands = x_true.shape[-1]
    t = x_true.reshape(-1, l_bands)
    e = x_est.reshape(-1, l_bands)

    dot = np.sum(t * e, axis=1)
    nt = np.linalg.norm(t, axis=1)
    ne = np.linalg.norm(e, axis=1)
    cosines = dot / (nt * ne + 1e-12)
    cosines = np.clip(cosines, -1.0, 1.0)
    return float(np.mean(np.arccos(cosines)))


def ergas_hsi(x_true: np.ndarray, x_est: np.ndarray) -> float:
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)
    h_l_ratio = 1.0

    rmse_per_band = np.sqrt(np.mean((x_est - x_true) ** 2, axis=(0, 1)))
    mean_ref_per_band = np.mean(x_true, axis=(0, 1))
    mean_ref_per_band = np.where(mean_ref_per_band == 0, 1e-10, mean_ref_per_band)

    return float(100.0 * h_l_ratio * np.sqrt(np.mean((rmse_per_band / mean_ref_per_band) ** 2)))


def abundance_rmse(a_true: np.ndarray, a_est: np.ndarray) -> float:
    a_true = np.asarray(a_true)
    a_est = np.asarray(a_est)
    if a_true.shape != a_est.shape:
        raise ValueError(f"Abundance shape mismatch: {a_true.shape} vs {a_est.shape}")
    return float(np.sqrt(np.mean((a_est - a_true) ** 2)))

def save_runtime_plots(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Save runtime comparison plots.

    Outputs:
        Output/runtime_plots/runtime_by_kernel_method.png
        Output/runtime_plots/runtime_by_method_overall.png
        Output/runtime_plots/runtime_by_kernel_method_logscale.png
    """
    if "runtime_seconds" not in df.columns:
        print("runtime_seconds column not found. Skipping runtime plots.")
        return

    runtime_dir = output_dir / "runtime_plots"
    ensure_dir(runtime_dir)

    # -------------------------------------------------------------------------
    # 1. Grouped bar plot: runtime for each method under each kernel
    # -------------------------------------------------------------------------
    runtime_summary = (
        df.groupby(["kernel_name", "method"])["runtime_seconds"]
        .agg(["mean", "std"])
        .reset_index()
    )

    kernels = runtime_summary["kernel_name"].unique()
    methods = runtime_summary["method"].unique()

    x = np.arange(len(kernels))
    width = 0.8 / len(methods)

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        method_data = runtime_summary[runtime_summary["method"] == method]

        means = []
        stds = []

        for kernel in kernels:
            row = method_data[method_data["kernel_name"] == kernel]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(row["mean"].values[0])
                stds.append(row["std"].values[0])

        plt.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method,
        )

    plt.xlabel("Kernel")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(x, kernels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(runtime_dir / "runtime_by_kernel_method.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Same grouped bar plot with log-scale y-axis
    # Useful when PnP/PnP-JDU are much slower than FCLS/UCLS/NNLS
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        method_data = runtime_summary[runtime_summary["method"] == method]

        means = []
        stds = []

        for kernel in kernels:
            row = method_data[method_data["kernel_name"] == kernel]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(row["mean"].values[0])
                stds.append(row["std"].values[0])

        plt.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method,
        )

    plt.xlabel("Kernel")
    plt.ylabel("Runtime (seconds, log scale)")
    plt.yscale("log")
    plt.xticks(x, kernels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(runtime_dir / "runtime_by_kernel_method_logscale.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Overall mean runtime per method across all kernels and seeds
    # -------------------------------------------------------------------------
    overall_runtime = (
        df.groupby("method")["runtime_seconds"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 6))
    plt.bar(
        overall_runtime["method"],
        overall_runtime["mean"],
        yerr=overall_runtime["std"],
        capsize=5,
    )

    plt.xlabel("Method")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(runtime_dir / "runtime_by_method_overall.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved runtime plots:")
    print(" -", runtime_dir / "runtime_by_kernel_method.png")
    print(" -", runtime_dir / "runtime_by_kernel_method_logscale.png")
    print(" -", runtime_dir / "runtime_by_method_overall.png")


def compute_metrics(
    method: str,
    kernel_id: str,
    kernel_name: str,
    seed: int,
    noise_sigma: float,
    x_true: np.ndarray,
    x_est: np.ndarray,
    a_true: np.ndarray | None = None,
    a_est: np.ndarray | None = None,
) -> Dict[str, float | str | int]:
    ssim_mean, _ = ssim_hsi(x_true, x_est, data_range=1.0)

    row = {
        "kernel_id": kernel_id,
        "kernel_name": kernel_name,
        "seed": int(seed),
        "noise_sigma": float(noise_sigma),
        "method": method,
        "RMSE": rmse_hsi(x_true, x_est),
        "PSNR": psnr_hsi(x_true, x_est, max_val=1.0),
        "SSIM": ssim_mean,
        "SAD": sad_hsi(x_true, x_est),
        "ERGAS": ergas_hsi(x_true, x_est),
    }

    if a_true is not None and a_est is not None:
        row["aRMSE_abundance"] = abundance_rmse(a_true, a_est)
    else:
        row["aRMSE_abundance"] = np.nan

    return row


# -----------------------------------------------------------------------------
# Plotting and saving helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hsi_to_matlab_LN(x_hwl: np.ndarray) -> np.ndarray:
    """Convert H x W x L HSI data to MATLAB format L x N."""
    x_hwl = np.asarray(x_hwl)
    h, w, l_bands = x_hwl.shape
    return np.transpose(x_hwl, (2, 0, 1)).reshape((l_bands, h * w), order="F")


def abundance_to_matlab_pN(a_hwp: np.ndarray) -> np.ndarray:
    """Convert H x W x p abundance maps to MATLAB format p x N."""
    a_hwp = np.asarray(a_hwp)
    h, w, p = a_hwp.shape
    return np.transpose(a_hwp, (2, 0, 1)).reshape((p, h * w), order="F")


def matlab_style_eigconv(kernel: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the circular convolution FFT exactly like the MATLAB ADMSolver runner:

        ConvMat = zeros(H, W)
        ConvMat(1:kh, 1:kw) = kernel
        ConvMat = circshift(ConvMat, -round([(kh - 1)/2, (kw - 1)/2]))
        EigConvMat = fft2(ConvMat)

    Returns EigConvMat, ConvMat, and the normalized small kernel.
    """
    k = np.asarray(kernel, dtype=np.float64).squeeze()

    if k.ndim != 2:
        raise ValueError(f"Kernel must be 2D after squeeze. Got shape {k.shape}")

    kh, kw = k.shape
    if kh > h or kw > w:
        raise ValueError(f"Kernel size {kh}x{kw} is larger than image size {h}x{w}")

    ksum = np.sum(k)
    if abs(ksum) < 1e-12:
        raise ValueError("Kernel sum is zero. Cannot normalize kernel.")

    kernel_norm = k / ksum

    conv_mat = np.zeros((h, w), dtype=np.float64)
    conv_mat[:kh, :kw] = kernel_norm

    shift_h = -int(np.round((kh - 1) / 2))
    shift_w = -int(np.round((kw - 1) / 2))
    conv_mat = np.roll(conv_mat, shift=(shift_h, shift_w), axis=(0, 1))

    eig_conv_mat = np.fft.fft2(conv_mat)
    return eig_conv_mat, conv_mat, kernel_norm


def blur_hsi_matlab_style_np(x_hwl: np.ndarray, eig_conv_mat: np.ndarray) -> np.ndarray:
    """Apply MATLAB-compatible circular blur to H x W x L HSI data."""
    x_hwl = np.asarray(x_hwl)
    return np.real(
        np.fft.ifft2(
            np.fft.fft2(x_hwl, axes=(0, 1)) * eig_conv_mat[:, :, None],
            axes=(0, 1),
        )
    )


def make_jax_blur_operator(eig_conv_mat: np.ndarray):
    """Return a JAX blur operator using the same FFT convention as MATLAB ADMSolver."""
    eig_jax = jnp.asarray(eig_conv_mat)

    def conv_operator(x):
        return jnp.real(
            jnp.fft.ifft2(
                jnp.fft.fft2(x, axes=(0, 1)) * eig_jax[:, :, None],
                axes=(0, 1),
            )
        )

    return conv_operator


def get_fixed_rgb_scale(x_ref: np.ndarray, bands: list[int], p_low: float = RGB_P_LOW, p_high: float = RGB_P_HIGH):
    """
    Compute fixed channel-wise RGB scaling from the ground-truth cube.

    This matches the logic in your separate RGB.py script:
        low/high are computed from Ground Truth only,
        then applied to Ground Truth, noisy image, and every method.
    """
    rgb_ref = np.asarray(x_ref)[..., bands].astype(np.float64)
    low = np.percentile(rgb_ref.reshape(-1, 3), p_low, axis=0)
    high = np.percentile(rgb_ref.reshape(-1, 3), p_high, axis=0)
    high = np.where(np.abs(high - low) < 1e-12, low + 1e-12, high)
    return low, high


def make_rgb_fixed(x: np.ndarray, bands: list[int], low: np.ndarray, high: np.ndarray):
    """
    Convert an HSI cube to RGB using fixed channel-wise low/high values.
    """
    rgb = np.asarray(x)[..., bands].astype(np.float64)
    rgb = (rgb - low.reshape(1, 1, 3)) / (high.reshape(1, 1, 3) - low.reshape(1, 1, 3))
    return np.clip(rgb, 0, 1)


def save_method_output_npz(
    out_file: Path,
    method: str,
    kernel_id: str,
    kernel_name: str,
    seed: int,
    x_hat: np.ndarray,
    abundance_hat: np.ndarray,
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    rgb_indices: Dict[str, int],
) -> None:
    """
    Save numerical outputs for your separate fixed-scale RGB.py script.

    Output path:
        Output/Kernel*/seed_00/method_outputs/METHOD_Kernel*_seed_00.npz
    """
    ensure_dir(out_file.parent)

    np.savez_compressed(
        out_file,
        method=method,
        kernel_id=kernel_id,
        kernel_name=kernel_name,
        seed=int(seed),
        x_hat=np.asarray(x_hat, dtype=np.float32),
        abundance_hat=np.asarray(abundance_hat, dtype=np.float32),
        x_true=np.asarray(x_true, dtype=np.float32),
        y_noisy=np.asarray(y_noisy, dtype=np.float32),
        rgb_R=int(rgb_indices["R"]),
        rgb_G=int(rgb_indices["G"]),
        rgb_B=int(rgb_indices["B"]),
    )


def save_reference_output_npz(
    out_file: Path,
    kernel_id: str,
    kernel_name: str,
    seed: int,
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    rgb_indices: Dict[str, int],
) -> None:
    """
    Save Ground Truth and Blurred+Noisy input for your separate fixed-scale RGB.py script.
    """
    ensure_dir(out_file.parent)

    np.savez_compressed(
        out_file,
        method="Reference",
        kernel_id=kernel_id,
        kernel_name=kernel_name,
        seed=int(seed),
        x_true=np.asarray(x_true, dtype=np.float32),
        y_noisy=np.asarray(y_noisy, dtype=np.float32),
        rgb_R=int(rgb_indices["R"]),
        rgb_G=int(rgb_indices["G"]),
        rgb_B=int(rgb_indices["B"]),
    )


def save_rgb_triplet(
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    x_est: np.ndarray,
    rgb_indices: Dict[str, int],
    title_est: str,
    out_file: Path,
) -> None:
    ensure_dir(out_file.parent)

    bands = [rgb_indices["R"], rgb_indices["G"], rgb_indices["B"]]
    rgb_true = get_rgb(np.asarray(x_true), bands)
    rgb_noisy = get_rgb(np.asarray(y_noisy), bands)
    rgb_est = get_rgb(np.asarray(x_est), bands)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_true)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(rgb_noisy)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(rgb_est)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_rgb_reconstruction_only(
    x_est: np.ndarray,
    rgb_indices: Dict[str, int],
    title: str,
    out_file: Path,
    rgb_low: np.ndarray | None = None,
    rgb_high: np.ndarray | None = None,
) -> None:
    """
    Save only the reconstructed RGB image.

    If rgb_low/rgb_high are provided, this uses the same fixed percentile scale
    as your separate RGB.py script. This prevents the darker-image issue caused
    by direct spectral.get_rgb scaling.
    """
    ensure_dir(out_file.parent)

    bands = [rgb_indices["R"], rgb_indices["G"], rgb_indices["B"]]

    if rgb_low is not None and rgb_high is not None:
        rgb_est = make_rgb_fixed(x_est, bands, rgb_low, rgb_high)
    else:
        # Fallback: old quick visualization style.
        rgb_est = get_rgb(np.asarray(x_est), bands)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_est)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_abundance_maps(a_est: np.ndarray, out_file: Path, title: str) -> None:
    ensure_dir(out_file.parent)
    a_est = np.asarray(a_est)
    num_endmembers = a_est.shape[2]
    ncols = 4
    nrows = int(np.ceil(num_endmembers / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()

    for i in range(num_endmembers):
        ax = axes[i]
        im = ax.imshow(a_est[:, :, i], cmap="viridis", vmin=0, vmax=1)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(num_endmembers, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_single_abundance_map(
    a_est: np.ndarray,
    endmember_number: int,
    out_file: Path,
    title: str,
) -> None:
    """
    Save one abundance map only.

    endmember_number is 1-based, so endmember_number=4 saves a_est[:, :, 3].
    """
    ensure_dir(out_file.parent)
    a_est = np.asarray(a_est)
    endmember_index = endmember_number - 1

    if endmember_index < 0 or endmember_index >= a_est.shape[2]:
        raise ValueError(
            f"Requested endmember {endmember_number}, but abundance map has only {a_est.shape[2]} endmembers."
        )

    plt.figure(figsize=(6, 6))
    im = plt.imshow(a_est[:, :, endmember_index], cmap="viridis", vmin=0, vmax=1)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_matlab_input(
    out_file: Path,
    y_noisy: np.ndarray,
    x_true: np.ndarray,
    endmember: np.ndarray,
    abundance_gt: np.ndarray,
    y_blur_clean: np.ndarray | None = None,
    kernel_original: np.ndarray | None = None,
    kernel_norm: np.ndarray | None = None,
    eig_conv_mat: np.ndarray | None = None,
    conv_mat: np.ndarray | None = None,
    y_blur_clean_scico: np.ndarray | None = None,
    y_blur_clean_matlab: np.ndarray | None = None,
    conv_mode: str | None = None,
    matlab_forward_model_matches_y: bool | None = None,
    kernel_id: str | None = None,
    kernel_name: str | None = None,
    seed: int | None = None,
    noise_sigma: float | None = None,
) -> None:
    """
    Save data in MATLAB-style format L x N and p x N.

    This preserves your original PUK variables:
        y, y_clean, M, A_gt, H, W, L, p

    It also optionally saves the additional variables needed by the new
    MATLAB-compatible ADMSolver workflow:
        y_blur_clean, EigConvMat, ConvMat, kernel, kernel_norm,
        noise_sigma, kernel_id, kernel_name, seed
    """
    ensure_dir(out_file.parent)

    y_noisy = np.asarray(y_noisy, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    endmember = np.asarray(endmember, dtype=np.float64)
    abundance_gt = np.asarray(abundance_gt, dtype=np.float64)

    h, w, l_bands = y_noisy.shape
    p = endmember.shape[1]

    data = {
        "y": hsi_to_matlab_LN(y_noisy),
        "y_clean": hsi_to_matlab_LN(x_true),
        "M": endmember,
        "A_gt": abundance_to_matlab_pN(abundance_gt),
        "H": h,
        "W": w,
        "L": l_bands,
        "p": p,
    }

    if y_blur_clean is not None:
        data["y_blur_clean"] = hsi_to_matlab_LN(np.asarray(y_blur_clean, dtype=np.float64))
    if y_blur_clean_scico is not None:
        data["y_blur_clean_scico"] = hsi_to_matlab_LN(np.asarray(y_blur_clean_scico, dtype=np.float64))
    if y_blur_clean_matlab is not None:
        data["y_blur_clean_matlab"] = hsi_to_matlab_LN(np.asarray(y_blur_clean_matlab, dtype=np.float64))
    if conv_mode is not None:
        data["conv_mode"] = conv_mode
    if matlab_forward_model_matches_y is not None:
        data["matlab_forward_model_matches_y"] = bool(matlab_forward_model_matches_y)
    if kernel_original is not None:
        data["kernel"] = np.asarray(kernel_original, dtype=np.float64)
    if kernel_norm is not None:
        data["kernel_norm"] = np.asarray(kernel_norm, dtype=np.float64)
    if eig_conv_mat is not None:
        data["EigConvMat"] = eig_conv_mat
    if conv_mat is not None:
        data["ConvMat"] = conv_mat
    if noise_sigma is not None:
        data["noise_sigma"] = float(noise_sigma)
    if kernel_id is not None:
        data["kernel_id"] = kernel_id
    if kernel_name is not None:
        data["kernel_name"] = kernel_name
    if seed is not None:
        data["seed"] = int(seed)

    savemat(out_file, data)


def save_reference_data_mat(
    out_file: Path,
    ground_truth: np.ndarray,
    y_blur_clean: np.ndarray,
    y_noisy: np.ndarray,
    endmember: np.ndarray,
    abundance_gt: np.ndarray,
    kernel_original: np.ndarray,
    kernel_norm: np.ndarray,
    eig_conv_mat: np.ndarray,
    conv_mat: np.ndarray,
    kernel_id: str,
    kernel_name: str,
    seed: int,
    noise_sigma: float,
    y_blur_clean_scico: np.ndarray | None = None,
    y_blur_clean_matlab: np.ndarray | None = None,
    conv_mode: str | None = None,
    matlab_forward_model_matches_y: bool | None = None,
) -> None:
    """
    Save reference HSI data separately from method results.

    This file is for checking/plotting the forward model:
        ground_truth  : clean reconstructed HSI, H x W x L
        y_blur_clean  : convolved clean HSI before noise, H x W x L
        y_noisy       : convolved + noisy HSI, H x W x L

    It also stores MATLAB-format versions:
        ground_truth_LN, y_blur_clean_LN, y_noisy_LN
    """
    ensure_dir(out_file.parent)

    ground_truth = np.asarray(ground_truth)
    y_blur_clean = np.asarray(y_blur_clean)
    y_noisy = np.asarray(y_noisy)
    y_blur_clean_scico = None if y_blur_clean_scico is None else np.asarray(y_blur_clean_scico)
    y_blur_clean_matlab = None if y_blur_clean_matlab is None else np.asarray(y_blur_clean_matlab)

    ref_data = {
            "ground_truth": ground_truth,
            "y_blur_clean": y_blur_clean,
            "y_noisy": y_noisy,
            "ground_truth_LN": hsi_to_matlab_LN(ground_truth),
            "y_blur_clean_LN": hsi_to_matlab_LN(y_blur_clean),
            "y_noisy_LN": hsi_to_matlab_LN(y_noisy),
            "endmember": np.asarray(endmember),
            "abundance_gt": np.asarray(abundance_gt),
            "abundance_gt_pN": abundance_to_matlab_pN(abundance_gt),
            "kernel": np.asarray(kernel_original),
            "kernel_norm": np.asarray(kernel_norm),
            "EigConvMat": eig_conv_mat,
            "ConvMat": conv_mat,
            "kernel_id": kernel_id,
            "kernel_name": kernel_name,
            "seed": int(seed),
            "noise_sigma": float(noise_sigma),
    }

    if y_blur_clean_scico is not None:
        ref_data["y_blur_clean_scico"] = y_blur_clean_scico
        ref_data["y_blur_clean_scico_LN"] = hsi_to_matlab_LN(y_blur_clean_scico)
    if y_blur_clean_matlab is not None:
        ref_data["y_blur_clean_matlab"] = y_blur_clean_matlab
        ref_data["y_blur_clean_matlab_LN"] = hsi_to_matlab_LN(y_blur_clean_matlab)
    if conv_mode is not None:
        ref_data["conv_mode"] = conv_mode
    if matlab_forward_model_matches_y is not None:
        ref_data["matlab_forward_model_matches_y"] = bool(matlab_forward_model_matches_y)

    savemat(out_file, ref_data)


def save_full_method_result_mat(
    out_file: Path,
    x_hat: np.ndarray,
    abundance_hat: np.ndarray,
    ground_truth: np.ndarray,
    abundance_gt: np.ndarray,
    endmember: np.ndarray,
    y_noisy: np.ndarray,
    y_blur_clean: np.ndarray,
    kernel_original: np.ndarray,
    kernel_norm: np.ndarray,
    eig_conv_mat: np.ndarray,
    conv_mat: np.ndarray,
    method: str,
    kernel_id: str,
    kernel_name: str,
    seed: int,
    noise_sigma: float,
    metrics_row: Dict,
    y_blur_clean_scico: np.ndarray | None = None,
    y_blur_clean_matlab: np.ndarray | None = None,
    conv_mode: str | None = None,
    matlab_forward_model_matches_y: bool | None = None,
) -> None:
    """Save the extra full .mat result file from the MATLAB-compatible workflow."""
    ensure_dir(out_file.parent)

    savemat(
        out_file,
        {
            "x_hat": np.asarray(x_hat),
            "abundance_hat": np.asarray(abundance_hat),
            "ground_truth": np.asarray(ground_truth),
            "abundance_gt": np.asarray(abundance_gt),
            "endmember": np.asarray(endmember),
            "y_noisy": np.asarray(y_noisy),
            "y_blur_clean": np.asarray(y_blur_clean),
            "y_blur_clean_scico": np.asarray(y_blur_clean_scico) if y_blur_clean_scico is not None else np.asarray([]),
            "y_blur_clean_matlab": np.asarray(y_blur_clean_matlab) if y_blur_clean_matlab is not None else np.asarray([]),
            "conv_mode": conv_mode if conv_mode is not None else "",
            "matlab_forward_model_matches_y": bool(matlab_forward_model_matches_y) if matlab_forward_model_matches_y is not None else False,
            "kernel": np.asarray(kernel_original),
            "kernel_norm": np.asarray(kernel_norm),
            "EigConvMat": eig_conv_mat,
            "ConvMat": conv_mat,
            "method": method,
            "kernel_id": kernel_id,
            "kernel_name": kernel_name,
            "seed": int(seed),
            "noise_sigma": float(noise_sigma),
            "metrics_RMSE": metrics_row.get("RMSE", np.nan),
            "metrics_PSNR": metrics_row.get("PSNR", np.nan),
            "metrics_SSIM": metrics_row.get("SSIM", np.nan),
            "metrics_SAD": metrics_row.get("SAD", np.nan),
            "metrics_ERGAS": metrics_row.get("ERGAS", np.nan),
            "metrics_aRMSE_abundance": metrics_row.get("aRMSE_abundance", np.nan),
            "runtime_seconds": metrics_row.get("runtime_seconds", np.nan),
            "noise_rmse": metrics_row.get("noise_rmse", np.nan),
        },
    )


# -----------------------------------------------------------------------------
# Methods
# -----------------------------------------------------------------------------

class SimplexProjection:
    has_eval = False
    has_prox = True

    def prox(self, v, lam=1.0, **kwargs):
        """Project every abundance vector onto the probability simplex."""
        v = snp.asarray(v, dtype=snp.float32)
        h, w, p = v.shape

        u = snp.sort(v, axis=-1)[..., ::-1]
        cssv = snp.cumsum(u, axis=-1)
        j = snp.arange(1, p + 1).reshape(1, 1, -1)

        cond = u - (cssv - 1) / j > 0
        rho = snp.sum(cond, axis=-1) - 1
        rho_clipped = snp.clip(rho, 0, p - 1).astype(snp.int32)

        batch_indices = snp.indices((h, w))
        cssv_rho = cssv[batch_indices[0], batch_indices[1], rho_clipped]
        theta = (cssv_rho - 1) / (rho + 1)
        return snp.maximum(v - theta[..., None], 0)


def run_admm_method(
    y_noisy: np.ndarray,
    endmember: np.ndarray,
    abundance_shape: Tuple[int, int, int],
    conv_operator,
    method: str,
    maxiter: int,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Run PnP-JDU or PnP and return reconstructed HSI and abundances."""
    if method not in {"PnP-JDU", "PnP"}:
        raise ValueError("method must be 'PnP-JDU' or 'PnP'")

    h, w, r = abundance_shape
    y_noisy_jax = jnp.asarray(y_noisy)
    endmember_jax = jnp.asarray(endmember)

    def afn_conv(a):
        return conv_operator(jnp.tensordot(a, endmember_jax.T, axes=([2], [0])))

    def afn_no_conv(a):
        return jnp.tensordot(a, endmember_jax.T, axes=([2], [0]))

    eval_fn = afn_conv if method == "PnP-JDU" else afn_no_conv
    rho_list = [1.5e-3, 2.0] if method == "PnP-JDU" else [1.5e-2, 2.0]

    a_operator = linop.LinearOperator(
        input_shape=(h, w, r),
        output_shape=y_noisy.shape,
        eval_fn=eval_fn,
    )

    f = loss.SquaredL2Loss(y=y_noisy_jax, A=a_operator)
    # lambda_l1 = 1.1e-2
    g1 = functional.DnCNN("17M")
    g2 = SimplexProjection()
    # g3 = lambda_l1 * functional.L1Norm()

    c1 = linop.Identity(input_shape=(h, w, r))
    c2 = linop.Identity(input_shape=(h, w, r))
    # c3 = linop.FiniteDifference(input_shape=(h, w, r), append=0)

    solver = ADMM(
        f=f,
        g_list=[g1, g2],
        C_list=[c1, c2],
        rho_list=rho_list,
        maxiter=maxiter,
        subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-8, "maxiter": 100}),
        itstat_options={"display": True, "period": 100},
    )

    print(f"Solving {method} on {device_info()} with maxiter={maxiter}")
    a_est = solver.solve()
    a_est = snp.clip(a_est, 0, 1)
    x_est = jnp.tensordot(a_est, endmember_jax.T, axes=([2], [0]))

    return np.asarray(x_est), np.asarray(a_est), solver


def run_linear_unmixing_method(
    y_noisy: np.ndarray,
    endmember: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run FCLS, UCLS, or NNLS from pysptools."""
    h, w, l_bands = y_noisy.shape
    y_flat = np.asarray(y_noisy).reshape(-1, l_bands)
    m = np.asarray(endmember).T  # p x L, as used in the notebooks

    if method == "FCLS":
        abundance_flat = FCLS(y_flat, m)
    elif method == "UCLS":
        abundance_flat = UCLS(y_flat, m)
    elif method == "NNLS":
        abundance_flat = NNLS(y_flat, m)
    else:
        raise ValueError("method must be FCLS, UCLS, or NNLS")

    x_hat_flat = abundance_flat @ m
    x_hat = x_hat_flat.reshape(h, w, l_bands)
    abundance_maps = abundance_flat.reshape(h, w, -1)

    return np.asarray(x_hat), np.asarray(abundance_maps)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_indian_pines(dataset_dir: Path) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    indian_pines = envi.open(str(dataset_dir / "indian_pines.hdr"))
    indian_pines_data = indian_pines.load().astype(np.float32)
    indian_pines_data = (indian_pines_data - np.min(indian_pines_data)) / (
        np.max(indian_pines_data) - np.min(indian_pines_data)
    )

    endmember_mat = loadmat(dataset_dir / "IndianPinesEndmembers.mat")
    endmember_raw = endmember_mat["endmembers"].astype(np.float32)
    endmember = (endmember_raw - endmember_raw.min()) / (endmember_raw.max() - endmember_raw.min())

    abundance_mat = loadmat(dataset_dir / "IndianPinesAbundancesGT.mat")
    abundance_gt = abundance_mat["abundanceMap"].astype(np.float32)

    ground_truth = np.asarray(jnp.tensordot(abundance_gt, endmember.T, axes=([2], [0])))

    print(indian_pines)
    print("IndianPines data shape:", indian_pines_data.shape)
    print("GroundTruthImage shape:", ground_truth.shape)
    print("Endmember shape:", endmember.shape)
    print("Abundance GT shape:", abundance_gt.shape)

    return indian_pines, ground_truth, endmember, abundance_gt


def get_rgb_band_indices(indian_pines, n_bands: int) -> Dict[str, int]:
    wavelengths = indian_pines.metadata.get("wavelength")
    if wavelengths is None:
        # fallback if wavelengths are missing
        return {"R": min(n_bands - 1, 100), "G": min(n_bands - 1, 60), "B": min(n_bands - 1, 30)}

    wavelengths = np.asarray(wavelengths, dtype=float)
    target_wavelengths = {"R": 650, "G": 550, "B": 450}
    return {
        color: int(np.argmin(np.abs(wavelengths - target)))
        for color, target in target_wavelengths.items()
    }


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_experiment(args) -> None:
    base_dir = Path(__file__).resolve().parent

    def resolve_path(path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()

    dataset_dir = resolve_path(args.dataset_dir)
    kernel_dir = resolve_path(args.kernel_dir)
    output_dir = resolve_path(args.output_dir)

    # Single output root. Everything is saved under output_dir.
    # Per-kernel/per-seed files are saved together in:
    #   output_dir / Kernel* / seed_*
    ensure_dir(output_dir)

    print("JAX devices:", jax.devices())
    print("Single output folder for all results:", output_dir.resolve())

    indian_pines, ground_truth, endmember, abundance_gt = load_indian_pines(dataset_dir)
    rgb_indices = get_rgb_band_indices(indian_pines, ground_truth.shape[-1])
    print("RGB band indices:", rgb_indices)

    rgb_bands = [rgb_indices["R"], rgb_indices["G"], rgb_indices["B"]]
    rgb_low, rgb_high = get_fixed_rgb_scale(ground_truth, rgb_bands, RGB_P_LOW, RGB_P_HIGH)
    print(f"Fixed RGB scale percentiles: low={rgb_low}, high={rgb_high}")

    methods = args.methods
    all_rows: List[Dict] = []
    diagnostic_rows: List[Dict] = []

    for kernel_spec in KERNEL_SPECS:
        kernel_id = kernel_spec["kernel_id"]
        kernel_name = kernel_spec["kernel_name"]
        kernel_file = kernel_dir / kernel_spec["file"]
        noise_sigma = float(kernel_spec["noise_sigma"])

        print("\n" + "=" * 80)
        print(f"Starting {kernel_id} / {kernel_name}: {kernel_file}, sigma={noise_sigma}")
        print("=" * 80)

        if not kernel_file.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file}")

        kernel_mat = scipy.io.loadmat(kernel_file)
        kernel = kernel_mat["kernel"]

        h, w, _ = ground_truth.shape

        # ------------------------------------------------------------------
        # SCICO convolution method.
        # This is your original PUK Python forward model and is the DEFAULT.
        # Use this for Python PnP-JDU/FCLS/UCLS/NNLS/PnP metric comparison.
        # ------------------------------------------------------------------
        kernel_3d = kernel[:, :, np.newaxis]
        conv_operator_scico = linop.CircularConvolve(h=kernel_3d, input_shape=ground_truth.shape)
        y_blurred_scico = np.asarray(conv_operator_scico(jnp.asarray(ground_truth))).astype(np.float32)

        # ------------------------------------------------------------------
        # MATLAB-compatible convolution method.
        # This is saved for diagnostics/ADMSolver compatibility. It is used
        # for the Python experiment only if you explicitly run:
        #     --conv_mode matlab
        # ------------------------------------------------------------------
        eig_conv_mat, conv_mat, kernel_norm = matlab_style_eigconv(kernel, h, w)
        y_blurred_matlab = blur_hsi_matlab_style_np(ground_truth, eig_conv_mat).astype(np.float32)

        if args.conv_mode == "scico":
            conv_operator = conv_operator_scico
            y_blurred = y_blurred_scico
            matlab_forward_model_matches_y = False
        elif args.conv_mode == "matlab":
            conv_operator = make_jax_blur_operator(eig_conv_mat)
            y_blurred = y_blurred_matlab
            matlab_forward_model_matches_y = True
        else:
            raise ValueError(f"Unknown conv_mode: {args.conv_mode}")

        rmse_scico_vs_matlab_blur = rmse_hsi(y_blurred_scico, y_blurred_matlab)
        blur_self_rmse = rmse_hsi(y_blurred, y_blurred)

        print(f"Convolution mode: {args.conv_mode}")
        print(f"RMSE(SCICO blur, MATLAB-style blur) = {rmse_scico_vs_matlab_blur:.12e}")
        if args.conv_mode == "scico":
            print("Using original SCICO CircularConvolve for Python metrics.")
            print("Note: saved EigConvMat is diagnostic only; it does NOT match y in scico mode.")
        else:
            print("Using MATLAB-style EigConvMat convolution for Python metrics and ADMSolver-compatible y.")

        diagnostic_rows.append(
            {
                "kernel_id": kernel_id,
                "kernel_name": kernel_name,
                "kernel_file": str(kernel_file),
                "conv_mode": args.conv_mode,
                "noise_sigma": noise_sigma,
                "kernel_sum_original": float(np.sum(kernel)),
                "kernel_sum_normalized": float(np.sum(kernel_norm)),
                "rmse_scico_vs_matlab_blur": rmse_scico_vs_matlab_blur,
                "matlab_forward_model_matches_y": bool(matlab_forward_model_matches_y),
                "blur_self_RMSE": blur_self_rmse,
            }
        )
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / "kernel_blur_diagnostics.csv", index=False)

        for seed in range(args.num_runs):
            print(f"\n--- {kernel_id}, seed {seed + 1}/{args.num_runs} ---")

            # Single folder for this kernel/seed.
            # This folder contains PUK-style outputs, ADMSolver input,
            # full method .mat results, and optional figures together.
            run_dir = output_dir / kernel_id / f"seed_{seed:02d}"
            ensure_dir(run_dir)

            # Keep outputs organized inside each kernel/seed folder.
            mat_dir = run_dir / "MAT_Files"
            rgb_dir = run_dir / "RGBs"
            abundance_fig_dir = run_dir / "Abundance_Figures"
            ensure_dir(mat_dir)
            ensure_dir(rgb_dir)
            ensure_dir(abundance_fig_dir)

            noise, _ = scico.random.randn(y_blurred.shape, seed=seed)
            noise = np.asarray(noise)
            y_noisy = np.asarray(y_blurred + noise_sigma * noise, dtype=np.float32)
            noise_rmse = rmse_hsi(y_noisy, y_blurred)
            print(f"Noise RMSE between y_noisy and y_blur_clean = {noise_rmse:.8f}")

            # Save reference .npz for your separate fixed-scale RGB.py script.
            # RGB.py reads Output/Kernel*/seed_00/method_outputs/*.npz.
            # We save seed_00 by default and also save figure_seed if different.
            if seed == 0 or seed == args.figure_seed:
                save_reference_output_npz(
                    out_file=run_dir / "method_outputs" / f"Reference_{kernel_id}_seed_{seed:02d}.npz",
                    kernel_id=kernel_id,
                    kernel_name=kernel_name,
                    seed=seed,
                    x_true=ground_truth,
                    y_noisy=y_noisy,
                    rgb_indices=rgb_indices,
                )

            if args.save_mat_inputs:
                # One combined MATLAB input file usable for both your PnP workflow
                # and MATLAB ADMSolver.
                #
                # It contains the original PUK variables:
                #   y, y_clean, M, A_gt, H, W, L, p
                # plus the ADMSolver/debug variables:
                #   y_blur_clean, EigConvMat, ConvMat, kernel, kernel_norm,
                #   noise_sigma, kernel_id, kernel_name, seed
                save_matlab_input(
                    mat_dir / f"IndianPines_{kernel_id}_seed_{seed:02d}_input.mat",
                    y_noisy=y_noisy,
                    x_true=ground_truth,
                    endmember=endmember,
                    abundance_gt=abundance_gt,
                    y_blur_clean=y_blurred,
                    kernel_original=kernel,
                    kernel_norm=kernel_norm,
                    eig_conv_mat=eig_conv_mat,
                    conv_mat=conv_mat,
                    y_blur_clean_scico=y_blurred_scico,
                    y_blur_clean_matlab=y_blurred_matlab,
                    conv_mode=args.conv_mode,
                    matlab_forward_model_matches_y=matlab_forward_model_matches_y,
                    kernel_id=kernel_id,
                    kernel_name=kernel_name,
                    seed=seed,
                    noise_sigma=noise_sigma,
                )

                # Separate reference file containing clean ground truth,
                # clean convolved HSI, and noisy convolved HSI.
                save_reference_data_mat(
                    mat_dir / f"ReferenceData_{kernel_id}_seed_{seed:02d}.mat",
                    ground_truth=ground_truth,
                    y_blur_clean=y_blurred,
                    y_noisy=y_noisy,
                    endmember=endmember,
                    abundance_gt=abundance_gt,
                    kernel_original=kernel,
                    kernel_norm=kernel_norm,
                    eig_conv_mat=eig_conv_mat,
                    conv_mat=conv_mat,
                    kernel_id=kernel_id,
                    kernel_name=kernel_name,
                    seed=seed,
                    noise_sigma=noise_sigma,
                    y_blur_clean_scico=y_blurred_scico,
                    y_blur_clean_matlab=y_blurred_matlab,
                    conv_mode=args.conv_mode,
                    matlab_forward_model_matches_y=matlab_forward_model_matches_y,
                )

            for method in methods:
                print(f"Running method: {method}")
                try:
                    runtime_start = time.perf_counter()

                    if method in {"PnP-JDU", "PnP"}:
                        x_hat, abundance_hat, solver = run_admm_method(
                            y_noisy=y_noisy,
                            endmember=endmember,
                            abundance_shape=abundance_gt.shape,
                            conv_operator=conv_operator,
                            method=method,
                            maxiter=args.maxiter,
                        )
                    elif method in {"FCLS", "UCLS", "NNLS"}:
                        x_hat, abundance_hat = run_linear_unmixing_method(
                            y_noisy=y_noisy,
                            endmember=endmember,
                            method=method,
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # Important for fair timing with JAX/GPU asynchronous execution.
                    if hasattr(x_hat, "block_until_ready"):
                        x_hat.block_until_ready()
                    if hasattr(abundance_hat, "block_until_ready"):
                        abundance_hat.block_until_ready()

                    runtime_seconds = time.perf_counter() - runtime_start

                    row = compute_metrics(
                        method=method,
                        kernel_id=kernel_id,
                        kernel_name=kernel_name,
                        seed=seed,
                        noise_sigma=noise_sigma,
                        x_true=ground_truth,
                        x_est=x_hat,
                        a_true=abundance_gt,
                        a_est=abundance_hat,
                    )

                    # Consistency check: x_hat should equal abundance_hat @ endmember.T.
                    x_from_abundance = np.asarray(
                        jnp.tensordot(jnp.asarray(abundance_hat), jnp.asarray(endmember).T, axes=([2], [0]))
                    )
                    recon_consistency_rmse = rmse_hsi(x_hat, x_from_abundance)

                    row["runtime_seconds"] = runtime_seconds
                    row["noise_rmse"] = noise_rmse
                    row["conv_mode"] = args.conv_mode
                    row["rmse_scico_vs_matlab_blur"] = rmse_scico_vs_matlab_blur
                    row["recon_consistency_rmse"] = recon_consistency_rmse
                    row["error"] = ""

                    if recon_consistency_rmse > 1e-6:
                        print(f"WARNING: RMSE(x_hat, abundance_hat @ M.T) = {recon_consistency_rmse:.6e}")

                    all_rows.append(row)
                    print(row)

                    # Save method .npz outputs for your separate fixed-scale RGB.py script.
                    # This restores the Output/Kernel*/seed_00/method_outputs/*.npz behavior.
                    if seed == 0 or seed == args.figure_seed:
                        save_method_output_npz(
                            out_file=run_dir / "method_outputs" / f"{method}_{kernel_id}_seed_{seed:02d}.npz",
                            method=method,
                            kernel_id=kernel_id,
                            kernel_name=kernel_name,
                            seed=seed,
                            x_hat=x_hat,
                            abundance_hat=abundance_hat,
                            x_true=ground_truth,
                            y_noisy=y_noisy,
                            rgb_indices=rgb_indices,
                        )

                    # Full .mat result saved in the SAME kernel/seed folder.
                    save_full_method_result_mat(
                        mat_dir / f"FullResult_{method}_{kernel_id}_seed_{seed:02d}.mat",
                        x_hat=x_hat,
                        abundance_hat=abundance_hat,
                        ground_truth=ground_truth,
                        abundance_gt=abundance_gt,
                        endmember=endmember,
                        y_noisy=y_noisy,
                        y_blur_clean=y_blurred,
                        kernel_original=kernel,
                        kernel_norm=kernel_norm,
                        eig_conv_mat=eig_conv_mat,
                        conv_mat=conv_mat,
                        method=method,
                        kernel_id=kernel_id,
                        kernel_name=kernel_name,
                        seed=seed,
                        noise_sigma=noise_sigma,
                        metrics_row=row,
                        y_blur_clean_scico=y_blurred_scico,
                        y_blur_clean_matlab=y_blurred_matlab,
                        conv_mode=args.conv_mode,
                        matlab_forward_model_matches_y=matlab_forward_model_matches_y,
                    )

                    # Save figure outputs only for the selected representative seed.
                    # If --figure_methods is provided, save figures only for those methods.
                    should_save_method_figures = (
                        args.save_figures
                        and seed == args.figure_seed
                        and (args.figure_methods is None or method in args.figure_methods)
                    )

                    if should_save_method_figures:
                        # Save reference RGBs and ground-truth abundance maps once per kernel/seed.
                        # These are separate from method reconstructions.
                        if method == methods[0]:
                            save_rgb_reconstruction_only(
                                x_est=ground_truth,
                                rgb_indices=rgb_indices,
                                title=f"",
                                out_file=rgb_dir / f"RGB_GroundTruth_{kernel_id}_seed_{seed:02d}.png",
                                rgb_low=rgb_low,
                                rgb_high=rgb_high,
                            )
                            save_rgb_reconstruction_only(
                                x_est=y_blurred,
                                rgb_indices=rgb_indices,
                                title=f"",
                                out_file=rgb_dir / f"RGB_ConvolvedClean_{kernel_id}_seed_{seed:02d}.png",
                                rgb_low=rgb_low,
                                rgb_high=rgb_high,
                            )
                            save_rgb_reconstruction_only(
                                x_est=y_noisy,
                                rgb_indices=rgb_indices,
                                title=f"",
                                out_file=rgb_dir / f"RGB_ConvolvedNoisy_{kernel_id}_seed_{seed:02d}.png",
                                rgb_low=rgb_low,
                                rgb_high=rgb_high,
                            )

                            # Ground-truth abundance maps: all endmembers in one figure.
                            save_abundance_maps(
                                abundance_gt,
                                abundance_fig_dir / f"GT_Abundance_AllEndmembers_{kernel_id}_seed_{seed:02d}.png",
                                title=f"",
                            )

                            # Ground-truth abundance maps: each endmember separately.
                            num_gt_endmembers = abundance_gt.shape[2]

                            for em in range(1, num_gt_endmembers + 1):
                                save_single_abundance_map(
                                    abundance_gt,
                                    endmember_number=em,
                                    out_file=abundance_fig_dir / f"GT_Abundance_Endmember{em}_{kernel_id}_seed_{seed:02d}.png",
                                    title=f"",
                                )

                        # Reconstructed RGB image only: no ground truth and no noisy image.
                        save_rgb_reconstruction_only(
                            x_est=x_hat,
                            rgb_indices=rgb_indices,
                            title=f"{method} Reconstruction, {kernel_id}, seed {seed}",
                            out_file=rgb_dir / f"ReconstructedRGB_{method}_{kernel_id}_seed_{seed:02d}.png",
                            rgb_low=rgb_low,
                            rgb_high=rgb_high,
                        )

                        # Estimated abundance maps: all endmembers in one figure.
                        save_abundance_maps(
                            abundance_hat,
                            abundance_fig_dir / f"Abundance_AllEndmembers_{method}_{kernel_id}_seed_{seed:02d}.png",
                            title=f"",
                        )

                        # Estimated abundance maps: each endmember separately.
                        num_endmembers = abundance_hat.shape[2]

                        for em in range(1, num_endmembers + 1):
                            save_single_abundance_map(
                                abundance_hat,
                                endmember_number=em,
                                out_file=abundance_fig_dir / f"Abundance_Endmember{em}_{method}_{kernel_id}_seed_{seed:02d}.png",
                                title=f"",
                            )
                except Exception as exc:
                    print(f"ERROR in {kernel_id}, seed={seed}, method={method}: {exc}")
                    traceback.print_exc()
                    all_rows.append(
                        {
                            "kernel_id": kernel_id,
                            "kernel_name": kernel_name,
                            "seed": int(seed),
                            "noise_sigma": noise_sigma,
                            "method": method,
                            "RMSE": np.nan,
                            "PSNR": np.nan,
                            "SSIM": np.nan,
                            "SAD": np.nan,
                            "ERGAS": np.nan,
                            "aRMSE_abundance": np.nan,
                            "runtime_seconds": np.nan,
                            "noise_rmse": noise_rmse,
                            "conv_mode": args.conv_mode,
                            "rmse_scico_vs_matlab_blur": rmse_scico_vs_matlab_blur,
                            "recon_consistency_rmse": np.nan,
                            "error": str(exc),
                        }
                    )

                pd.DataFrame(all_rows).to_csv(
                    output_dir / "all_runs_metrics_partial.csv",
                    index=False,
                )

    # Final results
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "all_runs_metrics.csv", index=False)

    metric_cols = [
        "RMSE", "PSNR", "SSIM", "SAD", "ERGAS", "aRMSE_abundance",
        "runtime_seconds", "noise_rmse", "rmse_scico_vs_matlab_blur",
        "recon_consistency_rmse",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(["kernel_id", "kernel_name", "method"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "summary_mean_std_metrics.csv", index=False)

    # Paper-friendly table: mean ± std strings.
    paper_rows = []
    for _, row in summary.iterrows():
        paper_row = {
            "kernel_id": row["kernel_id"],
            "kernel_name": row["kernel_name"],
            "method": row["method"],
        }
        for metric in metric_cols:
            mean_val = row[f"{metric}_mean"]
            std_val = row[f"{metric}_std"]
            if pd.isna(mean_val):
                paper_row[metric] = "NaN"
            else:
                paper_row[metric] = f"{mean_val:.6f} ± {std_val:.6f}"
        paper_rows.append(paper_row)

    pd.DataFrame(paper_rows).to_csv(output_dir / "summary_mean_std_metrics_paper_format.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "kernel_blur_diagnostics.csv", index=False)
    save_runtime_plots(df, summary, output_dir)

    print("\nFinished.")
    print("Saved all results in one output folder:")
    print(" -", output_dir.resolve())
    print("Per-kernel/per-seed folders contain MAT_Files/ for .mat files, RGBs/ for fixed-scale RGB images, Abundance_Figures/ for abundance maps, and method_outputs/*.npz for RGB.py.")
    print("Default conv_mode is scico, which reproduces the original PUK Python forward model.")

def parse_args():
    parser = argparse.ArgumentParser(description="Indian Pines 6-kernel multi-noise experiment")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Folder containing Indian Pines dataset files")
    parser.add_argument("--kernel_dir", type=str, default="Kernels", help="Folder containing kernel_1.mat ... kernel_6.mat")
    parser.add_argument("--output_dir", type=str, default="Output", help="Output folder; default is Output")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of independent noise realizations")
    parser.add_argument("--maxiter", type=int, default=400, help="ADMM max iterations for PnP-JDU and PnP")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Methods to run",
    )
    parser.add_argument("--save_figures", action="store_true", help="Save representative RGB and abundance figures")
    parser.add_argument("--figure_seed", type=int, default=0, help="Only save figures for this seed")
    parser.add_argument(
        "--figure_methods",
        nargs="+",
        default=None,
        choices=DEFAULT_METHODS,
        help=(
            "Save figure outputs only for these methods. "
            "Example: --figure_methods PnP-JDU. "
            "If omitted, figures are saved for all methods in --methods."
        ),
    )
    parser.add_argument("--save_mat_inputs", action="store_true", help="Save one combined MATLAB input .mat file per kernel/seed")
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="scico",
        choices=["scico", "matlab"],
        help=(
            "Forward convolution model. Use 'scico' for original PUK/Python metrics. "
            "Use 'matlab' only for MATLAB ADMSolver-compatible experiments."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
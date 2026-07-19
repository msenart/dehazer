"""Sweep the dark-channel-size parameter over the I-HAZE dataset and plot PSNR/SSIM.

Standalone evaluation script (parameters are module-level constants below rather
than a CLI). Run via ``python -m dehazer.psnrssim`` after downloading the I-HAZE
dataset into ``hazed_images/I-HAZE`` at the project root (see the README).
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

from . import core as dh
from . import u_guided_filter as gf
from .config import PROJECT_ROOT

# --- 1) Dehazing algorithm under evaluation ---
def dehaze_algorithm(image, param):
    """Run the core dehaze pipeline with a guided filter, sweeping dc_size=param."""
    kwargs = {
        "r":200, "eps":1e-3
    }
    image = dh.dehaze(image,gf.guided_filter, kwargs,dc_size = param, top_percent = 0.001, patch_avg = 3, omega = 0.95, t0 = 0.01, show_steps = False, custom_output_name=None)
    return image


# --- 2) Parameters ---
path_hazy = str(PROJECT_ROOT / "hazed_images" / "I-HAZE" / "train" / "hazy")
path_gt   = str(PROJECT_ROOT / "hazed_images" / "I-HAZE" / "train" / "clear")

parameter_values = [1 + 4*i for i in range(10)]

def to_uint8_image(arr):
    """Convert a float image (assumed in [0,1] or already [0,255]) to a clamped uint8 array."""
    if arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255)

    return arr.astype(np.uint8)

# --- 3) Metrics computation for one image ---
def compute_for_one_image(args):
    """Dehaze one hazy image with dehaze_algorithm and compute PSNR/SSIM against its ground truth."""
    hazy_filename, param, hazy_path, gt_path = args

    # The matching GT file: strip "_hazy"
    clear_filename = hazy_filename.replace("_hazy", "")

    hazy_img_path  = os.path.join(hazy_path, hazy_filename)
    gt_img_path    = os.path.join(gt_path, clear_filename)

    hazy_img = cv2.imread(hazy_img_path)
    gt_img   = cv2.imread(gt_img_path)

    if hazy_img is None:
        print(f"❌ Unable to read HAZY: {hazy_img_path}")
        return None

    if gt_img is None:
        print(f"❌ Unable to read CLEAR: {gt_img_path}")
        return None

    hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
    gt_img   = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    dehazed = dehaze_algorithm(hazy_img_path, param)
    dehazed = to_uint8_image(dehazed)

    p = psnr(gt_img, dehazed, data_range=255)
    s = ssim(gt_img, dehazed, data_range=255, channel_axis=2)

    return {
        "image": hazy_filename,
        "param": param,
        "psnr": p,
        "ssim": s
    }


# --- 4) Global metrics (dataset-wide PSNR and SSIM) ---
def compute_global_metrics(df):
    """Aggregate per-image PSNR/SSIM rows in df into dataset-wide PSNR and SSIM values."""
    global_ssim = df["ssim"].mean()

    mse_list = []
    for _, row in df.iterrows():
        psnr_value = row["psnr"]
        mse = (255 ** 2) / (10 ** (psnr_value / 10))
        mse_list.append(mse)

    mse_global = np.mean(mse_list)
    psnr_global = 10 * np.log10((255 ** 2) / mse_global)

    return psnr_global, global_ssim


# --- 5) Multiprocessing evaluation ---
def evaluate_with_multiprocessing(hazy_dir, gt_dir, params):
    """Evaluate every (image, parameter) combination in parallel and return a results DataFrame."""

    hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    gt_files   = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    assert len(hazy_files) == len(gt_files), "Both folders must contain the same number of images."

    tasks = []
    for p in params:
        for f in hazy_files:
            tasks.append((f, p, hazy_dir, gt_dir))

    print(f"\n>> Launching multiprocessing on {cpu_count()-1} CPUs... : number of tasks {len(tasks)}")
    with Pool(processes=cpu_count()-1) as pool:
        results = list(tqdm(pool.imap(compute_for_one_image, tasks), total=len(tasks)))

    df = pd.DataFrame(results)
    return df

# --- 6) Plots ---
def plot_metric(df, metric_name):
    """Plot metric_name vs. parameter, one line per image."""
    plt.figure(figsize=(10, 6))
    for image in df["image"].unique():
        sub = df[df["image"] == image]
        plt.plot(sub["param"], sub[metric_name], marker="o", label=image)

    plt.xlabel("Parameter")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} per image as a function of the parameter")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_global(df):
    """Plot the dataset-wide PSNR and SSIM curves vs. parameter."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["param"], df["psnr_global"], marker="o")
    plt.xlabel("Parameter")
    plt.ylabel("GLOBAL PSNR")
    plt.grid(True)
    plt.title("Global PSNR as a function of the parameter")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df["param"], df["ssim_global"], marker="o")
    plt.xlabel("Parameter")
    plt.ylabel("GLOBAL SSIM")
    plt.grid(True)
    plt.title("Global SSIM as a function of the parameter")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    df_results = evaluate_with_multiprocessing(path_hazy, path_gt, parameter_values)
    df_results.to_csv("results_psnr_ssim.csv", index=False)
    print("\n📁 CSV file generated: results_psnr_ssim.csv")

    # --- Global metrics ---
    global_stats = []
    for param in parameter_values:
        subdf = df_results[df_results["param"] == param]
        psnr_g, ssim_g = compute_global_metrics(subdf)
        global_stats.append({
            "param": param,
            "psnr_global": psnr_g,
            "ssim_global": ssim_g
        })

    df_global = pd.DataFrame(global_stats)
    df_global.to_csv("global_metrics.csv", index=False)
    print("\n📁 Global metrics: global_metrics.csv")

    print("\n📊 Generating plots...")
    plot_metric(df_results, "psnr")
    plot_metric(df_results, "ssim")
    plot_global(df_global)

    print("\n🎉 Program completed successfully!")

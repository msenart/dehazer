"""Plot PSNR/SSIM/UIQM vs. parameter value from comparison.json files produced by comparison.py."""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def try_float(s):
    """Parse s as a float, returning None instead of raising if it isn't numeric."""
    try:
        return float(s)
    except Exception:
        return None

def plot_metric(results, param_name, metric_key, out_path):
    """Plot metric_key vs. parameter value from results and save the figure to out_path."""
    items = []
    for r in results:
        v = r.get('value')
        metrics = r.get('metrics') or {}
        val = metrics.get(metric_key)
        items.append((v, try_float(v), val))
    items.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else x[0]))
    labels = [str(x[0]) for x in items]
    vals = [float(x[2]) if x[2] is not None else np.nan for x in items]
    x = np.arange(len(labels))

    plt.figure(figsize=(8,4.5))
    plt.plot(x, vals, marker='o', label=metric_key)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.xlabel(param_name)
    plt.ylabel(metric_key)
    plt.title(f'{param_name} — {metric_key}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")

def main(results_root, out_dir, json_name='comparison.json'):
    """Generate PSNR/SSIM/UIQM plots for every parameter subfolder found under results_root."""
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"results_root not found: {results_root}")
    os.makedirs(out_dir, exist_ok=True)
    for param_name in sorted(os.listdir(results_root)):
        param_dir = os.path.join(results_root, param_name)
        if not os.path.isdir(param_dir):
            continue
        json_path = os.path.join(param_dir, json_name)
        if not os.path.isfile(json_path):
            print(f"[WARN] {json_name} not found in {param_dir}, skipping")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data.get('results') or []
        if not results:
            print(f"[WARN] no results in {json_path}")
            continue

        # three separate plots
        plot_metric(results, param_name, 'PSNR', os.path.join(out_dir, f"{param_name}_PSNR.png"))
        plot_metric(results, param_name, 'SSIM', os.path.join(out_dir, f"{param_name}_SSIM.png"))
        plot_metric(results, param_name, 'UIQM', os.path.join(out_dir, f"{param_name}_UIQM.png"))

if __name__ == "__main__":
    from .config import PROJECT_ROOT

    default_results_root = str(PROJECT_ROOT / "seriespicturesoutput")
    default_out_dir = str(PROJECT_ROOT / "seriespicturesoutput" / "plots")

    parser = argparse.ArgumentParser(description="Plot PSNR/SSIM/UIQM per parameter from comparison.json files")
    parser.add_argument("--results_root", required=False,
                        default=default_results_root,
                        help="root folder containing parameter subfolders")
    parser.add_argument("--out_dir", required=False,
                        default=default_out_dir,
                        help="folder to save plots")
    parser.add_argument("--json_name", default="comparison.json", help="per-parameter json filename")
    args = parser.parse_args()
    main(args.results_root, args.out_dir, args.json_name)
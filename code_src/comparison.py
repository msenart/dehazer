import os
import cv2
import numpy as np
import json
from skimage import metrics, filters, feature
from scipy import ndimage

def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

def compute_mse(img1, img2):
    """Compute Mean Squared Error between two images"""
    return float(np.mean((img1 - img2) ** 2))

def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio"""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(1.0 / np.sqrt(mse)))

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index"""
    # Convert to grayscale for SSIM
    if img1.ndim == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    return float(metrics.structural_similarity(img1_gray, img2_gray, data_range=1.0))

def compute_uiqm(img):
    """
    Compute Underwater Image Quality Measure (UIQM)
    Higher values indicate better quality for underwater/hazy images
    """
    # Convert to float
    img_float = img.astype(np.float32)
    
    # Colorfulness measure (UICM)
    r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    uicm = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    
    # Sharpness measure (UISM) - using Sobel gradient
    if img_float.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_float
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    uism = np.mean(gradient_magnitude)
    
    # Contrast measure (UIConM)
    uiconm = np.std(gray)
    
    # Combined UIQM
    uiqm = 0.028 * uicm + 0.295 * uism + 3.575 * uiconm
    return float(uiqm)

def compute_all_metrics(img_gt, img_dehazed):
    """Compute all available metrics (expects images in [0,1] float32)"""
    return {
        'MSE': compute_mse(img_gt, img_dehazed),
        'PSNR': compute_psnr(img_gt, img_dehazed),
        'SSIM': compute_ssim(img_gt, img_dehazed),
        'UIQM': compute_uiqm(img_dehazed),
    }
    
def analyze_results_folder(gt_path, results_root, target_basename='14_hazy_final.png', output_json=None):
    """
    Scan results_root subfolders, find target_basename in each, compare to gt_path using metrics,
    and write JSON summary.

    Output JSON structure:
    {
      "root_folder": "<name>",
      "gt_path": "<path>",
      "target_basename": "<name>",
      "results": [
        {
          "param_folder": "<subfolder name>",
          "file_path": "<found file full path>",
          "metrics": { "MSE":.., "PSNR":.., "SSIM":.., "UIQM":.. }
        },
        ...
      ]
    }
    """
    gt = cv2.imread(gt_path)
    if gt is None:
        raise FileNotFoundError(f"Ground-truth image not found: {gt_path}")
    gt = gt.astype('float32') / 255.0
    results = []

    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results root folder not found: {results_root}")

    # iterate immediate subdirectories
    for name in sorted(os.listdir(results_root)):
        sub = os.path.join(results_root, name)
        if not os.path.isdir(sub):
            continue
        # search for target_basename inside sub (non-recursive then recursive)
        candidate = os.path.join(sub, target_basename)
        found_path = None
        if os.path.isfile(candidate):
            found_path = candidate
        else:
            # recursive search for file with that basename
            for root, _, files in os.walk(sub):
                if target_basename in files:
                    found_path = os.path.join(root, target_basename)
                    break
        if not found_path:
            # skip if not found
            print(f"[WARN] target '{target_basename}' not found in folder: {sub}")
            continue

        img = cv2.imread(found_path)
        if img is None:
            print(f"[WARN] Failed to read image: {found_path}")
            continue
        img = img.astype('float32') / 255.0

        # if sizes differ, resize dehazed to GT size
        if img.shape[:2] != gt.shape[:2]:
            h, w = gt.shape[:2]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        metrics_dict = compute_all_metrics(gt, img)
        results.append({
            'param_folder': name,
            'file_path': os.path.abspath(found_path),
            'metrics': metrics_dict
        })

    out = {
        'root_folder': os.path.basename(os.path.normpath(results_root)),
        'gt_path': os.path.abspath(gt_path),
        'target_basename': target_basename,
        'results': results
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(numpy_to_python(out), f, indent=2, ensure_ascii=False)

    return out

# Metric configuration for optimization
METRIC_CONFIG = {
    'MSE': {
        'direction': 'minimize',  # Lower is better
        'function': compute_mse
    },
    'PSNR': {
        'direction': 'maximize',  # Higher is better
        'function': compute_psnr
    },
    'SSIM': {
        'direction': 'maximize',  # Higher is better
        'function': compute_ssim
    },
    'UIQM': {
        'direction': 'maximize',  # Higher is better
        'function': lambda img1, img2: compute_uiqm(img2)
    },
}

def optimize_parameters(hazy_path, gt_path, optimization_metric='PSNR', param_ranges=None):
    """
    Find optimal parameters by comparing with ground truth
    
    Args:
        hazy_path: Path to hazy image
        gt_path: Path to ground truth image
        optimization_metric: Which metric to use for optimization
        param_ranges: dict of parameter ranges to try
    """
    if param_ranges is None:
        param_ranges = {
            'omega': [0.7, 0.8, 0.9, 0.95, 0.99],
            'w_radius': [1, 2],
            'eps': [1e-4, 1e-3],
            't0': [0.05, 0.1, 0.2]
        }
    
    # Validate optimization metric
    if optimization_metric not in METRIC_CONFIG:
        available_metrics = list(METRIC_CONFIG.keys())
        raise ValueError(f"Invalid optimization metric '{optimization_metric}'. Available: {available_metrics}")
    
    # Read images
    gt_img = cv2.imread(gt_path).astype('float32') / 255.0
    hazy_img = cv2.imread(hazy_path).astype('float32') / 255.0
    
    metric_config = METRIC_CONFIG[optimization_metric]
    
    # Initialize best values based on optimization direction
    if metric_config['direction'] == 'maximize':
        best_score = -float('inf')
    else:
        best_score = float('inf')
    
    best_params = None
    best_result = None
    results = []
    
    # Try all parameter combinations
    from itertools import product
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    total_combinations = np.prod([len(v) for v in param_values])
    print(f"Testing {total_combinations} parameter combinations using {optimization_metric} metric...")
    
    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        params.update({
            'img_path': hazy_path,
            'refine_method': 'guided',
            'custom_output_name': None
        })
        
        # Run dehazing
        try:
            J, _, _ = dehaze(**params)
            
            # Compute all metrics for comprehensive analysis
            all_metrics = compute_all_metrics(gt_img, J)
            
            # Get the optimization metric score
            score = metric_config['function'](gt_img, J)
            
            results.append({
                'params': params,
                'optimization_score': score,
                'all_metrics': all_metrics
            })
            
            print(f"Progress: {i+1}/{total_combinations}")
            print(f"Parameters: {params}")
            print(f"{optimization_metric}: {score:.4f}")
            print("-" * 50)
            
            # Update best result based on optimization direction
            if (metric_config['direction'] == 'maximize' and score > best_score) or \
               (metric_config['direction'] == 'minimize' and score < best_score):
                best_score = score
                best_params = params.copy()
                best_result = J
                
        except Exception as e:
            print(f"Failed with parameters {params}: {e}")
    
    # Sort results by optimization metric
    reverse = (metric_config['direction'] == 'maximize')
    results.sort(key=lambda x: x['optimization_score'], reverse=reverse)
    
    # Print top results
    print(f"\n=== Best Parameter Combinations (optimizing for {optimization_metric}) ===")
    for i, r in enumerate(results[:5]):
        print(f"\n{i+1}. {optimization_metric}: {r['optimization_score']:.4f}")
        for k, v in r['params'].items():
            if k not in ['img_path', 'custom_output_name']:
                print(f"   {k}: {v}")
        # Print other metrics for reference
        other_metrics = {k: v for k, v in r['all_metrics'].items() if k != optimization_metric}
        print("   Other metrics:", " | ".join([f"{k}: {v:.4f}" for k, v in other_metrics.items()]))
    
    return best_params, best_result, results

def multi_metric_optimization(hazy_path, gt_path, metrics_list=None, param_ranges=None):
    """
    Optimize parameters for multiple metrics and return comprehensive results
    """
    if metrics_list is None:
        metrics_list = ['PSNR', 'SSIM', 'UIQM']
    
    all_results = {}
    
    for metric in metrics_list:
        print(f"\n{'='*60}")
        print(f"Optimizing for metric: {metric}")
        print(f"{'='*60}")
        
        best_params, best_result, results = optimize_parameters(
            hazy_path, gt_path, metric, param_ranges
        )
        
        all_results[metric] = {
            'best_params': best_params,
            'best_score': results[0]['optimization_score'] if results else None,
            'all_results': results
        }
    
    # Print comprehensive comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    
    for metric, data in all_results.items():
        if data['best_params']:
            print(f"\n{metric}:")
            print(f"  Best score: {data['best_score']:.4f}")
            print(f"  Best parameters: {data['best_params']}")
    
    return all_results


def analyze_parameters_root(gt_path, results_root, target_basename='14_hazy_final.png', output_name='comparison.json'):
    """
    For each immediate subfolder in results_root (treated as a parameter name),
    scan its immediate child folders (treated as parameter values), look for the
    target_basename inside each child folder, compute metrics against gt_path,
    and write a JSON file into the parameter folder (output_name) containing all
    values and their metrics.

    Output JSON structure per parameter folder:
    {
      "parameter": "<param_folder_name>",
      "gt_path": "<gt abs path>",
      "target_basename": "<name>",
      "results": [
        { "value": "<child folder name>", "file_path": "<found file>", "metrics": {...} },
        ...
      ]
    }
    """
    gt = cv2.imread(gt_path)
    if gt is None:
        raise FileNotFoundError(f"Ground-truth image not found: {gt_path}")
    gt = gt.astype('float32') / 255.0

    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results root folder not found: {results_root}")

    summary = {
        'root_folder': os.path.basename(os.path.normpath(results_root)),
        'gt_path': os.path.abspath(gt_path),
        'target_basename': target_basename,
        'parameters': []
    }

    for param_name in sorted(os.listdir(results_root)):
        param_dir = os.path.join(results_root, param_name)
        if not os.path.isdir(param_dir):
            continue

        param_results = []
        # iterate immediate child folders (each child represents one parameter value)
        for val_name in sorted(os.listdir(param_dir)):
            val_dir = os.path.join(param_dir, val_name)
            if not os.path.isdir(val_dir):
                continue

            # look for the target file inside this value folder (first direct, then recursive)
            candidate = os.path.join(val_dir, target_basename)
            found_path = None
            if os.path.isfile(candidate):
                found_path = candidate
            else:
                for root, _, files in os.walk(val_dir):
                    if target_basename in files:
                        found_path = os.path.join(root, target_basename)
                        break

            if not found_path:
                # skip missing target in this value folder
                # still include entry with null metrics to indicate missing file
                param_results.append({
                    'value': val_name,
                    'file_path': None,
                    'metrics': None,
                    'note': f"'{target_basename}' not found in {val_dir}"
                })
                continue

            img = cv2.imread(found_path)
            if img is None:
                param_results.append({
                    'value': val_name,
                    'file_path': os.path.abspath(found_path),
                    'metrics': None,
                    'note': "failed to read image"
                })
                continue

            img = img.astype('float32') / 255.0
            # resize if needed
            if img.shape[:2] != gt.shape[:2]:
                h, w = gt.shape[:2]
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            metrics_dict = compute_all_metrics(gt, img)
            param_results.append({
                'value': val_name,
                'file_path': os.path.abspath(found_path),
                'metrics': metrics_dict
            })

        # write per-parameter json into the parameter folder
        out = {
            'parameter': param_name,
            'gt_path': os.path.abspath(gt_path),
            'target_basename': target_basename,
            'results': param_results
        }
        out_path = os.path.join(param_dir, output_name)
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(numpy_to_python(out), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Failed to write JSON for {param_name} -> {out_path}: {e}")

        # also add to top-level summary
        summary['parameters'].append({
            'parameter': param_name,
            'comparison_file': os.path.abspath(out_path),
            'n_values': len(param_results)
        })

    return summary

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Compare dehazed results in folder to a ground-truth image."
    )
    # Defaults requested by you
    default_results_root = r"C:\Users\22863\Desktop\git\ima1\ima_projet\dehazer\code_src\complet_dehazer\seriespicturesoutput"
    default_gt = r"C:\Users\22863\Desktop\git\ima1\ima_projet\dehazer\images\hazed_images\very_hazed_images\GT\14_GT.png"
    default_target = "14_hazy_final.png"
    default_out = os.path.join(default_results_root, "comparison_summary.json")

    parser.add_argument("--gt", default=default_gt, help="path to ground-truth (clean) image")
    parser.add_argument("--results_root", default=default_results_root,
                        help="folder containing subfolders per-parameter-result")
    parser.add_argument("--target", default=default_target, help="filename to look for inside each value subfolder")
    parser.add_argument("--out", default=default_out, help="top-level summary JSON (optional)")

    args = parser.parse_args()

    # ensure results_root exists
    if not os.path.isdir(args.results_root):
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")

    # create parent folder for top-level out if requested
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Run per-parameter analysis and write per-parameter JSONs
    top_summary = analyze_parameters_root(args.gt, args.results_root, target_basename=args.target, output_name='comparison.json')

    # write top-level summary if requested
    try:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(numpy_to_python(top_summary), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to write top-level summary {args.out}: {e}")

    # print concise summary
    print(f"Analyzed root: {top_summary['root_folder']}, parameter folders: {len(top_summary['parameters'])}")
    for p in top_summary['parameters']:
        print(f" - {p['parameter']}: comparison_file={p['comparison_file']}, n_values={p['n_values']}")
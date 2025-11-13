import os
import cv2
import numpy as np
import json
from skimage import metrics, filters, feature
from scipy import ndimage

from dehazer import dehaze

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

def compute_edge_preservation(img1, img2):
    """
    Compute edge preservation ratio using Canny edge detection
    Higher values indicate better edge preservation
    """
    if img1.ndim == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Detect edges
    edges1 = feature.canny(img1_gray, sigma=2)
    edges2 = feature.canny(img2_gray, sigma=2)
    
    # Calculate edge preservation
    edge_union = np.logical_or(edges1, edges2)
    edge_intersection = np.logical_and(edges1, edges2)
    
    if np.sum(edge_union) == 0:
        return 0.0
    
    return float(np.sum(edge_intersection) / np.sum(edge_union))

def compute_all_metrics(img1, img2):
    """Compute all available metrics"""
    metrics_dict = {
        'MSE': compute_mse(img1, img2),
        'PSNR': compute_psnr(img1, img2),
        'SSIM': compute_ssim(img1, img2),
        'UIQM': compute_uiqm(img2),  # Only for dehazed image
        'Edge_Preservation': compute_edge_preservation(img1, img2)
    }
    return metrics_dict

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
    'Edge_Preservation': {
        'direction': 'maximize',  # Higher is better
        'function': compute_edge_preservation
    }
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

# Example usage in main:
if __name__ == "__main__":
    hazy_path = r"C:\Users\22863\Desktop\git\ima1\ima_projet\dehazer\hazed_images\very_hazed_images\hazy\21_hazy.png"
    gt_path = r"C:\Users\22863\Desktop\git\ima1\ima_projet\dehazer\hazed_images\very_hazed_images\GT\21_GT.png"
    
    # Optional: custom parameter ranges
    param_ranges = {
        'omega': [0.8, 0.9, 0.95, 0.99],
        'w_radius': [1, 2],
        'eps': [1e-4, 1e-3],
        't0': [0.1, 0.15]
    }
    
    # Choose optimization method:
    # Option 1: Single metric optimization
    optimization_metric = 'UIQM'  # Change this to any metric in METRIC_CONFIG
    
    best_params, best_result, all_results = optimize_parameters(
        hazy_path, gt_path, optimization_metric, param_ranges
    )
    
    # Option 2: Multi-metric optimization (uncomment to use)
    # metrics_to_optimize = ['PSNR', 'SSIM', 'UIQM']
    # all_results = multi_metric_optimization(
    #     hazy_path, gt_path, metrics_to_optimize, param_ranges
    # )
    
    # Save results
    out_dir = "optimized_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save best result image
    if best_result is not None:
        cv2.imwrite(os.path.join(out_dir, "best_result.png"), 
                   (best_result * 255).astype('uint8'))
    
    # Save parameter search results (using conversion function)
    with open(os.path.join(out_dir, "parameter_search.json"), 'w') as f:
        json.dump(numpy_to_python([{
            'params': r['params'],
            'optimization_metric': optimization_metric,
            'optimization_score': r['optimization_score'],
            'all_metrics': r['all_metrics']
        } for r in all_results]), f, indent=2)
    
    print(f"\nAvailable metrics: {list(METRIC_CONFIG.keys())}")
    print(f"Used metric for optimization: {optimization_metric}")
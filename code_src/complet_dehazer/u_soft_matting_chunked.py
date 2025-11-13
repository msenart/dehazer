import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import cg
from multiprocessing import Pool, Manager
import logging

logger = logging.getLogger("widget_logger")

class chunk_soft_matting_data:
    ALGO_PARAMS = {
            "maxiter": "int", "n_cut_width": "int", "n_cut_height": "int",
            "win_radius": "int", "eps": "float", "lam": "float",
            "max_processes": "int", "ratio": "float"
        }
    DEFAULT_ALGO_PARAMS = {
            "maxiter": 5000,
            "n_cut_width": 1,
            "n_cut_height": 2,
            "win_radius": 3,
            "eps": 1e-7,
            "lam": 1e-4,
            "max_processes": 6,
            "ratio": 0.5
        }

def _chunk_soft_matting(I_rgb, t_coarse, win_radius=1, eps=1e-7, lam = 1e-4, maxiter = 5000):
    """
    Closed-form matting refinement of transmission (soft matting).
    I_rgb    : HxWx3 float32 in [0,1]  (guide: original color image)
    t_coarse : HxW    float32 in [0,1]  (initial transmission)
    win_radius : window radius r (window size = (2r+1)^2), typically 1 or 2
    eps      : regularization in covariance inversion (very small)
    lam      : data term weight (lambda in the paper; small ~1e-4)
    Returns:
        t_refined : HxW float32
    """
    H, W, _ = I_rgb.shape
    N = H * W
    win_size = (2 * win_radius + 1)
    K = win_size * win_size  # number of pixels per window
    results_size = (H - 2*win_radius) * (W - 2*win_radius) * K * K

    # flatten helpers
    inds = np.arange(N).reshape(H, W)

    rows = np.zeros(results_size)
    cols = np.zeros(results_size)
    vals = np.zeros(results_size)
    local_loop_idx = 0
    for y in range(win_radius, H - win_radius):
        for x in range(win_radius, W - win_radius):
            ys, ye = y - win_radius, y + win_radius + 1
            xs, xe = x - win_radius, x + win_radius + 1
            win_inds = inds[ys:ye, xs:xe].reshape(-1)
            win_I = I_rgb[ys:ye, xs:xe, :].reshape(K, 3)
            mu = win_I.mean(axis=0, keepdims=True)           
            cov = (win_I - mu).T @ (win_I - mu) / K          
            cov_reg = cov + (eps / K) * np.eye(3)            

            # inverse covariance
            inv = np.linalg.inv(cov_reg)

            # (I - 1/K) operator
            X = win_I - mu                                   
            M = np.eye(K) - np.ones((K, K)) / K              

            # L_w = M - X * inv * X^T / K
            # compute X * inv * X^T efficiently
            Xin = X @ inv                                    
            Q = (Xin @ X.T) / K                              
            Lw = M + Q * 0.0                                 
            Lw -= Q                                         

            # scatter-add to global Laplacian
            ii = np.repeat(win_inds, K)
            jj = np.tile(win_inds, K)
            kk = Lw.reshape(-1)

            start = local_loop_idx * len(ii)
            end = (local_loop_idx + 1) * len(ii)
            rows[start:end] = ii
            cols[start:end] = jj
            vals[start:end] = kk

    logger.info("loading laplacian : 100% !")
    logger.info("Putting it all in order (1/3)")
    # (slow !)
    L = csr_matrix((vals, (rows, cols)), shape=(N, N))
    logger.info("Putting it all in order (2/3)")
    # Data term: lambda * (t - t0)^2
    A = L + lam * eye(N, format='csr')
    logger.info("Putting it all in order (3/3)")
    b = lam * t_coarse.reshape(-1)

    logger.info("loading laplacian : 100% ! Inverting matrix...")

    # Solve sparse linear system (slow !)
    t_refined, info = cg(A, b, rtol=1e-6, maxiter = maxiter)
    if info != 0:
        logger.warning(f"⚠️ CG did not fully converge, info = {info}")
    t_refined = t_refined.reshape(H, W).astype(np.float32)

    # Clamp to [0,1]
    logger.info("soft matting completed ! (3/3)")
    return np.clip(t_refined, 0.0, 1.0)

def _patch_soft_matting_process(I_patch,t_patch,i_min,i_max,j_min,j_max,win_radius,eps,lam,started_counter,loading_counter,loading_total,maxiter):
    started_counter.value+=1
    logger.info(f"============= patch {t_patch.shape} started ! : {started_counter.value}/{loading_total} =============")
    refined_patch = _chunk_soft_matting(I_patch,t_patch, win_radius, eps, lam, maxiter)
    refined_patch = refined_patch.reshape(i_max - i_min, j_max - j_min)
    loading_counter.value+=1
    logger.info(f"============= patch {t_patch.shape} finished ! : {loading_counter.value}/{loading_total} =============")
    return refined_patch,i_min,i_max,j_min,j_max

### MAIN FUNCTION BELOW ========================================================================================================

def chunked_soft_matting(I_rgb : np.ndarray,t_coarse : np.ndarray , maxiter : int, n_cut_width : int ,n_cut_height : int, win_radius : int, eps : float ,lam : float, max_processes, ratio : float = 0.5) -> list[np.ndarray]:
    height,width = t_coarse.shape
    height_cut = height//n_cut_height
    width_cut = width//n_cut_width
    transmission_patches = []
    I_rgb_patches = []
    coords = []

    logger.info("cutting the image before doing soft matting")

    for i in range(n_cut_height):
        for j in range(n_cut_width):

            i_min_idx = max(0,i*height_cut)
            j_min_idx = max(0,j*width_cut)
            i_max_idx = min(height,(i+1)*height_cut)
            j_max_idx = min(width,(j+1)*width_cut)

            if (i == n_cut_height-1 and height%n_cut_height < ratio*height_cut) :
                i_max_idx = height
            if (j == n_cut_width-1 and width%n_cut_width < ratio*width_cut) :
                j_max_idx = width
            
            transmission_patches.append(t_coarse[i_min_idx:i_max_idx,j_min_idx:j_max_idx])
            I_rgb_patches.append(I_rgb[i_min_idx:i_max_idx,j_min_idx:j_max_idx,:])
            coords.append((i_min_idx, i_max_idx, j_min_idx, j_max_idx))

    logger.info([m.shape for m in transmission_patches])
    logger.info([m.shape for m in I_rgb_patches])

    loading_total = len(transmission_patches)
    loading_counter = 0
    t_refined_full = np.zeros((height, width), dtype=np.float32)

    # for t_patch,i_patch,coord in zip(transmission_patches,I_rgb_patches,coords):
    #     logger.info(f"============= patch {t_patch.shape} started ! : {loading_counter}/{loading_total} =============")
    #     i_min, i_max, j_min, j_max = coord
    #     refined_patch = soft_matting(i_patch,t_patch, maxiter, win_radius, eps, lam, max_processes)
    #     refined_patch = refined_patch.reshape(i_max - i_min, j_max - j_min)
    #     t_refined_full[i_min:i_max,j_min:j_max] = refined_patch
    #     loading_total+=1
    #     logger.info(f"============= patch {t_patch.shape} finished ! : {loading_counter}/{loading_total} =============")
    
    # return np.clip(t_refined_full,0.0,1.0)

    manager = Manager()
    started_counter = manager.Value('i', 0)
    loading_counter = manager.Value('i', 0)

    args_list = []
    for (I_patch ,t_patch, (i_min, i_max, j_min, j_max)) in zip(I_rgb_patches,transmission_patches, coords):
        args_list.append((I_patch,t_patch,i_min,i_max,j_min,j_max,win_radius,eps,lam,started_counter,loading_counter,loading_total,maxiter))
    
    with Pool(processes=max_processes) as pool:
        results = pool.starmap(_patch_soft_matting_process, args_list)

    logger.info("============= patch all treated ! =============")

    t_refined_full = np.zeros((height, width), dtype=np.float32)

    for patch, i_min, i_max, j_min, j_max in results:
        t_refined_full[i_min:i_max, j_min:j_max] = patch

    return np.clip(t_refined_full, 0.0, 1.0)

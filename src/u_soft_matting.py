import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import cg
from multiprocessing import Pool, cpu_count, Manager
import logging

logger = logging.getLogger("widget_logger")

class soft_matting_data:
    ALGO_PARAMS = {
            "maxiter": "int", "win_radius": "int", "eps": "float",
            "lam": "float", "max_processes": "int"
        }
    DEFAULT_ALGO_PARAMS = {
            "maxiter": 2000,
            "win_radius": 2,
            "eps": 1e-7,
            "lam": 1e-4,
            "max_processes": 6
        }

def _one_line_soft_matting(I_rgb,inds,win_radius,W,process_idx,n_processes,y_min,y_max,eps,K):
    '''
    The image is split into many strands, which are calculated separately to increase the speed. This is one process of them.
    '''
    results_size = (y_max - y_min) * (W - 2 * win_radius) * (2 * win_radius + 1)**2 * K

    rows = np.zeros(results_size)
    cols = np.zeros(results_size)
    vals = np.zeros(results_size)
    
    local_loop_idx = 0
    local_loop_total = (y_max-y_min)*(W-2*win_radius)
    print(f"============= process started ! {process_idx}/{n_processes} ==============",flush=True)
    for y in range(y_min, y_max):
        for x in range(win_radius, W - win_radius):
            if (local_loop_idx%int(local_loop_total/100) == int(local_loop_total/100)-1 and process_idx == 0):
                print(f"loading laplacian : process {process_idx} - {int((local_loop_idx/local_loop_total)*100)}%")

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
            local_loop_idx+=1
    print(f"============= process finished ! {process_idx}/{n_processes} ==============",flush=True)
    return [rows,cols,vals]

### MAIN FUNCTION BELOW ========================================================================================================

def soft_matting(I_rgb, t_coarse, maxiter, win_radius=1, eps=1e-7, lam=1e-4, max_processes = 4):
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

    # flatten helpers
    inds = np.arange(N).reshape(H, W)

    # allocate shared variables before multiprocessing

    process_idx = 0
    args_list = []

    n_processes = min(cpu_count(),max_processes)
    logger.info(f"n_processes : {n_processes} ")
    numbers_columns_share = H - 2*win_radius

    for idx_process in range(n_processes):
        if(idx_process == n_processes - 1):
            y_min,y_max = win_radius+(numbers_columns_share//n_processes)*idx_process, H-win_radius
        else :
            y_min,y_max = win_radius+(numbers_columns_share//n_processes)*idx_process, win_radius+(numbers_columns_share//n_processes)*(idx_process+1)
        args_list.append((I_rgb,inds,win_radius,W,process_idx,n_processes,y_min,y_max,eps,K))
        process_idx+=1

    logger.info(f"========== all processes are starting ! {n_processes} ===========")

    with Pool(processes=n_processes) as pool:
        results = pool.starmap(_one_line_soft_matting, args_list)

    rows = np.concatenate([r[0] for r in results]).astype(np.float32)
    cols = np.concatenate([r[1] for r in results]).astype(np.float32)
    vals = np.concatenate([r[2] for r in results]).astype(np.float32)

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



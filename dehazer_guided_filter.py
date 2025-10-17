#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Image Dehazing using Dark Channel Prior
----------------------------------------------
Implementation of He et al., 2009 (CVPR).
Comments are in English.
"""
import os
import cv2
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import cg
from multiprocessing import Pool, cpu_count, Manager, shared_memory
import traceback

def dark_channel(im, size=15):
    """Compute the dark channel of an image.
    im: [H,W,3], uint8 or float in [0,1]
    size: patch size
    """
    min_per_channel = np.min(im, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_per_channel, kernel)
    return dark

def estimate_atmospheric_light(im, dark, top_percent=0.001, patch_avg=3):
    """Estimate atmospheric light A
    im   : HxWx3 float32 in [0,1] (BGR if read by cv2)
    dark : HxW   float32 in [0,1] dark channel
    top_percent : fraction of pixels (e.g., 0.001 = top 0.1%) from dark channel
    patch_avg   : optional odd kernel size to average around the chosen pixel (denoise)
    Returns:
        A : (3,) atmospheric light vector in [0,1]
    """
    H, W = dark.shape
    N = H * W
    k = max(1, int(N * top_percent))

    # pick top-k brightest in dark channel (most haze-opaque)
    dark_flat = dark.reshape(-1)
    idxs = np.argpartition(dark_flat, -k)[-k:]      # top-k indices, O(N)
    # among these, choose the pixel with the highest intensity in the input image
    im_flat = im.reshape(-1, 3)
    # brightness proxy: channel sum (robust to BGR/RGB ordering)
    brightness = im_flat[idxs].sum(axis=1)
    best = idxs[np.argmax(brightness)]
    y, x = divmod(int(best), W)

    # 3) optionally average a small patch around (y,x) to reduce noise
    if patch_avg and patch_avg > 1:
        r = patch_avg // 2
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        A = im[y0:y1, x0:x1, :].reshape(-1, 3).mean(axis=0)
    else:
        A = im[y, x, :]

    return A

def estimate_transmission(I, dark, A, omega=0.95, size=15):
    """Estimate transmission  t̃(x) = 1 - ω * min_c( min_{y∈Ω(x)} I^c(y)/A^c ).
    I: HxWx3 float32 in [0,1]  (BGR if read by cv2)
    A: (3,) float32 in [0,1]   (same channel order as I)
    omega: keep-some-haze factor, typically 0.95
    size: patch size (odd), e.g. 15 for ~600x400 images
    """
    # --- safety & broadcasting ---
    I = I.astype(np.float32)
    A = A.reshape(1, 1, 3)                       

    # --- channel-wise normalization I^c / A^c ---
    norm = I / np.maximum(A, 1e-6)               
    norm = np.minimum(norm, 1.0)                 

    # --- final transmission:  t̃(x) = 1 - ω * dark_norm ---
    dark_norm = dark_channel(norm, size=size)
    t = 1.0 - omega * dark_norm
    t = np.clip(t, 0.0, 1.0)                      # keep in [0,1]
    return t

def guided_filter(I, p, r=20, eps=1e-3):
    """
    Edge-preserving guided filter (He et al., ECCV 2010).
    I: guidance image, HxW (gray) or HxWx3 (RGB), float32 [0,1]
    p: filtering input, HxW, float32 [0,1]
    r: radius
    eps: regularization
    """
    H, W = p.shape[:2]
    p = p.astype(np.float32)

    if I.ndim == 2:  # gray guidance
        I = I.astype(np.float32)
        ones = np.ones_like(p)

        mean_I = cv2.boxFilter(I, -1, (2*r+1, 2*r+1))
        mean_p = cv2.boxFilter(p, -1, (2*r+1, 2*r+1))
        mean_Ip = cv2.boxFilter(I*p, -1, (2*r+1, 2*r+1))
        cov_Ip  = mean_Ip - mean_I*mean_p

        mean_II = cv2.boxFilter(I*I, -1, (2*r+1, 2*r+1))
        var_I   = mean_II - mean_I*mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (2*r+1, 2*r+1))
        mean_b = cv2.boxFilter(b, -1, (2*r+1, 2*r+1))
        q = mean_a * I + mean_b
        return q.astype(np.float32)

    else:  # RGB guidance
        I = I.astype(np.float32)
        I_b, I_g, I_r = I[:,:,0], I[:,:,1], I[:,:,2]
        ones = np.ones_like(p)

        # means
        m_b = cv2.boxFilter(I_b, -1, (2*r+1, 2*r+1))
        m_g = cv2.boxFilter(I_g, -1, (2*r+1, 2*r+1))
        m_r = cv2.boxFilter(I_r, -1, (2*r+1, 2*r+1))
        m_p = cv2.boxFilter(p,   -1, (2*r+1, 2*r+1))

        # correlations
        m_bb = cv2.boxFilter(I_b*I_b, -1, (2*r+1, 2*r+1)); cov_bb = m_bb - m_b*m_b
        m_gg = cv2.boxFilter(I_g*I_g, -1, (2*r+1, 2*r+1)); cov_gg = m_gg - m_g*m_g
        m_rr = cv2.boxFilter(I_r*I_r, -1, (2*r+1, 2*r+1)); cov_rr = m_rr - m_r*m_r
        m_bg = cv2.boxFilter(I_b*I_g, -1, (2*r+1, 2*r+1)); cov_bg = m_bg - m_b*m_g
        m_br = cv2.boxFilter(I_b*I_r, -1, (2*r+1, 2*r+1)); cov_br = m_br - m_b*m_r
        m_gr = cv2.boxFilter(I_g*I_r, -1, (2*r+1, 2*r+1)); cov_gr = m_gr - m_g*m_r

        m_bp = cv2.boxFilter(I_b*p, -1, (2*r+1, 2*r+1)); cov_bp = m_bp - m_b*m_p
        m_gp = cv2.boxFilter(I_g*p, -1, (2*r+1, 2*r+1)); cov_gp = m_gp - m_g*m_p
        m_rp = cv2.boxFilter(I_r*p, -1, (2*r+1, 2*r+1)); cov_rp = m_rp - m_r*m_p

        # solve (Sigma + eps*I) a = cov_Ip (3x3 per-pixel)
        det = (cov_bb+eps)*(cov_gg+eps)*(cov_rr+eps) \
            + 2*cov_bg*cov_br*cov_gr \
            - (cov_bb+eps)*cov_gr*cov_gr \
            - (cov_gg+eps)*cov_br*cov_br \
            - (cov_rr+eps)*cov_bg*cov_bg
        inv_00 = (cov_gg+eps)*(cov_rr+eps) - cov_gr*cov_gr
        inv_01 = cov_br*cov_gr - cov_bg*(cov_rr+eps)
        inv_02 = cov_bg*cov_gr - (cov_gg+eps)*cov_br
        inv_11 = (cov_bb+eps)*(cov_rr+eps) - cov_br*cov_br
        inv_12 = cov_bg*cov_br - (cov_bb+eps)*cov_gr
        inv_22 = (cov_bb+eps)*(cov_gg+eps) - cov_bg*cov_bg

        a_b = ( inv_00*cov_bp + inv_01*cov_gp + inv_02*cov_rp ) / det
        a_g = ( inv_01*cov_bp + inv_11*cov_gp + inv_12*cov_rp ) / det
        a_r = ( inv_02*cov_bp + inv_12*cov_gp + inv_22*cov_rp ) / det

        b = m_p - a_b*m_b - a_g*m_g - a_r*m_r

        mean_ab = cv2.boxFilter(a_b, -1, (2*r+1, 2*r+1))
        mean_ag = cv2.boxFilter(a_g, -1, (2*r+1, 2*r+1))
        mean_ar = cv2.boxFilter(a_r, -1, (2*r+1, 2*r+1))
        mean_b  = cv2.boxFilter(b,   -1, (2*r+1, 2*r+1))
        q = mean_ab*I_b + mean_ag*I_g + mean_ar*I_r + mean_b
        return q.astype(np.float32)


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

    manager = Manager()
    process_idx = 0
    args_list = []
    image_tot_prog = (H - 2*win_radius)*(W - 2*win_radius)

    n_processes = min(cpu_count(),max_processes)
    numbers_columns_share = H - 2*win_radius

    for idx_process in range(n_processes):
        if(idx_process == n_processes - 1):
            y_min,y_max = win_radius+(numbers_columns_share//n_processes)*idx_process, H-win_radius
        else :
            y_min,y_max = win_radius+(numbers_columns_share//n_processes)*idx_process, win_radius+(numbers_columns_share//n_processes)*(idx_process+1)
        args_list.append((I_rgb,inds,win_radius,W,process_idx,n_processes,y_min,y_max,eps,K))
        process_idx+=1

    print(f"========== all processes are starting ! {n_processes} ===========")

    with Pool(processes=n_processes) as pool:
        results = pool.starmap(one_line_soft_matting, args_list)

    rows = np.concatenate([r[0] for r in results]).astype(np.float32)
    cols = np.concatenate([r[1] for r in results]).astype(np.float32)
    vals = np.concatenate([r[2] for r in results]).astype(np.float32)

    print("loading laplacian : 100% !")
    print("Putting it all in order (1/3)")
    # (slow !)
    L = csr_matrix((vals, (rows, cols)), shape=(N, N))
    print("Putting it all in order (2/3)")
    # Data term: lambda * (t - t0)^2
    A = L + lam * eye(N, format='csr')
    print("Putting it all in order (3/3)")
    b = lam * t_coarse.reshape(-1)

    print("loading laplacian : 100% ! Inverting matrix...")

    # Solve sparse linear system (slow !)
    t_refined, info = cg(A, b, rtol=1e-6, maxiter = maxiter)
    if info != 0:
        print("⚠️ CG did not fully converge, info =", info)
    t_refined = t_refined.reshape(H, W).astype(np.float32)

    # Clamp to [0,1]
    print("soft matting completed ! (3/3)")
    return np.clip(t_refined, 0.0, 1.0)

def one_line_soft_matting(I_rgb,inds,win_radius,W,process_idx,n_processes,y_min,y_max,eps,K):
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


def recover_radiance(I, A, t, t0=0.1):
    """Recover haze-free radiance J from hazy image I
    I: HxWx3 float32 in [0,1]
    A: (3,) float32 in [0,1]   (same channel order as I)
    t: HxW   float32 in [0,1]  (refined transmission)
    t0: lower bound to avoid noise amplification (typ. 0.1)
    """
    I = I.astype(np.float32)
    A = A.reshape(1, 1, 3).astype(np.float32)  
    t = np.clip(t, t0, 1.0)                    
    J = (I - A) / t[..., None] + A             
    return np.clip(J, 0.0, 1.0)         

def dehaze(img_path, custom_output_name, out_dir="dehazed_results",
           dc_size=15, top_percent=0.001, patch_avg=1,
           omega=0.95, w_radius=2, eps=1e-3, t0=0.1):
    """
    Full single-image dehazing pipeline (He et al. 2009).
    img_path : path to hazy image
    out_dir  : directory to save dehazed result
    """
    # 1. read image
    I = cv2.imread(img_path).astype('float32') / 255.0

    # 2. dark channel
    dark = dark_channel(I, dc_size)

    # 3. atmospheric light
    A = estimate_atmospheric_light(I, dark, top_percent, patch_avg)

    # 4. coarse transmission
    t_coarse = estimate_transmission(I, dark, A, omega)

    # 5. refine transmission
    try:
        # gray-scale guidance is faster and often sufficient; but RGB guidance can be used as well for potentially better results
        I_guide_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        t_refined = guided_filter(I_guide_gray, t_coarse, r=w_radius*10, eps=max(1e-4, eps))
        t_refined = np.clip(t_refined, 0.0, 1.0)
    except Exception as e:
        print(f"[WARN] Guided filtering failed ({e}), using coarse transmission.")
        t_refined = t_coarse

    # 6. recover scene radiance
    J = recover_radiance(I, A, t_refined, t0)

    # 7. save result
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    if custom_output_name is not None:
        out_path = os.path.join(out_dir, f"{custom_output_name}_dehazed.png")
    else :
        out_path = os.path.join(out_dir, f"{base_name}_dehazed.png")
    cv2.imwrite(out_path, (J * 255).astype('uint8'))
    print(f"[INFO] Dehazed image saved to {out_path}")
    return J, t_refined, A

def transmission_cut_apply_soft_matting(I_rgb : np.ndarray,t_coarse : np.ndarray , maxiter : int, n_cut_width : int ,n_cut_height : int, win_radius : int, eps : float ,lam : float, max_processes, ratio : float = 0.5) -> list[np.ndarray]:
    height,width = t_coarse.shape
    height_cut = height//n_cut_height
    width_cut = width//n_cut_width
    transmission_patches = []
    I_rgb_patches = []
    coords = []

    print("cutting the image before doing soft matting")

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

    print([m.shape for m in transmission_patches])
    print([m.shape for m in I_rgb_patches])

    loading_total = len(transmission_patches)
    loading_counter = 0
    t_refined_full = np.zeros((height, width), dtype=np.float32)

    for t_patch,i_patch,coord in zip(transmission_patches,I_rgb_patches,coords):
        print(f"============= patch {t_patch.shape} started ! : {loading_counter}/{loading_total} =============")
        i_min, i_max, j_min, j_max = coord
        refined_patch = soft_matting(i_patch,t_patch, maxiter, win_radius, eps, lam, max_processes)
        refined_patch = refined_patch.reshape(i_max - i_min, j_max - j_min)
        t_refined_full[i_min:i_max,j_min:j_max] = refined_patch
        loading_total+=1
        print(f"============= patch {t_patch.shape} finished ! : {loading_counter}/{loading_total} =============")
    
    return np.clip(t_refined_full,0.0,1.0)

# ==================== tried to cut the picture into pieces and multi-process this - ugly result =======================

#     manager = Manager()
#     started_counter = manager.Value('i', 0)
#     loading_counter = manager.Value('i', 0)

#     args_list = []
#     for (I_patch ,t_patch, (i_min, i_max, j_min, j_max)) in zip(I_rgb_patches,transmission_patches, coords):
#         args_list.append((I_patch,t_patch,i_min,i_max,j_min,j_max,win_radius,eps,lam,started_counter,loading_counter,loading_total))
    
#     with Pool(processes=1) as pool:
#         results = pool.starmap(patch_soft_matting_process, args_list)

#     print(f"============= patch all treated ! =============")

#     t_refined_full = np.zeros((height, width), dtype=np.float32)

#     for patch, i_min, i_max, j_min, j_max in results:
#         t_refined_full[i_min:i_max, j_min:j_max] = patch

#     t_refined_full = cv2.GaussianBlur(t_refined_full, (11,11), sigmaX=1.0)

#     return np.clip(t_refined_full, 0.0, 1.0)

# def patch_soft_matting_process(I_patch,t_patch,i_min,i_max,j_min,j_max,win_radius,eps,lam,started_counter,loading_counter,loading_total):
#     started_counter.value+=1
#     print(f"============= patch {t_patch.shape} started ! : {started_counter.value}/{loading_total} =============")
#     refined_patch = soft_matting(I_patch,t_patch, win_radius, eps, lam)
#     refined_patch = refined_patch.reshape(i_max - i_min, j_max - j_min)
#     loading_counter.value+=1
#     print(f"============= patch {t_patch.shape} finished ! : {loading_counter.value}/{loading_total} =============")
#     return refined_patch,i_min,i_max,j_min,j_max


# Example usage
if __name__ == "__main__":
    # dehaze(img_path = "./hazed_images/1.jpg",
    #        maxiter = 10000,                         # number of maximal iteration before attaining the inverse
    #        out_dir="./dehazed_results",             
    #        dc_size = 15,                            # size of convolution kernel of the dark channel
    #        top_percent = 0.001,                     # choose the top percent of the brightest pixels in the dark channel
    #        patch_avg = 1,                           # size of the patch to average the pixel chosen to be the ambient light around its place 
    #        omega = 0.95,                            # here to lessen the impact of the dark channel on the transmission light : the less it is, the less the dark channel influences the transmission map.                          # patch
    #        w_radius = 1,                            # window of convolution for soft-matting (2*win_radius+1)
    #        eps = 10E-7,                             # an epsilon in the formula of soft-matting
    #        lam = 10E-4,                             # an lambda in the soft-matting formula
    #        t0 = 0.1,                                # the inferior bound of the refined transmission
    #        max_processes = 6,
    #        custom_output_name = name                # custom saved file name if you want one
    #        )                       # maximum number of processes. Be careful of your logical core number and your RAM.

    img_path = r"C:\Users\22863\Desktop\git\ima1\ima_projet\dehazer\hazed_images\11.png"

    base_parameters = {
    "img_path": img_path,
    "out_dir": "seriespicturesoutput",
    "dc_size": 15,
    "top_percent": 0.001,
    "patch_avg": 1,
    "omega": 0.95,
    "w_radius": 1,
    "eps": 1e-3,  
    "t0": 0.1,
}

    modified_params_list = [
        {
    "omega" : 0.50,
    "t0" : 0.01
    }
    ,
            {
    "omega" : 0.70,
    "t0" : 0.01
    }
    ,
            {
    "omega" : 0.80,
    "t0" : 0.01
    }
    ,
            {
    "omega" : 0.90,
    "t0" : 0.01
    }
    ,
            {
    "omega" : 0.95,
    "t0" : 0.01
    }
    ,
            {
    "omega" : 0.99,
    "t0" : 0.01
    }
    ,
    ]

    full_params_list = [
        {**base_parameters, **mod} for mod in modified_params_list
    ]
    
    timings = []
    
    for i, params in enumerate(full_params_list):
        start = perf_counter()
        dehaze(**params, custom_output_name=f"{i}")
        elapsed = perf_counter() - start
        timings.append((i, elapsed, params["omega"], params["t0"]))

    # print the summary of the time
    print("\n========== Processing Summary ==========")
    print(f"Total images processed: {len(timings)}")
    for idx, time, omega, t0 in timings:
        print(f"Image #{idx}: {time:.2f}s (omega={omega}, t0={t0})")
    print("=====================================")
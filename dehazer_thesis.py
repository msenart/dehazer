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
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

def dark_channel(im, size=15): #size = 15 for ~600x400 images, can be adjusted for different resolutions
    """Compute the dark channel of an image.
    im: [H,W,3], uint8 or float in [0,1]
    size: patch size
    """
    min_per_channel = np.min(im, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_per_channel, kernel)
    return dark

def estimate_atmospheric_light(im, dark, top_percent=0.001, patch_avg=3): #by default patch(size = 3) averaging
    """Estimate atmospheric light A as in He et al. (2009).
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

    # 1) pick top-k brightest in dark channel (most haze-opaque)
    dark_flat = dark.reshape(-1)
    idxs = np.argpartition(dark_flat, -k)[-k:]      # top-k indices, O(N)
    # 2) among these, choose the pixel with the highest intensity in the input image
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

def estimate_transmission(I, A, omega=0.95, size=15):
    """Estimate transmission  t̃(x) = 1 - ω * min_c( min_{y∈Ω(x)} I^c(y)/A^c ).
    I: HxWx3 float32 in [0,1]  (BGR if read by cv2)
    A: (3,) float32 in [0,1]   (same channel order as I)
    omega: keep-some-haze factor, typically 0.95
    size: patch size (odd), e.g. 15 for ~600x400 images
    """
    # --- safety & broadcasting ---
    I = I.astype(np.float32)
    A = A.reshape(1, 1, 3)                        # shape -> [1,1,3] for broadcasting

    # --- channel-wise normalization I^c / A^c ---
    norm = I / np.maximum(A, 1e-6)               # avoid divide-by-zero
    norm = np.minimum(norm, 1.0)                 # clamp to [0,1] as in the paper's assumption

    # --- per-channel patch minimum:  min_{y∈Ω(x)} (I^c(y)/A^c) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    min_r = cv2.erode(norm[:, :, 0], kernel)     # channel R (or B if using cv2/BGR)
    min_g = cv2.erode(norm[:, :, 1], kernel)
    min_b = cv2.erode(norm[:, :, 2], kernel)

    # --- then take min over channels:  min_c ( · ) ---
    dark_norm = np.minimum(np.minimum(min_r, min_g), min_b)

    # --- final transmission:  t̃(x) = 1 - ω * dark_norm ---
    t = 1.0 - omega * dark_norm
    t = np.clip(t, 0.0, 1.0)                      # keep in [0,1]
    return t

def soft_matting(I_rgb, t_coarse, win_radius=1, eps=1e-7, lam=1e-4):
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

    # pre-allocate lists for sparse laplacian L
    rows, cols, vals = [], [], []
    loading_counter = 0
    loading_total = (H-2*win_radius)*(W-2*win_radius)

    # For each window, compute local statistics and add to Laplacian
    for y in range(win_radius, H - win_radius):
        for x in range(win_radius, W - win_radius):
            if (loading_counter%int(loading_total/100) == int(loading_total/100)-1):
                print(f"loading laplacian : {int((loading_counter/loading_total)*100)}%")
            ys, ye = y - win_radius, y + win_radius + 1
            xs, xe = x - win_radius, x + win_radius + 1

            win_inds = inds[ys:ye, xs:xe].reshape(-1)
            win_I = I_rgb[ys:ye, xs:xe, :].reshape(K, 3).astype(np.float64)

            mu = win_I.mean(axis=0, keepdims=True)           # 1x3
            cov = (win_I - mu).T @ (win_I - mu) / K          # 3x3
            cov_reg = cov + (eps / K) * np.eye(3)            # regularized

            # inverse covariance
            inv = np.linalg.inv(cov_reg)

            # (I - 1/K) operator
            X = win_I - mu                                   # Kx3
            M = np.eye(K) - np.ones((K, K)) / K              # KxK

            # L_w = M - X * inv * X^T / K
            # compute X * inv * X^T efficiently
            Xin = X @ inv                                    # Kx3
            Q = (Xin @ X.T) / K                              # KxK
            Lw = M + Q * 0.0                                 # start with M
            Lw -= Q                                          # Lw = M - Q

            # scatter-add to global Laplacian
            ii = np.repeat(win_inds, K)
            jj = np.tile(win_inds, K)
            rows.append(ii)
            cols.append(jj)
            vals.append(Lw.reshape(-1))
            loading_counter+=1

    print("loading laplacian : 100% ! Inverting matrix...")

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    L = csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Data term: lambda * (t - t0)^2
    A = L + lam * eye(N, format='csr')
    b = lam * t_coarse.flatten()
    
    # Solve sparse linear system
    t_refined = spsolve(A, b).reshape(H, W).astype(np.float32)

    # Clamp to [0,1]
    print("soft matting completed !")
    return np.clip(t_refined, 0.0, 1.0)

def recover_radiance(I, A, t, t0=0.1):
    """Recover haze-free radiance J from hazy image I using Eq.(16).
    I: HxWx3 float32 in [0,1]
    A: (3,) float32 in [0,1]   (same channel order as I)
    t: HxW   float32 in [0,1]  (refined transmission)
    t0: lower bound to avoid noise amplification (typ. 0.1)
    """
    I = I.astype(np.float32)
    A = A.reshape(1, 1, 3).astype(np.float32)  # broadcast A to each pixel
    t = np.clip(t, t0, 1.0)                    # max(t(x), t0)
    J = (I - A) / t[..., None] + A             # Eq.(16)
    return np.clip(J, 0.0, 1.0)                # keep valid range

def dehaze(img_path, out_dir="dehazed_results", dc_size = 15, top_percent = 0.001, patch_avg = 1, omega = 0.95, t_size = 15, w_radius = 1, eps = 10E-7, lam = 10E-4, t0 = 0.1):
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
    t_coarse = estimate_transmission(I, A, omega, t_size)

    # 5. refine transmission
    try:
        I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        t_refined = soft_matting(I_rgb, t_coarse, w_radius, eps, lam)
    except Exception as e:
        print(f"[WARN] Soft matting failed ({e}), using coarse transmission.")
        t_refined = t_coarse

    # 6. recover scene radiance
    J = recover_radiance(I, A, t_refined, t0)

    # 7. save result
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_dehazed.png")
    cv2.imwrite(out_path, (J * 255).astype('uint8'))
    print(f"[INFO] Dehazed image saved to {out_path}")
    return J, t_refined, A

# Example usage
if __name__ == "__main__":
    dehaze("images/haze_test.jpeg","images",
           dc_size = 5, w_radius = 4, patch_avg = 3)
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
import traceback
from typing import Callable, Any

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

def estimate_transmission(I, dark, A, omega=0.95):
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
    t = 1.0 - omega * dark
    t = np.clip(t, 0.0, 1.0)                      # keep in [0,1]
    return t

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

def dehaze(img_path, smoothing_method : Callable[...,Any], kwargs : dict[str,Any],
           dc_size, top_percent, patch_avg, omega, t0, show_steps = False,
           out_dir="dehazed_results", custom_output_name=None):
    """
    Full single-image dehazing pipeline (He et al. 2009) with soft matting refinement.
    Saves intermediate steps + composite comparison image.
    """

    def norm_gray(x):
        """Normalize to 8-bit grayscale"""
        x = np.clip(x, 0, 1)
        x = (x * 255).astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    def norm_color(x):
        """Normalize a color image (already BGR)"""
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    # --- 1. read image ---
    I = cv2.imread(img_path).astype('float32') / 255.0
    initial_image = norm_color(I)

    # --- 2. dark channel ---
    dark = dark_channel(I, dc_size)
    dark_channel_i = norm_gray(dark)

    # --- 3. atmospheric light ---
    A = estimate_atmospheric_light(I, dark, top_percent, patch_avg)

    # --- 4. coarse transmission ---
    t_coarse = estimate_transmission(I, dark, A, omega)
    t_coarse_i = norm_gray(t_coarse)
    # --- 5. refine transmission with soft matting ---
    try:
        t_refined = smoothing_method(I,t_coarse,**kwargs)
        t_refined = np.clip(t_refined, 0.0, 1.0)
        t_refined_i = norm_gray(t_refined)
    except Exception as e:
        print(f"[WARN] Soft matting failed ({e}), using coarse transmission.")
        traceback.print_exc()
        t_refined = t_coarse
        t_refined_i = norm_gray(t_refined)

    # --- 6. recover scene radiance ---
    J = recover_radiance(I, A, t_refined, t0)
    final_image = norm_color(J)

    # --- 7. create the big folder if doesn't exist ---
    os.makedirs(out_dir, exist_ok=True)

    # Save final dehazed result

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    total_path = os.path.join(out_dir,f"{base_name}_pipeline")
    print(f"[INFO] Dehazed image saved to {base_name}_dehazed.png")
    
    os.makedirs(total_path,exist_ok=True)
    cv2.imwrite(os.path.join(total_path,f"{base_name}_initial.png"),initial_image)
    cv2.imwrite(os.path.join(total_path,f"{base_name}_dc.png"),dark_channel_i)
    cv2.imwrite(os.path.join(total_path,f"{base_name}_tcoarse.png"),t_coarse_i)
    cv2.imwrite(os.path.join(total_path,f"{base_name}_trefined.png"),t_refined_i)
    cv2.imwrite(os.path.join(total_path,f"{base_name}_final.png"),final_image)
    
    return total_path





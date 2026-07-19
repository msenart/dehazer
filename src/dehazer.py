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
import logging
import json

logger = logging.getLogger("widget_logger")

class dehazer_data:
    DEFAULT_DEHAZE_PARAMS = {
    "dc_size": 15,
    "top_percent": 0.001,
    "patch_avg": 2,
    "omega": 0.95,
    "t0": 0.01
    }

def dark_channel(im, size=15):
    min_per_channel = np.min(im, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_per_channel, kernel)
    return dark

def estimate_atmospheric_light(im, dark, top_percent=0.001, patch_avg=3):
    H, W = dark.shape
    N = H * W
    k = max(1, int(N * top_percent))

    dark_flat = dark.reshape(-1)
    idxs = np.argpartition(dark_flat, -k)[-k:]
    im_flat = im.reshape(-1, 3)
    brightness = im_flat[idxs].sum(axis=1)
    best = idxs[np.argmax(brightness)]
    y, x = divmod(int(best), W)

    if patch_avg and patch_avg > 1:
        r = patch_avg // 2
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        A = im[y0:y1, x0:x1, :].reshape(-1, 3).mean(axis=0)
    else:
        A = im[y, x, :]

    return A

def estimate_transmission(I, dark, A, omega=0.95, size=15):
    I = I.astype(np.float32)
    A = A.reshape(1, 1, 3)

    norm = I / np.maximum(A, 1e-6)
    norm = np.clip(norm, 0, 1)

    dark_norm = cv2.erode(np.min(norm, axis=2), np.ones((size,size)))

    t = 1.0 - omega * dark_norm
    t = np.clip(t, 0.0, 1.0)
    return t

def recover_radiance(I, A, t, t0=0.1):
    I = I.astype(np.float32)
    A = A.reshape(1, 1, 3).astype(np.float32)  
    t = np.clip(t, t0, 1.0)                    
    J = (I - A) / t[..., None] + A             
    return np.clip(J, 0.0, 1.0)         

def dehaze(img_path, smoothing_method : Callable[...,Any], kwargs : dict[str,Any],
           dc_size, top_percent, patch_avg, omega, t0, show_steps = False,
           out_dir = None, custom_output_name=None):

    def norm_gray(x):
        x = np.clip(x, 0, 1)
        x = (x * 255).astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    def norm_color(x):
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    I = cv2.imread(img_path).astype('float32') / 255.0

    dark = dark_channel(I, dc_size)
    
    A = estimate_atmospheric_light(I, dark, top_percent, patch_avg)

    t_coarse = estimate_transmission(I, dark, A, omega)
    
    try:
        t_refined = smoothing_method(I,t_coarse,**kwargs)
        t_refined = np.clip(t_refined, 0.0, 1.0)
        t_refined_i = norm_gray(t_refined)
    except Exception as e:
        logger.info(f"Soft matting failed ({e}), using coarse transmission.")
        traceback.print_exc()
        t_refined = t_coarse
        
    J = recover_radiance(I, A, t_refined, t0)
    
    if out_dir :
        os.makedirs(out_dir, exist_ok=True)

        initial_image = norm_color(I)
        dark_channel_i = norm_gray(dark)
        t_coarse_i = norm_gray(t_coarse)
        t_refined_i = norm_gray(t_refined)
        final_image = norm_color(J)

        params_to_save = {
            "dehaze_params": {
                "smoothing algorithm" : smoothing_method.__name__,
                "dc_size": dc_size,
                "top_percent": top_percent,
                "patch_avg": patch_avg,
                "omega": omega,
                "t0": t0
            },
            "algo_params": kwargs
        }

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        i = 0
        while True:
            if i == 0:
                total_path = os.path.join(out_dir, f"{base_name}_pipeline")
            else:
                total_path = os.path.join(out_dir, f"{base_name}_pipeline_{i}")
            json_path = os.path.join(total_path, "params.json")

            if not os.path.exists(total_path):
                break
            else :
                if not os.path.exists(json_path):
                    break

            with open(json_path, "r") as f:
                older_params = json.load(f)

            if older_params != params_to_save:
                i += 1
                continue
            else:
                break

        logger.info(f"💾 Parameters saved to {json_path}")

        logger.info(f"Dehazed image saved to {base_name}_dehazed.png")
        
        os.makedirs(total_path,exist_ok=True)
        cv2.imwrite(os.path.join(total_path,f"{base_name}_initial.png"),initial_image)
        cv2.imwrite(os.path.join(total_path,f"{base_name}_dc.png"),dark_channel_i)
        cv2.imwrite(os.path.join(total_path,f"{base_name}_tcoarse.png"),t_coarse_i)
        cv2.imwrite(os.path.join(total_path,f"{base_name}_trefined.png"),t_refined_i)
        cv2.imwrite(os.path.join(total_path,f"{base_name}_final.png"),final_image)
        
        with open(json_path, "w") as f:
            json.dump(params_to_save, f, indent=4)

        return total_path
    else :
        return J





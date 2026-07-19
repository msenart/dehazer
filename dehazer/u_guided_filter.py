"""Guided filter transmission-map refinement (grayscale and color-guide variants)."""

import cv2
import numpy as np
import logging

logger = logging.getLogger("widget_logger")

class guided_filter_data:
    """Parameter schema and defaults for guided_filter(), shared with the GUI's parameter form."""

    ALGO_PARAMS = {
            "r": "int", "eps": "float"
        }
    DEFAULT_ALGO_PARAMS = {
            "r": 5,
            "eps": 0.01
        }

# --- Main function ---

def guided_filter(I, p, r=20, eps=1e-3):
    """Refine transmission map p using guidance image I (He et al.'s guided filter).

    Supports both a single-channel (grayscale) and a 3-channel (color) guidance
    image I; the color case solves the per-pixel 3x3 linear coefficients directly.
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    H, W = p.shape[:2]

    def box(x):
        return cv2.boxFilter(x, -1, (2*r+1, 2*r+1))

    if I.ndim == 2:
        mean_I = box(I)
        mean_p = box(p)
        mean_Ip = box(I*p)
        cov_Ip = mean_Ip - mean_I*mean_p
        mean_II = box(I*I)
        var_I = mean_II - mean_I*mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a*mean_I
        mean_a = box(a)
        mean_b = box(b)
        q = mean_a*I + mean_b
        return q

    else:
        I_b, I_g, I_r = I[:,:,0], I[:,:,1], I[:,:,2]

        mean_b = box(I_b)
        mean_g = box(I_g)
        mean_r = box(I_r)
        mean_p = box(p)

        cov_bb = box(I_b*I_b) - mean_b*mean_b + eps
        cov_gg = box(I_g*I_g) - mean_g*mean_g + eps
        cov_rr = box(I_r*I_r) - mean_r*mean_r + eps
        cov_bg = box(I_b*I_g) - mean_b*mean_g
        cov_br = box(I_b*I_r) - mean_b*mean_r
        cov_gr = box(I_g*I_r) - mean_g*mean_r

        cov_bp = box(I_b*p) - mean_b*mean_p
        cov_gp = box(I_g*p) - mean_g*mean_p
        cov_rp = box(I_r*p) - mean_r*mean_p

        det = cov_bb*cov_gg*cov_rr + 2*cov_bg*cov_br*cov_gr - cov_bb*cov_gr*cov_gr - cov_gg*cov_br*cov_br - cov_rr*cov_bg*cov_bg
        inv_00 = cov_gg*cov_rr - cov_gr*cov_gr
        inv_01 = cov_br*cov_gr - cov_bg*cov_rr
        inv_02 = cov_bg*cov_gr - cov_gg*cov_br
        inv_11 = cov_bb*cov_rr - cov_br*cov_br
        inv_12 = cov_bg*cov_br - cov_bb*cov_gr
        inv_22 = cov_bb*cov_gg - cov_bg*cov_bg

        a_b = (inv_00*cov_bp + inv_01*cov_gp + inv_02*cov_rp) / det
        a_g = (inv_01*cov_bp + inv_11*cov_gp + inv_12*cov_rp) / det
        a_r = (inv_02*cov_bp + inv_12*cov_gp + inv_22*cov_rp) / det

        b = mean_p - a_b*mean_b - a_g*mean_g - a_r*mean_r

        mean_a_b = box(a_b)
        mean_a_g = box(a_g)
        mean_a_r = box(a_r)
        mean_b = box(b)

        q = mean_a_b*I_b + mean_a_g*I_g + mean_a_r*I_r + mean_b
        return q
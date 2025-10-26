#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Image Dehazing using Dark Channel Prior
----------------------------------------------
Implementation of He et al., 2009 (CVPR).
Comments are in English.
"""
import cv2
import numpy as np

### MAIN FUNCTION BELOW ========================================================================================================

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

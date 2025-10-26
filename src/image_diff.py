import cv2
import numpy as np
import os

class ImageComparator:
    """Classe utilitaire pour comparer deux images et retourner les différences canal par canal."""

    @staticmethod
    def read_image(path: str):
        """Lit une image à partir d'un chemin."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image introuvable : {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image : {path}")
        return img

    @staticmethod
    def compare_images(path1: str, path2: str):
        """
        Compare deux images (même taille) et retourne les différences absolues
        canal par canal (B, G, R).
        """
        img1 = ImageComparator.read_image(path1)
        img2 = ImageComparator.read_image(path2)

        if img1.shape != img2.shape:
            raise ValueError("Les deux images doivent avoir la même taille et le même nombre de canaux.")

        diff_b = cv2.absdiff(img1[:, :, 0], img2[:, :, 0])
        diff_g = cv2.absdiff(img1[:, :, 1], img2[:, :, 1])
        diff_r = cv2.absdiff(img1[:, :, 2], img2[:, :, 2])

        return diff_b, diff_g, diff_r
    
    

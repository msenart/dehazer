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
        Compare deux images (même taille) et retourne les différences absolues.
        - Si les deux sont en niveaux de gris → 1 canal.
        - Si les deux sont en couleur → 3 canaux (B, G, R).
        - Si les formats diffèrent → convertit en niveaux de gris.
        """
        img1 = ImageComparator.read_image(path1)
        img2 = ImageComparator.read_image(path2)

        # Vérifie les dimensions
        if img1.shape != img2.shape:
            raise ValueError("Les deux images doivent avoir la même taille et le même nombre de canaux.")

        # Détection du nombre de canaux
        def is_really_grayscale(img):
            if len(img.shape) == 2:
                return True
            if img.shape[2] == 1:
                return True
            # Vérifie si tous les canaux sont identiques
            return np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])

        gray1 = is_really_grayscale(img1)
        gray2 = is_really_grayscale(img2)

        # Cas 1 : les deux images sont en niveaux de gris
        if gray1 and gray2:
            diff = cv2.absdiff(img1, img2)

            return {"mode": "grayscale", "diff": diff}

        # Cas 2 : les deux images sont en couleur
        elif not gray1 and not gray2:
            diff_b = cv2.absdiff(img1[:, :, 0], img2[:, :, 0])
            diff_g = cv2.absdiff(img1[:, :, 1], img2[:, :, 1])
            diff_r = cv2.absdiff(img1[:, :, 2], img2[:, :, 2])
            return {"mode": "color", "diff_b": diff_b, "diff_g": diff_g, "diff_r": diff_r}

        # Cas 3 : mélange couleur / gris → on convertit les deux en niveaux de gris
        else:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if not gray1 else img1
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if not gray2 else img2
            diff = cv2.absdiff(img1_gray, img2_gray)

            return {"mode": "converted_to_grayscale", "diff": diff}
    
    

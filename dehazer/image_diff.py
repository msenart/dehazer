"""Channel-wise comparison of two images, used by the GUI's difference tabs."""

import cv2
import numpy as np
import os

class ImageComparator:
    """Utility class to compare two images and return the differences channel by channel."""

    @staticmethod
    def read_image(path: str):
        """Read an image from a path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Unable to read image: {path}")
        return img

    @staticmethod
    def compare_images(path1: str, path2: str):
        """
        Compare two images (same size) and return the absolute differences.
        - If both are grayscale -> 1 channel.
        - If both are color -> 3 channels (B, G, R).
        - If formats differ -> convert to grayscale.
        """
        img1 = ImageComparator.read_image(path1)
        img2 = ImageComparator.read_image(path2)

        # Check dimensions
        if img1.shape != img2.shape:
            raise ValueError("Both images must have the same size and number of channels.")

        # Detect the number of channels
        def is_really_grayscale(img):
            if len(img.shape) == 2:
                return True
            if img.shape[2] == 1:
                return True
            # Check whether all channels are identical
            return np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])

        gray1 = is_really_grayscale(img1)
        gray2 = is_really_grayscale(img2)

        # Case 1: both images are grayscale
        if gray1 and gray2:
            diff = cv2.absdiff(img1, img2)

            return {"mode": "grayscale", "diff": diff}

        # Case 2: both images are color
        elif not gray1 and not gray2:
            diff_b = cv2.absdiff(img1[:, :, 0], img2[:, :, 0])
            diff_g = cv2.absdiff(img1[:, :, 1], img2[:, :, 1])
            diff_r = cv2.absdiff(img1[:, :, 2], img2[:, :, 2])
            return {"mode": "color", "diff_b": diff_b, "diff_g": diff_g, "diff_r": diff_r}

        # Case 3: mixed color / grayscale -> convert both to grayscale
        else:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if not gray1 else img1
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if not gray2 else img2
            diff = cv2.absdiff(img1_gray, img2_gray)

            return {"mode": "converted_to_grayscale", "diff": diff}

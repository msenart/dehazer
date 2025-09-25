from typing import List, Tuple, Callable
import numpy as np
import cupy as cp
from PIL import Image

class ImageTransformer :

    def __init__(self,image:Image):
        self.img = image
        self.img_array = np.array(image)
        self.darkchannel = np.zeros(np.shape(image))
        self.laplacian_matrix = np.zeros(np.shape(image))

    @staticmethod
    def get_patch_from_image(image: np.ndarray,center : Tuple[int,int],radius : int):
        n_row, n_column, n_channels = image.shape
        i,j = center
        rmin = max(i - radius, 0)
        rmax = min(i + radius + 1, n_row)
        cmin = max(j - radius, 0)
        cmax = min(j + radius + 1, n_column)

        patch = image[rmin:rmax, cmin:cmax, :]
        return patch


    def get_rough_darkchannel(self, radius: int):
        n_row, n_column, n_channels = self.img_array.shape
        darkchannel = np.zeros((n_row, n_column))

        for i in range(n_row):
            for j in range(n_column):

                patch = ImageTransformer.get_patch_from_image(self.img_array,(i,j),radius)

                darkpixel = patch.min()
                darkchannel[i, j] = darkpixel

        return darkchannel
    
    def get_rough_transmission(self,w : float = 0.95) -> np.ndarray:
        return 1 - self.darkchannel/self.get_atmospheric_light()*w

    def get_laplacian_matrix(self, radius: int, eps: float):
            n_row, n_column, n_channel = self.img_array.shape
            laplacian_matrix = np.zeros((n_row, n_column), dtype=np.float32)
            
            # Chaque pixel devient un vecteur RGB
            image_list = self.img_array.reshape(-1, 3)
            
            loading_counter = 0
            loading_total = n_row * n_column
            
            for i in range(n_row):
                for j in range(n_column):
                    print(f"laplacian : ({loading_counter}/{loading_total})")
                    patches_around = []
                    
                    # Récupération des patches autour en gérant les bords
                    for k in range(i - radius, i + radius + 1):
                        for l in range(j - radius, j + radius + 1):
                            patch = self.get_patch_from_image(self.img_array, (k, l), radius)
                            if patch.size > 0:
                                patches_around.append(patch)
                    
                    laplacian_term = float(i == j)
                    
                    for patch in patches_around:
                        h, w, c = patch.shape
                        patch_cardinal = h * w
                        patch_mean = np.mean(patch, axis=(0, 1))
                        
                        # Covariance 3x3 avec régularisation
                        pixels = patch.reshape(-1, 3).T
                        patch_covariance = np.cov(pixels)
                        eps_matrix = (eps / patch_cardinal) * np.eye(3)
                        inv_cov = np.linalg.inv(patch_covariance + eps_matrix)
                        
                        diff = image_list[i * n_column + j] - patch_mean
                        laplacian_term -= (1 / patch_cardinal) * (1 + diff.T @ inv_cov @ diff)
                    
                    laplacian_matrix[i, j] = laplacian_term
                    loading_counter += 1
            
            return laplacian_matrix

    def get_refined_transmission(self,rough_t : np.ndarray,  lmbd : float = 10E-4) -> np.ndarray:
        assert rough_t.shape == self.laplacian_matrix
        return lmbd*np.linalg.inv(self.laplacian_matrix+lmbd*np.eye(self.laplacian_matrix.shape))@rough_t
    
    def get_atmospheric_light(self) -> float:
        return np.max(self.darkchannel)

    def init_dehaze(self,radius : int = 5):
        self.darkchannel = self.get_rough_darkchannel(radius)
        print(f"dark channel loaded !")
        self.laplacian_matrix = self.get_laplacian_matrix(radius,0.001)
        print(f"laplacian matrix loaded !")
        self.atmospheric_light = self.get_atmospheric_light()
        print(f"atmospheric light loaded ! Ready to calculate !")

    def process_dehaze(self,radius : int = 5,t0 : float = 0.95) -> Image:
        print("processing dehazement...")
        self.transmission_map = self.get_refined_transmission(self.get_rough_transmission())
        self.dehazed_image = np.zeros(self.img_array.shape[:2])
        for i in range(len(self.dehazed_image)):
            for j in range(len(self.dehaze_image[0])):
                self.dehazed_image[i,j] = (self.img_array[i,j]-self.get_atmospheric_light())/max(self.transmission_map[i,j],t0)+self.atmospheric_light
        image = Image.fromarray(self.dehazed_image.astype(np.uint8))
        return image

def patching_image(image : np.ndarray,width : int,height : int) -> List[np.ndarray]:
    patches_list = []
    (n_row,n_column,n_channels) = image.shape
    for i in range(0,n_row,height):
        for j in range(0,n_column,width):
            new_patch = image[i:i+height,j:j+width,:]
            patches_list.append(new_patch)
            if (j + width >= n_column and i + height < n_row):
                new_patch = image[i:i+height,j:j:-1,:]
                patches_list.append(new_patch)
            if (j + width < n_column and i + height >= n_row):
                new_patch = image[i:-1,j:j+width,:]
                patches_list.append(new_patch)
            if (j + width >= n_column and i + height >= n_row):
                new_patch = image[i:-1,j:-1,:]
                patches_list.append(new_patch)
    return patches_list

def apply_contrast(image : np.ndarray,f: Callable[[float], float]) -> np.ndarray :
    (n_row,n_column,n_channels) = image.shape
    for i in range(n_row):
        for j in range(n_column):
            image[i,j,0] = f(image[i,j,0])
            image[i,j,1] = f(image[i,j,1])
            image[i,j,2] = f(image[i,j,2])
    return image

def monomial(x : float):
    alpha = 1

    x = x/255

    y = x**alpha

    y = y*255
    return y

def cosinus(x:float):

    x = x/255

    y = (1-np.cos(np.pi*(x)))/2

    y = y*255
    return y



if __name__ == "__main__":
    image_name = "test.jpg"
    image = Image.open(f"images/{image_name}")
    imageTransformer = ImageTransformer(image)
    imageTransformer.init_dehaze()
    result = imageTransformer.process_dehaze()
    result.show()
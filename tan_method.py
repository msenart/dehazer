from typing import List, Tuple, Callable
import numpy as np
from PIL import Image

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
    image_name = "ixion.jpg"
    image = Image.open(f"images/{image_name}")
    image_array = np.array(image)
    image_array = apply_contrast(image_array,cosinus)
    image_array = apply_contrast(image_array,cosinus)


    result = image_array.astype(np.uint8)
    result = Image.fromarray(image_array)
    result.show()
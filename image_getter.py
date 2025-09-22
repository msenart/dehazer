from PIL import Image;
import numpy as np;
import matplotlib as plt;


image_name = "ixion.jpg"
image = Image.open(f"images/{image_name}")

image.show()

image_array = np.array(image)

red_channel = np.zeros_like(image_array)

red_channel[:,:,0] = image_array[:,:,0]

red_image = Image.fromarray(red_channel)
red_image.show()

green_channel = np.zeros_like(image_array)

green_channel[:,:,1] = image_array[:,:,1]

green_image = Image.fromarray(green_channel)
green_image.show()


blue_channel = np.zeros_like(image_array)

blue_channel[:,:,2] = image_array[:,:,2]

blue_image = Image.fromarray(blue_channel)
blue_image.show()
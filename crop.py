from PIL import Image
import os

img_path = "C:/Users/22863/Desktop/git/ima1/ima_projet/dehazer/hazed_images/2.jpg"
save_dir = "C:/Users/22863/Desktop/git/ima1/ima_projet/dehazer/hazed_images" 

img = Image.open(img_path)
img = img.resize((600, int(600 * 1535 / 2733))) 
left = (img.width - 600) // 2
top = (img.height - 600) // 2
img = img.crop((left, top, left+600, top+600))

base, ext = os.path.splitext(os.path.basename(img_path))
out_name = f"{base}_crop{ext}"
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, out_name)
img.save(out_path)
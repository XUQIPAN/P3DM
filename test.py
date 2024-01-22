import os
from PIL import Image

from tqdm import tqdm 
def resize_images_in_folder(source_folder, target_folder, size=(64, 64)):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add/check file extensions as needed
            try:
                img_path = os.path.join(source_folder, filename)
                img = Image.open(img_path)
                img = img.resize(size, Image.ANTIALIAS)

                # Save to target folder
                img.save(os.path.join(target_folder, filename))

            except IOError as e:
                print(f"Error processing file {filename}: {e}")

# Usage
source_dir = '/data/local/xinxi/Project/DPgan_model/logs/exp_cub/datasets/cub/images'
target_dir = '/data/local/xinxi/Project/DPgan_model/logs/exp_cub/datasets/cub/images2'
resize_images_in_folder(source_dir, target_dir)
import os
from PIL import Image

# Directory path
directory_path = '/data/local/xinxi/Project/DPgan_model/logs/exp_celeba/datasets/celeba/celeba/img_align_celeba'

# List to hold images (optional, depends on what you want to do with the images)
images = []

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".jpg"):  # Check if the file is a JPEG image
        file_path = os.path.join(directory_path, filename)  # Full path to the file

        # Open and possibly process the image
        try:
            image = Image.open(file_path)
        except:
            print(filename)
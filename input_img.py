from model import classify_image
import os

img_path="images_test_run"

test_img_dir = os.path.join(img_path)
for image_file in os.listdir(test_img_dir):
    classify_image(os.path.join(test_img_dir, image_file))
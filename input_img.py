import cv2
import matplotlib.pyplot as plt
from model import test
import os

img_path="images_test_run"

test_img_dir = os.path.join(img_path)
for image_file in os.listdir(test_img_dir):
    if image_file.endswith('.png'):
        image = cv2.imread(os.path.join(test_img_dir, image_file))
        plt.imshow(image)  
        plt.show() 
        print(image_file)
        test(image)
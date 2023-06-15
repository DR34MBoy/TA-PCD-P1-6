import cv2
import matplotlib.pyplot as plt
import numpy as np
from model_2 import test

#### TESTING ####
img_path=r"images_test_run/Parasitized_(2027).png"
img=cv2.imread(img_path)  

plt.imshow(img)  
plt.show() 

test(img)

# resize_img = cv2.resize(img,(100,100))
# array_image=np.array(resize_img)
# array_image=array_image.reshape(1,30000)

# print(rmodel.predict(array_image).reshape(1,-1))

# plt.imshow(array_image.reshape(100,100,3)) 
# plt.show()
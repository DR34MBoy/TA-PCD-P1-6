import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

### TRAINING ###
directory_path=r'images'

# print(os.listdir(directory_path))
# print(os.path)

label=[]
for i in os.listdir(directory_path):    
    if i.split('_')[0]=='Parasitized':
        label.append("Parasitized")
    elif i.split('_')[0]=='Uninfected':     
        label.append("Uninfected")

features=[]     
for i in os.listdir(directory_path):
    f=cv2.imread(os.path.join(directory_path,i))  
    resized_f=cv2.resize(f,(100,100))
    features.append(resized_f)

print(np.array(features).shape)  

X=np.array(features)
Y=np.array(label)
X=X.reshape(2000,30000)   

xtrain,xtest,ytrain,ytest=train_test_split(X,Y)

rmodel=RandomForestClassifier()
rmodel.fit(xtrain,ytrain)

print(rmodel.score(xtrain,ytrain))
print(rmodel.score(xtest,ytest))



#### TESTING ####
img_path=r"images/Uninfected_(10).png"
img=cv2.imread(img_path)  

plt.imshow(img)  
plt.show() 

resize_img = cv2.resize(img,(100,100))
array_image=np.array(resize_img)
array_image=array_image.reshape(1,30000)

print(rmodel.predict(array_image).reshape(1,-1))

plt.imshow(array_image.reshape(100,100,3)) 
plt.show()
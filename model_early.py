import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

### TRAINING ###
directory_path=r'images_new'

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
    f=cv2.imread(os.path.join(directory_path,i), cv2.COLOR_RGB2GRAY)  
    resized_f=cv2.resize(f,(100,100))
    features.append(resized_f)

print(np.array(features).shape)  

X=np.array(features)
Y=np.array(label)
X=X.reshape(len(features), -1)   

xtrain,xtest,ytrain,ytest=train_test_split(X, Y)

rmodel=RandomForestClassifier()
rmodel.fit(xtrain,ytrain)


print("Train accuracy: ", rmodel.score(xtrain,ytrain))
print("Test accuracy: ", rmodel.score(xtest,ytest))


ypred = rmodel.predict(xtest)
print(accuracy_score(ypred,ytest))

def test(img) :
    resize_img = cv2.resize(img,(100,100))
    array_image=np.array(resize_img)
    array_image=array_image.reshape(1,30000)

    print(rmodel.predict(array_image).reshape(1,-1))

    plt.imshow(array_image.reshape(100,100,3)) 
    plt.show()

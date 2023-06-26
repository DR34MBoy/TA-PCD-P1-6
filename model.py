import sklearn
import os
import cv2
import mahotas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

directory_path = r'./images_folder'
print(len(os.listdir(directory_path)))

files=[]
for dirname, _, filenames in os.walk(directory_path):
    for filename in filenames:
        if '.db' not in filename:
            files.append(os.path.join(dirname,filename))

Parasitized_Dir='./images_folder/Parasitized'
Uninfected_Dir='./images_folder/Uninfected'

pd.DataFrame(files).sample(frac=1).reset_index(drop=True)

class DetectMalaria:
    def __init__(self,para_dir,uninfect_dir):
        self.parasitized_dir=para_dir
        self.uninfected_dir=uninfect_dir
    def dataset(self,ratio,files):
        Dataset=pd.DataFrame(files,columns=['Path'])
        Dataset=Dataset.sample(frac=1).reset_index(drop=True)
        trainfiles,testfiles=train_test_split(Dataset,test_size=ratio,random_state=None)
        return(trainfiles,testfiles)
    
x=DetectMalaria(Parasitized_Dir,Uninfected_Dir)

train_data,test_data=x.dataset(ratio=0.2,files=files)

def label(df):
    if 'Uninfected' in df:
        return 0
    else:
        return 1


train_data['label']=train_data['Path'].apply(label)
test_data['label']=test_data['Path'].apply(label)

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return(hist.flatten())

feature=[]
def dataframe(df):

    image=cv2.imread(df['Path'])
    global_feature = np.hstack([fd_haralick(image), fd_hu_moments(image),df['label']])
    feature.append(global_feature)
    
train_data.apply(dataframe,axis=1)

X_train=pd.DataFrame(feature).drop(columns=[20])
y_train=train_data['label']

feature=[]
test_data.apply(dataframe,axis=1)

X_test=pd.DataFrame(feature).drop(columns=[20])
y_test=test_data['label']

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

svc=SVC()
svc.fit(X_train,y_train)
pred=svc.predict(X_test)

print(accuracy_score(y_test, pred))

# Random Forest Classifier langsung menggunakan paramter terbaik
rf = RandomForestClassifier(n_estimators=90, criterion='entropy' ,max_depth=30, min_samples_leaf=2, max_features=40)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))

print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train, y_pred= y_pred_train))

print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test))

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Uninfected', 'Parasitized'])

disp.plot(cmap="Blues")
plt.show()


def classify_image(image_path):
    image = cv2.imread(image_path)
    global_feature = np.hstack([fd_haralick(image), fd_hu_moments(image)])
    scaled_feature = scaler.transform([global_feature])
    prediction = rf.predict(scaled_feature)
    
    if prediction == 0:
        print(f"citra '{image_path}' diklasifikasi sebegai Uninfected.")
    else:
        print(f"citra '{image_path}' diklasifikasi sebegai Parasitized.")
    

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Citra asli")
    plt.axis("off")
    plt.show()
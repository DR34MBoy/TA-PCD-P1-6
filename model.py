import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


### TRAINING ###
# Array untuk features dan label
features = []
label = []

# Folder gambar
image_folder_path = "./images_folder"

# Cek jumlah isi konten folder
print(len(os.listdir(image_folder_path)))

# Baca folder "./images_folder/Parasitized"
parasitized_dir = os.path.join(image_folder_path, 'Parasitized')
for image_file in os.listdir(parasitized_dir):
    if image_file.endswith('.png'):
        image = cv2.imread(os.path.join(parasitized_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ubah ke format warna RGB
        image = cv2.resize(image, (100, 100))
        features.append(image)
        label.append('Parasitized')

# Baca folder "./images_folder/Uninfected"
uninfected_dir = os.path.join(image_folder_path, 'Uninfected')
for image_file in os.listdir(uninfected_dir):
    if image_file.endswith('.png'):
        image = cv2.imread(os.path.join(uninfected_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ubah ke format warna RGB
        image = cv2.resize(image, (100, 100))
        features.append(image)
        label.append('Uninfected')

# Buat jadi Array
X = np.array(features)
Y = np.array(label)
X = X.reshape(len(features), -1)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Ubah dimensi data menjadi 1D (flatten)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalisasi nilai piksel menjadi range 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0


### Random Forest Classifier tanpa tuning ###

# rf = RandomForestClassifier() 
# rf.fit(x_train, y_train)

# y_pred_train = rf.predict(x_train)
#     # predictions for test
# y_pred_test = rf.predict(x_test)
#     # training metrics
# print("Training metrics:")
# print(sklearn.metrics.classification_report(y_true= y_train, y_pred= y_pred_train))
    
#     # test data metrics
# print("Test data metrics:")
# print(sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test))


### Random Forest Classifier dengan tuning ###
## Versi 1 ##
# parameters = {  'n_estimators':[50,60,70,80,90,100],
#                 'min_samples_leaf':[1,2,3,4],
#                 'max_depth':[10,20,30],
#                 'max_features':[10,20,30,40,50],
#                 'criterion':['gini','entropy']
#             }

# rf = RandomForestClassifier()
# rf = GridSearchCV(rf, param_grid=parameters, scoring='accuracy')

# feature_selector = SelectFromModel(rf)
# feature_selector.fit(x_train, y_train)

# x_train_selected = feature_selector.transform(x_train)
# X_test_selected = feature_selector.transform(x_test)

# rf.fit(x_train_selected, y_train)

# best_params = rf.best_params_
# print("Best Parameters:", best_params)

# y_pred_train = rf.predict(x_train)
#     # predictions for test
# y_pred_test = rf.predict(x_test)
#     # training metrics
# print("Training metrics:")
# print(sklearn.metrics.classification_report(y_true= y_train, y_pred= y_pred_train))
    
#     # test data metrics
# print("Test data metrics:")
# print(sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test))

## Versi 2 ##
rf = RandomForestClassifier(n_estimators=100, criterion='gini' ,max_depth=30,  min_samples_leaf=2, max_features=30)
feature_selector = SelectFromModel(rf)
feature_selector.fit(x_train, y_train)

x_train_selected = feature_selector.transform(x_train)
X_test_selected = feature_selector.transform(x_test)

rf.fit(x_train_selected, y_train)

y_pred_train = rf.predict(x_train_selected)
    # predictions for test
y_pred_test = rf.predict(X_test_selected)
    # training metrics
print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train, y_pred= y_pred_train))
    
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test))


## Fungsi untuk klasifikasi gambar ##
def test(img):
    resize_img = cv2.resize(img, (100, 100))
    array_image = np.array(resize_img)
    array_image = array_image.reshape(1, -1)

    print(rf.predict(array_image).reshape(1, -1))

    plt.imshow(array_image.reshape(100, 100, 3))    
    plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

### TRAINING ###
directory_path = r'images_new'

label = []
for i in os.listdir(directory_path):
    if i.split('_')[0] == 'Parasitized':
        label.append("Parasitized")
    elif i.split('_')[0] == 'Uninfected':
        label.append("Uninfected")

features = []
for i in os.listdir(directory_path):
    f = cv2.imread(os.path.join(directory_path, i))
    resized_f = cv2.resize(f, (227, 227))
    resized_f = cv2.cvtColor(resized_f, cv2.COLOR_BGR2HSV)
    features.append(resized_f)

X = np.array(features)
Y = np.array(label)
X = X.reshape(len(features), -1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest Classifier with regularization
rmodel = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rmodel.fit(x_train, y_train)

# Model evaluation
train_accuracy = rmodel.score(x_train, y_train)
test_accuracy = rmodel.score(x_test, y_test)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# Predict on test set
y_pred = rmodel.predict(x_test)
test_accuracy = accuracy_score(y_pred, y_test)
print("Test accuracy (using accuracy_score):", test_accuracy)

def test(img):
    resize_img = cv2.resize(img, (100, 100))
    array_image = np.array(resize_img)
    array_image = array_image.reshape(1, -1)

    print(rmodel.predict(array_image).reshape(1, -1))

    plt.imshow(array_image.reshape(100, 100, 3))    
    plt.show()
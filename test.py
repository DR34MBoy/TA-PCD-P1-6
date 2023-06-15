import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

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
    resized_f = cv2.resize(f, (100, 100))
    features.append(resized_f)

X = np.array(features)
Y = np.array(label)
X = X.reshape(len(features), -1)

# Shuffle the dataset
X, Y = shuffle(X, Y, random_state=42)

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest Classifier with regularization
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Model evaluation
train_accuracy = best_model.score(x_train, y_train)
test_accuracy = best_model.score(x_test, y_test)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# Predict on test set
y_pred = best_model.predict(x_test)
test_accuracy = accuracy_score(y_pred, y_test)
print("Test accuracy (using accuracy_score):", test_accuracy)

def test(img):
    resize_img = cv2.resize(img, (100, 100))
    array_image = np.array(resize_img)
    array_image = array_image.reshape(1, -1)

    print(best_model.predict(array_image).reshape(1, -1))

    plt.imshow(array_image.reshape(100, 100, 3))
    plt.show()
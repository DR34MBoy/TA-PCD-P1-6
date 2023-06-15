import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

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
    f = cv2.imread(os.path.join(directory_path, i), cv2.COLOR_BGR2GRAY)
    resized_f = cv2.resize(f, (100, 100))
    features.append(resized_f)

X = np.array(features)
Y = np.array(label)
X = X.reshape(len(features), -1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)


# Random Forest Classifier tanpa tuning
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


# Random Forest Classifier dengan tuning

parameters = {  'n_estimators':[50,60,70,80,90,100],
                # 'min_samples_leaf':[1,2,3,4],
                'max_depth':[10,20,30],
                # 'max_features':[10,20,30,40,50,60,70,80,90,100],
                # 'criterion':['gini','entropy']
            }

# rf = RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_leaf=1, max_features=100)
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

rf = RandomForestClassifier()
clf = GridSearchCV(rf, param_grid=parameters)
model = clf.fit(x_train,y_train)

y_pred_train = model.predict(x_train)
    # predictions for test
y_pred_test = model.predict(x_test)
    # training metrics
print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train, y_pred= y_pred_train))
    
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test))



def test(img):
    resize_img = cv2.resize(img, (100, 100))
    array_image = np.array(resize_img)
    array_image = array_image.reshape(1, -1)

    print(rf.predict(array_image).reshape(1, -1))

    plt.imshow(array_image.reshape(100, 100, 3))    
    plt.show()
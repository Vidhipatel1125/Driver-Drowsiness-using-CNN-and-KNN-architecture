## 📊 Feature Extraction with ResNet50 for KNN
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = resnet_model.predict(x)
    return features.flatten()

X_features = []
y_labels = []

for label in ['Open', 'Closed']:
    folder = f'data/{label}/'
    for file in os.listdir(folder):
        feat = extract_features(os.path.join(folder, file))
        X_features.append(feat)
        y_labels.append(label)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_features, y_labels)

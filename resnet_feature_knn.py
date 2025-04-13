## 📊 Feature Extraction with ResNet50 for KNN
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
def create_data_generators(train_dir, test_dir, batch_size, target_size, shuffle=False):
train_datagen = ImageDataGenerator(
rescale=1.0 / 255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=target_size,
batch_size=batch_size,
class_mode=None,
shuffle=shuffle
)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=target_size,
batch_size=batch_size,
class_mode=None,
shuffle=shuffle
)
return train_generator, test_generator
train_generator, test_generator = create_data_generators(
train_dir="/Users/vidhipatel/Downloads/train",
test_dir="/Users/vidhipatel/Downloads/test",
batch_size=32,
target_size=(32, 32),
shuffle=False
)
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
input_shape=(32, 32, 3))
train_features = base_model.predict(train_generator)
test_features = base_model.predict(test_generator)
train_features = np.reshape(train_features, (train_features.shape[0], -1))
test_features = np.reshape(test_features, (test_features.shape[0], -1))
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_features, train_generator.classes)
test_accuracy = knn_classifier.score(test_features, test_generator.classes)
print("Test Accuracy:", test_accuracy)
y_pred = knn_classifier.predict(test_features)
print("Classification Report:")
print(classification_report(test_generator.classes, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(test_generator.classes, y_pred))

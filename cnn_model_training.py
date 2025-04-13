
## 🧠 Model Training (CNN)

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
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
class_mode='binary',
shuffle=shuffle
)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=target_size,
batch_size=batch_size,
class_mode='binary',
shuffle=shuffle
)
return train_generator, test_generator
train_generator, test_generator = create_data_generators(
train_dir="/Users/vidhipatel/Downloads/train",
test_dir="/Users/vidhipatel/Downloads/test",
batch_size=32,
target_size=(24, 24),
shuffle=False
)
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,
24, 3)),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(1, activation='sigmoid')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
train_generator,
epochs=50,
validation_data=test_generator
)
test_loss, test_accuracy = model.evaluate(test_generator)
train_accuracy = history.history['accuracy'][-1]
print("Test Accuracy:", test_accuracy)
print("Training Accuracy:", train_accuracy)
y_pred = model.predict(test_generator)
y_pred = (y_pred > 0.5)
y_true = test_generator.classes
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

## 🎥 Real-Time Drowsiness Detection (OpenCV + CNN)
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame

model = load_model('cnn_model.h5')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

pygame.mixer.init()
sound = pygame.mixer.Sound('alarm.wav')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (224, 224))
            eye_img = eye_img / 255.0
            eye_img = eye_img.reshape(1, 224, 224, 1)

            pred = model.predict(eye_img)
            if pred[0][0] < 0.5:
                cv2.putText(frame, "DROWSY", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                sound.play()
            else:
                sound.stop()
            break

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

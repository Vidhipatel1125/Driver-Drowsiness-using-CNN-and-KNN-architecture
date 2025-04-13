# 🚀 Results & Demonstration
## 🔄 Real-Time Drowsiness Detection
The system continuously captures live video feed using an in-cabin camera and processes each frame using a trained Convolutional Neural Network (CNN). Eye states (open or closed) are classified in real time. A drowsiness score is incremented when eyes are closed for prolonged periods, simulating actual fatigue behavior. Once a threshold is reached, an alarm is triggered to alert the driver.

Eye activity is monitored using Haar cascades and fed into a 10-layer CNN.

The CNN runs inference on every frame, classifying the eye state as open or closed.

If both eyes are classified as closed consistently, a cumulative score increases.

Upon exceeding the threshold, an audio alert is played via the onboard system.

Live Camera Feed → Face & Eye Detection → CNN/KNN Inference → Drowsiness Score Logic → Alert System

## 📊 CNN Model Performance

| Metric        | Class 0 (Alert) | Class 1 (Drowsy) |
|---------------|----------------|------------------|
| Precision     | 1.00           | 0.97             |
| Recall        | 0.97           | 1.00             |
| F1 Score      | 0.99           | 0.99             |
| Accuracy      | 98.62%         | -                |
| Validation Loss | 0.0308       | -                |

High generalization: Consistently high training and validation accuracy.

Minimal overfitting: Training loss converged to 0.1075, validation loss to 0.0308.

Excellent classification ability for both drowsy and alert states.

## 📊 KNN Model Performance
| Metric           | Class 0 (Alert) | Class 1 (Drowsy) |
|------------------|----------------|------------------|
| Precision        | 0.96           | 0.95             |
| Recall           | 0.94           | 0.96             |
| F1 Score         | 0.95           | 0.95             |
| Accuracy         | 95.0%          | -                |
| False Alarms     | Moderate       | Moderate         |

## 📈 Training & Evaluation Curves
Training accuracy reached 95.92%, validation accuracy peaked at 98.62%.
Graphs show consistent convergence and strong model generalization.



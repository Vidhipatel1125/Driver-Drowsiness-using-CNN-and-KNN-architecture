# Driver-Drowsiness-using-CNN-and-KNN-architecture
This project presents a real-time drowsiness detection and alert system designed to improve road safety by identifying fatigue in drivers using computer vision and machine learning techniques. By integrating Convolutional Neural Networks (CNN) and K-Nearest Neighbors (KNN), the system analyzes live video feeds from in-vehicle cameras to detect signs of drowsiness and triggers alerts to mitigate risks.

🚗 Problem Statement
Drowsy driving is a leading cause of traffic accidents and fatalities. Traditional systems relying on physical sensors or manual observation are often ineffective. This project addresses the challenge by developing a camera-based AI system that autonomously detects driver fatigue through facial analysis.

💡 Solution Overview
CNN Model trained to detect closed and open eyes from facial imagery.

KNN Classifier used to refine predictions by analyzing features extracted using ResNet50.

Real-Time Detection Pipeline built with OpenCV to continuously monitor eye state.

Audible Alert System that triggers when signs of drowsiness persist.

🧠 Technologies Used
Python, TensorFlow, Keras
OpenCV, NumPy, Scikit-learn
CNN (Custom 10-layer) and ResNet50 for feature extraction
KNN for binary classification (Drowsy vs Alert)
Haar Cascades for face and eye detection
Pygame for audio alerts

## 📊 Performance

| Metric        | CNN Model | KNN Model  |
|---------------|-----------|------------|
| Accuracy      | 98.6%     | 95.0%      |
| F1 Score      | 0.99      | 0.95       |
| False Alarms  | Very Low  | Moderate   |


CNN significantly outperformed KNN in both recall and precision.

Real-world testing under varied lighting conditions confirmed model robustness.

🔧 Features
Real-time monitoring with minimal latency.

Eye-blink and eye-closure based drowsiness scoring system.

Robust alert mechanism using sound and visual feedback.

Modular code for ease of integration into vehicle systems.

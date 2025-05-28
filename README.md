# Hand Gesture Recognition Using Machine Learning

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Best Model Results](#best-model-results)
- [Real-Time Gesture Recognition](#real-time-gesture-recognition)
- [Usage](#usage)





## Overview
This system detects hand gestures in real time using a webcam and classifies them into predefined labels. It preprocesses the dataset, trains multiple models using GridSearchCV for hyperparameter tuning, and selects the best-performing model for deployment.

The input to the project is a CSV file containing hand landmarks (e.g., x, y, z coordinates of keypoints) extracted from the HaGRID dataset using MediaPipe. The output will be a trained machine learning model capable of classifying hand gestures into predefined classes.

We will gain hands-on experience in data preprocessing, visualization, machine learning model training, and performance evaluation.

https://github.com/user-attachments/assets/340bb5ce-ff58-481e-b283-6ca36b1e7786

## Installation
To run this project, install the required dependencies:

```bash
pip install opencv-python mediapipe pandas numpy joblib scikit-learn xgboost
```
## Dataset
The dataset consists of hand landmark coordinates (X, Y) extracted using MediaPipe Hands. It includes labeled gestures used for training the models.

## Model Training
Three machine learning models were trained using GridSearchCV for hyperparameter tuning:

### Support Vector Machine (SVM)
### Random Forest Classifier
### XGBoost Classifier
The dataset was split into training (90%) and testing (10%) subsets using train_test_split with stratification.

## Best Model Results
The best hyperparameters and evaluation metrics for each model:

The models were evaluated based on their accuracy, precision, recall, and F1-score. Below is a summary of the results:

| **Model**              | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-------------------------|--------------|---------------|------------|--------------|
| SVM                    | **0.97**     | **0.97**      | **0.97**   | **0.97**     |
| Logistic Regression     | 0.89         | 0.89          | 0.89       | 0.89         |
| Random Forest           | 0.85         | 0.85          | 0.85       | 0.85         |
| XGBoost                 | 0.72         | 0.72          | 0.72       | 0.72         |
| Decision Tree           | 0.65         | 0.66          | 0.63       | 0.63         |

### Performance Comparison Chart
![Model Performance Comparison](results/model_comparison.png)

## Real-Time Gesture Recognition
After training, the best model is used for real-time classification. MediaPipe extracts hand landmarks, and the trained model predicts the gesture. Results are displayed on the webcam feed.

## Usage
Run the real-time gesture detection script:
'''bash
python hand_gesture_detection.py
'''
Press q to exit.

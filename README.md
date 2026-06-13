# Medcial-AI-Heart-Disease-prediction
A medical AI project for automated cardiac function assessment from echocardiography videos using deep learning and machine learning techniques. The project includes data preprocessing, contour feature extraction, regression modeling, and evaluation on the EchoNet-Dynamic dataset.


# Cardiac Function Estimation using Echocardiography and Machine Learning

This research project focuses on developing an automated system for cardiac function assessment using the EchoNet-Dynamic dataset. The pipeline processes echocardiography videos, extracts clinically relevant information from ventricular tracings, and employs both deep learning and ensemble learning techniques to estimate cardiac parameters.

## Key Features
- End-to-end preprocessing pipeline for echocardiography videos.
- Contour extraction and validation using ventricular volume tracings.
- Automated extraction of end-diastolic and end-systolic cardiac phases.
- Deep learning models using EfficientNet for feature representation learning.
- Ensemble regression models including XGBoost and Extra Trees for cardiac parameter estimation.
- Comprehensive evaluation using metrics such as Mean Squared Error (MSE) and R² Score.

## Technologies Used
- Python
- PyTorch
- EfficientNet
- Scikit-learn
- XGBoost
- Pandas & NumPy
- Matplotlib

## Dataset
The project utilizes the **EchoNet-Dynamic** dataset, a large-scale collection of echocardiography videos with expert annotations for cardiac function analysis.

## Applications
This work contributes toward the development of AI-assisted clinical decision support systems for non-invasive cardiovascular assessment.

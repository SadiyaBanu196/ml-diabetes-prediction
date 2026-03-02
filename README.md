Diabetes Prediction using Machine Learning
Project Overview

This project builds a machine learning pipeline to predict whether a patient has diabetes based on medical features such as glucose, BMI, insulin, blood pressure, and skin thickness.

The goal is to compare multiple models, tune hyperparameters, and select the best performing classifier.

Dataset

Pima Indians Diabetes Dataset

Features include:

Glucose

BloodPressure

SkinThickness

Insulin

BMI

Age

Target:

Outcome (0 = No Diabetes, 1 = Diabetes)

Workflow

Data preprocessing:

Replace zero values with median

Train-test split (80/20)

Feature scaling using StandardScaler

Model training:

Logistic Regression

Random Forest

Decision Tree

Model optimization:

Hyperparameter tuning using GridSearchCV

Class imbalance handled using class_weight = balanced

Evaluation metrics:

Accuracy

Classification Report

Recall (important for medical prediction)

Results

Logistic Regression
Accuracy: 0.727

Random Forest (Best Model)
Accuracy: 0.746
Recall: 0.759

Decision Tree
Accuracy: 0.668

Random Forest performed best after tuning.

Tech Stack

Python

pandas

numpy

scikit-learn

How to Run

Install dependencies:
pip install pandas numpy scikit-learn

Run:
python diabetes.py

Project Structure

D2/
│── diabetes.py
│── diabetes.csv
│── README.md

Key Learnings

Data cleaning improves model quality

Scaling is important for linear models

Random Forest performs well on tabular data

Hyperparameter tuning improves performance

Model comparison is critical before final selection

Future Improvements

Cross-validation

Model saving with joblib

Streamlit/Flask deployment

Feature engineering
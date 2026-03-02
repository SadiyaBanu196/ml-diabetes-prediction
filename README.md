Diabetes Prediction using Machine Learning
Problem

Predict whether a patient has diabetes using medical diagnostic features.

Early detection is important because missing positive cases (false negatives) can be dangerous.

Dataset

Pima Indians Diabetes Dataset

Features include:

Glucose

Blood Pressure

BMI

Insulin

Age

etc.

Target:

Outcome (0 = Non-diabetic, 1 = Diabetic)

Preprocessing

Replaced invalid zeros with median values

Train-test split (80/20, stratified)

Feature scaling using StandardScaler

Models Implemented

Logistic Regression

Random Forest

Decision Tree

Random Forest was tuned using GridSearchCV.

Results
Model	Accuracy	Recall (Diabetic)
Logistic Regression	0.727	0.704
Random Forest (Tuned)	0.747	0.759
Decision Tree	0.675	0.500
Final Model

Random Forest

Reason:
Chosen for highest recall and better identification of diabetic patients.

How to Run
python diabetes.py
Tech Stack

Python
pandas
scikit-learn
NumPy
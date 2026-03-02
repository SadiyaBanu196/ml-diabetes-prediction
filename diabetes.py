import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

diabetes_df=pd.read_csv("diabetes.csv")

cols=['Glucose','BloodPressure','SkinThickness','Insulin','BMI',]

for col in cols:
    diabetes_df[col]=diabetes_df[col].replace(0,diabetes_df[col].median())
    

X=diabetes_df.drop('Outcome',axis=1)
y=diabetes_df['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

params={
    'n_estimators':[100,200,300,500],
    'max_depth':[None,3,5,7,10],
    'min_samples_split':[2,5,10]
}

grid=GridSearchCV(
    RandomForestClassifier(class_weight='balanced',random_state=42),
    params,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid.fit(X_train,y_train)
print("Best params: ",grid.best_params_)
best_rf = grid.best_estimator_

models={
    "LogisticRegression":LogisticRegression(max_iter=1000,class_weight='balanced'),
    "RandomForest":best_rf,
    "DecisionTree":DecisionTreeClassifier()
}

for name,m in models.items():
    m.fit(X_train,y_train)
    pred=m.predict(X_test)
    print("\n",name)
    print("Accuracy: ",accuracy_score(y_test,pred))
    print("Classification Report: \n",classification_report(y_test,pred,output_dict=True)['1']['recall'])

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('/kaggle/input/loan-prediction-casestudy-dataset/loan_prediction.csv')

df.drop(columns=['Loan_ID'], inplace=True)


cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=np.number).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


Q1 = df['LoanAmount'].quantile(0.25)
Q3 = df['LoanAmount'].quantile(0.75)
IQR = Q3 - Q1
upper_cap = Q3 + 1.5 * IQR
df['LoanAmount'] = np.where(df['LoanAmount'] > upper_cap, upper_cap, df['LoanAmount'])

df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
df['Married_Yes'] = (df['Married'] == 'Yes').astype(int)
df['Education_Not Graduate'] = (df['Education'] == 'Not Graduate').astype(int)


X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 
        'Gender_Male', 'Married_Yes', 'Education_Not Graduate']]
y = df['Loan_Status'].map({'N': 0, 'Y': 1})


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(x_train_scaled, y_train)


y_pred_rf = rf_model.predict(x_test_scaled)


print(" Random Forest Classifier (Simplified):")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


joblib.dump(rf_model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

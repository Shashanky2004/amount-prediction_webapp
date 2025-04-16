import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load the data
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Separate features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Convert categorical variables to numeric
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    # Fill NaN values with 'Missing' before encoding
    X[column] = X[column].fillna('Missing')
    X[column] = label_encoders[column].fit_transform(X[column])

# Handle numerical columns
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
imputer = SimpleImputer(strategy='median')
X[numerical_columns] = imputer.fit_transform(X[numerical_columns])

# Convert target variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and preprocessing objects
model_data = {
    'model': model,
    'label_encoders': label_encoders,
    'label_encoder_y': label_encoder_y,
    'imputer': imputer,
    'feature_names': list(X.columns)
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Print model accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Testing accuracy: {test_accuracy:.2f}") 
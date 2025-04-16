from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and preprocessing objects
try:
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    label_encoder_y = model_data['label_encoder_y']
    imputer = model_data['imputer']
    feature_names = model_data['feature_names']
    print("Model and preprocessing objects loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}. Using placeholder prediction.")
    model_data = None

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/prediction')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get values from the form
            input_data = {
                'Gender': request.form['gender'],
                'Married': request.form['married'],
                'Dependents': request.form['dependents'],
                'Education': request.form['education'],
                'Self_Employed': request.form['self_employed'],
                'ApplicantIncome': float(request.form['applicant_income']),
                'CoapplicantIncome': float(request.form['coapplicant_income']),
                'LoanAmount': float(request.form['loan_amount']),
                'Loan_Amount_Term': float(request.form['loan_amount_term']),
                'Credit_History': float(request.form['credit_history']),
                'Property_Area': request.form['property_area']
            }
            
            if model_data is not None:
                # Create a DataFrame with the input data
                input_df = pd.DataFrame([input_data])
                
                # Preprocess categorical features
                categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
                for column in categorical_columns:
                    input_df[column] = input_df[column].fillna('Missing')
                    input_df[column] = label_encoders[column].transform(input_df[column])
                
                # Preprocess numerical features
                numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
                input_df[numerical_columns] = imputer.transform(input_df[numerical_columns])
                
                # Reorder columns to match training data
                input_df = input_df[feature_names]
                
                # Make prediction
                prediction_encoded = model.predict(input_df)[0]
                prediction = label_encoder_y.inverse_transform([prediction_encoded])[0]
                
                result_text = "Loan Approved" if prediction == "Y" else "Loan Not Approved"
                return render_template('index.html', 
                                    prediction_text=result_text)
            else:
                # Placeholder prediction based on income and credit history
                total_income = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
                prediction = "Loan Approved" if (total_income * 0.3 * input_data['Credit_History']) > input_data['LoanAmount'] else "Loan Not Approved"
                return render_template('index.html', 
                                    prediction_text=f'{prediction} (Using placeholder logic)')

        except Exception as e:
            return render_template('index.html', 
                                prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
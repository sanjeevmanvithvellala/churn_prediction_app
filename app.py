from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature columns
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # List of 30 columns

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        # Basic inputs
        tenure = float(form['tenure'])
        monthlycharges = float(form['monthlycharges'])
        totalcharges = float(form['totalcharges'])

        # Initialize dictionary with 0s for all features
        input_dict = dict.fromkeys(feature_columns, 0)

        # Fill numeric values
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthlycharges
        input_dict['TotalCharges'] = totalcharges

        # Binary encoding
        input_dict['gender'] = 1 if form['gender'] == 'Male' else 0
        input_dict['SeniorCitizen'] = int(form['seniorcitizen'])
        input_dict['Partner'] = 1 if form['partner'] == 'Yes' else 0
        input_dict['Dependents'] = 1 if form['dependents'] == 'Yes' else 0
        input_dict['PhoneService'] = 1 if form['phoneservice'] == 'Yes' else 0

        # One-hot encoding: set the correct column to 1
        contract_col = f"Contract_{form['contract']}"
        if contract_col in input_dict:
            input_dict[contract_col] = 1

        payment_col = f"PaymentMethod_{form['paymentmethod']}"
        if payment_col in input_dict:
            input_dict[payment_col] = 1

        internet_col = f"InternetService_{form['internetservice']}"
        if internet_col in input_dict:
            input_dict[internet_col] = 1

        # Create DataFrame from dict in correct order
        final_df = pd.DataFrame([input_dict])

        # Scale numeric values
        final_scaled = scaler.transform(final_df)

        # Predict
        prediction = model.predict(final_scaled)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

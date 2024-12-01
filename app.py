from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('models/xgboost_model.pkl')  # Load the XGBoost model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Capture form data including all necessary features
        data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['education']),
            float(request.form['currentSmoker']),
            float(request.form['cigsPerDay']),
            float(request.form['BPMeds']),
            float(request.form['prevalentStroke']),
            float(request.form['prevalentHyp']),
            float(request.form['diabetes']),
            float(request.form['totChol']),
            float(request.form['sysBP']),
            float(request.form['diaBP']),
            float(request.form['BMI']),
            float(request.form['heartRate']),
            float(request.form['glucose']),
        ]
        
        # Convert data to a 2D array for prediction
        data = np.array([data])
        
        # Make a prediction
        prediction = model.predict(data)
        
        # Interpret the prediction result
        result = 'High risk of heart disease' if prediction[0] == 1 else 'Low risk of heart disease'
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

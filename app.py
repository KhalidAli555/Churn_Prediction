from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model (Churn_model.pkl)
with open('Churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler (scaler.pkl) if you used scaling during training
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = [float(request.form['CreditScore']), 
            float(request.form['Age']), 
            float(request.form['Tenure']), 
            float(request.form['Balance']),
            float(request.form['NumOfProducts']), 
            int(request.form['HasCrCard']), 
            int(request.form['IsActiveMember']),
            float(request.form['EstimatedSalary'])]

    # Convert data into a DataFrame (for consistency with training data)
    input_data = pd.DataFrame([data], columns=['CreditScore', 'Age', 'Tenure', 'Balance', 
                                               'NumOfProducts', 'HasCrCard', 
                                               'IsActiveMember', 'EstimatedSalary'])

    # Apply the scaler (if used during training)
    scaled_data = scaler.transform(input_data)

    # Make a prediction using the loaded model
    prediction = model.predict(scaled_data)
    
    # Binary output - interpreting model result
    result = 'Churn' if prediction[0] == 1 else 'No Churn'
    
    # Return the result on the web page
    return render_template('index.html', prediction_text=f'Customer Is Predicted To: {result}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

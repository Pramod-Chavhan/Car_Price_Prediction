from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved models and scaler
lin_reg_model = joblib.load('linear_regression_model.pkl')
lasso_reg_model = joblib.load('lasso_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # This will render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all 7 input features from the form
    present_price = float(request.form['Present_Price'])
    kms_driven = int(request.form['Kms_Driven'])
    fuel_type = int(request.form['Fuel_Type'])
    seller_type = int(request.form['Seller_Type'])
    transmission = int(request.form['Transmission'])
    owner = int(request.form['Owner'])  # Ensure you're collecting this 7th feature

    # Create the input array for prediction (with 7 features)
    input_data = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, owner]])

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = lin_reg_model.predict(scaled_data)

    return render_template('index.html', prediction_text=f'Predicted Car Price: {prediction[0]:.2f} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)

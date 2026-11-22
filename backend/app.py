
# Import necessary libraries
import numpy as np
import joblib 
import pandas as pd 
from flask import Flask, request, jsonify

# Flask app initialization
superkart_api = Flask("SuperKart")

# Loading the trained model
saved_model = joblib.load("superkart_model.joblib")

# Route for homepage
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Predictor!"

# API endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json()

    # Extract relevant features from the input data
    sample = {
    'Product_Weight': data['Product_Weight'],
    'Product_Sugar_Content': data['Product_Sugar_Content'],
    'Product_Allocated_Area': data['Product_Allocated_Area'],
    'Product_MRP': data['Product_MRP'],
    'Store_Size': data['Store_Size'],
    'Store_Location_City_Type': data['Store_Location_City_Type'],
    'Store_Type': data['Store_Type'],
    'Product_Id_char': data['Product_Id_char'],
    'Store_Age_Years': data['Store_Age_Years'],
    'Product_Type_Category': data['Product_Type_Category']
}

    # Convert the extracted data into a DataFrame, make prediction and return prediction as JSON
    input_data = pd.DataFrame([sample])
    prediction = saved_model.predict(input_data).tolist()[0]
    return jsonify({'Sales': prediction})

# API endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get CSV file from request
    file = request.files['file']
    batch_data = pd.read_csv(file)

    # Make predictions, convert output into dictionary and return output
    predictions = saved_model.predict(batch_data).tolist()
    output = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return output


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)

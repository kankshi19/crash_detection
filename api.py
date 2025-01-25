from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

# Load the model and scaler
model = joblib.load('crash_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert the received JSON to a DataFrame
    sensor_data = pd.DataFrame([data])
    
    # Preprocess timestamp
    sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
    sensor_data['hour_of_day'] = sensor_data['timestamp'].dt.hour
    sensor_data['day_of_week'] = sensor_data['timestamp'].dt.dayofweek

    # Select features
    features = ['accel_x', 'accel_y', 'accel_z', 
                'gyro_x', 'gyro_y', 'gyro_z', 
                'hour_of_day', 'day_of_week']
    
    # Scale the input data
    sensor_data_scaled = scaler.transform(sensor_data[features])
    
    # Predict
    prediction = model.predict(sensor_data_scaled)
    
    return jsonify({'crash_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

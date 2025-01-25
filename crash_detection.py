import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class CrashDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_data(self, df):
        # Convert timestamp to numeric features if needed
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Select features
        features = ['accel_x', 'accel_y', 'accel_z', 
                    'gyro_x', 'gyro_y', 'gyro_z', 
                    'hour_of_day', 'day_of_week']
        X = df[features]
        y = df['crash']
        
        return X, y

    def train_model(self, df, test_size=0.2, random_state=42):
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model and scaler
        joblib.dump(self.model, 'crash_detection_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Model and scaler saved as 'crash_detection_model.pkl' and 'scaler.pkl'")
        
        return self.model

    def predict_crash(self, sensor_data):
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(sensor_data, dict):
            sensor_data = pd.DataFrame([sensor_data])
        
        # Preprocess timestamp
        sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
        sensor_data['hour_of_day'] = sensor_data['timestamp'].dt.hour
        sensor_data['day_of_week'] = sensor_data['timestamp'].dt.dayofweek

        # Select features
        features = ['accel_x', 'accel_y', 'accel_z', 
                    'gyro_x', 'gyro_y', 'gyro_z', 
                    'hour_of_day', 'day_of_week']
        
        # Scale the input data using the loaded scaler
        scaler = joblib.load('scaler.pkl')
        sensor_data_scaled = scaler.transform(sensor_data[features])
        
        # Predict
        return self.model.predict(sensor_data_scaled)[0]

# Example usage
def main():
    # Load your dataset
    data = pd.read_csv('balanced_crash_detection_dataset.csv')
    
    # Create and train the model
    crash_detector = CrashDetectionModel()
    crash_detector.train_model(data)

if __name__ == "__main__":
    main()

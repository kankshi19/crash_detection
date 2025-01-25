import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)

from crash_detection import CrashDetectionModel

def test_crash_detection_model(model, X_test, y_test):
    """
    Comprehensive model testing function
    
    Args:
    model: Trained crash detection model
    X_test: Test features
    y_test: True crash labels
    
    Returns:
    Dict of model performance metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Performance metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    
    # Print detailed report
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return metrics

def cross_validation_test(model, X, y, cv=5):
    """
    Perform cross-validation testing
    
    Args:
    model: Crash detection model
    X: Features
    y: Crash labels
    cv: Number of cross-validation folds
    
    Returns:
    Dict of cross-validation scores
    """
    # Perform cross-validation
    cv_scores = {
        'Accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
        'Precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
        'Recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
        'F1': cross_val_score(model, X, y, cv=cv, scoring='f1')
    }
    
    # Print cross-validation results
    for metric, scores in cv_scores.items():
        print(f"{metric} Cross-Validation Scores: {scores}")
        print(f"{metric} Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_scores

def main():
    # Load your dataset
    data = pd.read_csv('balanced_crash_detection_dataset.csv')
    
    # Create and train the model
    crash_detector = CrashDetectionModel()
    X, y = crash_detector.preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    X_train_scaled = crash_detector.scaler.fit_transform(X_train)
    X_test_scaled = crash_detector.scaler.transform(X_test)
    
    # Train model
    model = crash_detector.model
    model.fit(X_train_scaled, y_train)
    
    # Perform testing
    print("Single Train-Test Split Testing:")
    test_metrics = test_crash_detection_model(model, X_test_scaled, y_test)
    
    print("\nCross-Validation Testing:")
    cross_validation_test(model, X, y)

if __name__ == "__main__":
    main()
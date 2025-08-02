"""
Utility functions for the MLOps pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os


def load_california_housing_data():
    """
    Load California Housing dataset from sklearn.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and test data
    """
    # Load the California Housing dataset
    california = fetch_california_housing()
    X = california.data
    y = california.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using R² score and MSE.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        tuple: (r2_score, mse) - Evaluation metrics
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return r2, mse


def save_model(model, filepath):
    """
    Save the trained model using joblib.
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load a saved model using joblib.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def print_model_info(model, r2_score, mse):
    """
    Print model information and performance metrics.
    
    Args:
        model: Trained model
        r2_score: R² score
        mse: Mean squared error
    """
    print("=" * 50)
    print("MODEL TRAINING RESULTS")
    print("=" * 50)
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of Features: {model.coef_.shape[0]}")
    print(f"R² Score: {r2_score:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
    print("=" * 50)


def create_sample_data():
    """
    Create sample data for prediction testing.
    
    Returns:
        numpy.ndarray: Sample feature data
    """
    # Create sample data with the same number of features as California Housing
    np.random.seed(42)
    sample_data = np.random.rand(5, 8)  # 5 samples, 8 features
    return sample_data 
"""
Model training script for Linear Regression on California Housing dataset.
"""
from sklearn.linear_model import LinearRegression
from utils import (
    load_california_housing_data,
    evaluate_model,
    save_model,
    print_model_info
)


def train_model():
    """
    Train Linear Regression model on California Housing dataset.
    
    Returns:
        tuple: (model, r2_score, mse) - Trained model and evaluation metrics
    """
    print("Loading California Housing dataset...")
    X_train, X_test, y_train, y_test = load_california_housing_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize and train the Linear Regression model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model performance...")
    r2, mse = evaluate_model(model, X_test, y_test)
    
    # Print model information
    print_model_info(model, r2, mse)
    
    # Save the trained model
    print("Saving trained model...")
    save_model(model, "models/linear_regression_model.joblib")
    print("Model saved successfully!")
    
    return model, r2, mse


if __name__ == "__main__":
    train_model() 
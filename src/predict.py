"""
Prediction script for model verification in Docker container.
"""
import numpy as np
from utils import load_model, load_california_housing_data, create_sample_data


def run_predictions():
    """
    Load trained model and run predictions on test data.
    
    Returns:
        dict: Prediction results and sample outputs
    """
    print("Loading trained model...")
    model_path = "models/linear_regression_model.joblib"
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model has been trained first.")
        return None
    
    # Load test data
    print("Loading test data...")
    _, X_test, _, y_test = load_california_housing_data()
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Calculate basic statistics
    print("Calculating prediction statistics...")
    mean_prediction = np.mean(predictions)
    std_prediction = np.std(predictions)
    min_prediction = np.min(predictions)
    max_prediction = np.max(predictions)
    
    # Print sample predictions
    print("=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Number of predictions: {len(predictions)}")
    print(f"Mean prediction: {mean_prediction:.4f}")
    print(f"Standard deviation: {std_prediction:.4f}")
    print(f"Min prediction: {min_prediction:.4f}")
    print(f"Max prediction: {max_prediction:.4f}")
    print()
    
    # Show sample predictions
    print("Sample Predictions (first 10):")
    print("-" * 40)
    for i in range(min(10, len(predictions))):
        print(f"Sample {i+1}: Actual = {y_test[i]:.4f}, Predicted = {predictions[i]:.4f}")
    
    # Test with sample data
    print("\nTesting with sample data...")
    sample_data = create_sample_data()
    sample_predictions = model.predict(sample_data)
    
    print("Sample Data Predictions:")
    print("-" * 40)
    for i, pred in enumerate(sample_predictions):
        print(f"Sample {i+1}: {pred:.4f}")
    
    # Verify model coefficients
    print("\nModel Information:")
    print("-" * 40)
    print(f"Number of features: {len(model.coef_)}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Coefficients: {model.coef_}")
    
    print("=" * 60)
    print("PREDICTION VERIFICATION COMPLETE")
    print("=" * 60)
    
    return {
        'predictions': predictions,
        'sample_predictions': sample_predictions,
        'statistics': {
            'mean': mean_prediction,
            'std': std_prediction,
            'min': min_prediction,
            'max': max_prediction
        }
    }


if __name__ == "__main__":
    results = run_predictions()
    if results:
        print("SUCCESS: Prediction script executed successfully!")
    else:
        print("FAILED: Prediction script failed!")
        exit(1) 
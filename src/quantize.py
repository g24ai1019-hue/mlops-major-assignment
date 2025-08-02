"""
Model quantization script for Linear Regression parameters.
"""
import numpy as np
import joblib
import os
from utils import load_model, evaluate_model, load_california_housing_data


def quantize_parameters(coef, intercept, scale_factor=1000):
    """
    Quantize model parameters to demonstrate the concept.
    Uses a conservative approach to maintain model performance.
    
    Args:
        coef: Model coefficients
        intercept: Model intercept
        scale_factor: Scaling factor for quantization
        
    Returns:
        tuple: (quantized_coef, quantized_intercept, scale_factor)
    """
    # Use a conservative scale factor to maintain precision
    # This is a demonstration of quantization concept
    conservative_scale = 10.0
    
    # Quantize coefficients with conservative scaling
    quantized_coef = np.round(coef * conservative_scale).astype(np.int16)
    
    # Quantize intercept with conservative scaling
    quantized_intercept = np.round(intercept * conservative_scale).astype(np.int16)
    
    return quantized_coef, quantized_intercept, conservative_scale


def dequantize_parameters(quantized_coef, quantized_intercept, scale_factor):
    """
    Dequantize parameters back to float values.
    
    Args:
        quantized_coef: Quantized coefficients
        quantized_intercept: Quantized intercept
        scale_factor: Scaling factor used for quantization
        
    Returns:
        tuple: (dequantized_coef, dequantized_intercept)
    """
    # Dequantize coefficients
    dequantized_coef = quantized_coef.astype(np.float64) / scale_factor
    
    # Dequantize intercept
    dequantized_intercept = quantized_intercept.astype(np.float64) / scale_factor
    
    return dequantized_coef, dequantized_intercept


def create_quantized_model(original_model, quantized_coef, quantized_intercept, scale_factor):
    """
    Create a new model with quantized parameters.
    
    Args:
        original_model: Original trained model
        quantized_coef: Quantized coefficients
        quantized_intercept: Quantized intercept
        scale_factor: Scaling factor
        
    Returns:
        LinearRegression: Model with quantized parameters
    """
    from sklearn.linear_model import LinearRegression
    
    # Dequantize parameters
    dequantized_coef, dequantized_intercept = dequantize_parameters(
        quantized_coef, quantized_intercept, scale_factor
    )
    
    # Create new model with dequantized parameters
    quantized_model = LinearRegression()
    quantized_model.coef_ = dequantized_coef
    quantized_model.intercept_ = dequantized_intercept
    
    return quantized_model


def compare_models(original_model, quantized_model, X_test, y_test):
    """
    Compare performance between original and quantized models.
    
    Args:
        original_model: Original trained model
        quantized_model: Quantized model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        dict: Comparison metrics
    """
    # Evaluate original model
    original_r2, original_mse = evaluate_model(original_model, X_test, y_test)
    
    # Evaluate quantized model
    quantized_r2, quantized_mse = evaluate_model(quantized_model, X_test, y_test)
    
    # Calculate differences
    r2_diff = quantized_r2 - original_r2
    mse_diff = quantized_mse - original_mse
    
    return {
        'original_r2': original_r2,
        'original_mse': original_mse,
        'quantized_r2': quantized_r2,
        'quantized_mse': quantized_mse,
        'r2_diff': r2_diff,
        'mse_diff': mse_diff
    }


def quantize_model():
    """
    Main function to quantize the trained model.
    
    Returns:
        dict: Quantization results and comparison metrics
    """
    print("Loading trained model...")
    model_path = "models/linear_regression_model.joblib"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    original_model = load_model(model_path)
    
    print("Extracting model parameters...")
    coef = original_model.coef_
    intercept = original_model.intercept_
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    
    # Save original parameters
    print("Saving original parameters...")
    original_params = {
        'coef': coef,
        'intercept': intercept
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(original_params, "models/unquant_params.joblib")
    
    # Quantize parameters
    print("Quantizing parameters...")
    quantized_coef, quantized_intercept, scale_factor = quantize_parameters(
        coef, intercept
    )
    
    print(f"Quantized coefficients: {quantized_coef}")
    print(f"Quantized intercept: {quantized_intercept}")
    print(f"Scale factor: {scale_factor}")
    
    # Save quantized parameters
    print("Saving quantized parameters...")
    quantized_params = {
        'coef': quantized_coef,
        'intercept': quantized_intercept,
        'scale_factor': scale_factor
    }
    joblib.dump(quantized_params, "models/quant_params.joblib")
    
    # Create quantized model and compare performance
    print("Creating quantized model...")
    quantized_model = create_quantized_model(
        original_model, quantized_coef, quantized_intercept, scale_factor
    )
    
    # Load test data for comparison
    print("Loading test data for comparison...")
    _, X_test, _, y_test = load_california_housing_data()
    
    # Compare models
    print("Comparing model performance...")
    comparison = compare_models(original_model, quantized_model, X_test, y_test)
    
    # Print comparison results
    print("=" * 60)
    print("QUANTIZATION RESULTS")
    print("=" * 60)
    print(f"Original R² Score: {comparison['original_r2']:.4f}")
    print(f"Quantized R² Score: {comparison['quantized_r2']:.4f}")
    print(f"R² Difference: {comparison['r2_diff']:.4f}")
    print(f"Original MSE: {comparison['original_mse']:.4f}")
    print(f"Quantized MSE: {comparison['quantized_mse']:.4f}")
    print(f"MSE Difference: {comparison['mse_diff']:.4f}")
    print("=" * 60)
    
    return comparison


if __name__ == "__main__":
    quantize_model() 
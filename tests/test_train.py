"""
Unit tests for the training pipeline.
"""
import pytest
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_california_housing_data, evaluate_model, save_model, load_model
from train import train_model


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_california_housing_data(self):
        """Test that California Housing dataset loads correctly."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Check data types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check shapes
        assert X_train.shape[1] == 8  # California Housing has 8 features
        assert X_test.shape[1] == 8
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check that data is not empty
        assert X_train.size > 0
        assert X_test.size > 0
        assert y_train.size > 0
        assert y_test.size > 0
        
        # Check that train and test sets are different
        assert not np.array_equal(X_train, X_test)


class TestModelCreation:
    """Test model creation and validation."""
    
    def test_linear_regression_instance(self):
        """Test that LinearRegression model can be created."""
        model = LinearRegression()
        assert isinstance(model, LinearRegression)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_model_training(self):
        """Test that model can be trained and has coefficients."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Check that model was trained (has coefficients)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert len(model.coef_) == X_train.shape[1]  # Number of features


class TestModelPerformance:
    """Test model performance and evaluation."""
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        r2, mse = evaluate_model(model, X_test, y_test)
        
        # Check that metrics are valid
        assert isinstance(r2, float)
        assert isinstance(mse, float)
        assert 0 <= r2 <= 1  # R² score should be between 0 and 1
        assert mse >= 0  # MSE should be non-negative
    
    def test_r2_score_threshold(self):
        """Test that R² score exceeds minimum threshold."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        r2, _ = evaluate_model(model, X_test, y_test)
        
        # R² score should be above a reasonable threshold for California Housing
        # This dataset typically achieves R² > 0.5 with Linear Regression
        assert r2 > 0.4, f"R² score {r2:.4f} is below threshold of 0.4"


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_model_save_load(self, tmp_path):
        """Test that model can be saved and loaded correctly."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        save_model(model, str(model_path))
        
        # Check that file exists
        assert model_path.exists()
        
        # Load model
        loaded_model = load_model(str(model_path))
        
        # Check that loaded model is the same type
        assert isinstance(loaded_model, LinearRegression)
        
        # Check that coefficients are the same
        np.testing.assert_array_almost_equal(model.coef_, loaded_model.coef_)
        assert abs(model.intercept_ - loaded_model.intercept_) < 1e-10
        
        # Check that predictions are the same
        original_predictions = model.predict(X_test)
        loaded_predictions = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestTrainingPipeline:
    """Test the complete training pipeline."""
    
    def test_train_model_function(self):
        """Test that train_model function works end-to-end."""
        model, r2, mse = train_model()
        
        # Check return types
        assert isinstance(model, LinearRegression)
        assert isinstance(r2, float)
        assert isinstance(mse, float)
        
        # Check that model was trained
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check performance metrics
        assert 0 <= r2 <= 1
        assert mse >= 0
        
        # Check that model file was created
        assert os.path.exists("models/linear_regression_model.joblib")


if __name__ == "__main__":
    pytest.main([__file__]) 
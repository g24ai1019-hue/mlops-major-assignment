# MLOps Pipeline - Linear Regression on California Housing Dataset

This repository contains a complete MLOps pipeline focused on Linear Regression using the California Housing dataset from sklearn.datasets. The pipeline includes training, testing, quantization, Dockerization, and CI/CD workflows.

## Project Structure

```
MLOps_Major/
├── src/
│   ├── train.py          # Model training script
│   ├── quantize.py       # Model quantization script
│   ├── predict.py        # Prediction script
│   └── utils.py          # Utility functions
├── tests/
│   └── test_train.py     # Unit tests
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD workflow
└── README.md            # This file
```

## Features

- **Model Training**: Linear Regression on California Housing dataset
- **Testing**: Comprehensive unit tests for data loading and model validation
- **Quantization**: Manual 8-bit quantization of model parameters
- **Dockerization**: Containerized application for deployment
- **CI/CD**: Automated testing and deployment pipeline

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python src/train.py`
4. Run tests: `pytest tests/`
5. Build Docker image: `docker build -t mlops-pipeline .`

## Model Performance Comparison

| Metric | Original Model | Quantized Model | Difference |
|--------|----------------|-----------------|------------|
| R² Score | 0.606 | 0.605 | -0.001 |
| MSE | 0.524 | 0.526 | +0.002 |
| Model Size | 1.2 KB | 0.8 KB | -33% |
| Inference Time | 0.15 ms | 0.12 ms | -20% |

## CI/CD Pipeline

The pipeline includes three main jobs:
1. **Test Suite**: Runs unit tests to validate code quality
2. **Train and Quantize**: Trains the model and performs quantization
3. **Build and Test Container**: Builds Docker image and validates predictions

## Requirements

- Python 3.8+
- scikit-learn
- numpy
- pandas
- joblib
- pytest
- docker

## License

MIT License 
#!/usr/bin/env python3
"""
Simple test runner for local testing.
"""
import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def main():
    """Run all tests and checks."""
    print("MLOps Pipeline - Local Test Runner")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/train.py"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Install dependencies
    success = run_command("pip install -r requirements.txt", "Installing dependencies")
    if not success:
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run unit tests
    success = run_command("python -m pytest tests/ -v", "Running unit tests")
    if not success:
        print("❌ Unit tests failed")
        sys.exit(1)
    
    # Train model
    success = run_command("python src/train.py", "Training model")
    if not success:
        print("❌ Model training failed")
        sys.exit(1)
    
    # Quantize model
    success = run_command("python src/quantize.py", "Quantizing model")
    if not success:
        print("❌ Model quantization failed")
        sys.exit(1)
    
    # Run predictions
    success = run_command("python src/predict.py", "Running predictions")
    if not success:
        print("❌ Predictions failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("Your MLOps pipeline is working correctly!")
    print("You can now:")
    print("1. Push to GitHub to trigger CI/CD")
    print("2. Build Docker image: docker build -t mlops-pipeline .")
    print("3. Run container: docker run mlops-pipeline")


if __name__ == "__main__":
    main() 
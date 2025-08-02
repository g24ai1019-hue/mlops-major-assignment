# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create models directory
RUN mkdir -p models

# Train the model during build
RUN python src/train.py

# Run quantization
RUN python src/quantize.py

# Set the default command to run predictions
CMD ["python", "src/predict.py"] 
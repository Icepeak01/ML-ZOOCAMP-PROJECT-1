# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and to buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Force TensorFlow to use CPU only
ENV CUDA_VISIBLE_DEVICES=-1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY app/requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application code, including the model
COPY app/ /app/

# Expose the port Flask is running on
EXPOSE 8080

# Define the default command to run the app
CMD gunicorn app.app:app --bind 0.0.0.0:$PORT --workers=1 --threads=1 --timeout=120

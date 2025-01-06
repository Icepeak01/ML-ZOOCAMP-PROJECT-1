# scripts/predict.py

import os
import sys
import numpy as np
import tensorflow as tf
from utils import load_model, preprocess_image, make_gradcam_heatmap, save_and_overlay_gradcam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import argparse

def main(image_path, output_path='gradcam_output.jpg'):
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'app', 'model', 'mobilenetv2_pneumonia_model.h5')

    # Load the model
    model = load_model(MODEL_PATH)
    if model is None:
        print("Failed to load the model.")
        sys.exit(1)

    # Preprocess the image
    img_array = preprocess_image(image_path, target_size=(224, 224))
    
    # Make prediction
    pred_prob = model.predict(img_array)[0][0]
    threshold = 0.5  # Adjust as needed
    if pred_prob > threshold:
        prediction = 'PNEUMONIA'
        confidence = pred_prob
    else:
        prediction = 'NORMAL'
        confidence = 1 - pred_prob
    prediction_text = f"Prediction: {prediction} (Confidence: {confidence:.2f})"
    print(prediction_text)

    # Generate Grad-CAM heatmap
    LAST_CONV_LAYER = 'Conv_1'  # Update if using a different model
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

    # Save Grad-CAM image
    save_and_overlay_gradcam(image_path, heatmap, output_path, alpha=0.4)
    print(f"Grad-CAM image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Pneumonia from Chest X-ray Image')
    parser.add_argument('image_path', type=str, help='Path to the input X-ray image')
    parser.add_argument('--output', type=str, default='gradcam_output.jpg', help='Path to save the Grad-CAM image')
    args = parser.parse_args()

    main(args.image_path, args.output)

import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from utils import load_model, preprocess_image, make_gradcam_heatmap, save_and_overlay_gradcam
import logging
import uuid
import sys

# Ensure TensorFlow uses CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize the Flask application
app = Flask(__name__)

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')  
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenetv2_pneumonia_model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
GRADCAM_FOLDER = os.path.join(BASE_DIR, 'static', 'gradcam')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Configure upload parameters
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define the last convolutional layer name for Grad-CAM
LAST_CONV_LAYER = 'Conv_1'  

# Define class labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    error = request.args.get('error')
    return render_template('index.html', error=error)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.warning("No file part in the request.")
        return redirect(url_for('index', error="No file part in the request."))

    file = request.files['file']

    if file.filename == '':
        logging.warning("No file selected for uploading.")
        return redirect(url_for('index', error="No file selected for uploading."))

    if file and allowed_file(file.filename):
        try:
            # Secure the filename and append UUID to prevent overwriting and caching issues
            filename = secure_filename(file.filename)
            unique_id = uuid.uuid4().hex
            filename = f"{unique_id}_{filename}"
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            logging.info(f"File {filename} saved successfully.")

            # Preprocess the image
            img_array = preprocess_image(img_path, target_size=(224, 224))
            logging.info("Image preprocessed successfully.")

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
            logging.info(f"Prediction made: {prediction_text}")

            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
            logging.info("Grad-CAM heatmap generated.")

            # Save Grad-CAM image
            gradcam_filename = f"gradcam_{filename}"
            gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
            superimposed_img = save_and_overlay_gradcam(img_path, heatmap, gradcam_path, alpha=0.4)
            logging.info(f"Grad-CAM image saved as {gradcam_filename}.")

            # Prepare paths for display
            original_image = url_for('static', filename=f'uploads/{filename}')
            gradcam_image = url_for('static', filename=f'gradcam/{gradcam_filename}')

            return render_template('result.html',
                                   prediction_text=prediction_text,
                                   original_image=original_image,
                                   gradcam_image=gradcam_image)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return redirect(url_for('index', error="An error occurred during prediction. Please try again."))
    else:
        logging.warning("Unsupported file type uploaded.")
        return redirect(url_for('index', error="Unsupported file type. Please upload an image file."))

if __name__ == '__main__':
    if model is not None:
        port = int(os.environ.get('PORT', 5000))  # Use PORT environment variable if it exists, otherwise default to 5000
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logging.error("Model not loaded. Flask app will not run.")
        exit(1)


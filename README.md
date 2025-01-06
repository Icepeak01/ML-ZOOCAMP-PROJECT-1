# Pneumonia-Detection-CNN

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Project Overview

This project aims to develop a Convolutional Neural Network (CNN) model to detect pneumonia from chest X-ray images. Leveraging transfer learning with MobileNetV2, the model achieves high accuracy while maintaining a small footprint, making it suitable for deployment in resource-constrained environments.

## Problem Description

Pneumonia is a leading cause of death worldwide, especially among children under five. Early and accurate detection is crucial for effective treatment and reducing mortality rates. Automated detection using deep learning can assist healthcare professionals by providing quick and reliable diagnostics.

**How the Model Helps:**
- **Speed:** Provides instant predictions, enabling faster diagnosis.
- **Accuracy:** Minimizes human error, ensuring reliable results.
- **Scalability:** Can be deployed across various platforms, aiding in widespread screening.

## Dataset

The dataset used for this project is sourced from [Kaggle's Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

### Dataset Details:
- **Classes:** 
  - `NORMAL` (0)
  - `PNEUMONIA` (1)
- **Distribution:**
  - **Training:** 5,216 images (Imbalanced: More `PNEUMONIA` than `NORMAL`)
  - **Validation:** 16 images (Reserved for final evaluation)
  - **Test:** 624 images

### Accessing the Data:
- **Committed in Repository:** The `chest_xray` directory contains all necessary images.
- **Alternatively:** Instructions to download from Kaggle are provided below.

## Project Structure


![image](https://github.com/user-attachments/assets/774d6404-55d0-4517-9c6f-623b3168da1f)


## Installation

### Prerequisites

- **Docker:** Ensure Docker is installed on your machine. [Installation Guide](https://docs.docker.com/get-docker/)
- **Python 3.9:** Required for running training and prediction scripts.

### Clone the Repository

```bash
git clone https://github.com/Icepeak01/ML-ZOOCAMP-PROJECT-.git
cd ML-ZOOCAMP-PROJECT
```

### Setup Python Environment
It's recommended to use a virtual environment
```bash
# Using virtualenv
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r app/requirements.txt
```
### Usage
Running the Flask Application Locally
```bash
# Activate Virtual Environment
source venv/bin/activate

# Navigate to the app Directory
cd app

# Run the Application
python app.py
```

### Docker Deployment
```bash
# Build the Docker Image:
docker build -t pneumonia-detection-cnn .

# Run the Docker Container:
docker run -d -p 8080:8080 pneumonia-detection-cnn

```
Access the App:

Open your browser and navigate to http://localhost:8080.

### Accessing the Deployed Application
The application is deployed on Render and can be accessed via the following link:

👉 [Pneumonia Detection App](https://ml-zoocamp-project-1.onrender.com/)

## Training the Model
The training process involves data preparation, model building using MobileNetV2, training with early stopping and learning rate reduction, and fine-tuning.

Running the Training Script
Ensure Dependencies are Installed:
``` bash
pip install -r app/requirements.txt
```
Run the Training Script:
``` bash
python scripts/train.py
```

Model Saving:

The trained model is saved in the app/model/ directory as mobilenetv2_pneumonia_model.h5.

## Model Evaluation
After training, the model was evaluated on the test dataset with the following results:

Classification Report:
``` markdown
              precision    recall  f1-score   support

      NORMAL       0.96      0.70      0.81       234
   PNEUMONIA       0.84      0.98      0.91       390

    accuracy                           0.88       624
   macro avg       0.90      0.84      0.86       624
weighted avg       0.89      0.88      0.87       624
```

### Training Performance:
``` markdown
  Epoch	    Accuracy	    Loss	   Val Accuracy	    Val Loss	 Learning Rate
      1	     0.7111	    0.5754	   0.9406	        0.1885        1e-04
      2	     0.9127	    0.2273	   0.9386	        0.1545	      1e-04
      3	     0.9283	    0.1706	   0.9473	        0.1349	      1e-04
      4      0.9415	    0.1415	   0.9703	        0.0926	      1e-04
      5	     0.9523	    0.1205	   0.9645	        0.0998	      1e-04
      6	     0.9521	    0.1182	   0.9703	        0.0882	      1e-04
```
### Fine-Tuning Performance:
``` markdown
  Epoch	    Accuracy	  Loss	      Val Accuracy	   Val Loss	  Learning Rate
      6	      0.9050	  0.3118	    0.9664	      0.0953	      1e-06
      7	      0.9252	  0.2134	    0.9655	      0.1013	      1e-06
      8	      0.9128	  0.2366	    0.9597	      0.1065	      1e-06
```

## Deployment

The application is containerized using Docker and deployed to [Render](https://render.com/).

### Access the Deployed Application

👉 [Pneumonia Detection App](https://ml-zoocamp-project-1.onrender.com)


### Deployment Steps

1. **Build the Docker Image:**

    ```bash
    docker build -t pneumonia-detection-cnn .
    ```

2. **Run Locally for Testing:**

    ```bash
    docker run -d -p 8080:8080 pneumonia-detection-cnn
    ```

3. **Push to GitHub:**

    Ensure all changes, including the Dockerfile and model, are pushed to your GitHub repository.

4. **Deploy to Render:**

    - Connect your GitHub repository to Render.
    - Render will automatically build and deploy the Docker container.
    - Ensure the `PORT` environment variable is set correctly if needed.

## Technologies Used

- **Programming Languages:** Python
- **Frameworks & Libraries:**
  - Flask
  - TensorFlow & Keras
  - OpenCV
  - Matplotlib
  - Seaborn
  - Gunicorn
- **Tools:**
  - Docker
  - Git & GitHub
  - Render (for deployment)


## Deployment

![image](https://github.com/user-attachments/assets/9668f398-fe79-451d-8371-9778bfdb76a2)
*Screenshot of the Pneumonia Detection App while uploading an X-ray image.*


![image](https://github.com/user-attachments/assets/e4f110a6-c488-4e94-b357-fdcb50ec9532)

*Screenshot of the Pneumonia Detection App showing an uploaded X-ray image and the Grad-CAM heatmap.*

## Dataset

The dataset used in this project is sourced from [Kaggle's Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

### Download Instructions:

1. Create a Kaggle account if you don't have one.
2. Navigate to the [dataset page](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
3. Click on the "Download" button to download the dataset.






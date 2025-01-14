# -*- coding: utf-8 -*-
"""PNEUMONIA CL DETECTION DATA EXPLORATION AND PREPROCESSING.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t1X8u7RxJ1doJYpaD_Up1YsfK1cy8WPj
"""

import os
import shutil

# Define base path (adjust if your project is in a different location)
base_path = '/content/drive/MyDrive/chest_xray/chest_xray'

# Verify current structure
!ls {base_path}

splits = ['train', 'val', 'test']

for split in splits:
    split_path = os.path.join(base_path, split)
    print(f"\nContents of {split_path}:")
    !ls {split_path}

# Define base path
base_path = '/content/drive/MyDrive/chest_xray/chest_xray'

# Define paths for each split
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')
test_path = os.path.join(base_path, 'test')

# Verify paths
print("Train Path:", train_path)
print("Validation Path:", val_path)
print("Test Path:", test_path)

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.metrics import classification_report, confusion_matrix

# Define image size and batch size
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Training data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='reflect',
    brightness_range=[0.5,1.5],
    validation_split=0.0  # Since we have separate validation set
)

# Validation and Testing data generators (no augmentation, only rescaling)
val_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Flow validation images in batches using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Flow test images in batches using test_datagen generator
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Print class indices for reference
print("Class Indices:", train_generator.class_indices)

def plot_class_distribution(generator, title):
    labels = generator.classes
    class_labels = list(generator.class_indices.keys())
    sns.countplot(x=labels)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(ticks=[0,1], labels=class_labels)
    plt.show()

# Plot for Training Set
plot_class_distribution(train_generator, 'Training Set Class Distribution')

# Plot for Validation Set
plot_class_distribution(validation_generator, 'Validation Set Class Distribution')

# Plot for Test Set
plot_class_distribution(test_generator, 'Test Set Class Distribution')

import matplotlib.image as mpimg

def display_sample_images(generator, class_label, num=5):
    class_indices = generator.class_indices
    class_name = [name for name, index in class_indices.items() if index == class_label][0]
    class_dir = os.path.join(generator.directory, class_name)
    images = os.listdir(class_dir)[:num]

    plt.figure(figsize=(15, 5))
    for i, img_name in enumerate(images):
        img_path = os.path.join(class_dir, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(1, num, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{class_name} #{i+1}")
        plt.axis('off')
    plt.show()

# Display sample Pneumonia images (class label 1)
display_sample_images(train_generator, class_label=1, num=5)

# Display sample Normal images (class label 0)
display_sample_images(train_generator, class_label=0, num=5)

# Select a sample image from the training set
sample_image_path = os.path.join(train_path, 'PNEUMONIA', os.listdir(os.path.join(train_path, 'PNEUMONIA'))[0])
sample_image = cv2.imread(sample_image_path)
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
sample_image = cv2.resize(sample_image, (IMG_WIDTH, IMG_HEIGHT))
sample_image = sample_image.reshape((1,) + sample_image.shape)  # Reshape to (1, height, width, channels)

# Create an augmented data generator (reuse train_datagen settings)
aug_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    brightness_range=[0.5,1.5]
)

# Generate augmented images
aug_iter = aug_datagen.flow(sample_image, batch_size=1)

# Plot augmented images
plt.figure(figsize=(10, 4))
for i in range(10):
    aug_image = next(aug_iter)[0].astype('uint8')
    plt.subplot(2, 5, i+1)
    plt.imshow(aug_image)
    plt.axis('off')
plt.suptitle('Data Augmentation Examples', fontsize=16)
plt.show()

"""#### Computing class weight to address imbalance in the train data"""

from sklearn.utils import class_weight
import numpy as np

# Retrieve the classes from the training generator
classes = train_generator.classes
class_labels = list(train_generator.class_indices.keys())

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)

# Create a dictionary mapping class indices to weights
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weights:", class_weights_dict)

"""#### Adjusting Data Generators to enhance data augmentation specifically for the minority class (normal) to further balance the dataset."""

# Separate directories
normal_train_dir = os.path.join(train_path, 'NORMAL')
pneumonia_train_dir = os.path.join(train_path, 'PNEUMONIA')

# Create a generator for NORMAL class with higher augmentation
normal_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.02,
    zoom_range=[0.8, 1.3],
    horizontal_flip=True,
    fill_mode='reflect',
    brightness_range=[0.4,1.6]
)

# Create a generator for PNEUMONIA class with standard augmentation
pneumonia_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    fill_mode='reflect',
    brightness_range=[0.5,1.5]
)

# Flow from directory for NORMAL
normal_generator = normal_datagen.flow_from_directory(
    directory=train_path,
    classes=['NORMAL'],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Flow from directory for PNEUMONIA
pneumonia_generator = pneumonia_datagen.flow_from_directory(
    directory=train_path,
    classes=['PNEUMONIA'],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

import itertools

# Calculate how many times to repeat the minority class to balance
steps_per_epoch = max(len(normal_generator), len(pneumonia_generator))

# Create a combined generator
def combined_generator(normal_gen, pneumonia_gen):
    while True:
        normal_batch = next(normal_gen)
        pneumonia_batch = next(pneumonia_gen)
        images = np.concatenate((normal_batch[0], pneumonia_batch[0]))
        labels = np.concatenate((normal_batch[1], pneumonia_batch[1]))
        yield images, labels

balanced_train_generator = combined_generator(normal_generator, pneumonia_generator)






















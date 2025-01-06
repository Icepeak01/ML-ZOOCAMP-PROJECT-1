import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_transfer_learning_model(input_shape=(224, 224, 3)):
    """
    Builds a transfer learning model using MobileNetV2 as the base.

    Parameters:
    - input_shape: Tuple specifying the input shape of the images.

    Returns:
    - model: The complete Keras model ready for training.
    - base_model: The pre-trained base MobileNetV2 model.
    """
    # Load the MobileNetV2 model, excluding the top layers
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')

    # Freeze the base model initially
    base_model.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

    # Construct the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model

def fine_tune_model(model, base_model, fine_tune_at=100):
    """
    Unfreeze the base model and freeze all layers before the specified layer.

    Parameters:
    - model: The compiled Keras model.
    - base_model: The pre-trained base model.
    - fine_tune_at: The layer from which to start fine-tuning.

    Returns:
    - model: The recompiled Keras model ready for fine-tuning.
    """
    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers before the 'fine_tune_at' layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile the model with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def plot_training_history(history, history_fine=None):
    """
    Plots training and validation accuracy and loss.

    Parameters:
    - history: History object from initial training.
    - history_fine: History object from fine-tuning (optional).
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.savefig('training_history.png')  # Save the plot
    plt.show()

def main():
    # Define file paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, '..', 'data', 'chest_xray')  # Update as per your data location
    TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
    VAL_DIR = os.path.join(DATASET_PATH, 'val')
    TEST_DIR = os.path.join(DATASET_PATH, 'test')
    MODEL_DIR = os.path.join(BASE_DIR, '..', 'app', 'model')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32

    # Data Generators with Augmentation for Training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    # Training Generator
    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )

    # Validation Split Generator
    validation_split_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True,
        seed=42
    )

    # Test Generator (Reserved for Final Evaluation)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        directory=TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Compute class weights to handle imbalance
    classes = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Class Weights:", class_weights_dict)

    # Build the model
    model, base_model = build_transfer_learning_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5, verbose=1)

    # Initial Training
    history = model.fit(
        train_generator,
        epochs=6,
        validation_data=validation_split_generator,
        class_weight=class_weights_dict,
        callbacks=[early_stop, reduce_lr]
    )

    # Fine-tuning the model
    model = fine_tune_model(model, base_model, fine_tune_at=100)

    # Continue Training with Fine-Tuning
    fine_tune_epochs = 4
    total_epochs = 6 + fine_tune_epochs

    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_split_generator,
        class_weight=class_weights_dict,
        callbacks=[early_stop, reduce_lr]
    )

    # Plot training history
    plot_training_history(history, history_fine)

    # Save the final model
    model_save_path = os.path.join(MODEL_DIR, 'mobilenetv2_pneumonia_model.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate on Test Data
    test_generator.reset()
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = (y_pred_prob > 0.65).astype(int).ravel()
    y_true = test_generator.classes

    # Classification Report
    class_labels = ['NORMAL', 'PNEUMONIA']
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig('confusion_matrix.png')  # Save the confusion matrix
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')  # Save the ROC curve
    plt.show()

if __name__ == "__main__":
    main()
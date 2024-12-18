# ml_trend.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and info logs

import pandas as pd
import numpy as np
import joblib
import logging
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv1D, LSTM, MultiHeadAttention, Flatten
from tensorflow.keras.regularizers import l2  # Import L2 regularizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Process

from Data.technical_indicators import calculate_ticks, calculate_adl, calculate_vold_ratio

# Load a Deep Neural Network model
def load_dnn_model(model_path="Models/dnn_SPY0DTE.keras"):
    try:
        model = load_model(model_path)
        #logging.info(f"Loaded DNN model from {model_path}")
        return model
    except Exception as e:
        #logging.error(f"Error loading DNN model: {e}", exc_info=True)
        return None
    
def preprocess_data(candles_df):
    """
    Preprocess data to include indicators and prepare for training.
    """
    try:
        # Calculate indicators
        candles_df['Ticks'] = calculate_ticks(candles_df)
        candles_df['ADL'] = calculate_adl(candles_df)
        candles_df['VOLD_Ratio'] = calculate_vold_ratio(candles_df)

        # Drop rows with missing values
        candles_df.dropna(inplace=True)

        # Generate labels (1 for upward trend, 0 for downward trend)
        candles_df['Future_Close'] = candles_df['close'].shift(-5)
        candles_df['Label'] = (candles_df['Future_Close'] > candles_df['close']).astype(int)
        candles_df.dropna(subset=['Future_Close'], inplace=True)

        # Select features and labels
        features = candles_df[['Ticks', 'ADL', 'VOLD_Ratio']]
        labels = candles_df['Label']

        # Scale features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        return features_scaled, labels, scaler

    except Exception as e:
        logging.error(f"Error in preprocess_data: {e}", exc_info=True)
        return None, None, None

def predict_trend(candles_df, model, scaler):
    """
    Predict upward or downward trend using the trained model.
    """
    try:
        # Calculate indicators
        candles_df['Ticks'] = calculate_ticks(candles_df)
        candles_df['ADL'] = calculate_adl(candles_df)
        candles_df['VOLD_Ratio'] = calculate_vold_ratio(candles_df)

        # Select features and scale them
        features = candles_df[['Ticks', 'ADL', 'VOLD_Ratio']].tail(1)
        features_scaled = scaler.transform(features)

        # Predict trend
        prediction = model.predict(features_scaled)
        return "BUY" if prediction[0][0] > 0.5 else "SELL"

    except Exception as e:
        logging.error(f"Error in predict_trend: {e}", exc_info=True)
        return None

# Train DNN model
def train_trends_model(candles_df, model_path="Models/dnn_SPY0DTE.keras"):
    """
    Train a DNN model for candlestick pattern detection.
    """
    # Preprocess the data
    X, y, scaler = preprocess_data(candles_df)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape features for Conv1D input
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train model
    input_shape = X_train.shape[1]
    model = Sequential([
        Input(shape=(input_shape,)),  # Input layer
        Dense(64, activation='relu'),  # Hidden layer 1
        Dropout(0.2),  # Dropout for regularization
        Dense(32, activation='relu'),  # Hidden layer 2
        Dropout(0.2),  # Dropout for regularization
        Dense(16, activation='relu'),  # Hidden layer 3
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        X_train_reshaped,
        y_train,
        epochs=500,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
    )

    # Save the model
    model.save(model_path)

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
    print(classification_report(y_test, y_pred))

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    return model, scaler

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = -y_true * alpha * tf.keras.backend.pow(1 - y_pred, gamma) * tf.keras.backend.log(y_pred) \
               - (1 - y_true) * (1 - alpha) * tf.keras.backend.pow(y_pred, gamma) * tf.keras.backend.log(1 - y_pred)
        return tf.keras.backend.mean(loss)
    return focal_loss_fixed

# Plot ROC Curve
def plot_roc_curve(y_test, y_scores, output_path="Models/Training/roc_curve_0dte.png"):
    """
    Generate and save the ROC curve plot.
    :param y_test: True binary labels.
    :param y_scores: Predicted probabilities or scores for the positive class.
    :param output_path: Path to save the ROC curve image.
    """
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels, output_path="Models/Training/confusion_matrix_0dte.png", normalize=False):
    """
    Generate and save the confusion matrix plot.
    :param y_true: Ground truth (actual) labels.
    :param y_pred: Predicted labels.
    :param labels: List of class labels (e.g., ['Non-Profitable', 'Profitable']).
    :param output_path: Path to save the confusion matrix image.
    :param normalize: Normalize values or not.
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
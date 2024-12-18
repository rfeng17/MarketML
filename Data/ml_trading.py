# ml_trading.py

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and info logs

import pandas as pd
import numpy as np
import joblib
import logging
import datetime
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv1D, LSTM, MultiHeadAttention, Flatten
from tensorflow.keras.regularizers import l2  # Import L2 regularizer
from tensorflow.keras.saving import register_keras_serializable
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

from Data.patterns import detect_patterns
from Data.technical_indicators import (
    calculate_MACD,
    calculate_bollinger_bands,
    calculate_parabolic_sar,
    calculate_ema,
    calculate_adl
)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Load a Deep Neural Network model
def load_dnn_model(model_path):
    try:
        model = load_model(model_path, custom_objects={"focal_loss_fixed": focal_loss_fixed})
        #logging.info(f"Loaded DNN model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading DNN model: {e}", exc_info=True)
        return None
    
def preprocess_data(candles_df, for_training=True):
    """
    Preprocess the data for machine learning.
    """
    try:
        # Ensure unique indices
        candles_df = candles_df[~candles_df.index.duplicated(keep='last')]
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not set(required_columns).issubset(candles_df.columns):
            raise ValueError(f"Missing required columns. Found: {candles_df.columns}")
        
        # Preserve the original datetime index
        original_index = candles_df.index
        
        candles_df.reset_index(drop=True, inplace=True)

        # Indicators
        candles_df['EMA'] = calculate_ema(candles_df['close'], 9)
        candles_df['ADL'] = calculate_adl(candles_df)
        candles_df['MACD'], candles_df['Signal Line'] = calculate_MACD(candles_df['close'])
        candles_df['MACD'] = candles_df['MACD'].fillna(0)
        candles_df['Signal Line'] = candles_df['Signal Line'].fillna(0)
        _, candles_df['Upper Band'], candles_df['Lower Band'] = calculate_bollinger_bands(candles_df['close'])

        # Fill NaN values for Bollinger Bands
        candles_df['Upper Band'] = candles_df['Upper Band'].bfill().ffill().fillna(0)
        candles_df['Lower Band'] = candles_df['Lower Band'].bfill().ffill().fillna(0)

        candles_df['SAR'] = calculate_parabolic_sar(candles_df).fillna(0)
        candles_df['Volatility'] = candles_df['close'].pct_change().rolling(window=10).std().fillna(0)
        candles_df['Momentum'] = candles_df['close'] - candles_df['close'].shift(5)

        # Other Features
        candles_df['Average Volume'] = candles_df['volume'].rolling(window=20).mean().fillna(0)
        candles_df['RVOL'] = candles_df['volume'] / candles_df['volume'].rolling(window=20).mean().fillna(1)
        candles_df['Volume Spike'] = (candles_df['volume'] > 1.5 * candles_df['Average Volume']).astype(int)

        candles_df['Price Change'] = candles_df['close'].pct_change().fillna(0)
        candles_df['Price Range'] = candles_df['high'] - candles_df['low']
        
        # Candlestick patterns
        
        # Add Slope feature
        candles_df['Slope'] = candles_df['close'].rolling(window=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        ).fillna(0)
        
        # Restore the original datetime index
        candles_df.set_index(original_index, inplace=True)
        
        # For training: Prepare labels
        if for_training:
            candles_df['Future Price'] = candles_df['close'].shift(-5)
            candles_df['Label'] = 0
            candles_df.loc[candles_df['Future Price'] > candles_df['close'], 'Label'] = 1
            candles_df.loc[candles_df['Future Price'] < candles_df['close'], 'Label'] = -1
            candles_df.dropna(subset=['Future Price'], inplace=True)
            
        
        feature_columns = [
            'EMA', 'ADL', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band', 'SAR',
            'Volatility', 'Momentum', 'Average Volume', 'RVOL',
            'Volume Spike', 'Price Change', 'Price Range', 'Slope'
        ]

        features = candles_df[feature_columns].fillna(0)
        
        if for_training:
        
            labels = candles_df['Label'].astype(int)

            # Ensure lengths match
            if len(features) != len(labels):
                logging.error(f"Mismatch between features and labels: {len(features)} vs {len(labels)}")
                return None, None, None
            
            # Scale features
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features.fillna(0))
            joblib.dump(scaler, 'scaler.pkl')
            
            return features_scaled, labels, scaler
        
        # For prediction
        return features.fillna(0)

    except Exception as e:
        logging.error(f"Error in preprocess_data: {e}", exc_info=True)
        return None, None
        
def predict_signal(candles_df, model, model_type):
    """
    Predict buy/sell/hold signal using the trained model efficiently.
    Adjusts feature set dynamically based on the symbol type (crypto or stock).
    """
    try:
        # Validate input data
        if model is None:
            logging.error("Model is not loaded or trained.")
            return 0  # Default to hold

        if candles_df is None or candles_df.empty or candles_df.shape[0] < 2:
            logging.error("Insufficient data in candles_df for prediction.")
            return 0  # Default to hold
        
        # Calculate PSAR signal
        psar_values = calculate_parabolic_sar(candles_df)
        psar_signal = 1 if psar_values.iloc[-1] < candles_df['close'].iloc[-1] else 0  # Buy if PSAR < Close

        features = preprocess_data(candles_df, for_training=False)

        # Apply scaler
        try:
            scaler = joblib.load('scaler.pkl')
            features_scaled = scaler.transform(features)
        except Exception as e:
            logging.error(f"Error loading or applying scaler: {e}")
            return 0

        # Reshape features for LSTM and Conv1D layers
        features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))
        
        raw_prediction = -1
        # Predict using the model
        if model_type == 'dnn':
            raw_prediction = model.predict(features_scaled)
            #logging.info(f"DNN raw prediction probabilities: {raw_prediction}")
            
            # Threshold-based prediction
            #threshold = 0.42
            #prediction = 1 if raw_prediction[0][0] > threshold else 0
        elif model_type == 'sgd':
            raw_prediction = model.predict(features_scaled)
            logging.info(f"SGD raw prediction: {raw_prediction}")
            #prediction = raw_prediction[0]
        else:
            logging.error(f"Unsupported model type: {model_type}")
            return 0

        #logging.info(f"Prediction signal: {prediction}")
        pattern = detect_patterns(candles_df).iloc[-1]['Pattern']
        logging.info(f"Detected pattern: {pattern}")
        
        # Combine DNN and PSAR signals using ensemble logic
        final_signal = ensemble_signal(raw_prediction[-1][0], psar_signal)
        
        if ("Bullish" in pattern) and final_signal == 0:
            final_signal = -1
        elif ("Bearish" in pattern) and final_signal == 1:
            final_signal =- 1
            
        
        logging.info(f"Final Ensemble Signal: {final_signal}")
        
        return final_signal
    
    except Exception as e:
        logging.error(f"Error in predict_signal: {e}", exc_info=True)
        return 0

def ensemble_signal(dnn_signal, psar_signal):
    """
    Combine DNN and PSAR signals for a final decision.
    """
    combined_signal = 1 * dnn_signal + 0 * psar_signal
    logging.info(f"psar signal: {psar_signal}, combined signal: {combined_signal}")
    
    return 1 if combined_signal > 0.46 else 0

def balance_classes(X, y):
    """
    Balance the dataset using SMOTE or manual balancing if only one class is present.
    """
    print("Original class distribution:", Counter(y))
    
    class_counts = Counter(y)
    max_class_size = max(class_counts.values())
    
    # Dynamically adjust the sampling strategy to ensure it's valid
    sampling_strategy = {
        class_label: max(max_class_size, class_count + 75)  # Ensure at least 50 additional samples
        for class_label, class_count in class_counts.items()
    }
    
    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("Balanced class distribution:", Counter(y_balanced))
    except ValueError as e:
        logging.error(f"SMOTE error: {e}. Falling back to manual duplication.")
        # Manual fallback if SMOTE fails
        X_balanced, y_balanced = manual_balance(X, y)
    
    return X_balanced, y_balanced


def manual_balance(X, y):
    """
    Manually balance the dataset by duplicating minority class samples.
    """
    class_counts = Counter(y)
    max_class_size = max(class_counts.values())
    X_balanced, y_balanced = X.copy(), y.copy()
    
    for class_label, class_count in class_counts.items():
        deficit = max_class_size - class_count
        if deficit > 0:
            class_indices = np.where(y == class_label)[0]
            additional_indices = np.random.choice(class_indices, size=deficit, replace=True)
            X_balanced = np.vstack([X_balanced, X[additional_indices]])
            y_balanced = np.hstack([y_balanced, y[additional_indices]])
    
    return X_balanced, y_balanced

def encode_labels(y):
    """
    Re-encode labels to match the model's output range.
    Convert:
    -1 (Sell) -> 0
    0 (Hold)  -> 0  (Merge Hold with Sell to have only two classes)
    1 (Buy)   -> 1
    """
    y = np.where(y <= 0, 0, y)  # Merge -1 and 0 into 0
    y = np.where(y == 1, 1, y)  # Keep 1 as 1
    return y

# Train DNN model
def train_dnn(candles_df, model_path):
    """
    Train a DNN model with balanced classes.
    """
    X, y, scaler = preprocess_data(candles_df)
    if X is None or y is None:
        logging.error("Invalid data for training.")
        return None

    # Re-encode labels
    y = encode_labels(y)
    
    # Balance classes
    X_balanced, y_balanced = balance_classes(X, y)
    
    X_balanced_reshaped = X_balanced.reshape((X_balanced.shape[0], X_balanced.shape[1], 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Compute class weights
    unique_classes = np.unique(y_balanced)
    class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y_balanced)
    class_weights = {int(cls): weight for cls, weight in zip(unique_classes, class_weights_array)}
    
    # Functional API model definition
    input_layer = Input(shape=(X_balanced_reshaped.shape[1], 1))
    
    # Conv1D layer for feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # LSTM layer
    x = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Attention layer
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)

    # Flatten before dense layers
    x = Flatten()(attention_output)

    # Fully connected layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.03))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.03))(x)

    # Output layer
    output_layer = Dense(1, activation='sigmoid')(x)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss(), metrics=['accuracy'])
    
    # Train model
    # Train model
    history = model.fit(
        X_train_reshaped, y_train, epochs=500, batch_size=64,
        class_weight=class_weights, validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    model.save(model_path)
    logging.info(f"DNN model saved to {model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Hold/Sell', 'Buy']))
    plot_confusion_matrix(y_test, y_pred, labels=['Hold/Sell', 'Buy'])
    plot_roc_curve(y_test, model.predict(X_test_reshaped))

    return model

@register_keras_serializable(package="Custom", name="focal_loss_fixed")
def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.25):
    """
    Compute focal loss for binary classification.
    """
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    loss = -y_true * alpha * tf.keras.backend.pow(1 - y_pred, gamma) * tf.keras.backend.log(y_pred) \
           - (1 - y_true) * (1 - alpha) * tf.keras.backend.pow(y_pred, gamma) * tf.keras.backend.log(1 - y_pred)
    return tf.keras.backend.mean(loss)

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        return focal_loss_fixed(y_true, y_pred, gamma=gamma, alpha=alpha)
    return loss

# Global plot function
def plot_training(history_path, history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(history_path)
    plt.close()

# Updated plot_training_history function
def plot_training_history(history):
    """
    Plot training and validation loss/accuracy curves using a separate process.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    history_path = f"Models/Training/{date}_training_history.png"

    # Convert history object to a dictionary
    history_dict = {key: history.history[key] for key in history.history.keys()}

    # Use a separate process for plotting
    p = Process(target=plot_training, args=(history_path, history_dict))
    p.start()
    p.join()

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", normalize=False):
    """
    Plot a confusion matrix using Matplotlib and Seaborn.

    :param y_true: Ground truth (actual) labels.
    :param y_pred: Predicted labels.
    :param labels: List of label names corresponding to the classes.
    :param title: Title of the plot.
    :param normalize: Whether to normalize the confusion matrix values.
    :param save_path: Path to save the plot. If None, the plot is shown instead.
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    
    # Normalize the confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    # Add labels and title
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Save or show the plot
    save_path = f"Models/Training/{date}_confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_test, y_scores):
    """
    Generate and plot the ROC curve.
    """
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"Models/Training/{date}_roc_curve.png")
    plt.close()
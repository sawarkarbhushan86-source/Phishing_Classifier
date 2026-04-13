"""
Deep Learning LSTM Model for Phishing Detection
Uses neural networks for advanced phishing website classification
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from datetime import datetime
import warnings
from typing import Tuple

warnings.filterwarnings('ignore')


class PhishingLSTMModel:
    """LSTM-based Deep Learning model for phishing detection"""
    
    def __init__(self, max_sequence_length: int = 100, embedding_dim: int = 64):
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None
        
    def build_model(self, vocab_size: int, input_shape: tuple = None) -> models.Model:
        """Build LSTM architecture"""
        print("Building LSTM model architecture...")
        
        if input_shape:
            # For numeric input
            self.model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
        else:
            # For sequence input (URL text)
            self.model = models.Sequential([
                layers.Embedding(vocab_size, self.embedding_dim, 
                               input_length=self.max_sequence_length),
                layers.LSTM(64, return_sequences=True, dropout=0.2),
                layers.LSTM(32, dropout=0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(self.model.summary())
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> None:
        """Train LSTM model"""
        print("\nTraining LSTM model...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ]
        
        if X_val is None:
            X_val = X_train
            y_val = y_train
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("[OK] Training complete")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        print("\nEvaluating LSTM model...")
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten() if isinstance(y_test, np.ndarray) else np.array(y_test).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, zero_division=0),
            'recall': recall_score(y_test_flat, y_pred, zero_division=0),
            'f1': f1_score(y_test_flat, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test_flat, y_pred_proba)
        }
        
        print(f"[OK] Accuracy: {metrics['accuracy']:.4f}")
        print(f"[OK] Precision: {metrics['precision']:.4f}")
        print(f"[OK] Recall: {metrics['recall']:.4f}")
        print(f"[OK] F1-Score: {metrics['f1']:.4f}")
        print(f"[OK] ROC-AUC: {metrics['roc_auc']:.4f}\n")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        return y_pred, y_pred_proba.flatten()
    
    def save(self, model_name: str = 'lstm_model') -> str:
        """Save model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/{model_name}_{timestamp}.keras"
        
        os.makedirs("models", exist_ok=True)
        self.model.save(model_path)
        
        print(f"[OK] LSTM Model saved to: {model_path}")
        return model_path


def train_deep_learning_model(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Tuple:
    """Train LSTM model on phishing dataset"""
    print("\n" + "="*70)
    print("DEEP LEARNING MODEL TRAINING - LSTM")
    print("="*70 + "\n")
    
    lstm_model = PhishingLSTMModel()
    
    # Build model (use dataset shape for input)
    input_shape = (X_train.shape[1],) if len(X_train.shape) > 1 else (1,)
    lstm_model.build_model(vocab_size=10000, input_shape=input_shape)
    
    # Train
    lstm_model.train(X_train, y_train, X_test, y_test, epochs=30, batch_size=32)
    
    # Evaluate
    metrics = lstm_model.evaluate(X_test, y_test)
    
    # Save
    model_path = lstm_model.save()
    
    return lstm_model, model_path, metrics


if __name__ == "__main__":
    print("LSTM Module loaded successfully")

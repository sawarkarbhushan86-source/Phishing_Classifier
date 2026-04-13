"""
Phishing Detection Prediction Pipeline
Handles single and batch predictions with probability scoring
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List
from pathlib import Path


class PredictionPipeline:
    """Complete prediction pipeline for phishing detection"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_names = None
        self.model_metadata = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model"""
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        try:
            self.model = joblib.load(model_path)
            print("[OK] Model loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def load_preprocessing_objects(self, scaler_path: str = None, encoders_path: str = None) -> None:
        """Load preprocessing objects (scaler, encoders)"""
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("[OK] Scaler loaded")
        
        if encoders_path and os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
            print("[OK] Encoders loaded")
    
    def preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        X_processed = X.copy()
        
        # Handle missing values
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_processed[col].fillna(X_processed[col].median(), inplace=True)
        
        # Encode categorical columns if encoders available
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        if self.encoders:
            for col in categorical_cols:
                if col in self.encoders:
                    X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))
        
        # Scale numeric features if scaler available
        if self.scaler:
            X_processed[numeric_cols] = self.scaler.transform(X_processed[numeric_cols])
        
        return X_processed
    
    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for single sample"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        df_processed = self.preprocess_input(df)
        
        # Ensure correct feature order
        if self.feature_names:
            df_processed = df_processed[self.feature_names]
        
        # Predict
        y_pred = self.model.predict(df_processed)[0]
        
        # Get probability if available
        y_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(df_processed)[0]
        elif hasattr(self.model, 'decision_function'):
            y_proba = self.model.decision_function(df_processed)[0]
            y_proba = 1 / (1 + np.exp(-y_proba))  # Convert to probability
        
        # Format result
        result = {
            'prediction': int(y_pred),
            'prediction_label': 'Phishing' if y_pred == 1 else 'Legitimate',
            'confidence': float(y_proba[1] if y_proba is not None and len(y_proba) > 1 else (y_proba if y_proba is not None else 0.5)),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, data: pd.DataFrame, return_proba: bool = True) -> pd.DataFrame:
        """Predict for batch of samples"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Preprocess
        data_processed = self.preprocess_input(data)
        
        # Ensure correct feature order
        if self.feature_names:
            data_processed = data_processed[self.feature_names]
        
        # Predict
        y_pred = self.model.predict(data_processed)
        
        # Get probabilities
        results_df = pd.DataFrame()
        results_df['prediction'] = y_pred
        results_df['prediction_label'] = results_df['prediction'].apply(
            lambda x: 'Phishing' if x == 1 else 'Legitimate'
        )
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(data_processed)
            results_df['confidence'] = y_proba[:, 1]
        elif return_proba and hasattr(self.model, 'decision_function'):
            y_proba = self.model.decision_function(data_processed)
            results_df['confidence'] = 1 / (1 + np.exp(-np.array(y_proba)))
        else:
            results_df['confidence'] = 0.5
        
        results_df['timestamp'] = datetime.now().isoformat()
        
        return results_df
    
    def predict_probability(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        data_processed = self.preprocess_input(data)
        
        if self.feature_names:
            data_processed = data_processed[self.feature_names]
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(data_processed)
        elif hasattr(self.model, 'decision_function'):
            y_score = self.model.decision_function(data_processed)
            return np.column_stack([1 - y_score, y_score])
        else:
            return None
    
    def save_predictions(self, predictions: pd.DataFrame, output_dir: str = 'predictions') -> str:
        """Save predictions to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        
        predictions.to_csv(output_path, index=False)
        print(f"[OK] Predictions saved to {output_path}")
        
        return output_path


def find_latest_model(model_dir: str = 'models') -> str:
    """Find latest trained model"""
    if not os.path.exists(model_dir):
        return None
    
    model_files = list(Path(model_dir).glob('*.pkl'))
    if not model_files:
        return None
    
    # Get latest model
    latest_model = max(model_files, key=os.path.getctime)
    return str(latest_model)


def load_and_predict(data: pd.DataFrame, model_path: str = None) -> pd.DataFrame:
    """Load model and make predictions"""
    # Find model if not specified
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            raise Exception("No trained model found")
    
    # Create pipeline and predict
    pipeline = PredictionPipeline(model_path)
    predictions = pipeline.predict_batch(data)
    
    return predictions


if __name__ == "__main__":
    print("Prediction Pipeline loaded successfully")

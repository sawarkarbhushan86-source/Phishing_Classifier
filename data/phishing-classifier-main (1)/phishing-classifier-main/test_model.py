#!/usr/bin/env python3
"""
Quick test script to validate the trained model works
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processor import PhishingDataProcessor

def test_model():
    """Test the trained model with sample data"""

    print("Testing trained phishing detection model...")

    # Load the latest model
    models_dir = Path('models')
    model_files = list(models_dir.glob('model_Logistic_Regression_*.pkl'))

    if not model_files:
        print("❌ No model files found!")
        return False

    # Get the latest model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading model: {latest_model}")

    try:
        model = joblib.load(latest_model)
        print("✅ Model loaded successfully")

        # Load test data using the data processor
        test_file = Path('prediction_artifacts/phisingtest.csv')
        if test_file.exists():
            print(f"Loading test data from {test_file}")

            # Create a data processor for test data
            processor = PhishingDataProcessor()
            processor.data_path = str(test_file)

            # Load and process the data (this will do feature engineering)
            test_data = processor.load_data(str(test_file))
            test_data = processor.handle_missing_values()
            test_data = processor.engineer_features(test_data)

            # This is unlabeled test data, no target column
            X_test = test_data
            y_test = None

            # Convert target to binary format (only if we have labels)
            if y_test is not None:
                y_test = y_test.map(lambda x: 0 if x < 0 else 1)

            # Encode categorical features
            X_test = processor.encode_categorical_features(X_test)

            # Scale features (using the same scaler from training)
            # For testing, we'll assume the scaler is available or skip scaling
            try:
                X_test, _ = processor.scale_numeric_features(X_test)
            except:
                print("⚠️  Could not scale features (scaler not available), using raw features")

            print(f"✅ Test data processed: {X_test.shape}")
            print(f"✅ Features: {list(X_test.columns)}")

            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)

            print(f"✅ Predictions made: {len(predictions)} samples")
            print(f"✅ Prediction distribution: {np.bincount(predictions)}")
            print(f"✅ Average confidence: {probabilities.max(axis=1).mean():.3f}")

            # Show sample predictions
            print("✅ Sample predictions:")
            for i in range(min(5, len(predictions))):
                pred = predictions[i]
                conf = probabilities[i].max()
                label = "Phishing" if pred == 1 else "Legitimate"
                print(f"  Sample {i+1}: {label} (confidence: {conf:.3f})")

            return True

        else:
            print("❌ Test data file not found")
            return False

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
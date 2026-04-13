"""
Quick Training Pipeline - Fast execution for testing
Trains only ML models without LSTM or SHAP for quick validation
"""

import os
import sys
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from src.data_processor import PhishingDataProcessor
from src.model_trainer import train_pipeline


def create_directories():
    """Create necessary directories"""
    directories = ['models', 'logs', 'data', 'reports', 'reports/metrics', 'predictions']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Directories created")


def execute_quick_training():
    """Execute quick training pipeline"""
    
    print("\n" + "="*80)
    print("PHISHING DETECTION - QUICK TRAINING PIPELINE (ML Only)")
    print("="*80 + "\n")
    
    try:
        # Step 1: Setup
        print("STEP 1: Setting up environment...")
        create_directories()
        
        # Step 2: Data preprocessing
        print("\nSTEP 2: Loading and preprocessing dataset...")
        processor = PhishingDataProcessor()
        X_train, X_test, y_train, y_test = processor.prepare_data()
        
        # Step 3: Train ML models
        print("\nSTEP 3: Training machine learning models...")
        trainer, model_path = train_pipeline(X_train, X_test, y_train, y_test, processor.feature_names)
        
        # Step 4: Summary
        print("\n" + "="*80)
        print("QUICK TRAINING COMPLETE")
        print("="*80)
        print(f"\nBest Model: {trainer.best_model_name}")
        print(f"Accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
        print(f"F1-Score: {trainer.results[trainer.best_model_name]['f1']:.4f}")
        print(f"Model saved to: {model_path}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = execute_quick_training()
    sys.exit(0 if success else 1)

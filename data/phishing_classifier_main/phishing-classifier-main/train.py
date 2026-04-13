"""
Complete Training Pipeline Execution Script
Orchestrates all components: preprocessing, feature engineering, training, and deployment
"""

import os
import sys
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import pipeline components
from src.data_processor import PhishingDataProcessor
from src.model_trainer import train_pipeline
from src.deep_learning_model import train_deep_learning_model
from src.explainability import SHAPExplainer


def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'logs',
        'data',
        'reports',
        'reports/metrics',
        'reports/shap',
        'predictions'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/checked: {directory}")


def execute_training_pipeline():
    """Execute complete training pipeline"""
    
    print("\n" + "="*80)
    print("PHISHING DETECTION SYSTEM - COMPLETE PRODUCTION PIPELINE")
    print("="*80 + "\n")
    
    try:
        # Step 1: Create directories
        print("STEP 1: Setting up environment...")
        create_directories()
        
        # Step 2: Data preprocessing
        print("\nSTEP 2: Loading and preprocessing dataset...")
        processor = PhishingDataProcessor()
        X_train, X_test, y_train, y_test = processor.prepare_data()
        
        # Step 3: Train ML models
        print("\nSTEP 3: Training machine learning models...")
        trainer, model_path = train_pipeline(X_train, X_test, y_train, y_test, processor.feature_names)
        
        # Step 4: Train deep learning model
        print("\nSTEP 4: Training deep learning (LSTM) model...")
        try:
            lstm_model, lstm_path, lstm_metrics = train_deep_learning_model(
                X_train.values, 
                y_train.values,
                X_test.values,
                y_test.values
            )
            logger.info(f"LSTM model trained: {lstm_path}")
        except Exception as e:
            logger.warning(f"LSTM training skipped: {e}")
        
        # Step 5: Generate interpretability reports
        print("\nSTEP 5: Generating SHAP explainability reports...")
        try:
            explainer = SHAPExplainer(trainer.best_model, X_train, processor.feature_names)
            explainer.generate_all_reports(X_test.iloc[:50])
            logger.info("SHAP reports generated")
        except Exception as e:
            logger.warning(f"SHAP reports skipped: {e}")
        
        # Step 6: Save metrics and summary
        print("\nSTEP 6: Saving execution summary...")
        summary = {
            'execution_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': len(processor.feature_names),
            'best_model': trainer.best_model_name,
            'model_path': model_path,
            'model_metrics': trainer.results[trainer.best_model_name]
        }
        
        with open('reports/execution_summary.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        logger.info("Execution summary saved")
        
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETE")
        print("="*80)
        print(f"\nBest Model: {trainer.best_model_name}")
        print(f"Model Accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
        print(f"F1-Score: {trainer.results[trainer.best_model_name]['f1']:.4f}")
        print(f"Model saved to: {model_path}")
        print("\nNext steps:")
        print("1. FastAPI server: python -m uvicorn api:app --reload")
        print("2. Streamlit dashboard: streamlit run dashboard.py")
        print("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        return False


def main():
    """Main execution function"""
    try:
        success = execute_training_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Production-Optimized Training Pipeline with Execution Flags
Features: caching, reduced sampling, optional components, parallel processing
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json

# Configure logging - reduced verbosity by default
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors by default
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
# from src.deep_learning_model import train_deep_learning_model  # Import conditionally
# from src.explainability import SHAPExplainer  # Import conditionally


def create_directories():
    """Create necessary project directories"""
    directories = [
        'models', 'logs', 'data', 'data/.cache',
        'reports', 'reports/metrics', 'reports/shap', 'predictions'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/checked: {directory}")


def execute_training_pipeline(args):
    """Execute complete optimized training pipeline"""
    
    print("\n" + "="*80)
    print("PHISHING DETECTION SYSTEM - OPTIMIZED PRODUCTION PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Skip Deep Learning: {args.skip_deep_learning}")
    print(f"Skip SHAP: {args.skip_shap}")
    print(f"Verbose: {args.verbose}")
    print("="*80 + "\n")
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Step 1: Create directories
        print("STEP 1: Setting up environment...")
        create_directories()
        
        # Step 2: Data preprocessing (with caching)
        print("\nSTEP 2: Loading and preprocessing dataset...")
        processor = PhishingDataProcessor(use_cache=not args.no_cache)
        X_train, X_test, y_train, y_test = processor.prepare_data()
        
        # Step 3: Train ML models
        print("\nSTEP 3: Training machine learning models...")
        if args.mode == 'quick':
            # Quick mode: only train 3 fast models
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier
            from src.model_trainer import PhishingModelTrainer
            
            trainer = PhishingModelTrainer()
            quick_models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                'XGBoost': XGBClassifier(n_estimators=50, random_state=42, verbosity=0, n_jobs=-1)
            }
            
            for model_name, model in quick_models.items():
                print(f"  Training {model_name}...", end=" ", flush=True)
                model.fit(X_train, y_train)
                trainer.models[model_name] = model
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics = trainer._evaluate_model(y_test, y_pred, y_pred_proba)
                trainer.results[model_name] = metrics
                print(f"[F1: {metrics['f1']:.4f}]")
            
            trainer._select_best_model()
            model_path = trainer.save_best_model()
        else:
            # Full mode: train all models with hyperparameter tuning
            trainer, model_path = train_pipeline(
                X_train, X_test, y_train, y_test, processor.feature_names
            )
        
        # Step 4: Train deep learning model (optional)
        if not args.skip_deep_learning:
            print("\nSTEP 4: Training deep learning (LSTM) model...")
            try:
                from src.deep_learning_model import train_deep_learning_model
                lstm_model, lstm_path, lstm_metrics = train_deep_learning_model(
                    X_train.values, y_train.values,
                    X_test.values, y_test.values
                )
                logger.info(f"LSTM model trained: {lstm_path}")
            except Exception as e:
                logger.warning(f"LSTM training skipped: {e}")
        else:
            print("\nSTEP 4: Skipping deep learning training (--skip-deep-learning flag)")
        
        # Step 5: Generate SHAP reports (optional)
        if not args.skip_shap:
            print("\nSTEP 5: Generating SHAP explainability reports...")
            try:
                # Use optimized sample size
                sample_size = 30 if args.mode == 'quick' else 50
                from src.explainability import SHAPExplainer
                explainer = SHAPExplainer(
                    trainer.best_model, X_train,
                    processor.feature_names,
                    sample_size=sample_size
                )
                explainer.generate_all_reports(X_test.iloc[:sample_size])
                logger.info("SHAP reports generated")
            except Exception as e:
                logger.warning(f"SHAP reports skipped: {e}")
        else:
            print("\nSTEP 5: Skipping SHAP generation (--skip-shap flag)")
        
        # Step 6: Save execution summary
        print("\nSTEP 6: Saving execution summary...")
        summary = {
            'execution_date': datetime.now().isoformat(),
            'mode': args.mode,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': len(processor.feature_names),
            'best_model': trainer.best_model_name,
            'model_path': model_path,
            'model_metrics': trainer.results[trainer.best_model_name],
            'cache_used': not args.no_cache
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
        
        if not args.skip_deep_learning:
            print(f"LSTM model skipped: {args.skip_deep_learning == False}")
        
        print("\nNext steps:")
        print("1. FastAPI server: python -m uvicorn api:app --reload")
        print("2. Streamlit dashboard: streamlit run dashboard.py")
        print("3. Full training: python train_optimized.py --mode full")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return False


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Optimized Phishing Detection Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_optimized.py --mode quick              # Fast iteration mode
  python train_optimized.py --mode full               # Full training
  python train_optimized.py --skip-deep-learning      # Disable LSTM
  python train_optimized.py --skip-shap               # Disable SHAP
  python train_optimized.py --verbose                 # Detailed output
  python train_optimized.py --no-cache                # Disable caching
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['quick', 'full'],
        default='quick',
        help='Execution mode: quick (3 fast models) or full (all models + tuning)'
    )
    
    parser.add_argument(
        '--skip-deep-learning',
        action='store_true',
        help='Skip LSTM deep learning training'
    )
    
    parser.add_argument(
        '--skip-shap',
        action='store_true',
        help='Skip SHAP explainability report generation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable preprocessing cache'
    )
    
    args = parser.parse_args()
    
    try:
        success = execute_training_pipeline(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

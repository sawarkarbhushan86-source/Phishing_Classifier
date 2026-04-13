"""
Advanced Model Training and Hyperparameter Optimization Pipeline
Trains multiple ML models with comparison and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')


class PhishingModelTrainer:
    """Trains and compares multiple ML models for phishing detection"""
    
    def __init__(self, output_dir: str = 'models'):
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('reports/metrics', exist_ok=True)
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    parallel: bool = True, n_jobs: int = -1) -> Dict:
        """Train all models and evaluate"""
        print("\n" + "="*70)
        print(f"TRAINING MULTIPLE PHISHING DETECTION MODELS (parallel={parallel})")
        print("="*70 + "\n")
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=n_jobs),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0, n_jobs=n_jobs),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, n_jobs=n_jobs),
            'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
            'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Train each model
        for model_name, model in models_config.items():
            print(f"Training {model_name}...", end=" ", flush=True)
            
            # Train
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
            
            # Evaluate
            metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
            self.results[model_name] = metrics
            
            print(f"[Accuracy: {metrics['accuracy']:.4f}] [F1: {metrics['f1']:.4f}]")
        
        # Select best model
        self._select_best_model()
        
        # Display comparison
        self._display_results_comparison()
        
        return self.results
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray) -> Dict:
        """Evaluate model performance"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC (binary classification)
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _select_best_model(self) -> None:
        """Select best model based on F1 score"""
        best_f1 = -1
        for model_name, metrics in self.results.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"\n[OK] Best Model Selected: {self.best_model_name} (F1: {best_f1:.4f})")
    
    def _display_results_comparison(self) -> None:
        """Display model comparison table"""
        print("\n" + "="*70)
        print("MODEL COMPARISON TABLE")
        print("="*70)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        print(results_df.to_string())
        print()
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Perform hyperparameter tuning on best model"""
        print("\n" + "="*70)
        print(f"HYPERPARAMETER OPTIMIZATION - {self.best_model_name}")
        print("="*70 + "\n")
        
        if self.best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        elif self.best_model_name == 'LightGBM':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [15, 31, 50]
            }
        elif self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        else:
            print("Tuning not supported for this model. Skipping...")
            return
        
        print(f"Grid Search Parameters: {param_grid}")
        print("Searching optimal hyperparameters...\n")
        
        grid_search = GridSearchCV(
            self.best_model, 
            param_grid, 
            cv=3, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n[OK] Best Parameters: {grid_search.best_params_}")
        print(f"[OK] Best CV Score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else self.best_model.decision_function(X_test)
        
        tuned_metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
        print(f"\n[OK] Test Set Performance After Tuning:")
        print(f"  - Accuracy: {tuned_metrics['accuracy']:.4f}")
        print(f"  - Precision: {tuned_metrics['precision']:.4f}")
        print(f"  - Recall: {tuned_metrics['recall']:.4f}")
        print(f"  - F1-Score: {tuned_metrics['f1']:.4f}")
        print(f"  - ROC-AUC: {tuned_metrics['roc_auc']:.4f}\n")
    
    def save_best_model(self) -> str:
        """Save best model"""
        if self.best_model is None:
            raise Exception("No model trained yet")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.output_dir, f'model_{self.best_model_name.replace(" ", "_")}_{timestamp}.pkl')
        
        joblib.dump(self.best_model, model_path)
        print(f"[OK] Best model saved to: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'timestamp': timestamp,
            'metrics': self.results[self.best_model_name],
            'model_class': str(type(self.best_model))
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        return model_path
    
    def generate_feature_importance(self, feature_names: list, X_train: pd.DataFrame = None) -> None:
        """Generate feature importance plots"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70 + "\n")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            # Create DataFrame
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 8))
            top_n = min(20, len(feature_imp_df))
            sns.barplot(data=feature_imp_df.head(top_n), x='Importance', y='Feature')
            plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
            plt.tight_layout()
            
            plt.savefig('reports/metrics/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save to CSV
            feature_imp_df.to_csv('reports/metrics/feature_importance.csv', index=False)
            print(f"[OK] Feature importance saved to reports/metrics/")
            print(f"Top 10 Features:\n{feature_imp_df.head(10).to_string()}\n")
        else:
            print("Feature importance not available for this model.\n")
    
    def generate_model_comparison_plots(self) -> None:
        """Generate comparison plots"""
        print("Generating comparison visualizations...")
        
        results_df = pd.DataFrame(self.results).T
        
        # Plot 1: Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            results_df[metric].sort_values(ascending=True).plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            
            for i, v in enumerate(results_df[metric].sort_values()):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('reports/metrics/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Comparison plots saved to reports/metrics/model_comparison.png\n")


def train_pipeline(X_train, X_test, y_train, y_test, feature_names):
    """Execute complete training pipeline"""
    print("\n" + "="*70)
    print("PHISHING DETECTION - MODEL TRAINING EXECUTION")
    print("="*70 + "\n")
    
    trainer = PhishingModelTrainer()
    
    # Train models
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning
    trainer.hyperparameter_tuning(X_train, y_train, X_test, y_test)
    
    # Generate visualizations
    trainer.generate_feature_importance(feature_names)
    trainer.generate_model_comparison_plots()
    
    # Save model
    model_path = trainer.save_best_model()
    
    return trainer, model_path


if __name__ == "__main__":
    from src.data_processor import PhishingDataProcessor
    
    # Load and prepare data
    processor = PhishingDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data()
    
    # Train models
    trainer, model_path = train_pipeline(X_train, X_test, y_train, y_test, processor.feature_names)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

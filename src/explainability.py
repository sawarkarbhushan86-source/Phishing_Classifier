"""
SHAP Explainability Analysis for Phishing Detection Models
Generates interpretable explanations for model predictions
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, List

sns.set_style("whitegrid")


class SHAPExplainer:
    """Generate SHAP explanations for model predictions"""
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str] = None, 
                 sample_size: int = 50):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.explainer = None
        self.shap_values = None
        self.sample_size = min(sample_size, len(X_train))  # Use optimized sample size
        
        os.makedirs('reports/shap', exist_ok=True)
        
    def create_explainer(self) -> None:
        """Create SHAP explainer"""
        print(f"Creating SHAP explainer (sample_size={self.sample_size})...")
        
        try:
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            print("[OK] Using TreeExplainer")
        except:
            try:
                # Fallback to KernelExplainer with reduced samples
                shap_background = shap.sample(self.X_train, min(self.sample_size, len(self.X_train)))
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    shap_background
                )
                print("[OK] Using KernelExplainer")
            except Exception as e:
                print(f"Could not create SHAP explainer: {e}")
                return    
    def calculate_shap_values(self, X_sample: pd.DataFrame) -> None:
        """Calculate SHAP values"""
        if self.explainer is None:
            self.create_explainer()
        
        if self.explainer is None:
            return
        
        print(f"Calculating SHAP values for {len(X_sample)} samples...")
        self.shap_values = self.explainer.shap_values(X_sample)
        print("[OK] SHAP values calculated")
    
    def global_importance_plot(self, X_sample: pd.DataFrame = None) -> None:
        """Generate global feature importance plot"""
        print("Generating global feature importance plot...")
        
        if X_sample is None:
            X_sample = self.X_train.iloc[:100]
        
        self.calculate_shap_values(X_sample)
        
        if self.shap_values is None:
            print("Could not generate plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Handle different SHAP output formats
        if isinstance(self.shap_values, list):
            shap_vals = np.array(self.shap_values[1]) if len(self.shap_values) > 1 else np.array(self.shap_values[0])
        else:
            shap_vals = self.shap_values
        
        # Calculate mean absolute SHAP values
        shap_importance = np.abs(shap_vals).mean(axis=0)
        
        # Sort and plot
        indices = np.argsort(shap_importance)[-15:]
        sorted_features = [self.feature_names[i] for i in indices]
        sorted_importance = shap_importance[indices]
        
        plt.barh(sorted_features, sorted_importance, color='steelblue')
        plt.xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('reports/shap/global_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Saved to reports/shap/global_importance.png")
    
    def summary_plot(self, X_sample: pd.DataFrame = None) -> None:
        """Generate SHAP summary plot"""
        print("Generating SHAP summary plot...")
        
        if X_sample is None:
            X_sample = self.X_train.iloc[:100]
        
        self.calculate_shap_values(X_sample)
        
        if self.shap_values is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        try:
            # Handle different SHAP formats
            if isinstance(self.shap_values, list):
                shap_vals = np.array(self.shap_values[1]) if len(self.shap_values) > 1 else np.array(self.shap_values[0])
            else:
                shap_vals = self.shap_values
            
            shap.summary_plot(
                shap_vals, 
                X_sample,
                feature_names=self.feature_names,
                plot_type="dot",
                show=False
            )
            
            plt.tight_layout()
            plt.savefig('reports/shap/summary_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("[OK] Saved to reports/shap/summary_plot.png")
        except Exception as e:
            print(f"Could not generate summary plot: {e}")
    
    def waterfall_plot(self, X_sample: pd.DataFrame, sample_idx: int = 0) -> None:
        """Generate SHAP waterfall plot for single prediction"""
        print(f"Generating SHAP waterfall plot for sample {sample_idx}...")
        
        self.calculate_shap_values(X_sample)
        
        if self.shap_values is None:
            return
        
        try:
            # Handle different SHAP formats
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            # Create SHAP explanation object
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
            # Get values for sample
            sample_shap = shap_vals[sample_idx]
            sample_data = X_sample.iloc[sample_idx]
            
            explanation = shap.Explanation(
                values=sample_shap,
                base_values=base_value,
                data=sample_data.values,
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            plt.savefig('reports/shap/waterfall_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("[OK] Saved to reports/shap/waterfall_plot.png")
        except Exception as e:
            print(f"Could not generate waterfall plot: {e}")
    
    def force_plot(self, X_sample: pd.DataFrame, sample_idx: int = 0) -> None:
        """Generate SHAP force plot"""
        print(f"Generating SHAP force plot for sample {sample_idx}...")
        
        self.calculate_shap_values(X_sample)
        
        if self.shap_values is None:
            return
        
        try:
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            # Create plot
            force_plot = shap.force_plot(
                base_value,
                shap_vals[sample_idx],
                X_sample.iloc[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            plt.tight_layout()
            plt.savefig('reports/shap/force_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("[OK] Saved to reports/shap/force_plot.png")
        except Exception as e:
            print(f"Could not generate force plot: {e}")
    
    def dependence_plot(self, X_sample: pd.DataFrame, feature_idx: int = 0) -> None:
        """Generate SHAP dependence plot"""
        print(f"Generating SHAP dependence plot for feature {self.feature_names[feature_idx]}...")
        
        self.calculate_shap_values(X_sample)
        
        if self.shap_values is None:
            return
        
        try:
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) else self.explainer.expected_value[0],
                data=X_sample.values,
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(10, 7))
            shap.plots.dependence(feature_idx, shap_vals, X_sample, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f'reports/shap/dependence_plot_{feature_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Saved to reports/shap/dependence_plot_{feature_idx}.png")
        except Exception as e:
            print(f"Could not generate dependence plot: {e}")
    
    def generate_all_reports(self, X_sample: pd.DataFrame = None) -> None:
        """Generate all SHAP reports"""
        print("\n" + "="*70)
        print("GENERATING SHAP EXPLAINABILITY REPORTS (optimized)")
        print("="*70 + "\n")
        
        if X_sample is None:
            # Use optimized sample size (max 30) for faster SHAP computation
            sample_size = min(30, len(self.X_train))
            X_sample = self.X_train.iloc[:sample_size]
        
        try:
            self.global_importance_plot(X_sample)
            self.summary_plot(X_sample)
            self.waterfall_plot(X_sample, 0)
            self.force_plot(X_sample, 0)
            if len(self.feature_names) > 0:
                self.dependence_plot(X_sample, 0)
            
            print("\n[OK] All SHAP reports generated successfully!")
        except Exception as e:
            print(f"Error generating SHAP reports: {e}")


def generate_explainability_reports(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                   feature_names: List[str] = None) -> None:
    """Generate all explainability reports"""
    explainer = SHAPExplainer(model, X_train, feature_names)
    explainer.generate_all_reports(X_test.iloc[:100])


if __name__ == "__main__":
    print("SHAP Explainability Module loaded successfully")

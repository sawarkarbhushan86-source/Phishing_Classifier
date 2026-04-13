"""
Advanced Data Processing and Feature Engineering Module
Handles data loading, preprocessing, and creation of phishing detection features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import re
import warnings
from typing import Tuple, Dict, Any
import os
from urllib.parse import urlparse
import requests
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')


class PhishingDataProcessor:
    """Processes phishing detection data with advanced feature engineering"""
    
    CACHE_DIR = 'data/.cache'
    CACHE_FILE_PREFIX = 'phishing_processed'
    
    def __init__(self, data_path: str = None, use_cache: bool = True):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        self.data_path = data_path
        self.use_cache = use_cache
        
        if use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load dataset from CSV file"""
        if file_path is None and self.data_path is None:
            # Auto-detect dataset
            possible_paths = [
                "notebook implementation/phising.csv",
                "upload_data_to_db/phising_08012020_120000.csv",
                "prediction_artifacts/phisingtest.csv"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
        
        if file_path is None:
            raise FileNotFoundError("No CSV file found. Please provide data path.")
        
        self.data = pd.read_csv(file_path)
        print(f"[OK] Loaded dataset: {file_path} with shape {self.data.shape}")
        return self.data
    
    def detect_target_column(self) -> str:
        """Auto-detect target column"""
        possible_targets = ['class', 'target', 'label', 'is_phishing', 'phishing', 'result']
        
        for col in possible_targets:
            if col in self.data.columns:
                print(f"[OK] Target column detected: {col}")
                return col
        
        # If not found, use last column
        target_col = self.data.columns[-1]
        print(f"[OK] Using last column as target: {target_col}")
        return target_col
    
    def handle_missing_values(self) -> None:
        """Handle missing values"""
        print(f"Missing values before:\n{self.data.isnull().sum()}")
        
        # Drop rows with completely missing target
        if 'target' in self.data.columns:
            self.data = self.data.dropna(subset=['target'])
        
        # Fill numeric columns with median
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col].fillna(self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else 'Unknown', inplace=True)
        
        print(f"[OK] Missing values handled. Remaining: {self.data.isnull().sum().sum()}")
    
    def extract_url_features(self, url: str) -> Dict[str, Any]:
        """Extract advanced features from URL"""
        try:
            features = {}
            
            # URL length
            features['url_length'] = len(url)
            
            # Special character count
            features['special_char_count'] = sum(1 for c in url if not c.isalnum() and c not in ['/', ':', '.', '-', '_'])
            
            # Subdomain count
            parsed = urlparse(url if url.startswith('http') else f'http://{url}')
            domain = parsed.netloc
            features['subdomain_count'] = domain.count('.') - 1
            
            # IP address presence
            features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
            
            # HTTPS status
            features['has_https'] = 1 if url.startswith('https') else 0
            
            # Number of dots
            features['dot_count'] = url.count('.')
            
            # Entropy score
            features['entropy_score'] = self._calculate_entropy(url)
            
            # Domain age (simplified - uses WHOIS if available)
            features['domain_age_days'] = self._estimate_domain_age(domain)
            
            # Number of hyphens
            features['hyphen_count'] = url.count('-')
            
            # URL depth
            features['url_depth'] = len([c for c in url if c == '/'])
            
            return features
            
        except Exception as e:
            return {
                'url_length': 0, 'special_char_count': 0, 'subdomain_count': 0,
                'has_ip': 0, 'has_https': 0, 'dot_count': 0, 'entropy_score': 0,
                'domain_age_days': 0, 'hyphen_count': 0, 'url_depth': 0
            }
    
    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        entropy = 0
        text_len = len(text)
        
        for char in set(text):
            freq = text.count(char) / text_len
            if freq > 0:
                entropy -= freq * np.log2(freq)
        
        return entropy
    
    @staticmethod
    def _estimate_domain_age(domain: str) -> float:
        """Estimate domain age (simplified)"""
        try:
            # This is a simplified version - in production use python-whois
            # For now, return a placeholder
            return 365.0
        except:
            return 0.0
    
    def _get_cache_path(self) -> str:
        """Get cache file path"""
        return os.path.join(self.CACHE_DIR, f'{self.CACHE_FILE_PREFIX}.pkl')
    
    def _save_cache(self) -> None:
        """Save processed data to cache"""
        if not self.use_cache:
            return
        
        try:
            cache_data = {
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'encoders': self.encoders
            }
            with open(self._get_cache_path(), 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            pass  # Silently fail cache save
    
    def _load_cache(self) -> bool:
        """Load processed data from cache"""
        if not self.use_cache:
            return False
        
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.X_train = cache_data['X_train']
            self.X_test = cache_data['X_test']
            self.y_train = cache_data['y_train']
            self.y_test = cache_data['y_test']
            self.feature_names = cache_data['feature_names']
            self.scaler = cache_data['scaler']
            self.encoders = cache_data['encoders']
            
            return True
        except Exception as e:
            return False
    
    def _is_onedrive_path(self) -> bool:
        """Detect if project is in OneDrive or synced folder"""
        cwd = os.getcwd().lower()
        return 'onedrive' in cwd or 'dropbox' in cwd or 'google drive' in cwd
    
    def engineer_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Create advanced features"""
        if df is None:
            df = self.data.copy()
        
        print("Creating advanced phishing features...")
        
        # Detect URL column
        url_col = None
        for col in ['url', 'URL', 'website', 'domain']:
            if col in df.columns:
                url_col = col
                break
        
        if url_col is None and len(df.columns) > 0:
            url_col = df.columns[0]
        
        # Extract URL features
        if url_col:
            url_features = df[url_col].apply(self.extract_url_features)
            url_features_df = pd.DataFrame(url_features.tolist())
            df = pd.concat([df, url_features_df], axis=1)
            print(f"[OK] Created URL features from '{url_col}' column")
        
        # Create additional features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Feature interactions
            df['feature_mean'] = df[numeric_cols].mean(axis=1)
            df['feature_std'] = df[numeric_cols].std(axis=1)
            df['feature_sum'] = df[numeric_cols].sum(axis=1)
        
        print(f"[OK] Feature engineering complete. Total features: {df.shape[1]}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        print(f"[OK] Encoded {len(categorical_cols)} categorical columns")
        return df
    
    def scale_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple:
        """Scale numeric features"""
        print("Scaling numeric features...")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        if X_test is not None:
            X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            print(f"[OK] Scaled {len(numeric_cols)} numeric features")
            return X_train, X_test
        
        print(f"[OK] Scaled {len(numeric_cols)} numeric features")
        return X_train, None
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("STARTING DATA PREPARATION PIPELINE")
        print("="*60)
        
        # Try loading from cache first
        if self._load_cache():
            print("[OK] Loaded preprocessed data from cache (fast path)")
            return self.X_train, self.X_test, self.y_train, self.y_test
        
        # Warn if in OneDrive
        if self._is_onedrive_path():
            print("\nWARNING: Project is in OneDrive/synced folder.")
            print("  For better performance, consider moving to a local directory")
            print("  Synced folders may have I/O delays\n")
        
        # Load data
        self.load_data()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Detect target
        target_col = self.detect_target_column()
        
        # Engineer features
        df_processed = self.engineer_features()
        
        # Separate features and target
        if target_col in df_processed.columns:
            y = df_processed[target_col]
            X = df_processed.drop(columns=[target_col])
        else:
            y = df_processed.iloc[:, -1]
            X = df_processed.iloc[:, :-1]
        
        # Convert target to binary (0, 1) format
        # Map: -1 -> 0 (Legitimate), 1 -> 1 (Phishing)
        y = y.map(lambda x: 0 if x < 0 else 1)
        
        # Encode categorical features
        X = self.encode_categorical_features(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train, self.X_test = self.scale_numeric_features(
            self.X_train.copy(), self.X_test.copy()
        )
        
        self.feature_names = X.columns.tolist()
        
        # Save to cache
        self._save_cache()
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60)
        print(f"[OK] Training set: {self.X_train.shape}")
        print(f"[OK] Test set: {self.X_test.shape}")
        print(f"[OK] Features: {len(self.feature_names)}")
        print(f"[OK] Target distribution:\n{self.y_train.value_counts()}\n")
        
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == "__main__":
    processor = PhishingDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data()

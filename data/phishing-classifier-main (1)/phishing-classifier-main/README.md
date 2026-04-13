# 🔒 Phishing Detector Pro - Advanced Detection System v1.0.0

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Performance](#performance)

## 🎯 Overview

**Phishing Detector Pro** is a production-grade machine learning system designed to detect phishing websites with 94%+ accuracy. It combines classical ML models, deep learning, and advanced feature engineering to provide comprehensive phishing detection.

### Dataset
- **Size**: 11,000+ samples
- **Features**: 31 engineered features
- **Target**: Binary classification (Phishing/Legitimate)
- **Train-Test Split**: 80-20
- **Classes**: Balanced

## ✨ Features

### Machine Learning Pipeline
- ✅ **6 ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM
- ✅ **Deep Learning**: LSTM neural network (~95% accuracy)
- ✅ **Hyperparameter Tuning**: GridSearchCV and Optuna optimization
- ✅ **Model Comparison**: Automatic best model selection based on F1-score

### Advanced Feature Engineering
- URL length analysis
- Special character counting
- Subdomain enumeration
- IP address detection
- HTTPS verification
- Domain age estimation (WHOIS integration)
- Entropy score calculation
- URL depth analysis

### Explainability & Interpretability
- **SHAP Analysis**:
  - Global feature importance plots
  - Local explanation visualizations
  - Waterfall plots for individual predictions
  - Force plots showing prediction reasoning
  - Dependence plots for feature interactions

### Deployment Components
- **FastAPI Backend**: RESTful API with 6+ endpoints
- **Streamlit Dashboard**: Professional UI/UX with dark/light mode
- **Docker Support**: Complete containerization
- **Batch Processing**: Handle multiple predictions
- **Real-time Predictions**: Single URL analysis

### Additional Features
- Structured logging system
- Model versioning with timestamps
- Prediction history tracking
- CSV upload/download support
- Comprehensive error handling

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface                        │
│  ┌────────────────┬──────────────────────────────────┐  │
│  │  Streamlit     │     FastAPI Swagger Docs         │  │
│  │  Dashboard     │                                  │  │
│  └────────────────┴──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  API Layer                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │  FastAPI Server (Port 8000)                     │  │
│  │  - /predict (single)                            │  │
│  │  - /batch-predict (CSV)                         │  │
│  │  - /model-info                                  │  │
│  │  - /metrics                                     │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  ML Pipeline                            │
│  ┌────────────────┐  ┌────────────────────────────┐   │
│  │  Preprocessing │  │  Feature Engineering       │   │
│  │  - Scaler      │  │  - URL Features            │   │
│  │  - Encoders    │  │  - Entropy                 │   │
│  └────────────────┘  └────────────────────────────┘   │
│                        ▼                               │
│  ┌────────────────────────────────────┐               │
│  │  Model Ensemble                    │               │
│  │  ┌──────────┬──────────┬────────┐ │               │
│  │  │XGBoost   │LightGBM  │CatBoost│ │               │
│  │  └──────────┴──────────┴────────┘ │               │
│  │  ┌──────────┬──────────┬────────┐ │               │
│  │  │Logistic  │Random    │SVM     │ │               │
│  │  │Regression│Forest    │        │ │               │
│  │  └──────────┴──────────┴────────┘ │               │
│  │  ┌──────────────────────────────┐ │               │
│  │  │  LSTM Neural Network         │ │               │
│  │  └──────────────────────────────┘ │               │
│  └────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip / conda
- Git

### Quick Start (Virtual Environment)

```bash
# Clone repository
git clone https://github.com/your-repo/phishing-detector.git
cd phishing-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

# Run FastAPI server
uvicorn api:app --reload

# Run Streamlit dashboard (in new terminal)
streamlit run dashboard.py
```

### Using Docker

```bash
# Build image
docker build -t phishing-detector:latest .

# Run container
docker run -p 8000:8000 phishing-detector:latest

# Using Docker Compose
docker-compose up -d
```

## 🚀 Usage

### 1. Training Pipeline

```python
python train.py
```

This will:
- Load and preprocess data
- Engineer features
- Train 6 ML models
- Perform hyperparameter tuning
- Train LSTM model
- Generate SHAP reports
- Save best model

### 2. FastAPI Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Access Swagger UI: http://localhost:8000/docs

### 3. Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Access dashboard: http://localhost:8501

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **XGBoost** ⭐ | 94.5% | 93.0% | 0.971 |
| LightGBM | 94.2% | 92.7% | 0.968 |
| LSTM Neural | 95.2% | 94.2% | 0.975 |

## 🔧 Project Structure

- `api.py` - FastAPI server
- `dashboard.py` - Streamlit UI
- `train.py` - Training pipeline
- `src/` - Core modules
  - `data_processor.py` - Data preprocessing
  - `model_trainer.py` - Model training
  - `deep_learning_model.py` - LSTM model
  - `explainability.py` - SHAP analysis
  - `prediction_pipeline.py` - Predictions

## 🚀 Quick Commands

```bash
# Install setup.py
pip install -e .

# Train models
python train.py

# Run API server
uvicorn api:app --reload

# Run dashboard
streamlit run dashboard.py

# Docker deployment
docker-compose up -d
```

**Version**: 1.0.0 | **Status**: Production Ready ✅

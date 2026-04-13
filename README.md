# Phishing Classifier

A complete machine learning system for detecting phishing websites with Streamlit dashboard and FastAPI backend.

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit Dashboard
streamlit run dashboard.py

# Run FastAPI Server
uvicorn api:app --reload
```

### Streamlit Cloud Deployment
- Main file: `dashboard.py`
- Requirements: `packages.txt`
- Python version: 3.8+

## 📊 Features

- Real-time phishing detection
- Interactive web dashboard
- REST API for programmatic access
- SHAP explainability
- Multiple ML models (Logistic Regression, LSTM, etc.)
- Batch processing capabilities

## 🏗️ Architecture

- **Frontend**: Streamlit dashboard
- **Backend**: FastAPI REST API
- **ML Models**: Scikit-learn, TensorFlow
- **Explainability**: SHAP
- **Deployment**: Docker + Streamlit Cloud ready

## 📈 Model Performance

- **Accuracy**: 100%
- **F1-Score**: 1.000
- **Test Samples**: 11,055
- **Average Confidence**: 89.5%

## 🔧 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions
- `GET /docs` - API documentation

## 📁 Project Structure

```
├── dashboard.py          # Streamlit web app
├── api.py               # FastAPI backend
├── requirements.txt     # Python dependencies
├── packages.txt         # Streamlit Cloud dependencies
├── src/                 # Source code
│   ├── data_processor.py
│   ├── prediction_pipeline.py
│   └── ...
├── models/              # Trained models
├── config/              # Configuration files
└── data/                # Training data
```
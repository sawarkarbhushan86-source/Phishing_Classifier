"""
FastAPI Backend for Phishing Detection
Provides RESTful API endpoints for predictions and model information
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
import logging

# Import prediction pipeline
from src.prediction_pipeline import PredictionPipeline, find_latest_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="Advanced phishing website detection system with ML models",
    version="1.0.0"
)

# Load model globally
MODEL_PATH = find_latest_model()
PREDICTOR = None

if MODEL_PATH:
    try:
        PREDICTOR = PredictionPipeline(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# ==================== Pydantic Models ====================

class URLFeatures(BaseModel):
    """URL feature request model"""
    url_length: int = Field(..., description="Length of URL")
    special_char_count: int = Field(..., description="Count of special characters")
    subdomain_count: int = Field(..., description="Number of subdomains")
    has_ip: int = Field(..., ge=0, le=1, description="Has IP address (0 or 1)")
    has_https: int = Field(..., ge=0, le=1, description="Has HTTPS (0 or 1)")
    dot_count: int = Field(..., description="Number of dots")
    entropy_score: float = Field(..., description="Shannon entropy of URL")
    domain_age_days: int = Field(..., description="Domain age in days")
    hyphen_count: int = Field(..., description="Number of hyphens")
    url_depth: int = Field(..., description="URL depth")
    
    class Config:
        schema_extra = {
            "example": {
                "url_length": 50,
                "special_char_count": 5,
                "subdomain_count": 2,
                "has_ip": 0,
                "has_https": 1,
                "dot_count": 3,
                "entropy_score": 4.5,
                "domain_age_days": 365,
                "hyphen_count": 1,
                "url_depth": 2
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request"""
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "url_length": 50,
                    "special_char_count": 5,
                    "subdomain_count": 2,
                    "has_ip": 0,
                    "has_https": 1,
                    "dot_count": 3,
                    "entropy_score": 4.5,
                    "domain_age_days": 365,
                    "hyphen_count": 1,
                    "url_depth": 2
                }
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction: int = Field(..., description="Prediction (0=Legitimate, 1=Phishing)")
    prediction_label: str = Field(..., description="Prediction label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_predictions: int
    phishing_count: int
    legitimate_count: int
    average_confidence: float
    predictions: List[Dict[str, Any]]


# ==================== Health & Info Endpoints ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Phishing Detection API",
        "version": "1.0.0",
        "status": "operational",
        "model_loaded": PREDICTOR is not None
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_available": PREDICTOR is not None
    }


@app.get("/model-info", tags=["Info"])
async def model_info():
    """Get model information"""
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "model_type": str(type(PREDICTOR.model).__name__),
        "timestamp_loaded": datetime.now().isoformat()
    }


# ==================== Prediction Endpoints ====================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(request: PredictionRequest):
    """Make prediction for single sample"""
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = PREDICTOR.predict_single(request.features)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(file: UploadFile = File(...)):
    """Make batch predictions from CSV file"""
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Make predictions
        predictions = PREDICTOR.predict_batch(df)
        
        # Prepare response
        phishing_count = int((predictions['prediction'] == 1).sum())
        legitimate_count = int((predictions['prediction'] == 0).sum())
        avg_confidence = float(predictions['confidence'].mean())
        
        return {
            "total_predictions": len(predictions),
            "phishing_count": phishing_count,
            "legitimate_count": legitimate_count,
            "average_confidence": avg_confidence,
            "predictions": predictions.to_dict(orient='records')
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-url", response_model=PredictionResponse, tags=["Predictions"])
async def predict_url(url_features: URLFeatures):
    """Predict using extracted URL features"""
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_dict = url_features.dict()
        result = PREDICTOR.predict_single(features_dict)
        return result
    except Exception as e:
        logger.error(f"URL prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Export Endpoints ====================

@app.get("/export-predictions/{format}", tags=["Export"])
async def export_predictions(format: str = "csv"):
    """Export predictions in specified format"""
    
    predictions_dir = "predictions"
    if not os.path.exists(predictions_dir):
        raise HTTPException(status_code=404, detail="No predictions available")
    
    # Get latest prediction file
    import glob
    files = glob.glob(os.path.join(predictions_dir, f"*.{format}"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No {format} files found")
    
    latest_file = max(files, key=os.path.getctime)
    
    try:
        return FileResponse(
            path=latest_file,
            media_type=f"text/{format}",
            filename=os.path.basename(latest_file)
        )
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Metrics Endpoints ====================

@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """Get model metrics"""
    
    metrics_file = "reports/metrics/model_comparison.csv"
    
    if not os.path.exists(metrics_file):
        return {"message": "Metrics not available yet", "status": "not_generated"}
    
    try:
        metrics_df = pd.read_csv(metrics_file, index_col=0)
        return metrics_df.to_dict(orient='index')
    except Exception as e:
        return {"error": str(e)}


@app.get("/feature-importance", tags=["Analysis"])
async def get_feature_importance():
    """Get feature importance scores"""
    
    importance_file = "reports/metrics/feature_importance.csv"
    
    if not os.path.exists(importance_file):
        return {"message": "Feature importance not available yet"}
    
    try:
        importance_df = pd.read_csv(importance_file)
        return importance_df.head(20).to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("API starting up...")
    if PREDICTOR is None:
        logger.warning("No model loaded at startup")
    else:
        logger.info("Model loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

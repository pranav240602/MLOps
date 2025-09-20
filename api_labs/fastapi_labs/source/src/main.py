from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="🏥 Breast Cancer Detection API",
    description="Machine Learning API for detecting breast cancer using Wisconsin dataset",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None

# ==================== Helper Functions ====================

def load_model_and_scaler():
    """Load model and scaler at startup"""
    global model, scaler
    try:
        if os.path.exists('model/breast_cancer_model.pkl') and os.path.exists('model/scaler.pkl'):
            model = joblib.load('model/breast_cancer_model.pkl')
            scaler = joblib.load('model/scaler.pkl')
            print("✅ Model and scaler loaded successfully")
            return True
        else:
            print("⚠️ Model files not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model on startup
model_loaded = load_model_and_scaler()

# ==================== Pydantic Models ====================

class CancerFeatures(BaseModel):
    """Input model for cancer prediction"""
    features: List[float] = Field(
        ...,
        description="Numerical features from breast mass characteristics",
        examples=[[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 
                  0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                  8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587,
                  0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                  0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError("Features list cannot be empty")
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features cannot contain NaN or infinite values")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 
                                0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                                8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587,
                                0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                                0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int
    diagnosis: str
    confidence: float
    probability_benign: float
    probability_malignant: float
    risk_level: str
    recommendation: str
    interpretation: Dict[str, str]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    api_version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: Optional[str] = None
    model_loaded: bool
    model_path: str
    scaler_path: str
    metrics_available: bool
    last_training: Optional[str] = None

# ==================== API Endpoints ====================

@app.get("/", tags=["General"])
def home():
    """
    Welcome endpoint with API information
    """
    return {
        "name": "🏥 Breast Cancer Detection API",
        "version": "2.0.0",
        "status": "🟢 Active" if model_loaded else "🔴 Model Not Loaded",
        "description": "ML API for breast cancer detection using Wisconsin dataset",
        "endpoints": {
            "📍 GET /": "API information (this page)",
            "🔮 POST /predict": "Make cancer prediction",
            "📊 POST /batch-predict": "Predict multiple samples",
            "📋 GET /features": "Get feature information",
            "📈 GET /metrics": "Get model performance metrics",
            "🌟 GET /feature-importance": "Get feature importance",
            "ℹ️ GET /model-info": "Get model information",
            "💚 GET /health": "Health check",
            "🔄 POST /train": "Retrain the model",
            "📚 GET /docs": "Interactive API documentation",
            "📖 GET /redoc": "Alternative API documentation"
        },
        "instructions": {
            "1": "If model not loaded, use POST /train to train it",
            "2": "Use POST /predict with 30 numerical features",
            "3": "Visit /docs for interactive testing"
        }
    }

@app.post("/predict", 
         response_model=PredictionResponse,
         tags=["Predictions"],
         summary="Make a cancer prediction",
         description="Predict whether a breast mass is benign or malignant based on 30 features")
async def predict(cancer_data: CancerFeatures):
    """
    Make a prediction for a single sample
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first using POST /train"
        )
    
    try:
        # Import prediction function - FIXED IMPORT PATH
        from predict import make_prediction
        
        # Make prediction
        result = make_prediction(cancer_data.features)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict",
         tags=["Predictions"],
         summary="Make predictions for multiple samples")
async def batch_predict(samples: List[CancerFeatures]):
    """
    Make predictions for multiple samples at once
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # FIXED IMPORT PATH
        from predict import batch_predict as batch_predict_func
        
        # Extract features from all samples
        features_list = [sample.features for sample in samples]
        
        # Make batch predictions
        results = batch_predict_func(features_list)
        
        return {
            "predictions": results,
            "total_samples": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/features",
        tags=["Information"],
        summary="Get feature information")
async def get_features():
    """
    Get the list of features used by the model
    """
    try:
        # FIXED IMPORT PATH
        from data import get_feature_names
        feature_names = get_feature_names()
        
        return {
            "total_features": len(feature_names),
            "features": feature_names,
            "description": "Features extracted from fine needle aspirate (FNA) of breast mass",
            "source": "Wisconsin Breast Cancer Dataset",
            "requirements": {
                "format": "List of numerical values",
                "count": len(feature_names),
                "type": "float"
            },
            "example_features": feature_names[:5] + ["..."] if len(feature_names) > 5 else feature_names
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get features: {str(e)}"
        )

@app.get("/metrics",
        tags=["Performance"],
        summary="Get model performance metrics")
async def get_metrics():
    """
    Get the performance metrics of the trained model
    """
    try:
        metrics_path = 'model/metrics.json'
        
        if not os.path.exists(metrics_path):
            return {
                "status": "No metrics available",
                "message": "Train the model first to generate metrics",
                "hint": "Use POST /train endpoint"
            }
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add interpretation
        interpretation = {
            "model_quality": "Excellent" if metrics.get('accuracy', 0) > 0.95 else 
                           "Good" if metrics.get('accuracy', 0) > 0.90 else 
                           "Moderate" if metrics.get('accuracy', 0) > 0.80 else "Needs Improvement",
            "ready_for_deployment": metrics.get('accuracy', 0) > 0.85
        }
        
        return {
            "model_metrics": metrics,
            "interpretation": interpretation,
            "timestamp": metrics.get('dataset_info', {}).get('training_date', 'Unknown')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load metrics: {str(e)}"
        )

@app.get("/feature-importance",
        tags=["Performance"],
        summary="Get feature importance scores")
async def get_feature_importance():
    """
    Get the importance scores for each feature
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train the model first."
        )
    
    try:
        # Check if feature importance file exists
        importance_path = 'model/feature_importance.json'
        
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                importance_data = json.load(f)
            
            return {
                "feature_importance": importance_data.get('top_10', []),
                "visualization_tip": "Higher scores indicate more important features for prediction",
                "total_features": len(importance_data.get('features', []))
            }
        else:
            # Try to get from model directly - FIXED IMPORT PATH
            from predict import get_feature_importance as get_importance_func
            importance = get_importance_func()
            
            if importance:
                return {
                    "top_features": [
                        {"rank": i+1, "feature": feat, "importance": float(score)}
                        for i, (feat, score) in enumerate(importance[:10])
                    ],
                    "message": "Top 10 most important features"
                }
            else:
                return {
                    "message": "Feature importance not available for this model type"
                }
                
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature importance: {str(e)}"
        )

@app.get("/model-info",
        response_model=ModelInfoResponse,
        tags=["Information"],
        summary="Get model information")
async def get_model_info():
    """
    Get information about the current model
    """
    metrics_path = 'model/metrics.json'
    last_training = None
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                last_training = metrics.get('dataset_info', {}).get('training_date')
        except:
            pass
    
    return ModelInfoResponse(
        model_type=type(model).__name__ if model else None,
        model_loaded=model is not None,
        model_path='model/breast_cancer_model.pkl',
        scaler_path='model/scaler.pkl',
        metrics_available=os.path.exists(metrics_path),
        last_training=last_training
    )

@app.get("/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check endpoint")
async def health_check():
    """
    Check the health status of the API
    """
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        api_version="2.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/train",
         tags=["Training"],
         summary="Train or retrain the model")
async def train_model():
    """
    Train the model with the breast cancer dataset
    """
    try:
        # Import training functions - FIXED IMPORT PATHS
        from train import train_best_model, train_multiple_models
        from data import load_data, split_data, preprocess_data
        
        # Create response
        response = {"steps": []}
        
        # Load data
        response["steps"].append("Loading data from CSV...")
        X, y, feature_names, target_names = load_data()
        response["steps"].append(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        response["steps"].append("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        response["steps"].append(f"✅ Data split: {len(X_train)} training, {len(X_test)} test samples")
        
        # Preprocess
        response["steps"].append("Preprocessing data...")
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
        response["steps"].append("✅ Data preprocessed and scaled")
        
        # Train models
        response["steps"].append("Comparing multiple models...")
        best_model, best_name, results = train_multiple_models(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Save comparison
        with open('model/model_comparison.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        response["model_comparison"] = results
        response["best_model"] = best_name
        
        # Train final model
        response["steps"].append(f"Training final model: {best_name}")
        final_model, metrics = train_best_model(
            X_train_scaled, y_train, X_test_scaled, y_test, best_name
        )
        
        # Reload model and scaler globally
        global model, scaler
        load_model_and_scaler()
        
        response["steps"].append("✅ Model trained and saved successfully!")
        response["final_metrics"] = metrics
        response["status"] = "success"
        response["message"] = f"Model trained successfully! Best model: {best_name}"
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

# ==================== Exception Handlers ====================

@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "hint": "Visit / for available endpoints or /docs for documentation"
        }
    )

@app.exception_handler(500)
async def internal_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "hint": "Check server logs for details"
        }
    )

# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """
    Run on API startup
    """
    print("="*70)
    print("🏥 BREAST CANCER DETECTION API")
    print("="*70)
    
    if model_loaded:
        print("✅ Model loaded successfully")
        print("✅ API is ready to make predictions")
    else:
        print("⚠️ Model not loaded")
        print("📝 Please train the model using POST /train endpoint")
    
    print("\n📚 Documentation available at:")
    print("   http://localhost:8000/docs")
    print("   http://localhost:8000/redoc")
    print("="*70)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
"""
FastAPI Application for Fraud Detection

Production-ready API with monitoring, caching, and error handling
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime
import yaml
from pathlib import Path
import joblib

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response

# Import prediction service
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.predict import PredictionService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Production ML API for credit card fraud detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics - use try-except to avoid duplicate registration
try:
    REQUEST_COUNT = Counter(
        'fraud_detection_requests_total',
        'Total number of requests',
        ['method', 'endpoint', 'status']
    )
except ValueError:
    REQUEST_COUNT = REGISTRY._names_to_collectors['fraud_detection_requests_total']

try:
    REQUEST_LATENCY = Histogram(
        'fraud_detection_request_latency_seconds',
        'Request latency in seconds',
        ['method', 'endpoint']
    )
except ValueError:
    REQUEST_LATENCY = REGISTRY._names_to_collectors['fraud_detection_request_latency_seconds']

try:
    PREDICTION_COUNT = Counter(
        'fraud_predictions_total',
        'Total number of predictions',
        ['prediction']
    )
except ValueError:
    PREDICTION_COUNT = REGISTRY._names_to_collectors['fraud_predictions_total']

try:
    MODEL_SCORE = Gauge(
        'fraud_detection_score',
        'Current fraud detection score'
    )
except ValueError:
    MODEL_SCORE = REGISTRY._names_to_collectors['fraud_detection_score']

# Pydantic models
class Transaction(BaseModel):
    Time: float = Field(..., description="Time in seconds from first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")

    class Config:
        schema_extra = {
            "example": {
                "Time": 12345.0,
                "V1": -1.5, "V2": 0.5, "V3": 1.2, "V4": -0.8,
                "V5": 0.3, "V6": -0.6, "V7": 0.9, "V8": -0.4,
                "V9": 1.1, "V10": -0.2, "V11": 0.7, "V12": -1.0,
                "V13": 0.4, "V14": -0.5, "V15": 0.8, "V16": -0.3,
                "V17": 1.3, "V18": -0.7, "V19": 0.6, "V20": -0.9,
                "V21": 0.2, "V22": -0.4, "V23": 0.5, "V24": -0.6,
                "V25": 0.7, "V26": -0.8, "V27": 0.9, "V28": -1.1,
                "Amount": 150.0
            }
        }


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    normal_probability: float
    prediction_label: str
    timestamp: str
    processing_time_ms: float


class BatchPredictionRequest(BaseModel):
    transactions: List[Transaction]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str


# Global prediction service
prediction_service: Optional[PredictionService] = None


def get_prediction_service():
    """Dependency to get prediction service"""
    global prediction_service
    
    if prediction_service is None:
        # Load model
        model_path = Path("models/fraud_detection_model.pkl")
        
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="No model available")
        
        logger.info(f"Loading model from {model_path}")
        
        prediction_service = PredictionService(
            model_path=str(model_path),
            feature_engineer_path="models",
            config_path="configs/config.yaml"
        )
    
    return prediction_service


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to track request metrics"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        service = get_prediction_service()
        model_loaded = service.model is not None
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: Transaction,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict fraud for a single transaction
    
    Returns fraud probability and classification
    """
    start_time = time.time()
    
    try:
        # Make prediction
        result = service.predict_single(transaction.dict())
        
        # Update metrics
        PREDICTION_COUNT.labels(
            prediction=result['prediction_label']
        ).inc()
        
        MODEL_SCORE.set(result['fraud_probability'])
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            **result,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def batch_predict(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict fraud for multiple transactions
    
    Returns predictions for all transactions
    """
    start_time = time.time()
    
    try:
        # Convert to list of dicts
        transactions = [t.dict() for t in request.transactions]
        
        # Make predictions
        results_df = service.batch_predict(transactions)
        
        # Update metrics
        for pred in results_df['prediction_label']:
            PREDICTION_COUNT.labels(prediction=pred).inc()
        
        # Convert to response
        results = results_df.to_dict('records')
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": processing_time
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info")
async def model_info(
    service: PredictionService = Depends(get_prediction_service)
):
    """Get model information"""
    return {
        "model_type": type(service.model).__name__,
        "model_loaded": service.model is not None,
        "scaler_loaded": service.scaler is not None,
        "feature_selector_loaded": service.feature_selector is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

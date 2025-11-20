import logging
import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from .config import settings

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sklearn Model Service",
    description="A FastAPI service for serving a trained sklearn model.",
    version=settings.version
)

model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Looking for model at: {settings.model_path}")
    logger.info(f"Model path exists: {os.path.exists(settings.model_path)}")
    
    model_abs_path = os.path.abspath(settings.model_path)
    logger.info(f"Absolute model path: {model_abs_path}")
    
    try:
        if os.path.exists(settings.model_path):
            model = joblib.load(settings.model_path)
            logger.info(f"Model loaded successfully from {settings.model_path}")
        else:
            logger.error(f"Model file not found at {settings.model_path}")
            model_dir = os.path.dirname(settings.model_path)
            if os.path.exists(model_dir):
                logger.info(f"Files in {model_dir}: {os.listdir(model_dir)}")
            else:
                logger.error(f"Model directory does not exist: {model_dir}")
        
        preprocessor_path = os.path.join(os.path.dirname(settings.model_path), "preprocessor.joblib")
        logger.info(f"Looking for preprocessor at: {preprocessor_path}")
        logger.info(f"Preprocessor path exists: {os.path.exists(preprocessor_path)}")
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
        else:
            logger.error(f"Preprocessor file not found at {preprocessor_path}")
            alt_preprocessor_path = "preprocessor.joblib"
            if os.path.exists(alt_preprocessor_path):
                preprocessor = joblib.load(alt_preprocessor_path)
                logger.info(f"Preprocessor loaded from alternative location: {alt_preprocessor_path}")
            else:
                logger.error(f"Preprocessor not found in current directory either")

        if model is not None:
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Model classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")
        
        if preprocessor is not None:
            logger.info(f"Preprocessor type: {type(preprocessor)}")
            logger.info(f"Preprocessor feature names: {preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else 'N/A'}")

    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}", exc_info=True)

class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="A dictionary of feature names and their corresponding values."
    )

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: List[float]
    status: str

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("Starting up application...")
    logger.info("=" * 50)
    load_artifacts()
    logger.info("=" * 50)
    logger.info(f"Startup complete. Model loaded: {model is not None}, Preprocessor loaded: {preprocessor is not None}")
    logger.info("=" * 50)

@app.get("/")
async def root():
    return {
        "service": settings.service_name,
        "version": settings.version,
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": settings.model_path,
        "cwd": os.getcwd()
    }

@app.get("/health")
async def health():
    health_status = {
        "service": "healthy" if (model is not None and preprocessor is not None) else "degraded",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": settings.model_path,
        "model_path_exists": os.path.exists(settings.model_path),
        "cwd": os.getcwd()
    }
    
    if preprocessor is not None and hasattr(preprocessor, 'feature_names'):
        health_status["expected_features"] = preprocessor.feature_names
    
    return health_status

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or preprocessor is None:
        error_details = {
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None,
            "model_path": settings.model_path,
            "model_exists": os.path.exists(settings.model_path)
        }
        logger.error(f"Prediction failed due to missing artifacts: {error_details}")
        raise HTTPException(
            status_code=503,
            detail=f"Model or preprocessor not available. Details: {error_details}"
        )
    
    try:
        logger.info(f"Received prediction request with features: {request.features}")
        
        if hasattr(preprocessor, 'feature_names') and preprocessor.feature_names:
            missing_features = set(preprocessor.feature_names) - set(request.features.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
        
        X = pd.DataFrame([request.features])
        logger.info(f"Created DataFrame with shape: {X.shape}, columns: {X.columns.tolist()}")
        
        X_processed = preprocessor.transform(X)
        logger.info(f"Processed features shape: {X_processed.shape}")

        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0].tolist()

        prediction_label = preprocessor.inverse_transform_target([prediction])[0]

        logger.info(f"Prediction successful: {prediction_label}, Probabilities: {probabilities}")

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=str(prediction_label),
            probability=probabilities,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port
    )
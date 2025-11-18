import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from .config import settings
import os


logging.basicConfig(
    level = settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title = "Sklearn Model Service",
    description = "A FastAPI service for serving a trained sklearn model.",
    version = settings.version
)


model = None
preprocessor = None

def load_artifacs():

    global model, preprocessor

    try:
        if os.path.exists(settings.model_path):
            model = joblib.load(settings.model_path)
            logger.info(f"Model loaded from {settings.model_path}")
        
        if os.path.exists("preprocessor.joblib"):
            preprocessor = joblib.load("preprocessor.joblib")
            logger.info("Preprocessor loaded from preprocessor.joblib")

    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")



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
    load_artifacs()



@app.get("/")
async def root():
    return {
        "service": settings.service_name,
        "version": settings.version,
        "model_loaded": model is not None
    }



@app.get("/health")
async def health():
    return {
        "service": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code = 503,
            detail = "Model or preprocessor not available."
        )
    
    try:

        X = pd.DataFrame([request.features])
        X_processed = preprocessor.transform(X)

        prediction = model.preduct(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0].tolist()

        prediction_label = preprocessor.inverse_transform_target([prediction])[0]

        logger.info(f"Prediction: {prediction_label}, Probabilities: {probabilities}")

        return PredictionResponse(
            prediction = int(prediction),
            prediction_label = str(prediction_label),
            probability = probabilities,
            status = "success"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code = 500,
            detail = f"Prediction error: {str(e)}"
        )
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host = settings.api_host,
        port = settings.api_port
    )
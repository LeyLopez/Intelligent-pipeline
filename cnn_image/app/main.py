import logging
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from PIL import Image
import numpy as np
import cv2
import os

from .config import settings
from .model import CNNImageClassifier
from ..filters.convolutions import ConvolutionFilters



logging.basicConfig(
    level=settings.log_level, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title = "CNN Image Classification Service",
    description = "A FastAPI service for image classification using a CNN model.",
    version = settings.version
)

 
classfier = CNNImageClassifier()
filters = ConvolutionFilters()


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    limitations: str
    status: str



class FilterResponse(BaseModel):
    filters_applied: List[str]
    images: Dict[str, str]


@app.on_event("startup")
async def startup_event():

    try:
        if os.path.exists(settings.model_path):
            classfier.load_model()
            logger.info("Model loaded successfully.")
        else:
            logger.warning("Model file not found. Training synthetic model...")
            X, y = classfier.create_synthetic_dataset(samples_per_class=50)
            classfier.train(X, y, epochs=5)
            classfier.save_model()
            logger.info("Synthetic model trained and saved.")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")




@app.get("/")
async def root():
    return {
        "service": settings.service_name,
        "version": settings.version,
        "model_loaded": classfier.model is not None,
        "supported_classes": settings.class_names
    }


@app.get("/health")
async def health():
    return {
        "service": "healthy" if classfier.model is not None else "degraded",
        "model_loaded": classfier.model is not None,
        "num_classes": settings.num_classes,
        "image_size": settings.image_size
    }



@app.get("/info")
async def info():
    return {
        "service": settings.service_name,
        "capabilities":{
            "classification": True,
            "filters": ["gaussian_blur", "edge_detection", "sharpening", "emboss"],
            "supported_classes": settings.class_names,
            "image_size": settings.image_size
        },
        "limitations":{
            "classes": f"Only supports classes: {settings.class_names}",
            "accuracy": "Model accuracy may vary based on input image quality and content - Limited precision on synthetic data",
            "image_format": "Supports common image formats (JPG, PNG) only. Recommended size is 128x128 pixels.",
            "restrictions":[
                "Does not recognize objects outside trained classes.",
                "Does not detect multiple objects in a single image.",
                "Result may be imprecise for images so different from training data."
            ]
        }
    }



@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):

    if classfier.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please try again later."
        )
    
    try:
        image_bytes = await file.read()
        predicted_class, probabilities = classfier.predict(image_bytes)

        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(settings.class_names, probabilities)
        }

        confidence = max(probabilities)

        limitations = (
            f"This model only recognize {len(settings.class_names)} classes."
            f"{', '.join(settings.class_names)}. "
            f"If the image has other object, the prediction will be incorrect. "
        )

        logger.info(f"Image classified as {predicted_class} with confidence {confidence:.2%}")


        return ClassificationResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=prob_dict,
            limitations=limitations,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"An error ocurred during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/apply-filters", response_model = FilterResponse)
async def apply_filters(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img_array = np.array(img)

        filtered_images = filters.apply_all_filters(img_array)

        encoded_images = {}
        for filter_name, filtered_img in filtered_images.items():
            
            pil_img = Image.fromarray(filtered_img.astype('uint8'))
            
            
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            encoded_images[filter_name] = img_str
        
        logger.info(f"Filtros aplicados exitosamente: {list(filtered_images.keys())}")
        
        return FilterResponse(
            filters_applied=list(filtered_images.keys()),
            images=encoded_images
        )
    
    except Exception as e:
        logger.error(f"Error aplicando filtros: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)







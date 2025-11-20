import logging
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from PIL import Image
import numpy as np

import os

from .config import settings
from .model import CNNImageClassifier
from filters.convolutions import ConvolutionFilters


logging.basicConfig(
    level=settings.log_level, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="CNN Image Classification Service",
    description="A FastAPI service for image classification using Transfer Learning (MobileNetV2).",
    version=settings.version
)

 
classifier = CNNImageClassifier()
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


class TrainingRequest(BaseModel):
    samples_per_class: int = 200
    epochs: int = 20
    batch_size: int = 32
    fine_tune: bool = False


class TrainingResponse(BaseModel):
    status: str
    message: str
    final_metrics: Dict[str, float]


@app.on_event("startup")
async def startup_event():

    try:
        if os.path.exists(settings.model_path):
            logger.info("Loading existing model...")
            classifier.load_model()
            logger.info("Model loaded successfully.")
        else:
            logger.warning("Model file not found. Training new model with Transfer Learning...")
            X, y = classifier.create_synthetic_dataset(samples_per_class=200)
            
            history = classifier.train(X, y, epochs=20, batch_size=32)
            
            classifier.save_model()
            
            final_acc = history['val_accuracy'][-1]
            logger.info(f"Model trained and saved. Final validation accuracy: {final_acc:.4f}")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error("Service will continue but model may not be available.")


@app.get("/")
async def root():
    return {
        "service": settings.service_name,
        "version": settings.version,
        "model_loaded": classifier.model is not None,
        "supported_classes": settings.class_names,
        "architecture": "MobileNetV2 Transfer Learning",
        "image_size": settings.image_size
    }


@app.get("/health")
async def health():
    return {
        "service": "healthy" if classifier.model is not None else "degraded",
        "model_loaded": classifier.model is not None,
        "num_classes": settings.num_classes,
        "image_size": settings.image_size,
        "architecture": "MobileNetV2"
    }


@app.get("/info")
async def info():
    return {
        "service": settings.service_name,
        "architecture": {
            "base_model": "MobileNetV2",
            "pretrained": "ImageNet",
            "transfer_learning": True,
            "data_augmentation": ["RandomFlip", "RandomRotation", "RandomZoom", "RandomContrast"]
        },
        "capabilities": {
            "classification": True,
            "filters": ["gaussian_blur", "edge_detection", "sharpening", "emboss"],
            "supported_classes": settings.class_names,
            "image_size": settings.image_size,
            "fine_tuning": True
        },
        "limitations": {
            "classes": f"Only supports {len(settings.class_names)} classes: {', '.join(settings.class_names)}",
            "accuracy": "Transfer Learning model trained on synthetic data. Performance on real images may vary.",
            "image_format": "Supports common image formats (JPG, PNG, WebP). Recommended size is 128x128 pixels.",
            "restrictions": [
                "Model recognizes only the trained classes.",
                "Single object classification (not multi-object detection).",
                "Best results with clear, centered objects.",
                "Synthetic training data may limit real-world accuracy."
            ]
        },
        "recommendations": {
            "image_quality": "Use clear, well-lit images with centered objects.",
            "image_size": "Images will be resized to 128x128 pixels.",
            "confidence_threshold": "Consider predictions with confidence > 0.6 as reliable.",
            "fine_tuning": "Use /train endpoint with fine_tune=true for better accuracy on specific data."
        }
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):

    if classifier.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please wait for model initialization or trigger /train endpoint."
        )
    
    try:
        image_bytes = await file.read()
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()  # Verificar que es una imagen válida
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Please upload a valid image file. Error: {str(e)}"
            )
        
        predicted_class, probabilities = classifier.predict(image_bytes)

        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(settings.class_names, probabilities)
        }

        confidence = float(max(probabilities))

        limitations = (
            f"This model recognizes {len(settings.class_names)} classes: "
            f"{', '.join(settings.class_names)}. "
            f"Predictions with confidence < 0.6 should be treated with caution. "
            f"The model uses Transfer Learning (MobileNetV2) but was trained on synthetic data."
        )

        logger.info(f"Image classified as '{predicted_class}' with confidence {confidence:.2%}")

        return ClassificationResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=prob_dict,
            limitations=limitations,
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/apply-filters", response_model=FilterResponse)
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
        
        logger.info(f"Filters applied successfully: {list(filtered_images.keys())}")
        
        return FilterResponse(
            filters_applied=list(filtered_images.keys()),
            images=encoded_images
        )
    
    except Exception as e:
        logger.error(f"Error applying filters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Filter application error: {str(e)}")


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):

    try:
        logger.info(f"Starting training with parameters: {request.dict()}")
        
        X, y = classifier.create_synthetic_dataset(
            samples_per_class=request.samples_per_class
        )
        
        history = classifier.train(
            X, y,
            epochs=request.epochs,
            batch_size=request.batch_size,
            fine_tune=request.fine_tune
        )
        
        classifier.save_model()
        
        final_metrics = {
            "train_accuracy": float(history['accuracy'][-1]),
            "val_accuracy": float(history['val_accuracy'][-1]),
            "train_loss": float(history['loss'][-1]),
            "val_loss": float(history['val_loss'][-1]),
            "best_val_accuracy": float(max(history['val_accuracy']))
        }
        
        logger.info(f"Training completed. Metrics: {final_metrics}")
        
        return TrainingResponse(
            status="success",
            message=f"Model trained successfully with {request.samples_per_class * settings.num_classes} samples",
            final_metrics=final_metrics
        )
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training error: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """
    Información detallada sobre el modelo actual.
    """
    if classifier.model is None:
        return {
            "status": "no_model_loaded",
            "message": "No model is currently loaded"
        }
    
    try:
        total_params = classifier.model.count_params()
        trainable_params = sum([
            np.prod(w.shape) for w in classifier.model.trainable_weights
        ])
        non_trainable_params = total_params - trainable_params
        
        return {
            "status": "loaded",
            "architecture": "MobileNetV2 Transfer Learning",
            "parameters": {
                "total": int(total_params),
                "trainable": int(trainable_params),
                "non_trainable": int(non_trainable_params)
            },
            "input_shape": list(classifier.model.input_shape),
            "output_shape": list(classifier.model.output_shape),
            "num_layers": len(classifier.model.layers),
            "classes": settings.class_names
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port,
        log_level="info"
    )
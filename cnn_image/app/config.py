from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):

    service_name: str = "cnn-image"
    version: str = "1.0.0"

    api_host: str = "0.0.0.0"
    api_port: int = 8003

    model_path: str = "./models/cnn_classifier.h5"
    image_size: tuple = (128, 128)
    num_classes: int = 3
    class_names: List[str] = ["cat", "dog", "bird"]

    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "cnn_image_classification"

    log_level: str = "INFO"

    model_config = {
        "protected_namespaces": ()
    }

    class Config:
        env_file = ".env"


settings = Settings()
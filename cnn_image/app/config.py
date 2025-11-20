from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Tuple
from typing import List

class Settings(BaseSettings):

    service_name: str = "cnn-image"
    version: str = "1.0.0"

    api_host: str = "0.0.0.0"
    api_port: int = 8003

    image_size: Tuple[int, int] = Field(default=(128, 128), description="Tamaño de imagen (alto, ancho)")
    num_classes: int = Field(default=3, description="Número de clases a clasificar")
    class_names: List[str] = Field(
        default=["dog", "cat", "bird"],
        description="Nombres de las clases"
    )
    
    model_path: str = Field(default="models/cnn_model.keras", description="Ruta del modelo guardado")
    


    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "cnn_image_classification"

    log_level: str = "INFO"

    model_config = {
        "protected_namespaces": ()
    }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


settings = Settings()
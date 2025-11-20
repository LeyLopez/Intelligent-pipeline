from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    service_name: str = "sklearn-model"
    version: str = "1.0.0"

    api_host: str = "0.0.0.0"
    api_port: int = 8002

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "sklearn_classification"


    model_path: str = "./models/classifier.joblib"
    model_type: str = "classification"

    log_level: str = "INFO"


    class Config:
        env_file = ".env"


settings = Settings()



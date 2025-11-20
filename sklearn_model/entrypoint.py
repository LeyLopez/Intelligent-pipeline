import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = "./models/classifier.joblib"
PREPROCESSOR_PATH = "./models/preprocessor.joblib"

def main():
    logger.info("=" * 50)
    logger.info("Initializing ML Model Service")
    logger.info("=" * 50)
    
    model_exists = os.path.exists(MODEL_PATH)
    preprocessor_exists = os.path.exists(PREPROCESSOR_PATH)
    
    logger.info(f"Model exists: {model_exists}")
    logger.info(f"Preprocessor exists: {preprocessor_exists}")
    
    if not model_exists or not preprocessor_exists:
        logger.info("Model or preprocessor not found. Training model...")
        
        try:
            # Crear directorio si no existe
            os.makedirs("./models", exist_ok=True)
            
            result = subprocess.run(
                [sys.executable, "-m", "pipeline.train"],
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info(result.stdout)
            
            if result.stderr:
                logger.warning(f"Training stderr: {result.stderr}")
            
            logger.info("Model trained successfully!")
            
            # Verificar que los archivos se crearon
            if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
                logger.info(f"Model size: {os.path.getsize(MODEL_PATH) / 1024:.2f} KB")
                logger.info(f"Preprocessor size: {os.path.getsize(PREPROCESSOR_PATH) / 1024:.2f} KB")
            else:
                logger.error("Model files not created after training!")
                logger.error(f"Files in ./models: {os.listdir('./models') if os.path.exists('./models') else 'directory does not exist'}")
                sys.exit(1)
                
        except subprocess.CalledProcessError as e:
            logger.error("Model training failed!")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            sys.exit(1)
    else:
        logger.info("Model already exists. Skipping training.")
        logger.info(f"Model size: {os.path.getsize(MODEL_PATH) / 1024:.2f} KB")
        logger.info(f"Preprocessor size: {os.path.getsize(PREPROCESSOR_PATH) / 1024:.2f} KB")
    
    logger.info("=" * 50)
    logger.info("Starting API server...")
    logger.info("=" * 50)
    
    try:
        subprocess.run(
            ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"],
            check=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
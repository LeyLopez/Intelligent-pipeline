import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        X, y = trainer.load_sample_data()
        metrics = trainer.train(X, y)
        trainer.save_model()
        logger.info(f"Training completed with metrics: {metrics}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)
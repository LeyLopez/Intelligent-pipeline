import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.datasets import load_iris
import pandas as pd
import joblib
import logging
from datetime import datetime
from .preprocess import DataPreprocessor
from ..app.config import settings

logger = logging.getLogger(__name__)

class ModelTrainer:

    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    

    def load_sample_data(self)-> Tuple[pd.DataFrame, pd.Series]:

        logger.info("Loading sample Iris dataset")
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        return X, y
    


    def train(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            test_size: float = 0.2,
            random_state: int = 42
    )->dict:
        
        logger.info("Starting model training")

        with mlflow.start_run(run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            X_processed, y_encoded = self.preprocessor.fit_transform(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_processed,
                y_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded
            )

            logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

            params ={
                "n_estimators":100,
                "max_depth":10,
                "min_samples_split":2,
                "min_samples_leaf":1,
                "random_state":random_state
            }

            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }

            cv_scores = cross_val_score(
                self.model, X_processed, y_encoded, cv=5
            )

            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                self.model, "model", registered_model_name="sklearn_classifier"
            )

            joblib.dump(
                self.preprocessor,
                "preprocessor.joblib"
            )
            mlflow.log_artifact("preprocessor.joblib")

            logger.info(f"Training completed with metrics: {metrics}")

            return metrics
        


    def save_model(self, model_path: str = None):
        if model_path is None:
            model_path = settings.model_path
        
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")


    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    trainer = ModelTrainer()
    X, y = trainer.load_sample_data()
    metrics = trainer.train(X, y)
    trainer.save_model()

    print(f"Training completed with metrics: \n {metrics}")
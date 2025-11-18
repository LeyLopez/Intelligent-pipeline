import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import sys
sys.path.append("..")

from pipeline.preprocess import DataPreprocessor
from pipeline.train import ModelTrainer


def test_preprocessor():
    iris = load_iris()
    X = pd.DataFrame(iris.data[:10], columns=iris.feature_names)
    y = pd.Series(iris.target[:10])

    preprocessor = DataPreprocessor()
    X_scaled, y_encoded = preprocessor.fit_transform(X, y)

    assert X_scaled.shape ==(10, 4)
    assert len(y_encoded) == 10
    assert X_scaled.mean() < 1.0



def test_model_trainer():
    trainer = ModelTrainer()
    X, y = trainer.load_sample_data()
    metrics = trainer.train(X[:50], y[:50])

    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.7
    assert trainer.model is not None


def test_prediction_shape():
    trainer = ModelTrainer()
    X, y = trainer.load_sample_data()
    trainer.train(X, y)

    X_processed, _ = trainer.preprocessor.fit_transform(X[:5])
    prediction = trainer.model.predict(X_processed)

    assert len(prediction) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
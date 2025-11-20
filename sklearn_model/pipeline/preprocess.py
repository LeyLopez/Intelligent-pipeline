import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = LabelEncoder()
        self.feature_names = None


    
    def fit_transform(
            self,
            X = pd.DataFrame,
            y: Optional[pd.Series] = None
    )-> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        logger.info(f"Preprocessing data:  {X.shape}")

        self.feature_names = X.columns.tolist()

        X_clean = X.fillna(X.mean())

        X_scaled = self.scaler.fit_transform(X_clean)

        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoders.fit_transform(y)
            logger.info(f"Encoding the target variable: {y.shape}")
            logger.info(f"Classes found: {self.label_encoders.classes_}")
        
        return X_scaled, y_encoded
    




    def transform(self, X: pd.DataFrame)-> np.ndarray:

        if self.feature_names and list(X.columns) != self.feature_names:
            raise ValueError(
                f"Expected features {self.feature_names}, but got {list(X.columns)}"
            )
        
        X_clean = X.fillna(X.mean())
        return self.scaler.transform(X_clean)
    




    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoders.inverse_transform(y_encoded)
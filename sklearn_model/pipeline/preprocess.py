import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}  # ← Diccionario
        self.feature_names = None

    def fit_transform(
            self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        logger.info(f"Preprocessing data: {X.shape}")

        self.feature_names = X.columns.tolist()

        X_clean = X.fillna(X.mean())

        X_scaled = self.scaler.fit_transform(X_clean)

        y_encoded = None
        if y is not None:
            # Inicializar si no existe
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
            
            y_encoded = self.label_encoders['target'].fit_transform(y)
            logger.info(f"Encoding the target variable: {y.shape}")
            logger.info(f"Classes found: {self.label_encoders['target'].classes_}")
        
        return X_scaled, y_encoded

    def transform(self, X: pd.DataFrame) -> np.ndarray:

        if self.feature_names and list(X.columns) != self.feature_names:
            raise ValueError(
                f"Expected features {self.feature_names}, but got {list(X.columns)}"
            )
        
        X_clean = X.fillna(X.mean())
        return self.scaler.transform(X_clean)

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        if 'target' not in self.label_encoders:
            raise ValueError("Target encoder not fitted yet")
        return self.label_encoders['target'].inverse_transform(y_encoded)
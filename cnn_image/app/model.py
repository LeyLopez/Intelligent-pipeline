import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import mlflow
import mlflow.keras
from PIL import Image
import io
import logging
from typing import Tuple, List
from .config import settings

logger = logging.getLogger(__name__)

class CNNImageClassifier:

    def __init__(self):
        self.model = None
        self.image_size = settings.image_size
        self.num_classes = settings.num_classes
        self.class_names = settings.class_names
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)


    
    def build_model(self) -> models.Sequential:
        
        logger.info("Building CNN model")

        model = models.Sequential([

            layers.Conv2D(
                32, (3, 3),
                activation='relu',
                input_shape=(self.image_size[0], self.image_size[1], 3),
                name='conv1'
            ),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.BatchNormalization(name='bn1'),


            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.BatchNormalization(name='bn2'),


            layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.BatchNormalization(name='bn3'),


            layers.Conv2D(128, (3, 3), activation='relu', name='conv4'),
            layers.MaxPooling2D((2, 2), name='pool4'),


            layers.Flatten(name='flatten'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(256, activation='relu', name='dense1'),
            layers.Dropout(0.3, name='dropout2'),
            layers.Dense(self.num_classes, activation='softmax', name='output')

        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


        logger.info(f"Model built with parameters: {model.count_params()}")

        return model
    


    def create_synthetic_dataset(self, samples_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:

        logger.info(f"Creating synthetic dataset with {samples_per_class} samples per class")

        X_list = []
        y_list = []

        for class_idx in range(self.num_classes):
            for _ in range(samples_per_class):
                img = np.random.rand(
                    self.image_size[0], 
                    self.image_size[1],
                3                     
                ) * 0.5  # Valores entre 0 y 0.5

                
                if class_idx == 0:
                    img[:, :, 0] += 0.3 
                elif class_idx == 1:
                    img[:, :, 1] += 0.3 
                else:
                    img[:, :, 2] += 0.3 

                img = np.clip(img, 0, 1)

                X_list.append(img)
                y_list.append(class_idx)

        X = np.array(X_list)
        y = np.array(y_list)


        indexes = np.random.permutation(len(X))
        X = X[indexes]
        y = y[indexes]

        return X, y

    


    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int = 10,
            batch_size: int = 32,
            validation_split: float = 0.2
    ) -> dict:
        
        logger.info(f"Starting training on {len(X_train)} samples for {epochs} epochs")

        with mlflow.start_run():
            
            self.model = self.build_model()

            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "image_size": self.image_size,
                "num_classes": self.num_classes,
                "optimizer": "adam",
            })


            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )


            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )


            final_metrics = {
                "train_accuracy": history.history['accuracy'][-1],
                "val_accuracy": history.history['val_accuracy'][-1],
                "train_loss": history.history['loss'][-1],
                "val_loss": history.history['val_loss'][-1]
            }
            mlflow.log_metrics(final_metrics)


            mlflow.keras.log_model(self.model, "cnn_image_classifier_model")


            logger.info(f"Training completed and model logged to MLflow with metrics: {final_metrics}")

            return history.history
        



    def predict(self, image_bytes: bytes) -> Tuple[str, List[float]]:

        logger.info("Predicting image class")

        if self.model is None:
            raise ValueError("Model is not loaded. Please load or train the model before prediction.")
        
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize(self.image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)


        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]

        logger.info(f"Predicted class: {predicted_class} with confidence {predictions[predicted_class_idx]:.4f}")

        return predicted_class, predictions.tolist()
    


    def save_model(self, path: str = None):
        if path is None:
            path = settings.model_path

        self.model.save(path)
        logger.info(f"Model saved at {path}")


    

    def load_model(self, path: str = None):
        if path is None:
            path = settings.model_path

        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
        




        
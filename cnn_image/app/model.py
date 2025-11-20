import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import mlflow
import mlflow.keras
from PIL import Image
import io
import logging
from typing import Tuple, List
from .config import settings
import cv2


logger = logging.getLogger(__name__)

class CNNImageClassifier:

    def __init__(self):
        self.model = None
        self.image_size = settings.image_size
        self.num_classes = settings.num_classes
        self.class_names = settings.class_names
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    
    def build_model(self) -> models.Model:

        logger.info("Building CNN model with Transfer Learning (MobileNetV2)")

        base_model = MobileNetV2(
            input_shape=(self.image_size[0], self.image_size[1], 3),
            include_top=False, 
            weights='imagenet')
        
 
        base_model.trainable = False

  
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ], name='data_augmentation')


        inputs = keras.Input(shape=(self.image_size[0], self.image_size[1], 3))
        

        x = data_augmentation(inputs)

        x = keras.applications.mobilenet_v2.preprocess_input(x)
        

        x = base_model(x, training=False)
        

        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.BatchNormalization(name='bn_final')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Model built with parameters: {model.count_params()}")
        logger.info(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights])}")

        return model
    
    def build_model_fine_tune(self) -> models.Model:
        logger.info("Building model with fine-tuning enabled")
        
        model = self.build_model()
        
        base_model = model.layers[3] 
        base_model.trainable = True
        
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Fine-tuning enabled. Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights])}")
        
        return model

    def create_synthetic_dataset(self, samples_per_class: int = 200) -> Tuple[np.ndarray, np.ndarray]:

        logger.info(f"Creating improved synthetic dataset with {samples_per_class} samples per class")

        X_list = []
        y_list = []

        for class_idx in range(self.num_classes):
            for i in range(samples_per_class):
                img = np.random.randint(200, 255, 
                    (self.image_size[0], self.image_size[1], 3), 
                    dtype=np.uint8)
                
                noise = np.random.normal(0, 15, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                if class_idx == 0:  # Clase 0: Círculos rojos
                    num_circles = np.random.randint(1, 4)
                    for _ in range(num_circles):
                        center = (
                            np.random.randint(20, self.image_size[1]-20),
                            np.random.randint(20, self.image_size[0]-20)
                        )
                        radius = np.random.randint(15, 35)
                        color = (
                            np.random.randint(180, 255),  # Rojo
                            np.random.randint(0, 50),
                            np.random.randint(0, 50)
                        )
                        thickness = np.random.choice([-1, 2, 3])
                        cv2.circle(img, center, radius, color, thickness)
                
                elif class_idx == 1:  # Clase 1: Rectángulos verdes
                    num_rects = np.random.randint(1, 3)
                    for _ in range(num_rects):
                        x1 = np.random.randint(10, self.image_size[1]//2)
                        y1 = np.random.randint(10, self.image_size[0]//2)
                        width = np.random.randint(30, 60)
                        height = np.random.randint(30, 60)
                        color = (
                            np.random.randint(0, 50),
                            np.random.randint(180, 255),  # Verde
                            np.random.randint(0, 50)
                        )
                        thickness = np.random.choice([-1, 2, 3])
                        cv2.rectangle(img, (x1, y1), (x1+width, y1+height), color, thickness)
                
                else:  # Clase 2: Triángulos azules
                    num_triangles = np.random.randint(1, 3)
                    for _ in range(num_triangles):
                        cx = np.random.randint(30, self.image_size[1]-30)
                        cy = np.random.randint(30, self.image_size[0]-30)
                        size = np.random.randint(20, 40)
                        pts = np.array([
                            [cx, cy - size],
                            [cx - size, cy + size],
                            [cx + size, cy + size]
                        ], np.int32)
                        color = (
                            np.random.randint(0, 50),
                            np.random.randint(0, 50),
                            np.random.randint(180, 255)  # Azul
                        )
                        cv2.fillPoly(img, [pts], color)
                
                if np.random.rand() > 0.5:

                    angle = np.random.randint(-30, 30)
                    M = cv2.getRotationMatrix2D(
                        (self.image_size[1]//2, self.image_size[0]//2), 
                        angle, 1.0
                    )
                    img = cv2.warpAffine(img, M, (self.image_size[1], self.image_size[0]))
                
                img = img.astype(np.float32) / 255.0
                
                X_list.append(img)
                y_list.append(class_idx)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        logger.info(f"Synthetic dataset created: X.shape={X.shape}, y.shape={y.shape}")

        return X, y

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int = 20,
            batch_size: int = 32,
            validation_split: float = 0.2,
            fine_tune: bool = False
    ) -> dict:

        logger.info(f"Starting training on {len(X_train)} samples for {epochs} epochs")

        with mlflow.start_run():
            
            if fine_tune and self.model is not None:
                logger.info("Fine-tuning existing model")
                self.model = self.build_model_fine_tune()
            else:
                self.model = self.build_model()

            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "image_size": self.image_size,
                "num_classes": self.num_classes,
                "optimizer": "adam",
                "architecture": "MobileNetV2_transfer_learning",
                "fine_tune": fine_tune
            })

            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,  # Aumentado para dar más tiempo al modelo
                restore_best_weights=True,
                verbose=1
            )

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )

            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            final_metrics = {
                "train_accuracy": float(history.history['accuracy'][-1]),
                "val_accuracy": float(history.history['val_accuracy'][-1]),
                "train_loss": float(history.history['loss'][-1]),
                "val_loss": float(history.history['val_loss'][-1]),
                "best_val_accuracy": float(max(history.history['val_accuracy']))
            }
            mlflow.log_metrics(final_metrics)

            mlflow.keras.log_model(self.model, "cnn_image_classifier_model")

            logger.info(f"Training completed. Best val_accuracy: {final_metrics['best_val_accuracy']:.4f}")

            return history.history

    def predict(self, image_bytes: bytes) -> Tuple[str, List[float]]:
        logger.info("Predicting image class")

        if self.model is None:
            raise ValueError("Model is not loaded. Please load or train the model before prediction.")
        
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize(self.image_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
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
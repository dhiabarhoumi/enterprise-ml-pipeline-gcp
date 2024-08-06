"""Model definition for the Enterprise ML Pipeline."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

logger = logging.getLogger(__name__)


class RetailPredictionModel:
    """Model for retail sales prediction."""

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        categorical_vocab: Dict[str, List[str]],
        output_type: str = "regression",
        hidden_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """Initialize the model.

        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            categorical_vocab: Dictionary mapping feature names to vocabulary lists
            output_type: Type of prediction task ('regression' or 'classification')
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.categorical_vocab = categorical_vocab
        self.output_type = output_type
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Build the model
        self.model = self._build_model()
        logger.info("Model initialized successfully")

    def _build_model(self) -> tf.keras.Model:
        """Build the TensorFlow model architecture.

        Returns:
            Compiled TensorFlow model
        """
        # Input layers
        inputs = {}
        encoded_features = []
        
        # Numeric features
        for feature_name in self.numeric_features:
            inputs[feature_name] = layers.Input(shape=(1,), name=feature_name)
            # Normalize numeric features
            encoded = layers.Normalization(axis=-1)(inputs[feature_name])
            encoded_features.append(encoded)
        
        # Categorical features
        for feature_name in self.categorical_features:
            vocab = self.categorical_vocab[feature_name]
            inputs[feature_name] = layers.Input(shape=(1,), name=feature_name, dtype=tf.string)
            # Convert strings to indices
            lookup = layers.StringLookup(vocabulary=vocab, output_mode="int")(inputs[feature_name])
            # Embed categorical features
            embedding_size = min(len(vocab) // 2, 50)  # Heuristic for embedding size
            embedding = layers.Embedding(input_dim=len(vocab) + 1, output_dim=embedding_size)(lookup)
            embedding = tf.squeeze(embedding, axis=1)
            encoded_features.append(embedding)
        
        # Concatenate all features
        x = layers.Concatenate()(encoded_features)
        
        # Hidden layers
        for units in self.hidden_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.output_type == "regression":
            output = layers.Dense(1)(x)
            loss = "mse"
            metrics = ["mae"]
        else:  # classification
            output = layers.Dense(1, activation="sigmoid")(x)
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        
        # Create and compile model
        model = models.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Built {self.output_type} model with {len(self.hidden_units)} hidden layers")
        return model

    def train(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            train_dataset: TensorFlow dataset for training
            validation_dataset: Optional TensorFlow dataset for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: Optional list of Keras callbacks

        Returns:
            Training history
        """
        logger.info(f"Starting model training for {epochs} epochs with batch size {batch_size}")
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss" if validation_dataset else "loss",
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if validation_dataset else "loss",
                    factor=0.5,
                    patience=2
                )
            ]
        
        # Train the model
        history = self.model.fit(
            train_dataset.batch(batch_size),
            validation_data=validation_dataset.batch(batch_size) if validation_dataset else None,
            epochs=epochs,
            callbacks=callbacks
        )
        
        logger.info("Model training completed")
        return history

    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_dataset: TensorFlow dataset for testing
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test data")
        results = self.model.evaluate(test_dataset.batch(batch_size), return_dict=True)
        logger.info(f"Evaluation results: {results}")
        return results

    def predict(
        self,
        input_data: Union[Dict[str, tf.Tensor], tf.data.Dataset],
        batch_size: int = 32
    ) -> tf.Tensor:
        """Make predictions with the model.

        Args:
            input_data: Input data as a dictionary of tensors or a TensorFlow dataset
            batch_size: Batch size for prediction

        Returns:
            Tensor of predictions
        """
        logger.info("Making predictions with the model")
        
        if isinstance(input_data, tf.data.Dataset):
            predictions = self.model.predict(input_data.batch(batch_size))
        else:
            predictions = self.model.predict(input_data, batch_size=batch_size)
            
        return predictions

    def save(self, export_dir: str) -> str:
        """Save the model to disk.

        Args:
            export_dir: Directory to save the model

        Returns:
            Path to the saved model
        """
        logger.info(f"Saving model to {export_dir}")
        self.model.save(export_dir)
        return export_dir

    @classmethod
    def load(cls, model_path: str) -> tf.keras.Model:
        """Load a saved model.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded TensorFlow model
        """
        logger.info(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)


def create_tf_dataset(
    data: Dict[str, List[Any]],
    target_feature: str,
    batch_size: int = 32,
    shuffle: bool = True,
    cache: bool = True
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from input data.

    Args:
        data: Dictionary of feature arrays
        target_feature: Name of the target feature
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        cache: Whether to cache the dataset

    Returns:
        TensorFlow dataset
    """
    # Separate features and labels
    features = {k: v for k, v in data.items() if k != target_feature}
    labels = data[target_feature]
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    # Apply transformations
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))
        
    if cache:
        dataset = dataset.cache()
    
    return dataset

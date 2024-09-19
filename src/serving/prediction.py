"""Prediction utilities for the Enterprise ML Pipeline."""

import logging
from typing import Dict, List, Any, Union
import os

import tensorflow as tf
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions with the trained model."""

    def __init__(self, model_path: str):
        """Initialize the prediction service.

        Args:
            model_path: Path to the saved model
        """
        self.model_path = model_path
        self.model = self._load_model()
        logger.info(f"Initialized PredictionService with model from {model_path}")

    def _load_model(self) -> tf.saved_model.SavedModel:
        """Load the TensorFlow model from disk.

        Returns:
            Loaded TensorFlow model
        """
        logger.info(f"Loading model from {self.model_path}")
        return tf.saved_model.load(self.model_path)

    def predict(self, instances: List[Dict[str, Any]]) -> np.ndarray:
        """Make predictions for multiple instances.

        Args:
            instances: List of feature dictionaries

        Returns:
            NumPy array of predictions
        """
        # Convert instances to TensorFlow examples
        examples = [self._create_tf_example(instance) for instance in instances]
        
        # Make predictions
        return self._predict(examples)

    def predict_single(self, instance: Dict[str, Any]) -> float:
        """Make a prediction for a single instance.

        Args:
            instance: Feature dictionary

        Returns:
            Prediction value
        """
        # Convert instance to TensorFlow example
        example = self._create_tf_example(instance)
        
        # Make prediction
        predictions = self._predict([example])
        
        return float(predictions[0][0])

    def _create_tf_example(self, instance: Dict[str, Any]) -> bytes:
        """Create a serialized TensorFlow Example from a feature dictionary.

        Args:
            instance: Feature dictionary

        Returns:
            Serialized TensorFlow Example
        """
        feature = {}
        
        # Process features
        for name, value in instance.items():
            if isinstance(value, (int, float)):
                feature[name] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(value)])
                )
            elif isinstance(value, str):
                feature[name] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value.encode()])
                )
            elif isinstance(value, list):
                if all(isinstance(x, (int, float)) for x in value):
                    feature[name] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[float(x) for x in value])
                    )
                elif all(isinstance(x, str) for x in value):
                    feature[name] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[x.encode() for x in value])
                    )
        
        # Create Example
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def _predict(self, examples: List[bytes]) -> np.ndarray:
        """Make predictions using the loaded model.

        Args:
            examples: List of serialized TensorFlow Examples

        Returns:
            NumPy array of predictions
        """
        # Create serving input
        serving_input = tf.constant(examples)
        
        # Get prediction signature
        infer = self.model.signatures["serving_default"]
        
        # Make prediction
        predictions = infer(examples=serving_input)
        
        # Extract prediction values
        return predictions["output_0"].numpy()

    def predict_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Make predictions for data in a CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            DataFrame with original data and predictions
        """
        # Load CSV data
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame to list of dictionaries
        instances = df.to_dict(orient="records")
        
        # Make predictions
        predictions = self.predict(instances)
        
        # Add predictions to DataFrame
        df["prediction"] = predictions
        
        return df

    def batch_predict(
        self, instances: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[float]:
        """Make predictions in batches.

        Args:
            instances: List of feature dictionaries
            batch_size: Batch size for prediction

        Returns:
            List of prediction values
        """
        # Process in batches
        results = []
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i + batch_size]
            batch_predictions = self.predict(batch)
            results.extend(batch_predictions.flatten().tolist())
        
        return results

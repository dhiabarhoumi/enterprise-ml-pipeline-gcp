"""Flask application for serving the trained model."""

import logging
import os
from typing import Dict, List, Any, Union

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None


def load_model(model_path: str) -> tf.keras.Model:
    """Load the TensorFlow model from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded TensorFlow model
    """
    logger.info(f"Loading model from {model_path}")
    return tf.saved_model.load(model_path)


def initialize():
    """Initialize the application by loading the model."""
    global model
    
    # Get model path from environment variable or use default
    model_path = os.environ.get("MODEL_PATH", "models/serving_model")
    
    # Load the model
    model = load_model(model_path)
    logger.info("Model loaded successfully")


# Initialize the model when the app starts
with app.app_context():
    initialize()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns:
        JSON response with status
    """
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint.

    Returns:
        JSON response with predictions
    """
    # Check if model is loaded
    global model
    if model is None:
        initialize()
    
    # Get request data
    request_json = request.get_json()
    
    if not request_json:
        return jsonify({"error": "No input data provided"}), 400
    
    try:
        # Convert input to TensorFlow examples
        examples = _create_tf_example(request_json)
        
        # Make predictions
        predictions = _predict(examples)
        
        # Format response
        response = {
            "predictions": predictions.tolist(),
            "input_features": request_json
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


def _create_tf_example(input_data: Dict[str, Any]) -> tf.train.Example:
    """Create TensorFlow Example from input data.

    Args:
        input_data: Dictionary of input features

    Returns:
        TensorFlow Example
    """
    feature = {}
    
    # Process numeric features
    for name, value in input_data.items():
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


def _predict(examples: Union[str, List[str]]) -> np.ndarray:
    """Make predictions using the loaded model.

    Args:
        examples: Serialized TensorFlow Example(s)

    Returns:
        NumPy array of predictions
    """
    # Ensure examples is a list
    if isinstance(examples, str):
        examples = [examples]
    
    # Create serving input
    serving_input = tf.constant(examples)
    
    # Get prediction signature
    infer = model.signatures["serving_default"]
    
    # Make prediction
    predictions = infer(examples=serving_input)
    
    # Extract prediction values
    return predictions["output_0"].numpy()


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Batch prediction endpoint.

    Returns:
        JSON response with predictions
    """
    # Check if model is loaded
    global model
    if model is None:
        initialize()
    
    # Get request data
    request_json = request.get_json()
    
    if not request_json or "instances" not in request_json:
        return jsonify({"error": "No instances provided"}), 400
    
    try:
        # Get instances
        instances = request_json["instances"]
        
        # Convert each instance to TensorFlow example
        examples = [_create_tf_example(instance) for instance in instances]
        
        # Make predictions
        predictions = _predict(examples)
        
        # Format response
        response = {
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

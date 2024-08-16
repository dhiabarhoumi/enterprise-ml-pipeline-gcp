"""TFX Trainer module for model training."""

import os
import logging
from typing import Dict, List, Text, Any, Callable

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor

from src.pipeline.transform import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    _input_fn,
    _build_keras_model,
    _get_serve_tf_examples_fn
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_fn(fn_args: FnArgs) -> None:
    """Train the model based on given args.

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    logger.info("Starting model training")
    
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # Get the training and evaluation datasets
    train_dataset = _input_fn(
        file_pattern=fn_args.train_files,
        tf_transform_output=tf_transform_output,
        batch_size=fn_args.custom_config["train_batch_size"]
    )
    
    eval_dataset = _input_fn(
        file_pattern=fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        batch_size=fn_args.custom_config["eval_batch_size"]
    )
    
    # Build the model
    model = _build_keras_model(
        hidden_units=fn_args.custom_config["hidden_units"],
        learning_rate=fn_args.custom_config["learning_rate"],
        dropout_rate=fn_args.custom_config["dropout_rate"]
    )
    
    # Define callbacks for training
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir, update_freq="batch"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(fn_args.model_run_dir, "model_checkpoint"),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        )
    ]
    
    # Train the model
    logger.info("Training model...")
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        callbacks=callbacks,
        epochs=100,  # We'll rely on early stopping
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps
    )
    
    # Define the serving signature
    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name="examples"
            )
        )
    }
    
    # Save the model
    logger.info(f"Saving model to {fn_args.serving_model_dir}")
    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures
    )
    
    logger.info("Model training completed successfully")


def _example_serving_receiver_fn(
    tf_transform_output: tft.TFTransformOutput,
    schema: schema_utils.schema_pb2.Schema
) -> tf.estimator.export.ServingInputReceiver:
    """Build the serving inputs.

    Args:
        tf_transform_output: A TFTransformOutput.
        schema: The schema of the input data.

    Returns:
        A ServingInputReceiver.
    """
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since it is not available during serving
    raw_feature_spec.pop(LABEL_KEY)
    
    # Create the serving input receiver
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None
    )
    serving_input_receiver = raw_input_fn()
    
    # Transform the raw features using the transform output
    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features
    )
    
    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors
    )

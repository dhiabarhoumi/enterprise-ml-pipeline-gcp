"""TFX Transform module for feature engineering."""

import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict, List, Text, Any


# Feature names according to the dataset schema
NUMERIC_FEATURES = [
    'quantity', 'unit_price', 'discount', 'customer_age', 'transaction_hour'
]

CATEGORICAL_FEATURES = [
    'customer_id', 'product_id', 'customer_gender', 'store_id', 
    'payment_method', 'customer_segment', 'transaction_day', 'transaction_month'
]

# Target feature to predict
LABEL_KEY = 'total_amount'


def _fill_in_missing(x):
    """Replace missing values with a default value.

    Args:
        x: Input tensor

    Returns:
        Tensor with missing values filled in
    """
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = '' if x.dtype == tf.string else 0
        return tf.sparse.to_dense(x, default_value=default_value)
    else:
        return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def preprocessing_fn(inputs: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """TFX Transform preprocessing function.

    This function defines the feature engineering logic for the model.

    Args:
        inputs: Dictionary of input tensors from the pipeline

    Returns:
        Dictionary of transformed tensors
    """
    outputs = {}

    # Handle numeric features: scale to [0, 1] range
    for feature_name in NUMERIC_FEATURES:
        # Fill in missing values
        outputs[feature_name] = _fill_in_missing(inputs[feature_name])
        # Scale numeric features to [0, 1]
        outputs[feature_name] = tft.scale_to_0_1(outputs[feature_name])

    # Handle categorical features: convert to indices
    for feature_name in CATEGORICAL_FEATURES:
        # Fill in missing values
        outputs[feature_name] = _fill_in_missing(inputs[feature_name])
        # Convert to categorical indices
        outputs[feature_name] = tft.compute_and_apply_vocabulary(
            outputs[feature_name], vocab_filename=feature_name)
    
    # Handle the label
    outputs[LABEL_KEY] = _fill_in_missing(inputs[LABEL_KEY])
    
    # Feature engineering: create new features
    
    # Create price per unit feature
    outputs['price_per_unit'] = outputs['unit_price'] * (1.0 - outputs['discount'])
    
    # Create weekend feature (1 if weekend, 0 otherwise)
    outputs['is_weekend'] = tf.cast(
        tf.logical_or(
            tf.equal(tf.cast(inputs['transaction_day'], tf.string), 'Saturday'),
            tf.equal(tf.cast(inputs['transaction_day'], tf.string), 'Sunday')
        ),
        tf.float32
    )
    
    # Create time of day feature (morning, afternoon, evening)
    hour = tf.cast(inputs['transaction_hour'], tf.int32)
    outputs['time_of_day'] = tf.cast(
        tf.case(
            [
                (tf.logical_and(hour >= 5, hour < 12), lambda: tf.constant(0)),  # morning
                (tf.logical_and(hour >= 12, hour < 18), lambda: tf.constant(1)),  # afternoon
            ],
            default=lambda: tf.constant(2)  # evening
        ),
        tf.int64
    )
    
    return outputs


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generates features and label for tuning/training."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        label_key=LABEL_KEY)

    return dataset


def _build_keras_model(hidden_units, learning_rate, dropout_rate=0.2):
    """Creates a DNN Keras model for predicting total amount."""
    # Define input layers
    input_layers = {}
    feature_layers = []

    # Numeric features
    for feature_name in NUMERIC_FEATURES:
        input_layers[feature_name] = tf.keras.layers.Input(
            shape=(1,), name=feature_name)
        feature_layers.append(input_layers[feature_name])

    # Categorical features
    for feature_name in CATEGORICAL_FEATURES:
        input_layers[feature_name] = tf.keras.layers.Input(
            shape=(1,), name=feature_name, dtype=tf.int64)
        embedding_size = max(1, int(tf.sqrt(float(3))))
        embedding = tf.keras.layers.Embedding(
            input_dim=1000,  # Will be set by TFT
            output_dim=embedding_size)(
                input_layers[feature_name])
        feature_layers.append(tf.keras.layers.Flatten()(embedding))

    # Additional engineered features
    input_layers['price_per_unit'] = tf.keras.layers.Input(
        shape=(1,), name='price_per_unit')
    feature_layers.append(input_layers['price_per_unit'])

    input_layers['is_weekend'] = tf.keras.layers.Input(
        shape=(1,), name='is_weekend')
    feature_layers.append(input_layers['is_weekend'])

    input_layers['time_of_day'] = tf.keras.layers.Input(
        shape=(1,), name='time_of_day', dtype=tf.int64)
    embedding = tf.keras.layers.Embedding(
        input_dim=3, output_dim=2)(
            input_layers['time_of_day'])
    feature_layers.append(tf.keras.layers.Flatten()(embedding))

    # Combine all features
    concatenated_features = tf.keras.layers.concatenate(feature_layers)
    deep = concatenated_features

    # Hidden layers
    for units in hidden_units:
        deep = tf.keras.layers.Dense(units, activation='relu')(deep)
        deep = tf.keras.layers.Dropout(dropout_rate)(deep)

    # Output layer
    output = tf.keras.layers.Dense(1)(deep)

    # Create model
    model = tf.keras.Model(inputs=input_layers, outputs=output)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model

"""Data validation module using TensorFlow Extended (TFX)."""

import logging
from typing import Dict, List, Optional, Union, Tuple
import os

import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import schema_util

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation using TensorFlow Data Validation."""

    def __init__(self, output_dir: str = "data_validation"):
        """Initialize the data validator.

        Args:
            output_dir: Directory to save validation artifacts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized DataValidator with output directory: {output_dir}")

    def generate_statistics(self, data: Union[pd.DataFrame, str]) -> tfdv.types.DatasetFeatureStatisticsList:
        """Generate statistics for a dataset.

        Args:
            data: Either a pandas DataFrame or path to a TFRecord file

        Returns:
            Dataset statistics
        """
        if isinstance(data, pd.DataFrame):
            stats = tfdv.generate_statistics_from_dataframe(data)
            logger.info(f"Generated statistics from DataFrame with {len(data)} rows")
        else:
            stats = tfdv.generate_statistics_from_tfrecord(data)
            logger.info(f"Generated statistics from TFRecord file: {data}")

        # Save statistics to output directory
        output_path = os.path.join(self.output_dir, "statistics.pb")
        tfdv.write_stats_text(stats, output_path)
        logger.info(f"Saved statistics to {output_path}")

        return stats

    def infer_schema(self, statistics: tfdv.types.DatasetFeatureStatisticsList) -> tfdv.types.Schema:
        """Infer a schema from dataset statistics.

        Args:
            statistics: Dataset statistics

        Returns:
            Inferred schema
        """
        schema = tfdv.infer_schema(statistics)
        
        # Save schema to output directory
        output_path = os.path.join(self.output_dir, "schema.pbtxt")
        tfdv.write_schema_text(schema, output_path)
        logger.info(f"Inferred and saved schema to {output_path}")
        
        return schema

    def validate_dataset(
        self, 
        data: Union[pd.DataFrame, str], 
        schema: Optional[tfdv.types.Schema] = None,
        environment: Optional[str] = None
    ) -> Tuple[tfdv.types.DatasetFeatureStatisticsList, List[tfdv.types.Anomaly]]:
        """Validate a dataset against a schema.

        Args:
            data: Either a pandas DataFrame or path to a TFRecord file
            schema: Schema to validate against (if None, will be inferred)
            environment: Optional environment to condition validation on

        Returns:
            Tuple of (statistics, anomalies)
        """
        # Generate statistics
        stats = self.generate_statistics(data)
        
        # Infer schema if not provided
        if schema is None:
            schema = self.infer_schema(stats)
            
        # Validate statistics against schema
        anomalies = tfdv.validate_statistics(stats, schema, environment=environment)
        
        # Save anomalies to output directory
        output_path = os.path.join(self.output_dir, "anomalies.pbtxt")
        tfdv.write_anomalies_text(anomalies, output_path)
        
        if anomalies.anomaly_info:
            logger.warning(f"Found {len(anomalies.anomaly_info)} anomalies in the dataset")
        else:
            logger.info("No anomalies found in the dataset")
            
        return stats, anomalies.anomaly_info

    def update_schema_for_production(
        self, 
        schema: tfdv.types.Schema,
        categorical_features: Optional[List[str]] = None,
        required_features: Optional[List[str]] = None,
        numeric_features_with_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> tfdv.types.Schema:
        """Update schema with production constraints.

        Args:
            schema: Base schema to update
            categorical_features: Features to mark as categorical
            required_features: Features to mark as required
            numeric_features_with_bounds: Dict mapping feature names to (min, max) bounds

        Returns:
            Updated schema
        """
        # Make a copy of the schema
        updated_schema = schema
        
        # Set features as categorical
        if categorical_features:
            for feature_name in categorical_features:
                try:
                    schema_util.set_domain(updated_schema, feature_name, schema_util.categorical_domain)
                    logger.info(f"Set {feature_name} as categorical feature")
                except Exception as e:
                    logger.error(f"Error setting {feature_name} as categorical: {e}")
        
        # Set features as required
        if required_features:
            for feature_name in required_features:
                try:
                    schema_util.get_feature(updated_schema, feature_name).presence.min_fraction = 1.0
                    logger.info(f"Set {feature_name} as required feature")
                except Exception as e:
                    logger.error(f"Error setting {feature_name} as required: {e}")
        
        # Set numeric bounds
        if numeric_features_with_bounds:
            for feature_name, (min_val, max_val) in numeric_features_with_bounds.items():
                try:
                    schema_util.set_domain(
                        updated_schema, 
                        feature_name, 
                        schema_util.float_domain(min_val, max_val)
                    )
                    logger.info(f"Set bounds for {feature_name}: [{min_val}, {max_val}]")
                except Exception as e:
                    logger.error(f"Error setting bounds for {feature_name}: {e}")
        
        # Save updated schema
        output_path = os.path.join(self.output_dir, "updated_schema.pbtxt")
        tfdv.write_schema_text(updated_schema, output_path)
        logger.info(f"Saved updated schema to {output_path}")
        
        return updated_schema

    def compare_statistics(
        self, 
        train_stats: tfdv.types.DatasetFeatureStatisticsList,
        eval_stats: tfdv.types.DatasetFeatureStatisticsList,
        schema: Optional[tfdv.types.Schema] = None
    ) -> List[tfdv.types.Anomaly]:
        """Compare statistics between training and evaluation datasets.

        Args:
            train_stats: Training dataset statistics
            eval_stats: Evaluation dataset statistics
            schema: Optional schema to use for comparison

        Returns:
            List of anomalies between the datasets
        """
        if schema is None:
            schema = self.infer_schema(train_stats)
        
        # Set environment in schema
        schema_util.set_domain(
            schema, 
            "environment", 
            schema_util.string_domain(["TRAINING", "EVALUATION"])
        )
        
        # Set default environment
        schema.default_environment.append("TRAINING")
        schema.default_environment.append("EVALUATION")
        
        # Compare statistics
        anomalies = tfdv.validate_statistics(eval_stats, schema, environment="EVALUATION")
        
        if anomalies.anomaly_info:
            logger.warning(
                f"Found {len(anomalies.anomaly_info)} anomalies between training and evaluation data"
            )
        else:
            logger.info("No anomalies found between training and evaluation data")
            
        return anomalies.anomaly_info

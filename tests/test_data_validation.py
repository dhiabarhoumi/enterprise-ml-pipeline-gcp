"""Tests for data validation module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

from src.data.data_validation import DataValidator


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        
        # Create a simple test dataframe
        self.test_df = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'integer_feature': [1, 2, 3, 4, 5],
            'categorical_feature': ['a', 'b', 'a', 'c', 'b'],
            'string_feature': ['text1', 'text2', 'text3', 'text4', 'text5'],
            'binary_feature': [True, False, True, False, True],
            'target': [10.5, 20.3, 30.1, 40.7, 50.2]
        })

    @patch('tensorflow_data_validation.generate_statistics_from_dataframe')
    def test_generate_statistics(self, mock_generate_stats):
        """Test generating statistics from a dataframe."""
        # Create a mock statistics proto
        mock_stats = MagicMock()
        mock_generate_stats.return_value = mock_stats
        
        # Call the method
        stats = self.validator.generate_statistics(self.test_df)
        
        # Verify the mock was called with the dataframe
        mock_generate_stats.assert_called_once()
        self.assertEqual(stats, mock_stats)

    @patch('tensorflow_data_validation.infer_schema')
    def test_infer_schema(self, mock_infer_schema):
        """Test inferring schema from statistics."""
        # Create mock objects
        mock_stats = MagicMock()
        mock_schema = MagicMock()
        mock_infer_schema.return_value = mock_schema
        
        # Call the method
        schema = self.validator.infer_schema(mock_stats)
        
        # Verify the mock was called with the stats
        mock_infer_schema.assert_called_once_with(mock_stats)
        self.assertEqual(schema, mock_schema)

    @patch('tensorflow_data_validation.validate_statistics')
    def test_validate_dataset(self, mock_validate_statistics):
        """Test validating dataset statistics against a schema."""
        # Create mock objects
        mock_stats = MagicMock()
        mock_schema = MagicMock()
        mock_anomalies = MagicMock()
        mock_validate_statistics.return_value = mock_anomalies
        
        # Call the method
        anomalies = self.validator.validate_dataset(mock_stats, mock_schema)
        
        # Verify the mock was called with the stats and schema
        mock_validate_statistics.assert_called_once_with(mock_stats, mock_schema)
        self.assertEqual(anomalies, mock_anomalies)

    @patch('tensorflow_data_validation.update_schema')
    def test_update_schema(self, mock_update_schema):
        """Test updating schema with anomalies."""
        # Create mock objects
        mock_schema = MagicMock()
        mock_stats = MagicMock()
        mock_anomalies = MagicMock()
        mock_updated_schema = MagicMock()
        mock_update_schema.return_value = mock_updated_schema
        
        # Call the method
        updated_schema = self.validator.update_schema(
            mock_schema, mock_stats, mock_anomalies)
        
        # Verify the mock was called with the schema, stats, and anomalies
        mock_update_schema.assert_called_once_with(
            mock_schema, mock_stats, mock_anomalies)
        self.assertEqual(updated_schema, mock_updated_schema)

    @patch('tensorflow_data_validation.validate_instance')
    def test_validate_instance(self, mock_validate_instance):
        """Test validating a single instance against a schema."""
        # Create mock objects
        mock_instance = {'feature1': 1, 'feature2': 'value'}
        mock_schema = MagicMock()
        mock_anomalies = MagicMock()
        mock_validate_instance.return_value = mock_anomalies
        
        # Call the method
        anomalies = self.validator.validate_instance(mock_instance, mock_schema)
        
        # Verify the mock was called with the instance and schema
        mock_validate_instance.assert_called_once()
        self.assertEqual(anomalies, mock_anomalies)

    @patch('tensorflow_data_validation.display_schema')
    def test_display_schema(self, mock_display_schema):
        """Test displaying schema."""
        # Create a mock schema
        mock_schema = MagicMock()
        
        # Call the method
        self.validator.display_schema(mock_schema)
        
        # Verify the mock was called with the schema
        mock_display_schema.assert_called_once_with(mock_schema)

    @patch('tensorflow_data_validation.display_stats')
    def test_display_statistics(self, mock_display_stats):
        """Test displaying statistics."""
        # Create a mock statistics proto
        mock_stats = MagicMock()
        
        # Call the method
        self.validator.display_statistics(mock_stats)
        
        # Verify the mock was called with the stats
        mock_display_stats.assert_called_once_with(mock_stats)

    @patch('tensorflow_data_validation.display_anomalies')
    def test_display_anomalies(self, mock_display_anomalies):
        """Test displaying anomalies."""
        # Create a mock anomalies proto
        mock_anomalies = MagicMock()
        
        # Call the method
        self.validator.display_anomalies(mock_anomalies)
        
        # Verify the mock was called with the anomalies
        mock_display_anomalies.assert_called_once_with(mock_anomalies)

    @patch('tensorflow_data_validation.compare_statistics')
    def test_compare_statistics(self, mock_compare_statistics):
        """Test comparing statistics."""
        # Create mock objects
        mock_stats1 = MagicMock()
        mock_stats2 = MagicMock()
        mock_schema = MagicMock()
        
        # Call the method
        self.validator.compare_statistics(mock_stats1, mock_stats2, mock_schema)
        
        # Verify the mock was called with the stats and schema
        mock_compare_statistics.assert_called_once_with(
            mock_stats1, mock_stats2, mock_schema)

    def test_save_and_load_schema(self):
        """Test saving and loading schema."""
        # Create a simple schema
        feature = schema_pb2.Feature(name="feature1")
        schema = schema_pb2.Schema(feature=[feature])
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pbtxt", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the schema
            self.validator.save_schema(schema, temp_path)
            
            # Load the schema
            loaded_schema = self.validator.load_schema(temp_path)
            
            # Verify the loaded schema has the same feature
            self.assertEqual(len(loaded_schema.feature), 1)
            self.assertEqual(loaded_schema.feature[0].name, "feature1")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

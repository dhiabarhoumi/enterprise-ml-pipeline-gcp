"""Tests for BigQuery utilities module."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.data.bigquery_utils import BigQueryClient


class TestBigQueryClient(unittest.TestCase):
    """Test cases for BigQueryClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_id = "test-project"
        
        # Create a patch for the BigQuery client
        self.bq_client_patcher = patch('google.cloud.bigquery.Client')
        self.mock_bq_client = self.bq_client_patcher.start()
        
        # Create the BigQueryClient instance
        self.client = BigQueryClient(project_id=self.project_id)
        
        # Create a test dataframe
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })

    def tearDown(self):
        """Tear down test fixtures."""
        self.bq_client_patcher.stop()

    def test_init(self):
        """Test initialization of BigQueryClient."""
        self.assertEqual(self.client.project_id, self.project_id)
        self.mock_bq_client.assert_called_once_with(project=self.project_id)

    def test_create_dataset(self):
        """Test creating a BigQuery dataset."""
        dataset_id = "test_dataset"
        
        # Call the method
        self.client.create_dataset(dataset_id)
        
        # Verify the client's create_dataset method was called
        self.client.client.create_dataset.assert_called_once()

    def test_execute_query(self):
        """Test executing a BigQuery query."""
        query = "SELECT * FROM `test-project.test_dataset.test_table`"
        
        # Set up the mock query job
        mock_query_job = MagicMock()
        mock_result = MagicMock()
        mock_query_job.result.return_value = mock_result
        self.client.client.query.return_value = mock_query_job
        
        # Call the method
        result = self.client.execute_query(query)
        
        # Verify the client's query method was called with the query
        self.client.client.query.assert_called_once_with(query)
        self.assertEqual(result, mock_result)

    def test_query_to_dataframe(self):
        """Test executing a query and returning the results as a DataFrame."""
        query = "SELECT * FROM `test-project.test_dataset.test_table`"
        
        # Set up the mock query job
        mock_query_job = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = self.test_df
        mock_query_job.result.return_value = mock_result
        self.client.client.query.return_value = mock_query_job
        
        # Call the method
        df = self.client.query_to_dataframe(query)
        
        # Verify the client's query method was called with the query
        self.client.client.query.assert_called_once_with(query)
        self.assertTrue(df.equals(self.test_df))

    def test_upload_dataframe_to_table(self):
        """Test uploading a DataFrame to a BigQuery table."""
        dataset_id = "test_dataset"
        table_id = "test_table"
        
        # Call the method
        self.client.upload_dataframe_to_table(self.test_df, dataset_id, table_id)
        
        # Verify the client's load_table_from_dataframe method was called
        self.client.client.load_table_from_dataframe.assert_called_once()

    def test_get_table_schema(self):
        """Test getting the schema of a BigQuery table."""
        dataset_id = "test_dataset"
        table_id = "test_table"
        
        # Set up the mock table reference and schema
        mock_table_ref = MagicMock()
        mock_table = MagicMock()
        mock_schema = MagicMock()
        mock_table.schema = mock_schema
        self.client.client.get_table.return_value = mock_table
        
        # Call the method
        schema = self.client.get_table_schema(dataset_id, table_id)
        
        # Verify the client's get_table method was called
        self.client.client.get_table.assert_called_once()
        self.assertEqual(schema, mock_schema)

    def test_optimize_query(self):
        """Test optimizing a BigQuery query."""
        query = "SELECT * FROM `test-project.test_dataset.test_table`"
        
        # Call the method
        optimized_query = self.client.optimize_query(query)
        
        # Verify the query was optimized
        self.assertIn("SELECT", optimized_query)
        self.assertIn("FROM", optimized_query)

    def test_check_table_exists(self):
        """Test checking if a table exists."""
        dataset_id = "test_dataset"
        table_id = "test_table"
        
        # Set up the mock table reference
        self.client.client.get_table.side_effect = Exception("Table not found")
        
        # Call the method
        exists = self.client.check_table_exists(dataset_id, table_id)
        
        # Verify the client's get_table method was called
        self.client.client.get_table.assert_called_once()
        self.assertFalse(exists)

    def test_delete_table(self):
        """Test deleting a BigQuery table."""
        dataset_id = "test_dataset"
        table_id = "test_table"
        
        # Call the method
        self.client.delete_table(dataset_id, table_id)
        
        # Verify the client's delete_table method was called
        self.client.client.delete_table.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""BigQuery utilities for the Enterprise ML Pipeline."""

import logging
from typing import Dict, List, Optional, Union

from google.cloud import bigquery
import pandas as pd

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Wrapper for BigQuery operations."""

    def __init__(self, project_id: str):
        """Initialize the BigQuery client.

        Args:
            project_id: GCP project ID
        """
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        logger.info(f"Initialized BigQuery client for project {project_id}")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a BigQuery SQL query and return results as a DataFrame.

        Args:
            query: SQL query string

        Returns:
            DataFrame containing query results
        """
        logger.info("Executing BigQuery query")
        query_job = self.client.query(query)
        return query_job.to_dataframe()

    def create_dataset(self, dataset_id: str, location: str = "US") -> None:
        """Create a new BigQuery dataset.

        Args:
            dataset_id: ID for the new dataset
            location: Dataset location (default: US)
        """
        dataset_ref = self.client.dataset(dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location

        try:
            self.client.create_dataset(dataset)
            logger.info(f"Created dataset {self.project_id}.{dataset_id}")
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise

    def upload_dataframe(
        self,
        dataframe: pd.DataFrame,
        dataset_id: str,
        table_id: str,
        schema: Optional[List[bigquery.SchemaField]] = None,
        write_disposition: str = "WRITE_TRUNCATE",
    ) -> None:
        """Upload a pandas DataFrame to BigQuery.

        Args:
            dataframe: Pandas DataFrame to upload
            dataset_id: Target dataset ID
            table_id: Target table ID
            schema: Optional BigQuery schema definition
            write_disposition: Write disposition (default: WRITE_TRUNCATE)
        """
        table_ref = self.client.dataset(dataset_id).table(table_id)
        job_config = bigquery.LoadJobConfig()

        if schema:
            job_config.schema = schema
        else:
            job_config.autodetect = True

        job_config.write_disposition = write_disposition

        try:
            job = self.client.load_table_from_dataframe(
                dataframe, table_ref, job_config=job_config
            )
            job.result()  # Wait for the job to complete
            logger.info(
                f"Uploaded DataFrame to {self.project_id}.{dataset_id}.{table_id}"
            )
        except Exception as e:
            logger.error(f"Error uploading DataFrame: {e}")
            raise

    def get_table_schema(self, dataset_id: str, table_id: str) -> List[bigquery.SchemaField]:
        """Get the schema for a BigQuery table.

        Args:
            dataset_id: Dataset ID
            table_id: Table ID

        Returns:
            List of SchemaField objects describing the table schema
        """
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        return table.schema

    def optimize_query_performance(self, query: str) -> str:
        """Apply optimization techniques to a BigQuery query.

        This is a simple implementation that applies some basic optimizations.
        In a production system, this would be more sophisticated.

        Args:
            query: Original SQL query

        Returns:
            Optimized SQL query
        """
        # This is a simplified example - in a real system, you'd have more
        # sophisticated query optimization logic
        optimized_query = query

        # Add query hints for better performance
        if "SELECT" in optimized_query and "FROM" in optimized_query:
            optimized_query = optimized_query.replace(
                "SELECT", "SELECT /*+ OPTIMIZE_JOIN_ORDER */ "
            )

        logger.info("Query optimized for performance")
        return optimized_query

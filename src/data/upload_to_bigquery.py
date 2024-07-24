"""Script to upload sample data to BigQuery."""

import argparse
import logging
import os
from typing import Optional

import pandas as pd
import numpy as np

from src.data.bigquery_utils import BigQueryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_retail_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic retail dataset for demonstration purposes.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with synthetic retail data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    
    # Generate product IDs (fewer products than customers)
    product_ids = [f"PROD_{i:04d}" for i in range(1, 101)]
    
    # Generate random data
    data = {
        "customer_id": np.random.choice(customer_ids, n_samples),
        "product_id": np.random.choice(product_ids, n_samples),
        "transaction_date": pd.date_range(
            start="2024-01-01", periods=n_samples, freq="H"
        ),
        "quantity": np.random.randint(1, 10, n_samples),
        "unit_price": np.round(np.random.uniform(5.0, 100.0, n_samples), 2),
        "discount": np.round(np.random.uniform(0.0, 0.3, n_samples), 2),
        "customer_age": np.random.randint(18, 80, n_samples),
        "customer_gender": np.random.choice(["M", "F", "Other"], n_samples),
        "store_id": np.random.choice([f"STORE_{i:02d}" for i in range(1, 11)], n_samples),
        "payment_method": np.random.choice(
            ["Credit Card", "Debit Card", "Cash", "Digital Wallet"], n_samples
        ),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate total amount
    df["total_amount"] = df["quantity"] * df["unit_price"] * (1 - df["discount"])
    
    # Add some categorical features
    df["customer_segment"] = pd.qcut(
        df["customer_age"], 
        q=4, 
        labels=["Young Adult", "Adult", "Middle Age", "Senior"]
    )
    
    # Add time-based features
    df["transaction_hour"] = df["transaction_date"].dt.hour
    df["transaction_day"] = df["transaction_date"].dt.day_name()
    df["transaction_month"] = df["transaction_date"].dt.month_name()
    
    logger.info(f"Generated {n_samples} synthetic retail data samples")
    return df


def save_sample_data(df: pd.DataFrame, output_dir: str) -> str:
    """Save sample data to CSV file.

    Args:
        df: DataFrame to save
        output_dir: Directory to save the CSV file

    Returns:
        Path to the saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "retail_data_sample.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sample data to {output_path}")
    return output_path


def upload_to_bigquery(
    project_id: str,
    dataset_id: str,
    table_id: str = "retail_transactions",
    data_path: Optional[str] = None,
    n_samples: int = 1000,
) -> None:
    """Upload sample data to BigQuery.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        data_path: Path to CSV file (if None, generates synthetic data)
        n_samples: Number of samples to generate if data_path is None
    """
    # Initialize BigQuery client
    bq_client = BigQueryClient(project_id)
    
    # Create dataset if it doesn't exist
    try:
        bq_client.create_dataset(dataset_id)
    except Exception as e:
        logger.warning(f"Dataset creation error (may already exist): {e}")
    
    # Load or generate data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info(f"Generating {n_samples} synthetic data samples")
        df = generate_sample_retail_data(n_samples)
        # Save generated data
        save_sample_data(df, "data")
    
    # Upload data to BigQuery
    logger.info(f"Uploading data to {project_id}.{dataset_id}.{table_id}")
    bq_client.upload_dataframe(
        dataframe=df,
        dataset_id=dataset_id,
        table_id=table_id,
        write_disposition="WRITE_TRUNCATE"
    )
    logger.info("Upload completed successfully")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Upload sample data to BigQuery")
    parser.add_argument(
        "--project_id", 
        required=True, 
        help="GCP project ID"
    )
    parser.add_argument(
        "--dataset", 
        default="retail_dataset", 
        help="BigQuery dataset ID"
    )
    parser.add_argument(
        "--table", 
        default="retail_transactions", 
        help="BigQuery table ID"
    )
    parser.add_argument(
        "--data_path", 
        help="Path to CSV file (if not provided, generates synthetic data)"
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=1000, 
        help="Number of samples to generate if data_path is not provided"
    )
    
    args = parser.parse_args()
    
    upload_to_bigquery(
        project_id=args.project_id,
        dataset_id=args.dataset,
        table_id=args.table,
        data_path=args.data_path,
        n_samples=args.n_samples
    )


if __name__ == "__main__":
    main()

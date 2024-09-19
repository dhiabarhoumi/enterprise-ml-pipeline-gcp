"""Example script for model inference using the trained model."""

import argparse
import json
import logging
import os
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import tensorflow as tf

from src.serving.prediction import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_data(num_samples: int = 5) -> List[Dict[str, Any]]:
    """Generate sample data for inference.

    Args:
        num_samples: Number of samples to generate

    Returns:
        List of feature dictionaries
    """
    # Define possible values for categorical features
    customer_ids = [f"CUST_{i:04d}" for i in range(1, 101)]
    product_ids = [f"PROD_{i:04d}" for i in range(1, 501)]
    genders = ["M", "F"]
    store_ids = [f"STORE_{i:02d}" for i in range(1, 11)]
    payment_methods = ["Credit Card", "Debit Card", "Cash", "Mobile Payment"]
    customer_segments = ["Regular", "Premium", "New", "VIP"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["January", "February", "March", "April", "May", "June", 
             "July", "August", "September", "October", "November", "December"]
    
    # Generate samples
    samples = []
    for _ in range(num_samples):
        sample = {
            "quantity": np.random.randint(1, 20),
            "unit_price": round(np.random.uniform(10.0, 200.0), 2),
            "discount": round(np.random.uniform(0.0, 0.5), 2),
            "customer_age": np.random.randint(18, 80),
            "transaction_hour": np.random.randint(8, 22),
            "customer_id": np.random.choice(customer_ids),
            "product_id": np.random.choice(product_ids),
            "customer_gender": np.random.choice(genders),
            "store_id": np.random.choice(store_ids),
            "payment_method": np.random.choice(payment_methods),
            "customer_segment": np.random.choice(customer_segments),
            "transaction_day": np.random.choice(days),
            "transaction_month": np.random.choice(months)
        }
        samples.append(sample)
    
    return samples


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Model inference example")
    parser.add_argument(
        "--model_path", 
        default="models/serving_model",
        help="Path to the saved model"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5,
        help="Number of sample instances to generate"
    )
    parser.add_argument(
        "--output_file", 
        help="Path to save prediction results as JSON"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        logger.info("This is an example script. You need to train a model first.")
        logger.info("Run the pipeline with: python src/pipeline/run_pipeline.py")
        return
    
    # Initialize prediction service
    logger.info(f"Loading model from {args.model_path}")
    prediction_service = PredictionService(args.model_path)
    
    # Generate sample data
    logger.info(f"Generating {args.num_samples} sample instances")
    samples = generate_sample_data(args.num_samples)
    
    # Make predictions
    logger.info("Making predictions")
    try:
        predictions = prediction_service.predict(samples)
        
        # Create results
        results = []
        for i, (sample, prediction) in enumerate(zip(samples, predictions)):
            result = {
                "instance": sample,
                "prediction": float(prediction[0]),
                "expected_total": round(sample["quantity"] * sample["unit_price"] * (1 - sample["discount"]), 2)
            }
            results.append(result)
            logger.info(f"Sample {i+1}: Predicted=${result['prediction']:.2f}, Expected=${result['expected_total']:.2f}")
        
        # Save results if output file is specified
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")


if __name__ == "__main__":
    main()

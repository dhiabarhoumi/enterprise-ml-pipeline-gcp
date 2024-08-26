"""Script to submit the ML pipeline to Vertex AI."""

import argparse
import logging
import os
import uuid
from typing import Dict, List, Optional, Any

from google.cloud import aiplatform
from google.cloud.aiplatform.pipeline_jobs import PipelineJob
from kfp.v2 import compiler
from google.cloud.aiplatform import telemetry

from src.pipeline.config import PipelineConfig
from src.pipeline.run_pipeline import create_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compile_pipeline(config: PipelineConfig, output_path: str) -> None:
    """Compile the TFX pipeline to a JSON file.

    Args:
        config: Pipeline configuration
        output_path: Path to save the compiled pipeline
    """
    logger.info(f"Compiling pipeline to {output_path}")
    
    # Create the pipeline
    pipeline = create_pipeline(
        config=config,
        data_path=config.data_root,
        enable_cache=True
    )
    
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=lambda: pipeline,
        package_path=output_path
    )
    
    logger.info(f"Pipeline compiled successfully to {output_path}")


def submit_pipeline_job(
    config: PipelineConfig,
    pipeline_path: str,
    job_id: Optional[str] = None,
    enable_caching: bool = True
) -> PipelineJob:
    """Submit a pipeline job to Vertex AI.

    Args:
        config: Pipeline configuration
        pipeline_path: Path to the compiled pipeline
        job_id: Optional job ID
        enable_caching: Whether to enable caching

    Returns:
        The submitted pipeline job
    """
    # Initialize Vertex AI with telemetry
    telemetry.start()
    aiplatform.init(
        project=config.project_id,
        location=config.region,
        staging_bucket=config.vertex_ai_config["staging_bucket"]
    )
    
    # Set job display name
    if job_id:
        display_name = f"{config.pipeline_name}-{job_id}"
    else:
        display_name = f"{config.pipeline_name}-{uuid.uuid4().hex[:8]}"
    
    # Set pipeline parameters
    pipeline_parameters = {
        "data_path": config.data_root,
        "train_steps": config.train_steps,
        "eval_steps": config.eval_steps,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "hidden_units": str(config.hidden_units)  # Convert list to string
    }
    
    # Set service account if available
    service_account = config.vertex_ai_config.get("service_account", None)
    
    # Submit the pipeline job
    logger.info(f"Submitting pipeline job: {display_name}")
    job = PipelineJob(
        display_name=display_name,
        template_path=pipeline_path,
        pipeline_root=config.pipeline_root,
        parameter_values=pipeline_parameters,
        enable_caching=enable_caching,
        service_account=service_account,
        labels={"environment": "production", "pipeline": "enterprise-ml"}
    )
    
    job.submit(sync=False)  # Async submission
    logger.info(f"Pipeline job submitted with ID: {job.name}")
    
    return job


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Submit ML pipeline to Vertex AI"
    )
    parser.add_argument(
        "--project_id", 
        required=True, 
        help="GCP project ID"
    )
    parser.add_argument(
        "--region", 
        required=True, 
        help="GCP region"
    )
    parser.add_argument(
        "--pipeline_name", 
        default="enterprise-ml-pipeline", 
        help="Name of the pipeline"
    )
    parser.add_argument(
        "--job_id", 
        help="Optional job ID suffix"
    )
    parser.add_argument(
        "--data_path", 
        help="Path to the data directory"
    )
    parser.add_argument(
        "--output_path", 
        default="pipeline.json", 
        help="Path to save the compiled pipeline"
    )
    parser.add_argument(
        "--disable_cache", 
        action="store_true", 
        help="Disable caching"
    )
    
    args = parser.parse_args()
    
    # Create pipeline configuration
    config = PipelineConfig(
        project_id=args.project_id,
        region=args.region,
        pipeline_name=args.pipeline_name
    )
    
    # Set data path if provided
    if args.data_path:
        config.data_root = args.data_path
    
    # Compile the pipeline
    compile_pipeline(config, args.output_path)
    
    # Submit the pipeline job
    job = submit_pipeline_job(
        config=config,
        pipeline_path=args.output_path,
        job_id=args.job_id,
        enable_caching=not args.disable_cache
    )
    
    # Print job information
    print(f"\nPipeline job submitted successfully!")
    print(f"Job ID: {job.name}")
    print(f"Job URL: https://console.cloud.google.com/vertex-ai/locations/{config.region}/pipelines/runs/{job.name}?project={config.project_id}")
    
    # Add monitoring instructions
    print(f"\nTo monitor the pipeline run:")
    print(f"  - View in Google Cloud Console: https://console.cloud.google.com/vertex-ai/locations/{config.region}/pipelines/runs/{job.name}?project={config.project_id}")
    print(f"  - Check status with gcloud: gcloud ai pipeline-jobs describe {job.name} --project={config.project_id} --region={config.region}")
    print(f"  - Stream logs: gcloud ai pipeline-jobs stream-logs {job.name} --project={config.project_id} --region={config.region}")


if __name__ == "__main__":
    main()

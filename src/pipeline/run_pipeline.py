"""Pipeline runner for the Enterprise ML Pipeline."""

import argparse
import logging
import os
from typing import Dict, List, Optional, Any

import tensorflow_model_analysis as tfma
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.vertex import VertexDagRunner
from tfx.proto import trainer_pb2

from src.pipeline.config import PipelineConfig
from src.pipeline.components import PipelineComponents, create_eval_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_pipeline(
    config: PipelineConfig,
    data_path: str,
    enable_cache: bool = False
) -> pipeline.Pipeline:
    """Create a TFX pipeline.

    Args:
        config: Pipeline configuration
        data_path: Path to the data
        enable_cache: Whether to enable caching

    Returns:
        TFX pipeline
    """
    # Create pipeline components factory
    components = PipelineComponents(
        pipeline_name=config.pipeline_name,
        pipeline_root=config.pipeline_root
    )
    
    # Create individual components
    example_gen = components.create_example_gen(data_path)
    statistics_gen = components.create_statistics_gen(example_gen.outputs["examples"])
    schema_gen = components.create_schema_gen(statistics_gen.outputs["statistics"])
    example_validator = components.create_example_validator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )
    
    # Create transform component
    transform = components.create_transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        transform_module=config.transform_module_file
    )
    
    # Create trainer component
    trainer = components.create_trainer(
        transformed_examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        trainer_module=config.trainer_module_file,
        train_args=config.get_train_args(),
        eval_args=config.get_eval_args(),
        custom_config=config.get_custom_config()
    )
    
    # Create evaluator component
    evaluator = components.create_evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        eval_config=create_eval_config(),
        schema=schema_gen.outputs["schema"]
    )
    
    # Create pusher component
    pusher = components.create_pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        serving_model_dir=config.serving_model_dir
    )
    
    # Define pipeline components in execution order
    pipeline_components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        evaluator,
        pusher
    ]
    
    # Create and return the pipeline
    return pipeline.Pipeline(
        pipeline_name=config.pipeline_name,
        pipeline_root=config.pipeline_root,
        components=pipeline_components,
        enable_cache=enable_cache
    )


def run_pipeline(
    config: PipelineConfig,
    data_path: str,
    mode: str = "local",
    enable_cache: bool = False
) -> None:
    """Run the TFX pipeline.

    Args:
        config: Pipeline configuration
        data_path: Path to the data
        mode: Execution mode ('local' or 'cloud')
        enable_cache: Whether to enable caching
    """
    # Create the pipeline
    tfx_pipeline = create_pipeline(
        config=config,
        data_path=data_path,
        enable_cache=enable_cache
    )
    
    # Run the pipeline
    if mode == "local":
        logger.info("Running pipeline in local mode")
        LocalDagRunner().run(tfx_pipeline)
    elif mode == "cloud":
        logger.info("Running pipeline in cloud mode (Vertex AI)")
        # Set up Vertex AI-specific configurations
        vertex_config = {
            "project": config.project_id,
            "region": config.region,
            "staging_bucket": config.vertex_ai_config["staging_bucket"]
        }
        VertexDagRunner(vertex_config).run(tfx_pipeline)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'local' or 'cloud'.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run the Enterprise ML Pipeline")
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
        "--data_path", 
        help="Path to the data directory"
    )
    parser.add_argument(
        "--pipeline_name", 
        default="enterprise-ml-pipeline", 
        help="Name of the pipeline"
    )
    parser.add_argument(
        "--mode", 
        choices=["local", "cloud"], 
        default="local", 
        help="Execution mode (local or cloud)"
    )
    parser.add_argument(
        "--enable_cache", 
        action="store_true", 
        help="Enable caching"
    )
    
    args = parser.parse_args()
    
    # Create pipeline configuration
    config = PipelineConfig(
        project_id=args.project_id,
        region=args.region,
        pipeline_name=args.pipeline_name
    )
    
    # Determine data path
    data_path = args.data_path or os.path.join(config.data_root, "retail_data")
    
    # Run the pipeline
    run_pipeline(
        config=config,
        data_path=data_path,
        mode=args.mode,
        enable_cache=args.enable_cache
    )


if __name__ == "__main__":
    main()

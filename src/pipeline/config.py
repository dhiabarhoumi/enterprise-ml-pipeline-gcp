"""Configuration for the Enterprise ML Pipeline."""

import os
from typing import Dict, List, Optional, Any

from tfx.proto import trainer_pb2


class PipelineConfig:
    """Configuration for the TFX pipeline."""

    def __init__(
        self,
        project_id: str,
        region: str,
        pipeline_name: str = "enterprise-ml-pipeline",
        pipeline_root: Optional[str] = None,
        data_root: Optional[str] = None,
        module_file_root: Optional[str] = None,
        serving_model_dir: Optional[str] = None,
        vertex_ai_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the pipeline configuration.

        Args:
            project_id: GCP project ID
            region: GCP region
            pipeline_name: Name of the pipeline
            pipeline_root: Root directory for pipeline artifacts
            data_root: Root directory for data
            module_file_root: Root directory for module files
            serving_model_dir: Directory for serving models
            vertex_ai_config: Configuration for Vertex AI
        """
        self.project_id = project_id
        self.region = region
        self.pipeline_name = pipeline_name
        
        # Set default paths if not provided
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.pipeline_root = pipeline_root or os.path.join(
            "gs://", f"{project_id}-ml-pipeline", "pipeline_root"
        )
        self.data_root = data_root or os.path.join(base_dir, "data")
        self.module_file_root = module_file_root or os.path.join(base_dir, "src")
        self.serving_model_dir = serving_model_dir or os.path.join(
            "gs://", f"{project_id}-ml-pipeline", "serving_model"
        )
        
        # Vertex AI configuration
        self.vertex_ai_config = vertex_ai_config or {
            "project": project_id,
            "region": region,
            "staging_bucket": f"gs://{project_id}-ml-pipeline"
        }
        
        # Module file paths
        self.transform_module_file = os.path.join(
            self.module_file_root, "pipeline", "transform.py"
        )
        self.trainer_module_file = os.path.join(
            self.module_file_root, "pipeline", "trainer.py"
        )
        
        # Training and evaluation arguments
        self.train_steps = 1000
        self.eval_steps = 200
        self.train_batch_size = 32
        self.eval_batch_size = 32

    def get_train_args(self) -> trainer_pb2.TrainArgs:
        """Get training arguments.

        Returns:
            TrainArgs proto
        """
        return trainer_pb2.TrainArgs(num_steps=self.train_steps)

    def get_eval_args(self) -> trainer_pb2.EvalArgs:
        """Get evaluation arguments.

        Returns:
            EvalArgs proto
        """
        return trainer_pb2.EvalArgs(num_steps=self.eval_steps)

    def get_custom_config(self) -> Dict[str, Any]:
        """Get custom configuration for components.

        Returns:
            Dictionary of custom configuration
        """
        return {
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "hidden_units": [64, 32],
            "learning_rate": 0.001,
            "dropout_rate": 0.2
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "project_id": self.project_id,
            "region": self.region,
            "pipeline_name": self.pipeline_name,
            "pipeline_root": self.pipeline_root,
            "data_root": self.data_root,
            "module_file_root": self.module_file_root,
            "serving_model_dir": self.serving_model_dir,
            "vertex_ai_config": self.vertex_ai_config,
            "transform_module_file": self.transform_module_file,
            "trainer_module_file": self.trainer_module_file,
            "train_steps": self.train_steps,
            "eval_steps": self.eval_steps,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size
        }

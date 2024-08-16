"""TFX pipeline components for the Enterprise ML Pipeline."""

import logging
from typing import Dict, List, Optional, Union, Any
import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor

logger = logging.getLogger(__name__)


class PipelineComponents:
    """Factory for TFX pipeline components."""

    def __init__(self, pipeline_name: str, pipeline_root: str):
        """Initialize the pipeline components factory.

        Args:
            pipeline_name: Name of the pipeline
            pipeline_root: Root directory for pipeline artifacts
        """
        self.pipeline_name = pipeline_name
        self.pipeline_root = pipeline_root
        logger.info(f"Initialized PipelineComponents for pipeline {pipeline_name}")

    def create_example_gen(self, data_path: str) -> CsvExampleGen:
        """Create an ExampleGen component for CSV data.

        Args:
            data_path: Path to the CSV data file or directory

        Returns:
            CsvExampleGen component
        """
        logger.info(f"Creating ExampleGen component for data at {data_path}")
        
        # Set up input config
        input_config = example_gen_pb2.Input(
            splits=[
                example_gen_pb2.Input.Split(name="train", pattern="*train*"),
                example_gen_pb2.Input.Split(name="eval", pattern="*eval*"),
            ]
        )
        
        # Create the component
        example_gen = CsvExampleGen(input_base=data_path, input_config=input_config)
        
        return example_gen

    def create_statistics_gen(self, examples: Channel) -> StatisticsGen:
        """Create a StatisticsGen component.

        Args:
            examples: Channel of examples from ExampleGen

        Returns:
            StatisticsGen component
        """
        logger.info("Creating StatisticsGen component")
        return StatisticsGen(examples=examples)

    def create_schema_gen(self, statistics: Channel) -> SchemaGen:
        """Create a SchemaGen component.

        Args:
            statistics: Channel of statistics from StatisticsGen

        Returns:
            SchemaGen component
        """
        logger.info("Creating SchemaGen component")
        return SchemaGen(statistics=statistics)

    def create_example_validator(
        self, statistics: Channel, schema: Channel
    ) -> ExampleValidator:
        """Create an ExampleValidator component.

        Args:
            statistics: Channel of statistics from StatisticsGen
            schema: Channel of schema from SchemaGen

        Returns:
            ExampleValidator component
        """
        logger.info("Creating ExampleValidator component")
        return ExampleValidator(statistics=statistics, schema=schema)

    def create_transform(
        self, examples: Channel, schema: Channel, transform_module: str
    ) -> Transform:
        """Create a Transform component.

        Args:
            examples: Channel of examples from ExampleGen
            schema: Channel of schema from SchemaGen
            transform_module: Path to the transform module file

        Returns:
            Transform component
        """
        logger.info(f"Creating Transform component with module {transform_module}")
        return Transform(
            examples=examples,
            schema=schema,
            module_file=transform_module
        )

    def create_trainer(
        self,
        transformed_examples: Channel,
        transform_graph: Channel,
        schema: Channel,
        trainer_module: str,
        train_args: trainer_pb2.TrainArgs,
        eval_args: trainer_pb2.EvalArgs,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Trainer:
        """Create a Trainer component.

        Args:
            transformed_examples: Channel of transformed examples from Transform
            transform_graph: Channel of transform graph from Transform
            schema: Channel of schema from SchemaGen
            trainer_module: Path to the trainer module file
            train_args: Training arguments
            eval_args: Evaluation arguments
            custom_config: Optional custom configuration

        Returns:
            Trainer component
        """
        logger.info(f"Creating Trainer component with module {trainer_module}")
        
        trainer = Trainer(
            module_file=trainer_module,
            transformed_examples=transformed_examples,
            transform_graph=transform_graph,
            schema=schema,
            train_args=train_args,
            eval_args=eval_args,
            custom_config=custom_config
        )
        
        return trainer

    def create_evaluator(
        self,
        examples: Channel,
        model: Channel,
        eval_config: tfma.EvalConfig,
        schema: Optional[Channel] = None
    ) -> Evaluator:
        """Create an Evaluator component.

        Args:
            examples: Channel of examples from ExampleGen
            model: Channel of model from Trainer
            eval_config: Evaluation configuration
            schema: Optional channel of schema from SchemaGen

        Returns:
            Evaluator component
        """
        logger.info("Creating Evaluator component")
        
        return Evaluator(
            examples=examples,
            model=model,
            eval_config=eval_config,
            schema=schema
        )

    def create_pusher(
        self,
        model: Channel,
        model_blessing: Optional[Channel] = None,
        serving_model_dir: Optional[str] = None
    ) -> Pusher:
        """Create a Pusher component.

        Args:
            model: Channel of model from Trainer
            model_blessing: Optional channel of model blessing from Evaluator
            serving_model_dir: Optional directory to push the model to

        Returns:
            Pusher component
        """
        logger.info(f"Creating Pusher component with serving dir {serving_model_dir}")
        
        if serving_model_dir is None:
            serving_model_dir = os.path.join(self.pipeline_root, "serving_model")
        
        pusher_args = {
            "model": model,
            "push_destination": pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=serving_model_dir
                )
            )
        }
        
        if model_blessing:
            pusher_args["model_blessing"] = model_blessing
        
        return Pusher(**pusher_args)


def create_eval_config() -> tfma.EvalConfig:
    """Create a default evaluation configuration.

    Returns:
        Evaluation configuration
    """
    # Define metrics for evaluation
    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name="BinaryAccuracy"),
                tfma.MetricConfig(class_name="BinaryCrossentropy"),
                tfma.MetricConfig(class_name="AUC"),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
            ]
        )
    ]
    
    # Define slicing specs (optional)
    slicing_specs = [
        tfma.SlicingSpec(),  # Overall metrics
        tfma.SlicingSpec(feature_keys=["customer_segment"]),  # Slice by segment
    ]
    
    # Create the evaluation config
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key="total_amount", prediction_key="prediction")
        ],
        metrics_specs=metrics_specs,
        slicing_specs=slicing_specs
    )
    
    return eval_config

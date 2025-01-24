"""Tests for pipeline components."""

import os
import unittest
from unittest.mock import patch, MagicMock

import tensorflow as tf
from tfx.proto import trainer_pb2

from src.pipeline.components import PipelineComponents, create_eval_config
from src.pipeline.config import PipelineConfig


class TestPipelineComponents(unittest.TestCase):
    """Test cases for pipeline components."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline_name = "test-pipeline"
        self.pipeline_root = "/tmp/test-pipeline-root"
        self.components = PipelineComponents(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root
        )

    def test_create_example_gen(self):
        """Test creating ExampleGen component."""
        data_path = "/tmp/data"
        example_gen = self.components.create_example_gen(data_path)
        self.assertIsNotNone(example_gen)

    def test_create_statistics_gen(self):
        """Test creating StatisticsGen component."""
        examples = MagicMock()
        statistics_gen = self.components.create_statistics_gen(examples)
        self.assertIsNotNone(statistics_gen)

    def test_create_schema_gen(self):
        """Test creating SchemaGen component."""
        statistics = MagicMock()
        schema_gen = self.components.create_schema_gen(statistics)
        self.assertIsNotNone(schema_gen)

    def test_create_example_validator(self):
        """Test creating ExampleValidator component."""
        statistics = MagicMock()
        schema = MagicMock()
        example_validator = self.components.create_example_validator(
            statistics=statistics,
            schema=schema
        )
        self.assertIsNotNone(example_validator)

    def test_create_transform(self):
        """Test creating Transform component."""
        examples = MagicMock()
        schema = MagicMock()
        transform_module = "src.pipeline.transform"
        transform = self.components.create_transform(
            examples=examples,
            schema=schema,
            transform_module=transform_module
        )
        self.assertIsNotNone(transform)

    def test_create_trainer(self):
        """Test creating Trainer component."""
        transformed_examples = MagicMock()
        transform_graph = MagicMock()
        schema = MagicMock()
        trainer_module = "src.pipeline.trainer"
        train_args = trainer_pb2.TrainArgs(num_steps=100)
        eval_args = trainer_pb2.EvalArgs(num_steps=50)
        custom_config = {"learning_rate": 0.001}
        
        trainer = self.components.create_trainer(
            transformed_examples=transformed_examples,
            transform_graph=transform_graph,
            schema=schema,
            trainer_module=trainer_module,
            train_args=train_args,
            eval_args=eval_args,
            custom_config=custom_config
        )
        self.assertIsNotNone(trainer)

    def test_create_evaluator(self):
        """Test creating Evaluator component."""
        examples = MagicMock()
        model = MagicMock()
        eval_config = create_eval_config()
        schema = MagicMock()
        
        evaluator = self.components.create_evaluator(
            examples=examples,
            model=model,
            eval_config=eval_config,
            schema=schema
        )
        self.assertIsNotNone(evaluator)

    def test_create_pusher(self):
        """Test creating Pusher component."""
        model = MagicMock()
        model_blessing = MagicMock()
        serving_model_dir = "/tmp/serving_model"
        
        pusher = self.components.create_pusher(
            model=model,
            model_blessing=model_blessing,
            serving_model_dir=serving_model_dir
        )
        self.assertIsNotNone(pusher)


class TestPipelineConfig(unittest.TestCase):
    """Test cases for pipeline configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_id = "test-project"
        self.region = "us-central1"
        self.pipeline_name = "test-pipeline"
        self.config = PipelineConfig(
            project_id=self.project_id,
            region=self.region,
            pipeline_name=self.pipeline_name
        )

    def test_get_train_args(self):
        """Test getting training arguments."""
        train_args = self.config.get_train_args()
        self.assertIsInstance(train_args, trainer_pb2.TrainArgs)
        self.assertEqual(train_args.num_steps, self.config.train_steps)

    def test_get_eval_args(self):
        """Test getting evaluation arguments."""
        eval_args = self.config.get_eval_args()
        self.assertIsInstance(eval_args, trainer_pb2.EvalArgs)
        self.assertEqual(eval_args.num_steps, self.config.eval_steps)

    def test_get_custom_config(self):
        """Test getting custom configuration."""
        custom_config = self.config.get_custom_config()
        self.assertIsInstance(custom_config, dict)
        self.assertIn("hidden_units", custom_config)
        self.assertIn("learning_rate", custom_config)
        self.assertIn("dropout_rate", custom_config)
        self.assertIn("train_batch_size", custom_config)
        self.assertIn("eval_batch_size", custom_config)


if __name__ == "__main__":
    unittest.main()

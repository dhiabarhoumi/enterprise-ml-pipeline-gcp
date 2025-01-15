"""Tests for model evaluation module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from src.models.evaluation import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Create test data for regression
        self.y_true_regression = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_regression = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
        
        # Create test data for classification
        self.y_true_classification = np.array([0, 1, 0, 1, 0])
        self.y_pred_classification = np.array([0, 1, 1, 1, 0])
        self.y_pred_proba_classification = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.7, 0.3]
        ])

    def test_calculate_regression_metrics(self):
        """Test calculating regression metrics."""
        metrics = self.evaluator.calculate_regression_metrics(
            self.y_true_regression, self.y_pred_regression)
        
        # Verify metrics are calculated correctly
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # Check specific metric values
        expected_mse = mean_squared_error(self.y_true_regression, self.y_pred_regression)
        expected_r2 = r2_score(self.y_true_regression, self.y_pred_regression)
        self.assertAlmostEqual(metrics['mse'], expected_mse)
        self.assertAlmostEqual(metrics['r2'], expected_r2)

    def test_calculate_classification_metrics(self):
        """Test calculating classification metrics."""
        metrics = self.evaluator.calculate_classification_metrics(
            self.y_true_classification, self.y_pred_classification)
        
        # Verify metrics are calculated correctly
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Check specific metric values
        expected_accuracy = accuracy_score(self.y_true_classification, self.y_pred_classification)
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_regression_results(self, mock_close, mock_savefig):
        """Test plotting regression results."""
        # Create a temporary file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Call the method
            self.evaluator.plot_regression_results(
                self.y_true_regression, 
                self.y_pred_regression, 
                output_path=output_path
            )
            
            # Verify the plot was saved
            mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
            mock_close.assert_called_once()
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_confusion_matrix(self, mock_close, mock_savefig):
        """Test plotting confusion matrix."""
        # Create a temporary file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Call the method
            self.evaluator.plot_confusion_matrix(
                self.y_true_classification, 
                self.y_pred_classification, 
                output_path=output_path
            )
            
            # Verify the plot was saved
            mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
            mock_close.assert_called_once()
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_roc_curve(self, mock_close, mock_savefig):
        """Test plotting ROC curve."""
        # Create a temporary file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Call the method
            self.evaluator.plot_roc_curve(
                self.y_true_classification, 
                self.y_pred_proba_classification[:, 1], 
                output_path=output_path
            )
            
            # Verify the plot was saved
            mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches='tight')
            mock_close.assert_called_once()
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_generate_regression_report(self):
        """Test generating regression report."""
        # Create a temporary directory for the report
        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the method
            report = self.evaluator.generate_regression_report(
                self.y_true_regression, 
                self.y_pred_regression, 
                output_dir=temp_dir,
                model_name="test_model"
            )
            
            # Verify the report contains metrics
            self.assertIn('metrics', report)
            self.assertIn('mae', report['metrics'])
            self.assertIn('mse', report['metrics'])
            self.assertIn('rmse', report['metrics'])
            self.assertIn('r2', report['metrics'])
            
            # Verify the report contains file paths
            self.assertIn('plots', report)
            self.assertIn('actual_vs_predicted', report['plots'])
            self.assertIn('residuals', report['plots'])

    def test_generate_classification_report(self):
        """Test generating classification report."""
        # Create a temporary directory for the report
        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the method
            report = self.evaluator.generate_classification_report(
                self.y_true_classification, 
                self.y_pred_classification, 
                self.y_pred_proba_classification, 
                output_dir=temp_dir,
                model_name="test_model"
            )
            
            # Verify the report contains metrics
            self.assertIn('metrics', report)
            self.assertIn('accuracy', report['metrics'])
            self.assertIn('precision', report['metrics'])
            self.assertIn('recall', report['metrics'])
            self.assertIn('f1', report['metrics'])
            
            # Verify the report contains file paths
            self.assertIn('plots', report)
            self.assertIn('confusion_matrix', report['plots'])
            self.assertIn('roc_curve', report['plots'])

    def test_compare_models_regression(self):
        """Test comparing regression models."""
        # Create model results
        model_results = {
            'model1': {
                'y_true': self.y_true_regression,
                'y_pred': self.y_pred_regression
            },
            'model2': {
                'y_true': self.y_true_regression,
                'y_pred': self.y_pred_regression * 1.1  # Slightly worse predictions
            }
        }
        
        # Call the method
        comparison = self.evaluator.compare_models(
            model_results, task_type='regression')
        
        # Verify the comparison contains both models
        self.assertIn('model1', comparison)
        self.assertIn('model2', comparison)
        
        # Verify each model has metrics
        self.assertIn('metrics', comparison['model1'])
        self.assertIn('metrics', comparison['model2'])
        
        # Verify the comparison includes a ranking
        self.assertIn('ranking', comparison)

    def test_compare_models_classification(self):
        """Test comparing classification models."""
        # Create model results
        model_results = {
            'model1': {
                'y_true': self.y_true_classification,
                'y_pred': self.y_pred_classification,
                'y_pred_proba': self.y_pred_proba_classification
            },
            'model2': {
                'y_true': self.y_true_classification,
                'y_pred': np.array([0, 0, 0, 1, 0]),  # Different predictions
                'y_pred_proba': self.y_pred_proba_classification * 0.9
            }
        }
        
        # Call the method
        comparison = self.evaluator.compare_models(
            model_results, task_type='classification')
        
        # Verify the comparison contains both models
        self.assertIn('model1', comparison)
        self.assertIn('model2', comparison)
        
        # Verify each model has metrics
        self.assertIn('metrics', comparison['model1'])
        self.assertIn('metrics', comparison['model2'])
        
        # Verify the comparison includes a ranking
        self.assertIn('ranking', comparison)


if __name__ == "__main__":
    unittest.main()

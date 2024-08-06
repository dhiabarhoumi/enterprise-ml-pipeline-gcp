"""Model evaluation utilities for the Enterprise ML Pipeline."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation utilities."""

    def __init__(self, model_type: str = "regression", output_dir: str = "evaluation_results"):
        """Initialize the model evaluator.

        Args:
            model_type: Type of model ('regression' or 'classification')
            output_dir: Directory to save evaluation results
        """
        self.model_type = model_type
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized ModelEvaluator for {model_type} model")

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None,
        feature_importance: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model predictions and calculate metrics.

        Args:
            y_true: Ground truth values
            y_pred: Model predictions
            feature_names: Optional list of feature names
            feature_importance: Optional array of feature importance scores

        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure predictions are the right shape
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        
        # Calculate metrics based on model type
        if self.model_type == "regression":
            metrics = self._evaluate_regression(y_true, y_pred)
        else:  # classification
            # For binary classification, threshold predictions
            if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_class = y_pred
            metrics = self._evaluate_classification(y_true, y_pred, y_pred_class)
        
        # Save metrics to file
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        # Plot feature importance if provided
        if feature_names is not None and feature_importance is not None:
            self.plot_feature_importance(feature_names, feature_importance)
        
        return metrics

    def _evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.

        Args:
            y_true: Ground truth values
            y_pred: Model predictions

        Returns:
            Dictionary of regression metrics
        """
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mean_prediction": float(np.mean(y_pred)),
            "mean_actual": float(np.mean(y_true)),
        }
        
        # Plot regression results
        self._plot_regression_results(y_true, y_pred)
        
        return metrics

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_class: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics.

        Args:
            y_true: Ground truth values
            y_pred_proba: Predicted probabilities
            y_pred_class: Predicted classes

        Returns:
            Dictionary of classification metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred_class)),
            "precision": float(precision_score(y_true, y_pred_class, average="weighted")),
            "recall": float(recall_score(y_true, y_pred_class, average="weighted")),
            "f1": float(f1_score(y_true, y_pred_class, average="weighted")),
        }
        
        # Add AUC if we have probabilities
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_pred_proba))
        except Exception:
            logger.warning("Could not calculate AUC score")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_true, y_pred_class)
        
        return metrics

    def _plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot regression results.

        Args:
            y_true: Ground truth values
            y_pred: Model predictions
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "regression_results.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved regression results plot to {plot_path}")

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix.

        Args:
            y_true: Ground truth values
            y_pred: Model predictions
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {plot_path}")

    def plot_feature_importance(self, feature_names: List[str], importance: np.ndarray) -> None:
        """Plot feature importance.

        Args:
            feature_names: List of feature names
            importance: Array of feature importance scores
        """
        # Sort features by importance
        indices = np.argsort(importance)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {plot_path}")

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        primary_metric: str = "rmse" if "regression" else "f1"
    ) -> Tuple[str, Dict[str, float]]:
        """Compare multiple models based on evaluation metrics.

        Args:
            model_results: Dictionary mapping model names to metric dictionaries
            primary_metric: Primary metric for comparison

        Returns:
            Tuple of (best_model_name, best_model_metrics)
        """
        # Determine if we're minimizing or maximizing the metric
        minimize_metric = primary_metric in ["mse", "rmse", "mae"]
        
        # Find the best model
        best_model = None
        best_score = float('inf') if minimize_metric else float('-inf')
        
        for model_name, metrics in model_results.items():
            if primary_metric not in metrics:
                logger.warning(f"Metric {primary_metric} not found for model {model_name}")
                continue
                
            score = metrics[primary_metric]
            
            if (minimize_metric and score < best_score) or (not minimize_metric and score > best_score):
                best_score = score
                best_model = model_name
        
        if best_model is None:
            logger.error(f"No models found with metric {primary_metric}")
            return None, {}
            
        logger.info(f"Best model: {best_model} with {primary_metric} = {best_score}")
        
        # Plot comparison
        self._plot_model_comparison(model_results, primary_metric)
        
        return best_model, model_results[best_model]

    def _plot_model_comparison(self, model_results: Dict[str, Dict[str, float]], primary_metric: str) -> None:
        """Plot model comparison.

        Args:
            model_results: Dictionary mapping model names to metric dictionaries
            primary_metric: Primary metric for comparison
        """
        models = list(model_results.keys())
        scores = [model_results[model].get(primary_metric, 0) for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, scores)
        plt.xlabel('Model')
        plt.ylabel(primary_metric)
        plt.title(f'Model Comparison by {primary_metric}')
        plt.xticks(rotation=45)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved model comparison plot to {plot_path}")

    def generate_evaluation_report(self, metrics: Dict[str, float], output_format: str = "markdown") -> str:
        """Generate a formatted evaluation report.

        Args:
            metrics: Dictionary of evaluation metrics
            output_format: Format of the report ('markdown' or 'text')

        Returns:
            Formatted report string
        """
        if output_format == "markdown":
            report = "# Model Evaluation Report\n\n"
            report += "## Metrics\n\n"
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"
            
            for metric, value in metrics.items():
                report += f"| {metric} | {value:.4f} |\n"
                
            # Add links to plots
            report += "\n## Visualizations\n\n"
            
            if self.model_type == "regression":
                report += "- [Regression Results](regression_results.png)\n"
            else:
                report += "- [Confusion Matrix](confusion_matrix.png)\n"
                
            report += "- [Feature Importance](feature_importance.png)\n"
            
        else:  # text format
            report = "Model Evaluation Report\n\n"
            report += "Metrics:\n"
            
            for metric, value in metrics.items():
                report += f"  {metric}: {value:.4f}\n"
        
        # Save report
        report_path = os.path.join(self.output_dir, f"evaluation_report.{output_format}")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved evaluation report to {report_path}")
        
        return report

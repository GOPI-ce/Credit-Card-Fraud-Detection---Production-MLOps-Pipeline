"""
Model Evaluation Module

Comprehensive model evaluation and performance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config_path: str = "configs/config.yaml", threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            config_path: Path to config file
            threshold: Classification threshold
        """
        self.threshold = threshold
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model and return metrics
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        return self.calculate_metrics(y, y_pred, y_pred_proba)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba)
        }
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.4f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"PR curve saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't have feature importance")
            return
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def generate_evaluation_report(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
        output_dir: str = "metrics"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            output_dir: Directory to save outputs
            
        Returns:
            Evaluation report dictionary
        """
        logger.info("Generating evaluation report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate plots
        self.plot_confusion_matrix(
            y_test, y_pred,
            save_path=output_path / "confusion_matrix.png"
        )
        
        self.plot_roc_curve(
            y_test, y_pred_proba,
            save_path=output_path / "roc_curve.png"
        )
        
        self.plot_precision_recall_curve(
            y_test, y_pred_proba,
            save_path=output_path / "pr_curve.png"
        )
        
        self.plot_feature_importance(
            model, feature_names,
            save_path=output_path / "feature_importance.png"
        )
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create full report
        evaluation_report = {
            'metrics': metrics,
            'classification_report': report,
            'threshold': self.threshold
        }
        
        # Save report
        report_path = output_path / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return evaluation_report


def main():
    """Main function to run model evaluation"""
    import joblib
    from pathlib import Path
    
    # Load trained model (adjust path as needed)
    models_path = Path("models")
    model_files = list(models_path.glob("model_*.pkl"))
    
    if not model_files:
        logger.error("No trained model found")
        return
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model from {latest_model}")
    
    model = joblib.load(latest_model)
    
    # Load test data (you'll need to implement this based on your data pipeline)
    # For now, this is a placeholder
    logger.info("Load test data using your data pipeline")
    
    return


if __name__ == "__main__":
    main()

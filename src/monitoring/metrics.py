"""
Monitoring Metrics Module

Custom metrics for model monitoring and data drift detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from prometheus_client import Gauge, Counter, Histogram
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Prometheus metrics
PREDICTION_DISTRIBUTION = Histogram(
    'prediction_distribution',
    'Distribution of prediction scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'KS statistic for data drift detection',
    ['feature']
)

MODEL_PERFORMANCE = Gauge(
    'model_performance_metric',
    'Model performance metrics',
    ['metric_name']
)

FEATURE_STATISTICS = Gauge(
    'feature_statistics',
    'Statistical properties of features',
    ['feature', 'statistic']
)


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize monitor
        
        Args:
            reference_data: Reference dataset for drift detection
        """
        self.reference_data = reference_data
        self.reference_stats = None
        
        if reference_data is not None:
            self.calculate_reference_stats()
    
    def calculate_reference_stats(self):
        """Calculate statistics on reference data"""
        logger.info("Calculating reference statistics...")
        
        self.reference_stats = {}
        
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_stats[col] = {
                'mean': self.reference_data[col].mean(),
                'std': self.reference_data[col].std(),
                'min': self.reference_data[col].min(),
                'max': self.reference_data[col].max(),
                'q25': self.reference_data[col].quantile(0.25),
                'q50': self.reference_data[col].quantile(0.50),
                'q75': self.reference_data[col].quantile(0.75)
            }
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect data drift using Kolmogorov-Smirnov test
        
        Args:
            current_data: Current dataset
            threshold: Significance threshold
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return {}
        
        logger.info("Detecting data drift...")
        
        drift_results = {}
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            drift_detected = p_value < threshold
            
            drift_results[col] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected
            }
            
            # Update Prometheus metric
            DATA_DRIFT_SCORE.labels(feature=col).set(ks_statistic)
            
            if drift_detected:
                logger.warning(f"Data drift detected in {col}: KS={ks_statistic:.4f}, p={p_value:.4f}")
        
        return drift_results
    
    def calculate_feature_statistics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature statistics
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of statistics per feature
        """
        logger.info("Calculating feature statistics...")
        
        stats_dict = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'missing_pct': float(data[col].isnull().mean() * 100)
            }
            
            # Update Prometheus metrics
            for stat_name, stat_value in stats_dict[col].items():
                FEATURE_STATISTICS.labels(
                    feature=col,
                    statistic=stat_name
                ).set(stat_value)
        
        return stats_dict
    
    def monitor_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Monitor prediction distribution
        
        Args:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary with monitoring results
        """
        logger.info("Monitoring predictions...")
        
        # Update histogram
        for prob in probabilities:
            PREDICTION_DISTRIBUTION.observe(float(prob))
        
        results = {
            'fraud_rate': float(predictions.mean()),
            'avg_fraud_probability': float(probabilities[predictions == 1].mean()) if predictions.sum() > 0 else 0,
            'avg_normal_probability': float(probabilities[predictions == 0].mean()) if (predictions == 0).sum() > 0 else 0,
            'prediction_distribution': {
                'mean': float(probabilities.mean()),
                'std': float(probabilities.std()),
                'min': float(probabilities.min()),
                'max': float(probabilities.max())
            }
        }
        
        return results
    
    def generate_monitoring_report(
        self,
        current_data: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report
        
        Args:
            current_data: Current dataset
            predictions: Predictions
            probabilities: Prediction probabilities
            
        Returns:
            Monitoring report
        """
        logger.info("Generating monitoring report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': current_data.shape,
            'feature_statistics': self.calculate_feature_statistics(current_data),
            'prediction_monitoring': self.monitor_predictions(predictions, probabilities)
        }
        
        # Add drift detection if reference data available
        if self.reference_data is not None:
            report['data_drift'] = self.detect_data_drift(current_data)
        
        return report


def main():
    """Main function for testing monitoring"""
    # This would typically be called from the API or training pipeline
    logger.info("Model monitoring module loaded")


if __name__ == "__main__":
    main()

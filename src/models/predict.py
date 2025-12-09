"""
Prediction Service Module

Handles model inference and predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Tuple
import logging
import joblib
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    """Model prediction service"""
    
    def __init__(
        self,
        model_path: str,
        feature_engineer_path: str = "models/feature_engineering",
        config_path: str = "configs/config.yaml"
    ):
        """
        Initialize prediction service
        
        Args:
            model_path: Path to trained model
            feature_engineer_path: Path to feature engineering artifacts
            config_path: Path to configuration file
        """
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load feature engineering artifacts
        self.load_feature_artifacts(feature_engineer_path)
    
    def load_feature_artifacts(self, artifacts_path: str):
        """Load feature engineering artifacts"""
        artifacts_path = Path(artifacts_path)
        
        # Load scaler
        scaler_path = artifacts_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            self.scaler = None
            logger.warning("No scaler found")
        
        # No feature selector needed for this model
        self.feature_selector = None
        self.selected_features = None
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data for prediction
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed features
        """
        from features.engineering import FeatureEngineer
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Apply feature engineering (same as training)
        df_processed = engineer.create_time_features(df)
        df_processed = engineer.create_amount_features(df_processed)
        
        # Select only numeric columns (exclude categorical like Amount_category)
        df_numeric = df_processed.select_dtypes(include=[np.number])
        
        # Scale features if scaler available
        if self.scaler:
            X = self.scaler.transform(df_numeric)
        else:
            X = df_numeric.values
        
        return X
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions (and probabilities if requested)
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        logger.info(f"Making predictions for {len(df)} samples")
        
        # Preprocess
        X = self.preprocess(df)
        
        # Predict
        predictions = self.model.predict(X)
        
        if return_proba:
            probabilities = self.model.predict_proba(X)
            return predictions, probabilities
        
        return predictions
    
    def predict_single(
        self,
        transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict for a single transaction
        
        Args:
            transaction: Transaction data
            
        Returns:
            Prediction result with details
        """
        predictions, probabilities = self.predict([transaction], return_proba=True)
        
        result = {
            'is_fraud': bool(predictions[0]),
            'fraud_probability': float(probabilities[0][1]),
            'normal_probability': float(probabilities[0][0]),
            'prediction_label': 'Fraud' if predictions[0] else 'Normal'
        }
        
        return result
    
    def batch_predict(
        self,
        transactions: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Predict for batch of transactions
        
        Args:
            transactions: List of transactions
            
        Returns:
            DataFrame with predictions
        """
        predictions, probabilities = self.predict(transactions, return_proba=True)
        
        results_df = pd.DataFrame({
            'is_fraud': predictions,
            'fraud_probability': probabilities[:, 1],
            'normal_probability': probabilities[:, 0],
            'prediction_label': ['Fraud' if p else 'Normal' for p in predictions]
        })
        
        return results_df


def main():
    """Main function for testing predictions"""
    from pathlib import Path
    
    # Find latest model
    models_path = Path("models")
    model_files = list(models_path.glob("model_*.pkl"))
    
    if not model_files:
        logger.error("No trained model found")
        return
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # Initialize service
    service = PredictionService(str(latest_model))
    
    # Example prediction
    sample_transaction = {
        'Time': 12345.0,
        'V1': -1.5, 'V2': 0.5, 'V3': 1.2, 'V4': -0.8,
        'V5': 0.3, 'V6': -0.6, 'V7': 0.9, 'V8': -0.4,
        'V9': 1.1, 'V10': -0.2, 'V11': 0.7, 'V12': -1.0,
        'V13': 0.4, 'V14': -0.5, 'V15': 0.8, 'V16': -0.3,
        'V17': 1.3, 'V18': -0.7, 'V19': 0.6, 'V20': -0.9,
        'V21': 0.2, 'V22': -0.4, 'V23': 0.5, 'V24': -0.6,
        'V25': 0.7, 'V26': -0.8, 'V27': 0.9, 'V28': -1.1,
        'Amount': 150.0
    }
    
    result = service.predict_single(sample_transaction)
    
    print("\n" + "="*50)
    print("Prediction Result")
    print("="*50)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Normal Probability: {result['normal_probability']:.4f}")
    
    return result


if __name__ == "__main__":
    main()

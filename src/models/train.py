"""
Model Training Module

Handles model training, hyperparameter tuning, and experiment tracking with MLflow
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import joblib
from datetime import datetime

# ML Libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Hyperparameter tuning
import optuna
from optuna.samplers import TPESampler

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and tune ML models"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.hyperparams = self.config['hyperparameters']
        self.paths = self.config['paths']
        
        # Setup MLflow (disabled by default, use local file storage)
        mlflow_config = self.config.get('mlflow', {})
        
        # Use local file-based tracking instead of server
        import os
        mlruns_path = os.path.join(os.getcwd(), 'mlruns')
        mlflow.set_tracking_uri(f'file:///{mlruns_path}')
        
        self.experiment_name = self.training_config['experiment_name']
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Continuing without experiment tracking.")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Class'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X, y)
        """
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """
        Create model instance
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Model instance
        """
        if model_name == 'xgboost':
            return xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        elif model_name == 'catboost':
            return CatBoostClassifier(**params, random_state=42, verbose=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Optuna objective function for hyperparameter tuning
        
        Args:
            trial: Optuna trial
            X: Features
            y: Target
            
        Returns:
            Validation score
        """
        model_name = self.model_config['name']
        params_space = self.hyperparams.get(model_name, {})
        
        # Sample hyperparameters
        params = {}
        for param_name, param_values in params_space.items():
            if isinstance(param_values, list):
                if all(isinstance(v, int) for v in param_values):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif all(isinstance(v, float) for v in param_values):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values), log=True
                    )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
        
        # Create and evaluate model
        model = self.create_model(model_name, params)
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.training_config['cv_folds'],
            shuffle=True,
            random_state=42
        )
        
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Best parameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.training_config['n_trials'],
            show_progress_bar=True
        )
        
        logger.info(f"Best trial score: {study.best_trial.value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Train model with given parameters
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Model parameters
            
        Returns:
            Trained model
        """
        logger.info("Training model...")
        
        model_name = self.model_config['name']
        
        if params is None:
            # Use default parameters
            params = self.hyperparams.get(model_name, {})
            # Take first value if list
            params = {k: v[0] if isinstance(v, list) else v for k, v in params.items()}
        
        model = self.create_model(model_name, params)
        
        # Train with early stopping if validation data provided
        if X_val is not None and y_val is not None:
            if model_name == 'xgboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.training_config['early_stopping_rounds'],
                    verbose=False
                )
            elif model_name == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(self.training_config['early_stopping_rounds'])]
                )
            elif model_name == 'catboost':
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.training_config['early_stopping_rounds']
                )
        else:
            model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X: Features
            y: Target
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating model on {dataset_name} set...")
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            f'{dataset_name}_accuracy': accuracy_score(y, y_pred),
            f'{dataset_name}_precision': precision_score(y, y_pred, zero_division=0),
            f'{dataset_name}_recall': recall_score(y, y_pred, zero_division=0),
            f'{dataset_name}_f1': f1_score(y, y_pred, zero_division=0),
            f'{dataset_name}_roc_auc': roc_auc_score(y, y_pred_proba),
            f'{dataset_name}_pr_auc': average_precision_score(y, y_pred_proba)
        }
        
        # Log metrics
        logger.info(f"Metrics for {dataset_name} set:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Classification report
        report = classification_report(y, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        return metrics
    
    def train_baseline_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Train a baseline model with default parameters
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        logger.info("Training baseline model...")
        
        # Default XGBoost parameters for fraud detection
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train)  # Handle imbalance
        }
        
        model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        logger.info("Baseline model training completed")
        return model
    
    def save_model(self, model: Any, model_path: str) -> str:
        """
        Save trained model
        
        Args:
            model: Trained model
            model_path: Path for the saved model
            
        Returns:
            Path to saved model
        """
        model_path_obj = Path(model_path)
        model_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        tune: bool = True
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Run the full training pipeline
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame (optional)
            tune: Whether to tune hyperparameters
            
        Returns:
            Tuple of (model, metrics)
        """
        logger.info("Starting model training pipeline...")
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_df)
        X_val, y_val = self.prepare_data(val_df)
        
        with mlflow.start_run(run_name=f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(self.model_config)
            mlflow.log_params(self.training_config)
            
            # Tune hyperparameters
            if tune:
                best_params = self.tune_hyperparameters(X_train, y_train)
                mlflow.log_params(best_params)
            else:
                best_params = None
            
            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val, best_params)
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(model, X_val, y_val, "validation")
            
            # Evaluate on test set if provided
            if test_df is not None:
                X_test, y_test = self.prepare_data(test_df)
                test_metrics = self.evaluate_model(model, X_test, y_test, "test")
                val_metrics.update(test_metrics)
            
            # Log metrics
            mlflow.log_metrics(val_metrics)
            
            # Log model
            if self.model_config['name'] == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            elif self.model_config['name'] == 'lightgbm':
                mlflow.lightgbm.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Save model
            model_path = self.save_model(model)
            mlflow.log_artifact(model_path)
        
        logger.info("Model training pipeline completed successfully")
        
        return model, val_metrics


def main():
    """Main function to run model training"""
    from data.ingestion import DataIngestion
    from data.cleaning import DataCleaner
    from features.engineering import FeatureEngineer
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    
    ingestion = DataIngestion()
    df = ingestion.ingest_from_csv()
    
    cleaner = DataCleaner()
    cleaned_df = cleaner.run(df, save=False)
    
    engineer = FeatureEngineer()
    features_df = engineer.run(cleaned_df, fit=True, save=True)
    
    # Split data
    train_df, val_df, test_df = ingestion.split_data(features_df, save=False)
    
    # Train model
    trainer = ModelTrainer()
    model, metrics = trainer.run(train_df, val_df, test_df, tune=True)
    
    print("\n" + "="*50)
    print("Model Training Summary")
    print("="*50)
    print(f"Model: {trainer.model_config['name']}")
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    return model, metrics


if __name__ == "__main__":
    main()

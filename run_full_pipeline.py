"""
Complete MLOps Pipeline Runner
Executes the full fraud detection pipeline from data ingestion to model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

from src.data.cleaning import DataCleaner
from src.data.validation import DataValidator
from src.features.engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Run the complete ML pipeline"""
    
    logger.info("="*80)
    logger.info("Starting Fraud Detection MLOps Pipeline")
    logger.info("="*80)
    
    # Step 1: Data Ingestion
    logger.info("\n[STEP 1] Data Ingestion")
    df = pd.read_csv('data/raw/creditcard.csv')
    logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")
    
    # Step 2: Data Cleaning
    logger.info("\n[STEP 2] Data Cleaning")
    cleaner = DataCleaner('configs/config.yaml')
    df_clean = cleaner.handle_missing_values(df)
    logger.info(f"Data after cleaning: {df_clean.shape}")
    
    # Step 3: Data Validation
    logger.info("\n[STEP 3] Data Validation")
    validator = DataValidator('configs/config.yaml')
    schema = validator.create_schema(df_clean)
    df_validated = schema.validate(df_clean)
    logger.info("Data validation successful")
    
    # Step 4: Feature Engineering
    logger.info("\n[STEP 4] Feature Engineering")
    engineer = FeatureEngineer('configs/config.yaml')
    df_feat = engineer.create_time_features(df_clean)
    df_feat = engineer.create_amount_features(df_feat)
    logger.info(f"Data after feature engineering: {df_feat.shape}")
    
    # Step 5: Train-Test Split
    logger.info("\n[STEP 5] Train-Test Split")
    
    # Drop categorical columns for now (Amount_category is a string)
    df_feat_numeric = df_feat.select_dtypes(include=[np.number])
    
    X = df_feat_numeric.drop(['Class'], axis=1)
    y = df_feat_numeric['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Fraud rate - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")
    
    # Step 6: Feature Scaling
    logger.info("\n[STEP 6] Feature Scaling")
    X_train_scaled, scaler = engineer.scale_features(X_train, fit=True)
    X_test_scaled, _ = engineer.scale_features(X_test, fit=False)
    logger.info("Feature scaling complete")
    
    # Step 7: Model Training (Simple baseline without extensive tuning)
    logger.info("\n[STEP 7] Model Training")
    trainer = ModelTrainer('configs/config.yaml')
    
    # Train a baseline XGBoost model
    logger.info("Training baseline XGBoost model...")
    model = trainer.train_baseline_model(X_train_scaled, y_train)
    logger.info("Model training complete")
    
    # Step 8: Model Evaluation
    logger.info("\n[STEP 8] Model Evaluation")
    evaluator = ModelEvaluator('configs/config.yaml')
    metrics = evaluator.evaluate_model(model, X_test_scaled, y_test)
    
    logger.info("\nModel Performance Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    
    # Step 9: Save Model and Artifacts
    logger.info("\n[STEP 9] Saving Model and Artifacts")
    model_path = Path('models/fraud_detection_model.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model, str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    scaler_path = Path('models/scaler.pkl')
    engineer.save_scaler(str(scaler_path))
    logger.info(f"Scaler saved to {scaler_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline Execution Complete!")
    logger.info("="*80)
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path)
    }


if __name__ == '__main__':
    try:
        results = run_pipeline()
        logger.info("\nPipeline executed successfully!")
        logger.info(f"Model saved at: {results['model_path']}")
        logger.info(f"ROC-AUC Score: {results['metrics']['roc_auc']:.4f}")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

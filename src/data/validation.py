"""
Data Validation Module

Validates data quality and schema using Great Expectations and Pandera
"""

import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and schema"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths = self.config['paths']
        
    def create_schema(self, df: pd.DataFrame) -> DataFrameSchema:
        """
        Create a Pandera schema from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Pandera DataFrameSchema
        """
        logger.info("Creating data schema...")
        
        schema_dict = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                schema_dict[col] = Column(
                    df[col].dtype,
                    checks=[
                        Check.greater_than_or_equal_to(df[col].min()),
                        Check.less_than_or_equal_to(df[col].max()),
                    ],
                    nullable=df[col].isnull().any()
                )
            elif df[col].dtype == 'object':
                schema_dict[col] = Column(
                    str,
                    nullable=df[col].isnull().any()
                )
        
        schema = DataFrameSchema(schema_dict, coerce=True)
        return schema
    
    def validate_schema(
        self, 
        df: pd.DataFrame,
        schema: Optional[DataFrameSchema] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame against schema
        
        Args:
            df: DataFrame to validate
            schema: Pandera schema (optional)
            
        Returns:
            Tuple of (is_valid, errors)
        """
        logger.info("Validating data schema...")
        
        errors = []
        
        try:
            if schema is None:
                schema = self.create_schema(df)
            
            schema.validate(df, lazy=True)
            logger.info("Schema validation passed")
            return True, errors
            
        except pa.errors.SchemaErrors as e:
            logger.error(f"Schema validation failed: {e}")
            errors = e.failure_cases.to_dict('records')
            return False, errors
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality checks
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info("Checking data quality...")
        
        quality_report = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_values': {},
            'duplicates': 0,
            'data_types': {},
            'unique_values': {},
            'outliers': {},
            'class_distribution': {}
        }
        
        # Missing values
        missing = df.isnull().sum()
        quality_report['missing_values'] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Duplicates
        quality_report['duplicates'] = int(df.duplicated().sum())
        
        # Data types
        quality_report['data_types'] = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }
        
        # Unique values
        for col in df.columns:
            n_unique = df[col].nunique()
            quality_report['unique_values'][col] = int(n_unique)
        
        # Class distribution (for classification)
        if 'Class' in df.columns:
            class_dist = df['Class'].value_counts().to_dict()
            quality_report['class_distribution'] = {
                str(k): int(v) for k, v in class_dist.items()
            }
        
        # Outliers detection (IQR method)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
            if outliers > 0:
                quality_report['outliers'][col] = int(outliers)
        
        return quality_report
    
    def validate_fraud_detection_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Specific validation for fraud detection dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, errors)
        """
        logger.info("Validating fraud detection data...")
        
        errors = []
        
        # Check required columns
        required_cols = ['Time', 'Amount', 'Class']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for V1-V28 features (PCA components)
        v_features = [f'V{i}' for i in range(1, 29)]
        missing_v_features = [col for col in v_features if col not in df.columns]
        
        if missing_v_features:
            errors.append(f"Missing V features: {missing_v_features}")
        
        # Validate Class column
        if 'Class' in df.columns:
            valid_classes = df['Class'].isin([0, 1]).all()
            if not valid_classes:
                errors.append("Class column contains values other than 0 and 1")
        
        # Validate Amount column
        if 'Amount' in df.columns:
            if (df['Amount'] < 0).any():
                errors.append("Amount column contains negative values")
        
        # Check data size
        if len(df) < 1000:
            errors.append(f"Dataset too small: {len(df)} rows (minimum 1000 recommended)")
        
        # Check class imbalance
        if 'Class' in df.columns:
            class_balance = df['Class'].value_counts(normalize=True)
            if class_balance.min() < 0.001:
                logger.warning(f"Severe class imbalance detected: {class_balance.to_dict()}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Fraud detection data validation passed")
        else:
            logger.error(f"Validation failed with {len(errors)} errors")
        
        return is_valid, errors
    
    def generate_validation_report(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive validation report
        
        Args:
            df: Input DataFrame
            output_path: Path to save report (optional)
            
        Returns:
            Validation report dictionary
        """
        logger.info("Generating validation report...")
        
        # Run all validations
        schema_valid, schema_errors = self.validate_schema(df)
        fraud_valid, fraud_errors = self.validate_fraud_detection_data(df)
        quality_metrics = self.check_data_quality(df)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_shape': df.shape,
            'schema_validation': {
                'passed': schema_valid,
                'errors': schema_errors
            },
            'fraud_validation': {
                'passed': fraud_valid,
                'errors': fraud_errors
            },
            'quality_metrics': quality_metrics,
            'overall_status': 'PASS' if (schema_valid and fraud_valid) else 'FAIL'
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def run(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Run full validation pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, report)
        """
        logger.info("Starting data validation pipeline...")
        
        report = self.generate_validation_report(df)
        is_valid = report['overall_status'] == 'PASS'
        
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.error("✗ Data validation failed")
        
        return is_valid, report


def main():
    """Main function to run data validation"""
    from data.ingestion import DataIngestion
    from data.cleaning import DataCleaner
    
    # Ingest and clean data
    ingestion = DataIngestion()
    df = ingestion.ingest_from_csv()
    
    cleaner = DataCleaner()
    cleaned_df = cleaner.run(df, save=False)
    
    # Validate data
    validator = DataValidator()
    is_valid, report = validator.run(cleaned_df)
    
    print("\n" + "="*50)
    print("Data Validation Report")
    print("="*50)
    print(json.dumps(report, indent=2))
    
    return is_valid, report


if __name__ == "__main__":
    main()

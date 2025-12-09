"""
Data Ingestion Module

Handles data loading from various sources (CSV, Database, API, S3)
"""

import os
import pandas as pd
import yaml
from typing import Tuple, Optional
from pathlib import Path
import logging
from sqlalchemy import create_engine
import boto3
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Data ingestion from multiple sources"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data_ingestion']
        self.paths = self.config['paths']
        
    def ingest_from_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame containing the data
        """
        if file_path is None:
            file_path = self.data_config['source_path']
        
        logger.info(f"Ingesting data from CSV: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} records from CSV")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def ingest_from_database(self) -> pd.DataFrame:
        """
        Ingest data from PostgreSQL database
        
        Returns:
            DataFrame containing the data
        """
        db_config = self.config['database']
        
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['name']}"
        )
        
        logger.info("Connecting to database...")
        
        try:
            engine = create_engine(connection_string)
            query = "SELECT * FROM transactions"  # Modify as needed
            
            df = pd.read_sql(query, engine)
            logger.info(f"Successfully loaded {len(df)} records from database")
            
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def ingest_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """
        Ingest data from AWS S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            DataFrame containing the data
        """
        logger.info(f"Ingesting data from S3: s3://{bucket}/{key}")
        
        try:
            s3_client = boto3.client('s3')
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            
            df = pd.read_csv(obj['Body'])
            logger.info(f"Successfully loaded {len(df)} records from S3")
            
            return df
        except Exception as e:
            logger.error(f"Error loading from S3: {str(e)}")
            raise
    
    def split_data(
        self, 
        df: pd.DataFrame,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            save: Whether to save the splits to disk
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        val_split = self.data_config['validation_split']
        test_split = self.data_config['test_split']
        random_state = self.data_config['random_state']
        
        logger.info(f"Splitting data: train/val/test")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_split, 
            random_state=random_state,
            stratify=df['Class'] if 'Class' in df.columns else None
        )
        
        # Second split: separate validation from training
        val_size = val_split / (1 - test_split)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_df['Class'] if 'Class' in train_val_df.columns else None
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        if save:
            self._save_splits(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _save_splits(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> None:
        """Save data splits to disk"""
        raw_path = Path(self.paths['data']['raw'])
        raw_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_path = raw_path / f"train_{timestamp}.csv"
        val_path = raw_path / f"val_{timestamp}.csv"
        test_path = raw_path / f"test_{timestamp}.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved splits to {raw_path}")
    
    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get basic statistics about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # For classification tasks
        if 'Class' in df.columns:
            stats['class_distribution'] = df['Class'].value_counts().to_dict()
            stats['class_balance'] = df['Class'].value_counts(normalize=True).to_dict()
        
        return stats
    
    def run(self, source_type: Optional[str] = None) -> pd.DataFrame:
        """
        Run the full ingestion pipeline
        
        Args:
            source_type: Type of source (csv, database, s3)
            
        Returns:
            Ingested DataFrame
        """
        if source_type is None:
            source_type = self.data_config['source_type']
        
        logger.info(f"Starting data ingestion from {source_type}")
        
        if source_type == 'csv':
            df = self.ingest_from_csv()
        elif source_type == 'database':
            df = self.ingest_from_database()
        elif source_type == 's3':
            # Example: modify with actual bucket and key
            df = self.ingest_from_s3(
                bucket='your-bucket',
                key='data/creditcard.csv'
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Get and log statistics
        stats = self.get_data_statistics(df)
        logger.info(f"Data statistics: {stats}")
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        logger.info("Data ingestion completed successfully")
        
        return df


def main():
    """Main function to run data ingestion"""
    ingestion = DataIngestion()
    df = ingestion.run()
    
    print("\n" + "="*50)
    print("Data Ingestion Summary")
    print("="*50)
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    main()

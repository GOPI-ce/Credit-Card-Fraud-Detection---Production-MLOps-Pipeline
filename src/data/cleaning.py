"""
Data Cleaning Module

Handles missing values, outliers, duplicates, and data quality issues
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess data"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cleaning_config = self.config['data_cleaning']
        self.paths = self.config['paths']
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        strategy: str = 'auto'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                     'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill'
                     
        Returns:
            Cleaned DataFrame
        """
        logger.info("Handling missing values...")
        
        df = df.copy()
        missing_threshold = self.cleaning_config['missing_threshold']
        
        # Log missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Found missing values:\n{missing[missing > 0]}")
        
        # Drop columns with too many missing values
        cols_to_drop = missing[missing / len(df) > missing_threshold].index.tolist()
        if cols_to_drop:
            logger.info(f"Dropping columns with >{missing_threshold*100}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Handle remaining missing values
        if strategy == 'auto':
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        
        logger.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")
        
        return df
    
    def detect_outliers_iqr(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers using IQR method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: IQR multiplier
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outliers
    
    def detect_outliers_zscore(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers using Z-score method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: Z-score threshold
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            outliers[col] = z_scores > threshold
        
        return outliers
    
    def detect_outliers_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Detect outliers using Isolation Forest
        
        Args:
            df: Input DataFrame
            columns: Columns to use for outlier detection
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean array indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
        predictions = iso_forest.fit_predict(df[columns])
        outliers = predictions == -1
        
        return outliers
    
    def handle_outliers(
        self, 
        df: pd.DataFrame,
        method: Optional[str] = None,
        action: str = 'cap'
    ) -> pd.DataFrame:
        """
        Handle outliers in the dataset
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            action: Action to take ('cap', 'remove', 'keep')
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Handling outliers using {method} method...")
        
        df = df.copy()
        
        if method is None:
            method = self.cleaning_config['outlier_method']
        
        threshold = self.cleaning_config['outlier_threshold']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')  # Don't treat target as outlier
        
        if method == 'iqr':
            outliers = self.detect_outliers_iqr(df, numeric_cols, threshold)
            outlier_mask = outliers.any(axis=1)
        elif method == 'zscore':
            outliers = self.detect_outliers_zscore(df, numeric_cols, threshold)
            outlier_mask = outliers.any(axis=1)
        elif method == 'isolation_forest':
            outlier_mask = self.detect_outliers_isolation_forest(df, numeric_cols)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        logger.info(f"Found {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(df)*100:.2f}%)")
        
        if action == 'remove':
            df = df[~outlier_mask]
            logger.info(f"Removed outliers. New shape: {df.shape}")
        elif action == 'cap':
            for col in numeric_cols:
                if method in ['iqr', 'zscore']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    df[col] = df[col].clip(lower_bound, upper_bound)
            logger.info("Capped outliers to acceptable range")
        
        return df
    
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        logger.info("Handling duplicates...")
        
        df = df.copy()
        handling = self.cleaning_config['duplicate_handling']
        
        initial_count = len(df)
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows ({duplicates/initial_count*100:.2f}%)")
            
            if handling == 'drop':
                df = df.drop_duplicates()
            elif handling == 'keep_first':
                df = df.drop_duplicates(keep='first')
            elif handling == 'keep_last':
                df = df.drop_duplicates(keep='last')
            
            logger.info(f"After handling: {len(df)} rows (removed {initial_count - len(df)})")
        else:
            logger.info("No duplicates found")
        
        return df
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect and suggest appropriate data types
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to suggested types
        """
        suggested_types = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(df[col])
                    suggested_types[col] = 'numeric'
                except:
                    # Check if it's datetime
                    try:
                        pd.to_datetime(df[col])
                        suggested_types[col] = 'datetime'
                    except:
                        suggested_types[col] = 'categorical'
            else:
                suggested_types[col] = str(df[col].dtype)
        
        return suggested_types
    
    def run(
        self, 
        df: pd.DataFrame,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Run the full data cleaning pipeline
        
        Args:
            df: Input DataFrame
            save: Whether to save cleaned data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        logger.info(f"Initial shape: {df.shape}")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle duplicates
        df = self.handle_duplicates(df)
        
        # Handle outliers
        df = self.handle_outliers(df)
        
        logger.info(f"Final shape: {df.shape}")
        logger.info("Data cleaning completed successfully")
        
        if save:
            output_path = Path(self.paths['data']['processed']) / 'cleaned_data.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")
        
        return df


def main():
    """Main function to run data cleaning"""
    from data.ingestion import DataIngestion
    
    # Ingest data first
    ingestion = DataIngestion()
    df = ingestion.ingest_from_csv()
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_df = cleaner.run(df)
    
    print("\n" + "="*50)
    print("Data Cleaning Summary")
    print("="*50)
    print(f"Cleaned samples: {len(cleaned_df)}")
    print(f"Features: {len(cleaned_df.columns)}")
    print("\nData info:")
    print(cleaned_df.info())
    
    return cleaned_df


if __name__ == "__main__":
    main()

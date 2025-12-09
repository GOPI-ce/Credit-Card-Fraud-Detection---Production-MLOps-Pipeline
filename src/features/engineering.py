"""
Feature Engineering Module

Handles feature creation, transformation, scaling, and selection
"""

import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering and transformation"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['feature_engineering']
        self.paths = self.config['paths']
        self.scalers = {}
        self.feature_selectors = {}
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from Time column
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time features...")
        
        df = df.copy()
        
        if 'Time' in df.columns:
            # Convert seconds to hours
            df['Hour'] = (df['Time'] / 3600) % 24
            df['Day'] = (df['Time'] / 86400).astype(int)
            
            # Cyclical encoding for hour
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            
            # Time periods
            df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
            df['Is_Weekend'] = (df['Day'] % 7 >= 5).astype(int)
            
            logger.info(f"Created {6} time-based features")
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with amount features
        """
        logger.info("Creating amount features...")
        
        df = df.copy()
        
        if 'Amount' in df.columns:
            # Log transformation
            df['Amount_log'] = np.log1p(df['Amount'])
            
            # Amount categories
            df['Amount_category'] = pd.cut(
                df['Amount'],
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            ).astype(str)
            
            # Statistical features
            df['Amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
            df['Is_High_Amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
            df['Is_Low_Amount'] = (df['Amount'] < df['Amount'].quantile(0.05)).astype(int)
            
            logger.info(f"Created {6} amount-based features")
        
        return df
    
    def create_v_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from V1-V28 PCA components
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with V-based features
        """
        logger.info("Creating V features...")
        
        df = df.copy()
        v_cols = [col for col in df.columns if col.startswith('V')]
        
        if v_cols:
            # Statistical aggregations
            df['V_mean'] = df[v_cols].mean(axis=1)
            df['V_std'] = df[v_cols].std(axis=1)
            df['V_min'] = df[v_cols].min(axis=1)
            df['V_max'] = df[v_cols].max(axis=1)
            df['V_range'] = df['V_max'] - df['V_min']
            
            # Percentiles
            df['V_median'] = df[v_cols].median(axis=1)
            df['V_q25'] = df[v_cols].quantile(0.25, axis=1)
            df['V_q75'] = df[v_cols].quantile(0.75, axis=1)
            
            # Count features
            df['V_positive_count'] = (df[v_cols] > 0).sum(axis=1)
            df['V_negative_count'] = (df[v_cols] < 0).sum(axis=1)
            df['V_zero_count'] = (df[v_cols] == 0).sum(axis=1)
            
            logger.info(f"Created {11} V-based features")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        if 'Amount' in df.columns and 'Time' in df.columns:
            df['Amount_Time_interaction'] = df['Amount'] * df['Time']
        
        if 'Amount_log' in df.columns and 'V_mean' in df.columns:
            df['Amount_V_interaction'] = df['Amount_log'] * df['V_mean']
        
        logger.info("Created interaction features")
        
        return df
    
    def scale_features(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[Any]]:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame or numpy array
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler
            
        Returns:
            Tuple of (scaled data, scaler if fit=True else None)
        """
        logger.info(f"Scaling features...")
        
        # Handle DataFrame or array input
        if isinstance(df, pd.DataFrame):
            data = df.copy()
            is_dataframe = True
        else:
            data = df
            is_dataframe = False
        
        if method is None:
            method = self.feature_config['scaling_method']
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        if fit:
            if is_dataframe:
                scaled_data = pd.DataFrame(
                    scaler.fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                scaled_data = scaler.fit_transform(data)
            self.scalers[method] = scaler
            logger.info(f"Fitted and transformed using {method} scaler")
            return scaled_data, scaler
        else:
            if method not in self.scalers:
                raise ValueError(f"Scaler {method} not fitted yet")
            if is_dataframe:
                scaled_data = pd.DataFrame(
                    self.scalers[method].transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                scaled_data = self.scalers[method].transform(data)
            logger.info(f"Transformed using {method} scaler")
            return scaled_data, None
    
    def save_scaler(self, scaler_path: str, method: str = 'standard') -> None:
        """
        Save scaler to disk
        
        Args:
            scaler_path: Path to save the scaler
            method: Scaling method name
        """
        if method not in self.scalers:
            raise ValueError(f"Scaler {method} not fitted yet")
        
        scaler_path_obj = Path(scaler_path)
        scaler_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scalers[method], scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    
    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'Class',
        n_features: Optional[int] = None,
        method: str = 'f_classif',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Select top k features
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            n_features: Number of features to select
            method: Selection method ('f_classif', 'mutual_info')
            fit: Whether to fit the selector
            
        Returns:
            DataFrame with selected features
        """
        if not self.feature_config['feature_selection']:
            return df
        
        logger.info(f"Selecting features using {method}...")
        
        df = df.copy()
        
        if n_features is None:
            n_features = self.feature_config['n_features']
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Select score function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Fit and transform
        if fit:
            selector = SelectKBest(score_func=score_func, k=min(n_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selectors[method] = {
                'selector': selector,
                'features': selected_features
            }
        else:
            if method not in self.feature_selectors:
                raise ValueError(f"Feature selector {method} not fitted yet")
            
            selector = self.feature_selectors[method]['selector']
            selected_features = self.feature_selectors[method]['features']
            X_selected = selector.transform(X)
        
        logger.info(f"Selected {len(selected_features)} features")
        
        # Create result DataFrame
        result_df = pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=df.index
        )
        result_df[target_col] = y
        
        return result_df
    
    def apply_pca(
        self,
        df: pd.DataFrame,
        n_components: Optional[int] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            df: Input DataFrame
            n_components: Number of components
            fit: Whether to fit PCA
            
        Returns:
            DataFrame with PCA components
        """
        if n_components is None:
            n_components = self.feature_config.get('pca_components')
        
        if n_components is None:
            return df
        
        logger.info(f"Applying PCA with {n_components} components...")
        
        df = df.copy()
        
        # Separate target
        target_col = 'Class'
        X = df.drop(columns=[target_col]) if target_col in df.columns else df
        
        # Fit and transform
        if fit:
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X)
            self.pca = pca
            
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        else:
            if not hasattr(self, 'pca'):
                raise ValueError("PCA not fitted yet")
            X_pca = self.pca.transform(X)
        
        # Create result DataFrame
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        result_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        if target_col in df.columns:
            result_df[target_col] = df[target_col]
        
        return result_df
    
    def save_artifacts(self, output_dir: str = "models/feature_engineering"):
        """Save feature engineering artifacts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = output_path / f"scaler_{name}.pkl"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature selectors
        for name, selector_info in self.feature_selectors.items():
            selector_path = output_path / f"selector_{name}.pkl"
            joblib.dump(selector_info, selector_path)
            logger.info(f"Saved selector to {selector_path}")
        
        # Save PCA if exists
        if hasattr(self, 'pca'):
            pca_path = output_path / "pca.pkl"
            joblib.dump(self.pca, pca_path)
            logger.info(f"Saved PCA to {pca_path}")
    
    def load_artifacts(self, input_dir: str = "models/feature_engineering"):
        """Load feature engineering artifacts"""
        input_path = Path(input_dir)
        
        # Load scalers
        for scaler_file in input_path.glob("scaler_*.pkl"):
            name = scaler_file.stem.replace("scaler_", "")
            self.scalers[name] = joblib.load(scaler_file)
            logger.info(f"Loaded scaler: {name}")
        
        # Load selectors
        for selector_file in input_path.glob("selector_*.pkl"):
            name = selector_file.stem.replace("selector_", "")
            self.feature_selectors[name] = joblib.load(selector_file)
            logger.info(f"Loaded selector: {name}")
        
        # Load PCA
        pca_path = input_path / "pca.pkl"
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            logger.info("Loaded PCA")
    
    def run(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers
            save: Whether to save results and artifacts
            
        Returns:
            Engineered features DataFrame
        """
        logger.info("Starting feature engineering pipeline...")
        logger.info(f"Initial shape: {df.shape}")
        
        # Create features
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_v_features(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Shape after feature creation: {df.shape}")
        
        # Handle categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            logger.info(f"Encoded {len(categorical_cols)} categorical features")
        
        # Scale features
        df = self.scale_features(df, fit=fit)
        
        # Feature selection
        if 'Class' in df.columns:
            df = self.select_features(df, fit=fit)
        
        logger.info(f"Final shape: {df.shape}")
        logger.info("Feature engineering completed successfully")
        
        if save:
            output_path = Path(self.paths['data']['features']) / 'features.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved features to {output_path}")
            
            if fit:
                self.save_artifacts()
        
        return df


def main():
    """Main function to run feature engineering"""
    from data.ingestion import DataIngestion
    from data.cleaning import DataCleaner
    
    # Ingest and clean data
    ingestion = DataIngestion()
    df = ingestion.ingest_from_csv()
    
    cleaner = DataCleaner()
    cleaned_df = cleaner.run(df, save=False)
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.run(cleaned_df)
    
    print("\n" + "="*50)
    print("Feature Engineering Summary")
    print("="*50)
    print(f"Final samples: {len(features_df)}")
    print(f"Final features: {len(features_df.columns)}")
    print("\nFeature columns:")
    print(features_df.columns.tolist())
    
    return features_df


if __name__ == "__main__":
    main()

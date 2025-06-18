import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values dengan berbagai strategi"""
        logger.info(f"Handling missing values with strategy: {strategy}")
        df_processed = df.copy()
        
        if strategy == 'drop':
            df_processed = df_processed.dropna()
        elif strategy == 'mean':
            numeric_columns = df_processed.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if df_processed[col].isnull().any():
                    mean_value = df_processed[col].mean()
                    df_processed[col].fillna(mean_value, inplace=True)
        elif strategy == 'median':
            numeric_columns = df_processed.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if df_processed[col].isnull().any():
                    median_value = df_processed[col].median()
                    df_processed[col].fillna(median_value, inplace=True)
        elif strategy == 'mode':
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    mode_value = df_processed[col].mode()
                    if len(mode_value) > 0:
                        df_processed[col].fillna(mode_value[0], inplace=True)
        
        return df_processed
    
    def normalize_data(self, df, method='standard'):
        """Normalisasi data numerik"""
        logger.info(f"Normalizing data with method: {method}")
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=['number']).columns
        
        if len(numeric_columns) > 0:
            if method == 'standard':
                self.scaler = StandardScaler()
                df_processed[numeric_columns] = self.scaler.fit_transform(df_processed[numeric_columns])
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
                df_processed[numeric_columns] = self.scaler.fit_transform(df_processed[numeric_columns])
        
        return df_processed
    
    def remove_outliers(self, df, method='iqr', threshold=1.5):
        """Remove outliers dari data numerik"""
        logger.info(f"Removing outliers with method: {method}")
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=['number']).columns
        
        if method == 'iqr':
            for col in numeric_columns:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
        
        return df_processed
    
    def preprocess_pipeline(self, df, missing_strategy='mean', normalize_method='none', 
                          remove_outliers_flag=False, outlier_method='iqr'):
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df, missing_strategy)
        
        # Step 2: Remove outliers if requested
        if remove_outliers_flag:
            df_processed = self.remove_outliers(df_processed, outlier_method)
        
        # Step 3: Normalize data if requested
        if normalize_method != 'none':
            df_processed = self.normalize_data(df_processed, normalize_method)
        
        # Clean any remaining infinity/NaN values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.dropna()
        
        logger.info(f"Preprocessing completed. Shape: {df_processed.shape}")
        return df_processed

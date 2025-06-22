import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
import math

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None

    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values dengan berbagai strategi dan validasi ketat"""
        logger.info(f"Handling missing values with strategy: {strategy}")
        df_processed = df.copy()
        
        if strategy == 'drop':
            initial_rows = len(df_processed)
            df_processed = df_processed.dropna()
            logger.info(f"Dropped {initial_rows - len(df_processed)} rows with missing values")
            
        elif strategy == 'mean':
            numeric_columns = df_processed.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if df_processed[col].isnull().any():
                    # Calculate mean dengan handling infinity
                    col_data = df_processed[col].replace([np.inf, -np.inf], np.nan)
                    mean_value = col_data.mean()
                    
                    # Validasi mean value
                    if pd.isna(mean_value) or np.isinf(mean_value):
                        # Fallback ke median jika mean tidak valid
                        median_value = col_data.median()
                        if pd.isna(median_value) or np.isinf(median_value):
                            # Last resort: fill dengan 0
                            fill_value = 0.0
                        else:
                            fill_value = median_value
                    else:
                        fill_value = mean_value
                    
                    df_processed[col].fillna(fill_value, inplace=True)
                    logger.info(f"Filled missing values in {col} with {fill_value}")
                    
        elif strategy == 'median':
            numeric_columns = df_processed.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if df_processed[col].isnull().any():
                    col_data = df_processed[col].replace([np.inf, -np.inf], np.nan)
                    median_value = col_data.median()
                    
                    if pd.isna(median_value) or np.isinf(median_value):
                        # Fallback ke mean
                        mean_value = col_data.mean()
                        if pd.isna(mean_value) or np.isinf(mean_value):
                            fill_value = 0.0
                        else:
                            fill_value = mean_value
                    else:
                        fill_value = median_value
                    
                    df_processed[col].fillna(fill_value, inplace=True)
                    logger.info(f"Filled missing values in {col} with median {fill_value}")
                    
        elif strategy == 'mode':
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    # Clean infinity values first
                    if df_processed[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                    
                    mode_values = df_processed[col].mode()
                    if len(mode_values) > 0:
                        fill_value = mode_values[0]
                        df_processed[col].fillna(fill_value, inplace=True)
                        logger.info(f"Filled missing values in {col} with mode {fill_value}")
                    else:
                        # No mode available, use default value
                        if df_processed[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            df_processed[col].fillna(0, inplace=True)
                        else:
                            df_processed[col].fillna('Unknown', inplace=True)

        return df_processed

    def normalize_data(self, df, method='standard'):
        """Normalisasi data numerik dengan validasi ketat"""
        logger.info(f"Normalizing data with method: {method}")
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=['number']).columns

        if len(numeric_columns) > 0:
            # Clean infinity values sebelum normalisasi
            for col in numeric_columns:
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN dengan mean untuk kolom yang akan dinormalisasi
                if df_processed[col].isnull().any():
                    mean_val = df_processed[col].mean()
                    if pd.isna(mean_val):
                        mean_val = 0.0
                    df_processed[col].fillna(mean_val, inplace=True)

            try:
                if method == 'standard':
                    self.scaler = StandardScaler()
                    scaled_data = self.scaler.fit_transform(df_processed[numeric_columns])
                    
                elif method == 'minmax':
                    self.scaler = MinMaxScaler()
                    scaled_data = self.scaler.fit_transform(df_processed[numeric_columns])
                else:
                    return df_processed
                
                # Validasi hasil scaling
                if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
                    logger.warning("Scaling produced invalid values, using original data")
                    return df_processed
                
                # Replace dengan data yang sudah di-scale
                df_processed[numeric_columns] = scaled_data
                logger.info(f"Successfully normalized {len(numeric_columns)} columns")
                
            except Exception as e:
                logger.error(f"Error during normalization: {str(e)}")
                logger.warning("Using original data without normalization")
                return df_processed

        return df_processed

    def remove_outliers(self, df, method='iqr', threshold=1.5):
        """Remove outliers dari data numerik dengan validasi"""
        logger.info(f"Removing outliers with method: {method}")
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=['number']).columns
        initial_rows = len(df_processed)

        if method == 'iqr':
            for col in numeric_columns:
                # Clean infinity values first
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                
                # Calculate quartiles dengan handling NaN
                col_data = df_processed[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                try:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    
                    if pd.isna(Q1) or pd.isna(Q3):
                        logger.warning(f"Cannot calculate quartiles for {col}, skipping outlier removal")
                        continue
                        
                    IQR = Q3 - Q1
                    if IQR == 0:
                        logger.warning(f"IQR is 0 for {col}, skipping outlier removal")
                        continue
                        
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Create mask untuk outliers
                    outlier_mask = (
                        (df_processed[col] < lower_bound) | 
                        (df_processed[col] > upper_bound) |
                        df_processed[col].isnull()
                    )
                    
                    # Remove outliers
                    df_processed = df_processed[~outlier_mask]
                    
                except Exception as e:
                    logger.error(f"Error removing outliers from {col}: {str(e)}")
                    continue

        elif method == 'zscore':
            from scipy import stats
            for col in numeric_columns:
                try:
                    df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                    col_data = df_processed[col].dropna()
                    
                    if len(col_data) == 0:
                        continue
                        
                    z_scores = np.abs(stats.zscore(col_data))
                    outlier_indices = col_data[z_scores > threshold].index
                    df_processed = df_processed.drop(outlier_indices)
                    
                except Exception as e:
                    logger.error(f"Error removing outliers from {col} using z-score: {str(e)}")
                    continue

        removed_rows = initial_rows - len(df_processed)
        logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df_processed

    def preprocess_pipeline(self, df, missing_strategy='mean', normalize_method='none',
                           remove_outliers_flag=False, outlier_method='iqr'):
        """Complete preprocessing pipeline dengan validasi ketat"""
        logger.info("Starting preprocessing pipeline")
        
        # Validasi input DataFrame
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        initial_shape = df.shape
        logger.info(f"Initial data shape: {initial_shape}")
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df, missing_strategy)
        logger.info(f"After missing values handling: {df_processed.shape}")
        
        # Step 2: Remove outliers if requested
        if remove_outliers_flag:
            df_processed = self.remove_outliers(df_processed, outlier_method)
            logger.info(f"After outlier removal: {df_processed.shape}")
        
        # Step 3: Normalize data if requested
        if normalize_method != 'none':
            df_processed = self.normalize_data(df_processed, normalize_method)
            logger.info(f"After normalization: {df_processed.shape}")
        
        # Step 4: VALIDASI FINAL - Clean infinity/NaN values
        logger.info("Performing final data validation and cleaning")
        
        # Replace infinity dengan NaN terlebih dahulu
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        
        # Count invalid values
        invalid_values = df_processed.isnull().sum().sum()
        if invalid_values > 0:
            logger.warning(f"Found {invalid_values} invalid values after preprocessing")
            
            # Drop rows dengan NaN
            initial_rows = len(df_processed)
            df_processed = df_processed.dropna()
            final_rows = len(df_processed)
            
            if final_rows < initial_rows:
                logger.info(f"Dropped {initial_rows - final_rows} rows with invalid values")
        
        # Validasi final - pastikan tidak ada nilai invalid
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_invalid = df_processed[col].isnull().sum() + np.isinf(df_processed[col]).sum()
            if col_invalid > 0:
                logger.error(f"Column {col} still contains {col_invalid} invalid values")
                # Force clean
                mean_val = df_processed[col].mean()
                if pd.isna(mean_val) or np.isinf(mean_val):
                    mean_val = 0.0
                df_processed[col] = df_processed[col].fillna(mean_val)
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], mean_val)
        
        # Final validation
        final_shape = df_processed.shape
        if final_shape[0] == 0:
            raise ValueError("All data was removed during preprocessing. Please check your data quality.")
        
        # Check for any remaining invalid values
        total_invalid = 0
        for col in df_processed.select_dtypes(include=[np.number]).columns:
            invalid_count = np.sum(~np.isfinite(df_processed[col]))
            total_invalid += invalid_count
            
        if total_invalid > 0:
            logger.error(f"Still found {total_invalid} invalid values after final cleaning")
            # Last resort: replace all invalid values with 0
            for col in df_processed.select_dtypes(include=[np.number]).columns:
                df_processed[col] = np.where(np.isfinite(df_processed[col]), df_processed[col], 0)
        
        logger.info(f"Preprocessing completed successfully. Final shape: {final_shape}")
        logger.info(f"Data reduction: {initial_shape[0] - final_shape[0]} rows removed ({((initial_shape[0] - final_shape[0])/initial_shape[0]*100):.2f}%)")
        
        return df_processed

    def get_preprocessing_summary(self, original_df, processed_df):
        """Generate summary of preprocessing steps"""
        summary = {
            "original_shape": original_df.shape,
            "processed_shape": processed_df.shape,
            "rows_removed": original_df.shape[0] - processed_df.shape[0],
            "columns_processed": len(processed_df.select_dtypes(include=[np.number]).columns),
            "missing_values_before": original_df.isnull().sum().sum(),
            "missing_values_after": processed_df.isnull().sum().sum(),
            "data_quality_score": self._calculate_data_quality_score(processed_df)
        }
        return summary

    def _calculate_data_quality_score(self, df):
        """Calculate data quality score (0-100)"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            if total_cells == 0:
                return 0
            
            # Count invalid values
            invalid_count = 0
            for col in df.select_dtypes(include=[np.number]).columns:
                invalid_count += np.sum(~np.isfinite(df[col]))
            
            # Calculate completeness
            completeness = (total_cells - invalid_count) / total_cells * 100
            
            # Calculate consistency (percentage of numeric columns)
            numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns) * 100
            
            # Overall score
            quality_score = (completeness * 0.7 + numeric_ratio * 0.3)
            return min(100, max(0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {str(e)}")
            return 50  # Default score

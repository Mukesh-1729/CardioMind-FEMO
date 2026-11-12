import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataCleaner:
    """Handles missing values, duplicates, and outliers in dataset."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full cleaning pipeline."""
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
        
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.validate_ranges(df)
        df = self.detect_outliers(df)
        
        logger.info(f"Completed data cleaning. Final shape: {df.shape}")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicated rows."""
        initial_len = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_len - len(df)} duplicate rows.")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values with median for numeric and mode for categorical."""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_val = df_clean[col].median()
                    logger.info(f"Filling missing values in {col} with median {fill_val}")
                else:
                    fill_val = df_clean[col].mode()[0]
                    logger.info(f"Filling missing values in {col} with mode {fill_val}")
                df_clean[col] = df_clean[col].fillna(fill_val)
                
        return df_clean

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates basic feature ranges to remove impossible values."""
        df_clean = df.copy()
        
        # Validating specific columns if they exist
        if 'age' in df_clean.columns:
            # Assuming age could be in days (like in cardio_data) or years
            if df_clean['age'].max() > 200:
                # convert days to years if needed
                df_clean['age'] = df_clean['age'] / 365.25
            df_clean = df_clean[(df_clean['age'] >= 1) & (df_clean['age'] <= 120)]
            
        if 'ap_hi' in df_clean.columns:
            df_clean = df_clean[(df_clean['ap_hi'] >= 40) & (df_clean['ap_hi'] <= 300)]
        if 'ap_lo' in df_clean.columns:
            df_clean = df_clean[(df_clean['ap_lo'] >= 20) & (df_clean['ap_lo'] <= 200)]
            
        logger.info(f"Rows remaining after range validation: {len(df_clean)}")
        return df_clean

    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Clips outliers using the IQR method."""
        df_clean = df.copy()
        
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if len(df_clean[col].unique()) > 10:  # Skip categorical/binary numericals
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clip values instead of removing rows
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
        return df_clean

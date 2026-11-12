import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates data integrity, removes invalid physiological anomalies,
    handles missingness, and clips extreme outliers.
    """
    def validate_and_clean(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        initial_len = len(df)
        df_clean = df.copy()

        # 1. Deduplication
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_len - len(df_clean)} exact duplicate rows.")

        # 2. Convert age from days to years if needed
        if 'age' in df_clean.columns:
            if df_clean['age'].max() > 200:
                df_clean['age'] = df_clean['age'] / 365.25

        # 3. Filter impossible physiological ranges
        if 'age' in df_clean.columns:
            df_clean = df_clean[(df_clean['age'] >= 1) & (df_clean['age'] <= 120)]
        if 'ap_hi' in df_clean.columns:
            df_clean = df_clean[(df_clean['ap_hi'] >= 50) & (df_clean['ap_hi'] <= 280)]
        if 'ap_lo' in df_clean.columns:
            df_clean = df_clean[(df_clean['ap_lo'] >= 30) & (df_clean['ap_lo'] <= 180)]
        if 'height' in df_clean.columns:
            df_clean = df_clean[(df_clean['height'] >= 80) & (df_clean['height'] <= 250)]
        if 'weight' in df_clean.columns:
            df_clean = df_clean[(df_clean['weight'] >= 25) & (df_clean['weight'] <= 250)]

        # If SBP < DBP due to data recording swap, swap them back
        if 'ap_hi' in df_clean.columns and 'ap_lo' in df_clean.columns:
            swap_mask = df_clean['ap_hi'] < df_clean['ap_lo']
            if swap_mask.sum() > 0:
                logger.info(f"Swapping inverted SBP/DBP in {swap_mask.sum()} records.")
                df_clean.loc[swap_mask, ['ap_hi', 'ap_lo']] = df_clean.loc[swap_mask, ['ap_lo', 'ap_hi']].values

        # 4. Impute any remaining missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    val = df_clean[col].median()
                else:
                    val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(val)

        logger.info(f"Cleaned dataset: {len(df_clean)} rows remaining ({initial_len - len(df_clean)} discarded).")
        return df_clean

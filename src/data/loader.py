import os
import logging
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AutoDataLoader:
    """
    Automatically detects:
    - CSV structure
    - Target column
    - Numerical columns
    - Categorical columns
    - Missing values
    - Duplicates
    Without hardcoding assumptions before inspecting dataset.
    """
    def __init__(self, target_hints: List[str] = None):
        self.target_hints = target_hints or ['cardio', 'target', 'TenYearCHD', 'condition', 'output', 'label']

    def load_and_inspect(self, file_path: str) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        candidate_paths = [
            file_path,
            os.path.join('datasets', file_path),
            os.path.join('datasets', 'raw', file_path),
            os.path.join('datasets', 'raw', 'cardio_data_processed.csv'),
            os.path.join('datasets', 'raw', 'cardio_train.csv')
        ]
        resolved_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                resolved_path = p
                break

        if not resolved_path:
            raise FileNotFoundError(f"Dataset file not found at: {file_path} (checked candidates: {candidate_paths})")

        logger.info(f"Loading dataset from: {resolved_path}")
        df = pd.read_csv(resolved_path)
        logger.info(f"Initial raw shape: {df.shape}")

        # 1. Identify Target Column
        target_col = None
        for hint in self.target_hints:
            for col in df.columns:
                if col.lower() == hint.lower():
                    target_col = col
                    break
            if target_col:
                break

        if not target_col:
            # Fallback: check binary columns at the end of DataFrame
            for col in reversed(df.columns):
                if df[col].nunique() == 2:
                    target_col = col
                    break

        if not target_col:
            raise ValueError(f"Could not automatically detect binary target column in {df.columns.tolist()}")

        logger.info(f"Detected target column: '{target_col}'")

        # 2. Check and report duplicates
        num_duplicates = df.duplicated().sum()
        if num_duplicates > 0:
            logger.info(f"Found {num_duplicates} duplicate rows in raw dataset.")

        # 3. Separate features and target
        feature_cols = [c for c in df.columns if c != target_col and c.lower() not in ['id', 'unnamed: 0']]

        # 4. Detect numerical vs categorical columns
        numerical_cols = []
        categorical_cols = []

        for col in feature_cols:
            n_unique = df[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            # If numeric but <= 5 distinct values (e.g. gender, cholesterol grade, smoking), treated as categorical
            if not is_numeric or n_unique <= 5:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

        # 5. Missing values profile
        missing_counts = df[feature_cols].isnull().sum().to_dict()
        total_missing = sum(missing_counts.values())

        inspection_report = {
            'total_rows': len(df),
            'total_features': len(feature_cols),
            'target_column': target_col,
            'target_distribution': df[target_col].value_counts(normalize=True).to_dict(),
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'duplicate_rows': int(num_duplicates),
            'total_missing_values': int(total_missing),
            'missing_by_column': {k: int(v) for k, v in missing_counts.items() if v > 0}
        }

        logger.info(f"Inspection Summary: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical, {total_missing} missing values.")
        return df, target_col, inspection_report

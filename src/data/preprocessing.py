import logging
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles strict, leak-free train/val/test splitting and transformations:
    - Split: 80% Train, 10% Validation, 10% Test (Locked)
    - Fit scaler/encoder ONLY on Train
    - Transform Train, Validation, and Test
    """
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.numerical_cols = []
        self.categorical_cols = []
        self.feature_names = []

    def split_and_transform(
        self, df: pd.DataFrame, target_col: str, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        
        y = df[target_col].values.astype(int)
        X_df = df.drop(columns=[target_col], errors='ignore')
        
        # Remove ID-like columns
        drop_cols = [c for c in X_df.columns if c.lower() in ['id', 'unnamed: 0']]
        if drop_cols:
            X_df = X_df.drop(columns=drop_cols)

        # 1. Split out Test set FIRST (10% strictly held out)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_df, y, test_size=self.test_ratio, stratify=y, random_state=seed
        )

        # 2. Split Train and Validation from TrainVal (80% train / 10% val)
        val_relative_ratio = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_relative_ratio, stratify=y_train_val, random_state=seed
        )

        logger.info(f"Split shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape} (Locked)")

        # 3. Identify column types on Train
        self.numerical_cols = []
        self.categorical_cols = []

        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]) and X_train[col].nunique() > 5:
                self.numerical_cols.append(col)
            else:
                self.categorical_cols.append(col)

        logger.info(f"Fitting transformers on Train: {len(self.numerical_cols)} numerical, {len(self.categorical_cols)} categorical")

        # 4. Fit transformers STRICTLY on X_train
        if self.numerical_cols:
            self.scaler.fit(X_train[self.numerical_cols])

        if self.categorical_cols:
            self.encoder.fit(X_train[self.categorical_cols])

        # 5. Transform all partitions
        def _transform_split(split_df: pd.DataFrame) -> np.ndarray:
            parts = []
            if self.numerical_cols:
                num_scaled = self.scaler.transform(split_df[self.numerical_cols])
                parts.append(num_scaled)
            if self.categorical_cols:
                cat_encoded = self.encoder.transform(split_df[self.categorical_cols])
                parts.append(cat_encoded)
            return np.hstack(parts).astype(np.float32)

        X_train_enc = _transform_split(X_train)
        X_val_enc = _transform_split(X_val)
        X_test_enc = _transform_split(X_test)

        # Build feature names
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        if self.categorical_cols:
            cat_out = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
            feature_names.extend(cat_out)
        self.feature_names = feature_names

        logger.info(f"Transformation complete. Encoded dimension: {len(feature_names)}")
        return X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test, feature_names

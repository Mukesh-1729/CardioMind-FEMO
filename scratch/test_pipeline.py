import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

PROJECT_ROOT = Path("e:/CardioMind")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AutoDataLoader
from src.data.validator import DataValidator
from src.data.feature_engineering import FeatureEngineering
from src.data.preprocessing import DataPreprocessor
from src.data.partition import NonIIDPartitioner

print("Testing data loading and partition pipeline...")
loader = AutoDataLoader()
raw_df, target_col, _ = loader.load_and_inspect('datasets/raw/cardio_data_processed.csv')
validator = DataValidator()
clean_df = validator.validate_and_clean(raw_df, target_col)
feat_eng = FeatureEngineering()
df_eng = feat_eng.transform(clean_df)

preprocessor = DataPreprocessor(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
X_train, y_train, X_val, y_val, X_test, y_test, feat_names = preprocessor.split_and_transform(df_eng, target_col, seed=42)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print("Data pipeline check passed!")

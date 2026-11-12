import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatasetAdapter:
    """Adapter to handle loading and standardizing various cardiovascular datasets."""

    def __init__(self):
        self.standard_target = 'cardio'
        self.standard_features = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]

    def load(self, path: str) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
        """Loads a dataset and standardizes its columns based on file name or columns."""
        path_obj = Path(path)
        logger.info(f"Loading dataset from {path}")
        
        if path_obj.name == 'cardio_data_processed.csv':
            return self._load_cardio_data(path_obj)
        elif path_obj.name == 'framingham.csv':
            return self._load_framingham(path_obj)
        elif path_obj.name == 'HeartDiseaseTrain-Test.csv':
            return self._load_heart_disease(path_obj)
        elif path_obj.name == 'heart.dat':
            return self._load_heart_dat(path_obj)
        else:
            raise ValueError(f"Unknown dataset format for {path_obj.name}")

    def _load_cardio_data(self, path: Path) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
        df = pd.read_csv(path)
        y = df['cardio']
        X = df.drop(columns=['cardio', 'id', 'bp_category', 'bp_category_encoded'], errors='ignore')
        
        dataset_info = {
            'name': 'CardioData',
            'samples': len(df),
            'features': len(X.columns),
            'pos_ratio': y.mean()
        }
        return X, y, list(X.columns), dataset_info

    def _load_framingham(self, path: Path) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
        df = pd.read_csv(path).dropna()
        y = df['TenYearCHD']
        X = df.drop(columns=['TenYearCHD'])
        
        dataset_info = {
            'name': 'Framingham',
            'samples': len(df),
            'features': len(X.columns),
            'pos_ratio': y.mean()
        }
        return X, y, list(X.columns), dataset_info

    def _load_heart_disease(self, path: Path) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
        df = pd.read_csv(path)
        target_col = 'target' if 'target' in df.columns else 'condition' # generic handling
        # Using typical UCI columns mapping for binary prediction
        y = (df[target_col] > 0).astype(int)
        X = df.drop(columns=[target_col])
        
        dataset_info = {
            'name': 'UCI_HeartDisease',
            'samples': len(df),
            'features': len(X.columns),
            'pos_ratio': y.mean()
        }
        return X, y, list(X.columns), dataset_info

    def _load_heart_dat(self, path: Path) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
        df = pd.read_csv(path, sep=' ', header=None)
        
        y = (df.iloc[:, -1] == 2).astype(int)
        X = df.iloc[:, :-1]
        
        # Approximate feature names for statlog heart
        X.columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        dataset_info = {
            'name': 'Statlog_Heart',
            'samples': len(df),
            'features': len(X.columns),
            'pos_ratio': y.mean()
        }
        return X, y, list(X.columns), dataset_info

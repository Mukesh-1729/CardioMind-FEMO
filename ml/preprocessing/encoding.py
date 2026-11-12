import logging
import joblib
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

class FeatureEncoder:
    """Encodes categorical and scales numerical features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.num_cols = []
        self.cat_cols = []
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the encoder and scaler on data, then transforms it."""
        self.num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Hardcode some known categoricals that might be loaded as ints
        for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
            if col in self.num_cols and col in df.columns:
                if len(df[col].unique()) <= 5:
                    self.num_cols.remove(col)
                    self.cat_cols.append(col)
        
        df_encoded = pd.DataFrame()
        
        if self.num_cols:
            scaled_vals = self.scaler.fit_transform(df[self.num_cols])
            df_encoded = pd.concat([df_encoded, pd.DataFrame(scaled_vals, columns=self.num_cols, index=df.index)], axis=1)
            
        if self.cat_cols:
            encoded_vals = self.encoder.fit_transform(df[self.cat_cols])
            cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols)
            df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_vals, columns=cat_feature_names, index=df.index)], axis=1)
            
        return df_encoded
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms data using fitted parameters."""
        df_encoded = pd.DataFrame()
        
        if self.num_cols:
            scaled_vals = self.scaler.transform(df[self.num_cols])
            df_encoded = pd.concat([df_encoded, pd.DataFrame(scaled_vals, columns=self.num_cols, index=df.index)], axis=1)
            
        if self.cat_cols:
            encoded_vals = self.encoder.transform(df[self.cat_cols])
            cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols)
            df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_vals, columns=cat_feature_names, index=df.index)], axis=1)
            
        return df_encoded
        
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms scaled and encoded data."""
        # Reconstruct numericals
        df_inv = pd.DataFrame(index=df.index)
        if self.num_cols:
            inv_scaled = self.scaler.inverse_transform(df[self.num_cols])
            df_inv[self.num_cols] = inv_scaled
            
        # Reconstruct categoricals (simplified)
        if self.cat_cols:
            cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols)
            inv_cat = self.encoder.inverse_transform(df[cat_feature_names])
            for i, col in enumerate(self.cat_cols):
                df_inv[col] = inv_cat[:, i]
                
        return df_inv
        
    def save(self, path: str):
        joblib.dump({
            'scaler': self.scaler,
            'encoder': self.encoder,
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols
        }, path)
        logger.info(f"Saved encoder to {path}")
        
    def load(self, path: str):
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.encoder = data['encoder']
        self.num_cols = data['num_cols']
        self.cat_cols = data['cat_cols']
        logger.info(f"Loaded encoder from {path}")

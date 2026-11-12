import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Computes domain-specific cardiovascular features:
    - BMI = weight / (height/100)^2
    - PP = SBP - DBP
    - MAP = (SBP + 2 * DBP) / 3
    - BP_ratio = SBP / (DBP + eps)
    - Hypertension flag: (SBP >= 140) | (DBP >= 90)
    - Lifestyle risk score
    """
    def transform(self, df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
        df_feat = df.copy()

        # 1. Body Mass Index (BMI)
        if 'weight' in df_feat.columns and 'height' in df_feat.columns:
            height_m = df_feat['height'] / 100.0
            df_feat['bmi'] = df_feat['weight'] / (height_m ** 2)

        # 2. Blood Pressure Metrics (Pulse Pressure, MAP, BP Ratio, Hypertension Flag)
        sbp_col = 'ap_hi' if 'ap_hi' in df_feat.columns else ('systolic_bp' if 'systolic_bp' in df_feat.columns else None)
        dbp_col = 'ap_lo' if 'ap_lo' in df_feat.columns else ('diastolic_bp' if 'diastolic_bp' in df_feat.columns else None)

        if sbp_col and dbp_col:
            sbp = df_feat[sbp_col]
            dbp = df_feat[dbp_col]

            df_feat['pulse_pressure'] = sbp - dbp
            df_feat['map'] = (sbp + 2.0 * dbp) / 3.0
            df_feat['bp_ratio'] = sbp / (dbp + eps)
            df_feat['hypertension_flag'] = ((sbp >= 140) | (dbp >= 90)).astype(float)

        # 3. Lifestyle Score
        lifestyle_cols = ['smoke', 'alco', 'active']
        if all(c in df_feat.columns for c in lifestyle_cols):
            smk = (df_feat['smoke'] > 0).astype(float)
            alc = (df_feat['alco'] > 0).astype(float)
            act = (df_feat['active'] > 0).astype(float)
            df_feat['lifestyle_score'] = (smk + alc + (1.0 - act)) / 3.0

        logger.info(f"Feature engineering added new features. Total columns: {len(df_feat.columns)}")
        return df_feat

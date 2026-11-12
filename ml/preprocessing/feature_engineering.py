import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Generates new features for cardiovascular risk prediction."""

    def compute_bmi(self, weight_kg: pd.Series, height_cm: pd.Series) -> pd.Series:
        """BMI = weight / (height/100)^2"""
        height_m = height_cm / 100.0
        return weight_kg / (height_m ** 2)

    def compute_pulse_pressure(self, sbp: pd.Series, dbp: pd.Series) -> pd.Series:
        """PP = SBP - DBP"""
        return sbp - dbp

    def compute_map(self, sbp: pd.Series, dbp: pd.Series) -> pd.Series:
        """Mean Arterial Pressure = (SBP + 2*DBP) / 3"""
        return (sbp + 2 * dbp) / 3

    def compute_bp_ratio(self, sbp: pd.Series, dbp: pd.Series, eps: float = 1e-6) -> pd.Series:
        """SBP / max(DBP, eps)"""
        return sbp / np.maximum(dbp, eps)

    def compute_hypertension_flag(self, sbp: pd.Series, dbp: pd.Series, 
                                 sbp_thresh: float = 140, dbp_thresh: float = 90) -> pd.Series:
        """Flag for clinical hypertension."""
        return ((sbp >= sbp_thresh) | (dbp >= dbp_thresh)).astype(int)

    def compute_lifestyle_score(self, smoking: pd.Series, alcohol: pd.Series, activity: pd.Series) -> pd.Series:
        """
        Bounded 0-1 score representing unhealthy lifestyle.
        Formula: (smoking + alcohol + (1-activity)) / 3
        """
        # Convert all to binary if not already
        smk = (smoking > 0).astype(float)
        alc = (alcohol > 0).astype(float)
        act = (activity > 0).astype(float)
        
        score = (smk + alc + (1.0 - act)) / 3.0
        return np.clip(score, 0.0, 1.0)

    def compute_family_risk(self, family_history: pd.Series) -> pd.Series:
        """Extract family risk if column exists."""
        if family_history is not None:
            return family_history.astype(float)
        return None

    def transform(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Applies feature engineering based on available columns."""
        df_new = df.copy()
        
        # Assuming cardio data standard columns
        if 'weight' in df.columns and 'height' in df.columns:
            df_new['bmi'] = self.compute_bmi(df['weight'], df['height'])
            
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df_new['pulse_pressure'] = self.compute_pulse_pressure(df['ap_hi'], df['ap_lo'])
            df_new['map'] = self.compute_map(df['ap_hi'], df['ap_lo'])
            df_new['bp_ratio'] = self.compute_bp_ratio(df['ap_hi'], df['ap_lo'])
            
            ht_thresh = config.get('hypertension_thresholds', {'systolic': 140, 'diastolic': 90})
            df_new['hypertension_flag'] = self.compute_hypertension_flag(
                df['ap_hi'], df['ap_lo'], ht_thresh['systolic'], ht_thresh['diastolic']
            )
            
        if 'smoke' in df.columns and 'alco' in df.columns and 'active' in df.columns:
            df_new['lifestyle_score'] = self.compute_lifestyle_score(df['smoke'], df['alco'], df['active'])
            
        logger.info(f"Added new features. Total features now: {len(df_new.columns)}")
        return df_new

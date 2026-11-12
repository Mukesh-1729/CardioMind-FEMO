"""
Tests for CardioMind-FEMO: Feature Engineering

Tests the FeatureEngineer class with known input-output pairs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np
import pandas as pd


class TestBMI:
    """Test BMI computation."""

    def test_bmi_basic(self):
        """BMI = weight_kg / (height_m)^2"""
        # 70kg, 175cm -> 70 / (1.75)^2 = 70 / 3.0625 ≈ 22.86
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        bmi = fe.compute_bmi(weight_kg=70.0, height_cm=175.0)
        assert abs(bmi - 22.857) < 0.01

    def test_bmi_obesity_class(self):
        """Test obese BMI value > 30."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        bmi = fe.compute_bmi(weight_kg=100.0, height_cm=170.0)
        assert bmi > 30.0

    def test_bmi_underweight(self):
        """Test underweight BMI value < 18.5."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        bmi = fe.compute_bmi(weight_kg=45.0, height_cm=175.0)
        assert bmi < 18.5


class TestPulsePressure:
    """Test Pulse Pressure computation."""

    def test_pp_basic(self):
        """PP = SBP - DBP"""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        pp = fe.compute_pulse_pressure(sbp=120.0, dbp=80.0)
        assert pp == 40.0

    def test_pp_hypertensive(self):
        """Test hypertensive pulse pressure > 60."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        pp = fe.compute_pulse_pressure(sbp=160.0, dbp=90.0)
        assert pp == 70.0


class TestMAP:
    """Test Mean Arterial Pressure computation."""

    def test_map_basic(self):
        """MAP = (SBP + 2*DBP) / 3"""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        map_val = fe.compute_map(sbp=120.0, dbp=80.0)
        expected = (120 + 2 * 80) / 3  # 93.33...
        assert abs(map_val - expected) < 0.01


class TestHypertensionFlag:
    """Test hypertension indicator."""

    def test_normal_bp(self):
        """Normal BP should not trigger hypertension flag."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        flag = fe.compute_hypertension_flag(sbp=120.0, dbp=80.0, sbp_thresh=140, dbp_thresh=90)
        assert flag == 0

    def test_high_systolic(self):
        """High systolic should trigger flag."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        flag = fe.compute_hypertension_flag(sbp=150.0, dbp=80.0, sbp_thresh=140, dbp_thresh=90)
        assert flag == 1

    def test_high_diastolic(self):
        """High diastolic should trigger flag."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        flag = fe.compute_hypertension_flag(sbp=130.0, dbp=95.0, sbp_thresh=140, dbp_thresh=90)
        assert flag == 1

    def test_configurable_thresholds(self):
        """Custom thresholds should work correctly."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        # Normal BP under standard thresholds, high under custom thresholds
        flag = fe.compute_hypertension_flag(sbp=125.0, dbp=82.0, sbp_thresh=120, dbp_thresh=80)
        assert flag == 1


class TestLifestyleScore:
    """Test lifestyle score computation."""

    def test_healthy_lifestyle(self):
        """Never smoker, no alcohol, active -> low score (low risk)."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        score = fe.compute_lifestyle_score(smoking=0, alcohol=0, activity=1)
        assert 0.0 <= score <= 1.0
        # Active non-smoker should have lower score than inactive smoker
        smoker_score = fe.compute_lifestyle_score(smoking=2, alcohol=1, activity=0)
        assert score < smoker_score

    def test_score_bounds(self):
        """Lifestyle score must always be in [0, 1]."""
        from ml.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        for smoking in [0, 1, 2]:
            for alcohol in [0, 1]:
                for activity in [0, 1]:
                    score = fe.compute_lifestyle_score(smoking=smoking, alcohol=alcohol, activity=activity)
                    assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for ({smoking},{alcohol},{activity})"

"""
Tests for CardioMind-FEMO: APCRS Calculator

Verifies APCRS score computation, clamping, and risk classification.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np


class TestAPCRS:
    """Test APCRS calculation."""

    def setup_method(self):
        from ml.apcrs.risk_score import APCRSCalculator
        self.calc = APCRSCalculator()
        self.default_config = {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}

    def test_apcrs_range(self):
        """APCRS must always be in [0, 100]."""
        for _ in range(100):
            p_t = np.random.uniform(0, 1)
            R_t = np.random.uniform(0, 1)
            delta_R = np.random.uniform(-1, 1)
            score = self.calc.compute_apcrs(p_t, R_t, delta_R, self.default_config)
            assert 0 <= score <= 100, f"APCRS {score} out of [0,100] range"

    def test_apcrs_weights_sum_constraint(self):
        """alpha + beta + gamma must equal 1.0."""
        config = self.default_config
        total = config['alpha'] + config['beta'] + config['gamma']
        assert abs(total - 1.0) < 1e-9, f"Alpha+Beta+Gamma = {total}, must be 1.0"

    def test_high_risk_patient(self):
        """Patient with p_t=1.0, R_t=1.0, delta_R=0.5 should have very high APCRS."""
        score = self.calc.compute_apcrs(p_t=1.0, R_t=1.0, delta_R=0.5, config=self.default_config)
        assert score >= 80, f"Expected high APCRS but got {score}"

    def test_low_risk_patient(self):
        """Patient with p_t=0.0, R_t=0.0, delta_R=-0.5 should have very low APCRS."""
        score = self.calc.compute_apcrs(p_t=0.0, R_t=0.0, delta_R=-0.5, config=self.default_config)
        assert score <= 20, f"Expected low APCRS but got {score}"

    def test_apcrs_formula(self):
        """Verify formula: APCRS = 100 * clip(alpha*p + beta*R + gamma*dR, 0, 1)"""
        alpha, beta, gamma = 0.5, 0.3, 0.2
        p_t, R_t, delta_R = 0.7, 0.6, 0.1
        config = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        expected_inner = alpha * p_t + beta * R_t + gamma * delta_R
        expected_inner = max(0, min(1, expected_inner))
        expected = 100 * expected_inner
        score = self.calc.compute_apcrs(p_t, R_t, delta_R, config)
        assert abs(score - expected) < 0.01


class TestRiskClassification:
    """Test risk classification from APCRS."""

    def setup_method(self):
        from ml.apcrs.risk_score import APCRSCalculator
        self.calc = APCRSCalculator()
        self.thresholds = {'low_max': 30, 'moderate_max': 60, 'high_max': 80, 'very_high_max': 100}

    def test_low_risk(self):
        assert self.calc.classify_risk(20, self.thresholds) == 'Low'

    def test_moderate_risk(self):
        assert self.calc.classify_risk(45, self.thresholds) == 'Moderate'

    def test_high_risk(self):
        assert self.calc.classify_risk(70, self.thresholds) == 'High'

    def test_very_high_risk(self):
        assert self.calc.classify_risk(90, self.thresholds) == 'Very High'


class TestRiskTrend:
    """Test trend calculation."""

    def setup_method(self):
        from ml.apcrs.risk_trend import RiskTrendAnalyzer
        self.analyzer = RiskTrendAnalyzer()

    def test_worsening_trend(self):
        history = [40, 45, 50, 55, 65]
        result = self.analyzer.analyze_trend(history)
        assert result['direction'] == 'Worsening'

    def test_improving_trend(self):
        history = [70, 65, 55, 45, 35]
        result = self.analyzer.analyze_trend(history)
        assert result['direction'] == 'Improving'

    def test_stable_trend(self):
        history = [50, 51, 49, 50, 50]
        result = self.analyzer.analyze_trend(history)
        assert result['direction'] == 'Stable'

from app.services.predictor import build_risk_assessment
from app.schemas.risk import RiskAssessmentRequest


def test_risk_assessment_response_shape(monkeypatch):
    class DummyModel:
        @staticmethod
        def predict(_):
            return [0.82]

    monkeypatch.setattr('app.services.predictor.load_xgboost_model', lambda: DummyModel())

    assessment = build_risk_assessment(
        RiskAssessmentRequest(
            age=62,
            sex='male',
            systolic_bp=154,
            diastolic_bp=94,
            cholesterol=242,
            smoking_status='current',
            diabetes=True,
            family_history=True,
            bmi=29.4,
            heart_rate=86
        )
    )

    assert 'risk_score' in assessment
    assert 'prediction' in assessment
    assert 'confidence' in assessment
    assert 'risk_level' in assessment
    assert assessment['prediction'] == 'positive'
    assert assessment['risk_level'] == 'high'

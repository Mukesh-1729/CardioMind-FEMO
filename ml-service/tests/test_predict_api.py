import time

from fastapi.testclient import TestClient


class DummyModel:
    @staticmethod
    def predict(_):
        return [0.67]


def test_predict_endpoint_contract_and_latency(monkeypatch):
    monkeypatch.setattr('app.main.load_xgboost_model', lambda: DummyModel())
    monkeypatch.setattr('app.services.predictor.load_xgboost_model', lambda: DummyModel())

    from app.main import app

    client = TestClient(app)

    payload = {
        'age': 58,
        'sex': 'male',
        'systolic_bp': 148,
        'diastolic_bp': 90,
        'cholesterol': 230,
        'smoking_status': 'former',
        'diabetes': True,
        'family_history': True,
        'bmi': 28.1,
        'heart_rate': 84
    }

    started = time.perf_counter()
    response = client.post('/predict/risk', json=payload)
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 2.0

    body = response.json()

    assert set(body.keys()) == {'risk_score', 'prediction', 'confidence', 'risk_level'}
    assert isinstance(body['risk_score'], float)
    assert isinstance(body['prediction'], str)
    assert isinstance(body['confidence'], float)
    assert isinstance(body['risk_level'], str)

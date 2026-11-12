import os
import time
from threading import Lock
from typing import Any, Dict

import numpy as np
import xgboost as xgb
from fastapi import HTTPException

from app.core.config import settings
from app.schemas.risk import RiskAssessmentRequest

_MODEL: xgb.Booster | None = None
_MODEL_LOCK = Lock()


def _resolve_model_path() -> str:
    model_path = settings.xgboost_model_path
    if os.path.isabs(model_path):
        return model_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.abspath(os.path.join(project_root, model_path))


def load_xgboost_model(force_reload: bool = False) -> xgb.Booster:
    global _MODEL

    if _MODEL is not None and not force_reload:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None and not force_reload:
            return _MODEL

        model_path = _resolve_model_path()
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail=(
                    'XGBoost model artifact not found. '
                    f'Expected file at: {model_path}'
                )
            )

        booster = xgb.Booster()
        booster.load_model(model_path)
        _MODEL = booster
        return _MODEL


def _encode_categorical(value: str, mapping: Dict[str, float], field_name: str) -> float:
    encoded = mapping.get(value)
    if encoded is None:
        raise HTTPException(status_code=422, detail=f'Unsupported value for {field_name}: {value}')
    return encoded


def _to_feature_vector(payload: RiskAssessmentRequest) -> np.ndarray:
    sex_code = _encode_categorical(
        payload.sex,
        {'female': 0.0, 'male': 1.0, 'other': 2.0},
        'sex'
    )
    smoking_code = _encode_categorical(
        payload.smoking_status,
        {'never': 0.0, 'former': 1.0, 'current': 2.0},
        'smoking_status'
    )

    return np.array([
        float(payload.age),
        sex_code,
        float(payload.systolic_bp),
        float(payload.diastolic_bp),
        float(payload.cholesterol),
        smoking_code,
        1.0 if payload.diabetes else 0.0,
        1.0 if payload.family_history else 0.0,
        float(payload.bmi or 0.0),
        float(payload.heart_rate or 72.0)
    ], dtype=np.float32).reshape(1, -1)


def _risk_level(probability: float) -> str:
    if probability >= settings.high_risk_threshold:
        return 'high'
    if probability >= settings.risk_threshold:
        return 'moderate'
    return 'low'


def build_risk_assessment(payload: RiskAssessmentRequest) -> Dict[str, Any]:
    start_time = time.perf_counter()
    model = load_xgboost_model()
    features = _to_feature_vector(payload)

    dmatrix = xgb.DMatrix(features)
    risk_probability = float(model.predict(dmatrix)[0])
    risk_probability = float(np.clip(risk_probability, 0.0, 1.0))

    elapsed = time.perf_counter() - start_time
    if elapsed >= 2.0:
        raise HTTPException(status_code=503, detail='Inference exceeded 2 second SLA.')

    predicted_positive = risk_probability >= settings.risk_threshold
    confidence = risk_probability if predicted_positive else (1.0 - risk_probability)

    return {
        'risk_score': round(risk_probability, 4),
        'prediction': 'positive' if predicted_positive else 'negative',
        'confidence': round(float(confidence), 4),
        'risk_level': _risk_level(risk_probability)
    }

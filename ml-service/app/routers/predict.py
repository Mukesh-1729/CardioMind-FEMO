from fastapi import APIRouter
from app.schemas.risk import RiskAssessmentRequest, RiskAssessmentResponse
from app.services.predictor import build_risk_assessment

router = APIRouter()


@router.post('/risk', response_model=RiskAssessmentResponse)
def predict_risk(payload: RiskAssessmentRequest):
    return build_risk_assessment(payload)

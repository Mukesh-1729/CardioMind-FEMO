from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RiskAssessmentRequest(BaseModel):
    patient_id: Optional[str] = None
    age: int = Field(ge=0, le=130)
    sex: Literal['male', 'female', 'other']
    systolic_bp: float = Field(ge=60, le=280)
    diastolic_bp: float = Field(ge=30, le=180)
    cholesterol: float = Field(ge=50, le=600)
    smoking_status: Literal['never', 'former', 'current']
    diabetes: bool
    family_history: bool
    bmi: Optional[float] = Field(default=0.0, ge=0, le=120)
    heart_rate: Optional[float] = Field(default=72.0, ge=20, le=250)

    @field_validator('sex', mode='before')
    @classmethod
    def normalize_sex(cls, value: str) -> str:
        return str(value).strip().lower()

    @field_validator('smoking_status', mode='before')
    @classmethod
    def normalize_smoking_status(cls, value: str) -> str:
        return str(value).strip().lower()


class RiskAssessmentResponse(BaseModel):
    risk_score: float
    prediction: str
    confidence: float
    risk_level: str

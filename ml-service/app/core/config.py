from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = 'CardioMind ML Service'
    environment: str = 'development'
    xgboost_model_path: str = 'app/models/xgboost_cardiovascular_model.json'
    risk_threshold: float = 0.5
    high_risk_threshold: float = 0.75


settings = Settings()

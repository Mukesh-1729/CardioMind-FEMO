Place the trained XGBoost model artifact at:

app/models/xgboost_cardiovascular_model.json

The FastAPI service loads this model at startup. You can override the path with the XGBOOST_MODEL_PATH environment variable.

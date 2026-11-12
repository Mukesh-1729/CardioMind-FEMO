import lightgbm as lgb
import joblib
from typing import Dict, Any
from .base_model import BaseCardioModel

class LightGBMModel(BaseCardioModel):
    def train(self, X, y, config: Dict[str, Any]):
        params = config.get('models', {}).get('lightgbm', {}).copy()
        params.setdefault('verbose', -1)
        self.model = lgb.LGBMClassifier(**params, random_state=config.get('random_seeds', [42])[0])
        self.model.fit(X, y)
        self.metadata = {'model_name': 'LightGBM', 'params': params}
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, path: str):
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        self.model = joblib.load(path)

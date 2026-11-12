from catboost import CatBoostClassifier
from typing import Dict, Any
from .base_model import BaseCardioModel

class CatBoostModel(BaseCardioModel):
    def train(self, X, y, config: Dict[str, Any]):
        params = config.get('models', {}).get('catboost', {})
        self.model = CatBoostClassifier(**params, random_state=config.get('random_seeds', [42])[0], verbose=0)
        self.model.fit(X, y)
        self.metadata = {'model_name': 'CatBoost', 'params': params}
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, path: str):
        self.model.save_model(path)
        
    def load(self, path: str):
        self.model = CatBoostClassifier()
        self.model.load_model(path)

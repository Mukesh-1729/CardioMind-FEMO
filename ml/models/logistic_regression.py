import joblib
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from .base_model import BaseCardioModel

class LogisticRegressionModel(BaseCardioModel):
    def train(self, X, y, config: Dict[str, Any]):
        params = config.get('models', {}).get('logistic_regression', {})
        self.model = LogisticRegression(**params, random_state=config.get('random_seeds', [42])[0])
        self.model.fit(X, y)
        self.metadata = {'model_name': 'LogisticRegression', 'params': params}
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, path: str):
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        self.model = joblib.load(path)

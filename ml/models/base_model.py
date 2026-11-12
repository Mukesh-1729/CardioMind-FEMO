import os
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseCardioModel(ABC):
    """Abstract base class for all cardiovascular risk models."""
    
    def __init__(self):
        self.model = None
        self.metadata = {}
        
    @abstractmethod
    def train(self, X, y, config: Dict[str, Any]):
        pass
        
    @abstractmethod
    def predict_proba(self, X):
        pass
        
    @abstractmethod
    def save(self, path: str):
        pass
        
    @abstractmethod
    def load(self, path: str):
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

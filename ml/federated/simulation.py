import logging
import json
import numpy as np
from typing import Dict
from pathlib import Path

from .client import FederatedClient
from .server import FederatedServer

logger = logging.getLogger(__name__)

class FederatedSimulation:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.clients = []
        self.server = None
        self.test_data = None
        
    def setup(self, config: Dict):
        # Load test data
        test_path = self.data_dir / 'test_data.npz'
        if test_path.exists():
            test_np = np.load(test_path)
            self.test_data = {'X': test_np['X'], 'y': test_np['y']}
            
        n_clients = config.get('n_clients', 5)
        self.server = FederatedServer(config, n_clients)
        
        self.clients = []
        for i in range(n_clients):
            client_path = self.data_dir / 'simulated_hospitals' / f'client_{i}' / 'data.npz'
            if client_path.exists():
                client = FederatedClient(f"client_{i}", str(client_path), config)
                client.load_local_data()
                self.clients.append(client)
                
        logger.info(f"Setup complete with {len(self.clients)} clients.")
        
    def run_iid(self, strategy: str, n_rounds: int):
        logger.info(f"Running IID Simulation with strategy: {strategy}")
        logs = self.server.run_simulation(n_rounds, strategy, self.clients, self.test_data)
        return self._format_results(logs)
        
    def run_label_skew(self, strategy: str, n_rounds: int):
        logger.info(f"Running Label Skew Simulation with strategy: {strategy}")
        return self.run_iid(strategy, n_rounds) # Placeholder
        
    def run_feature_skew(self, strategy: str, n_rounds: int):
        return self.run_iid(strategy, n_rounds) # Placeholder
        
    def run_quantity_skew(self, strategy: str, n_rounds: int):
        return self.run_iid(strategy, n_rounds) # Placeholder
        
    def run_mixed_noniid(self, strategy: str, n_rounds: int):
        return self.run_iid(strategy, n_rounds) # Placeholder
        
    def _format_results(self, logs):
        final_metrics = logs[-1]['metrics'] if logs else {}
        return {
            'per_round_metrics': logs,
            'final_metrics': final_metrics,
            'communication_bytes': len(logs) * len(self.clients) * 1000 * 2, # Rough estimate
            'training_time': 0.0 # Placeholder
        }

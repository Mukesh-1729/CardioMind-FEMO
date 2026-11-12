import logging
import copy
from typing import Dict, List

logger = logging.getLogger(__name__)

class AblationStudy:
    """
    5 variants:
    - A: L_pred only (no regularization)
    - B: L_pred + L_prox
    - C: L_pred + L_cal
    - D: L_pred + L_cal + L_prox
    - E: Full FEMO (L_pred + L_cal + L_prox + L_risk)
    """
    
    def run_variant(self, variant: str, clients: List, server, config: Dict, n_rounds: int, test_data) -> Dict:
        logger.info(f"Running Ablation Variant {variant}")
        
        # Create a specific config for this variant
        v_config = copy.deepcopy(config)
        femo_cfg = v_config['federated']['femo']
        
        if variant == 'A':
            femo_cfg['lambda_cal'] = 0.0
            femo_cfg['lambda_prox'] = 0.0
            femo_cfg['lambda_risk'] = 0.0
        elif variant == 'B':
            femo_cfg['lambda_cal'] = 0.0
            femo_cfg['lambda_risk'] = 0.0
        elif variant == 'C':
            femo_cfg['lambda_prox'] = 0.0
            femo_cfg['lambda_risk'] = 0.0
        elif variant == 'D':
            femo_cfg['lambda_risk'] = 0.0
        elif variant == 'E':
            pass # Use original
            
        # Run simulation using FEMO strategy but altered config
        # Setup clean server
        from ..federated.server import FederatedServer
        v_server = FederatedServer(v_config, len(clients))
        v_server.initialize_global_model(dim=test_data['X'].shape[1])
        
        logs = v_server.run_simulation(n_rounds, 'femo', clients, test_data)
        
        final_metrics = logs[-1]['metrics']
        return final_metrics
        
    def run_all_variants(self, clients, server, config, n_rounds, test_data):
        results = {}
        for var in ['A', 'B', 'C', 'D', 'E']:
            results[var] = self.run_variant(var, clients, server, config, n_rounds, test_data)
        return results

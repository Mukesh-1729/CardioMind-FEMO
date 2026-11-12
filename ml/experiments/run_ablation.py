import os
import sys
import json
import copy
import logging
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.federated.client import FederatedClient
from ml.federated.server import FederatedServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / 'ml' / 'config.json')

    with open(config_path, 'r') as f:
        base_config = json.load(f)

    data_dir = PROJECT_ROOT / 'ml' / 'data' / 'processed'
    test_path = data_dir / 'test_data.npz'
    hospitals_dir = data_dir / 'simulated_hospitals'

    test_data = np.load(test_path, allow_pickle=True)
    n_clients = base_config.get('n_clients', 5)
    n_rounds = base_config.get('federated', {}).get('n_rounds', 20)
    seeds = base_config.get('random_seeds', [42, 123, 456, 789, 1024])

    variants = {
        'A (L_pred only)': {'lambda_cal': 0.0, 'lambda_prox': 0.0, 'lambda_risk': 0.0},
        'B (L_pred + L_prox)': {'lambda_cal': 0.0, 'lambda_prox': 0.01, 'lambda_risk': 0.0},
        'C (L_pred + L_cal)': {'lambda_cal': 0.1, 'lambda_prox': 0.0, 'lambda_risk': 0.0},
        'D (L_pred + L_cal + L_prox)': {'lambda_cal': 0.1, 'lambda_prox': 0.01, 'lambda_risk': 0.0},
        'E (Full FEMO)': {'lambda_cal': 0.1, 'lambda_prox': 0.01, 'lambda_risk': 0.05}
    }

    logger.info(f"=== Running Systematic Ablation Study across {len(variants)} variants ===")

    all_ablation_results = []
    summary_by_variant = {}

    for var_name, lambdas in variants.items():
        logger.info(f"\n--- Running Ablation Variant {var_name} ---")
        var_seed_metrics = []

        for seed in seeds:
            np.random.seed(seed)
            var_config = copy.deepcopy(base_config)
            var_config['federated']['femo'].update(lambdas)

            clients = [
                FederatedClient(
                    client_id=f"Hospital_{i+1}",
                    data_path=str(hospitals_dir / f"client_{i}" / "data.npz"),
                    config=var_config
                )
                for i in range(n_clients)
            ]

            server = FederatedServer(var_config, n_clients)
            logs = server.run_simulation(n_rounds=n_rounds, strategy='femo', clients=clients, test_data=test_data)

            final_m = logs[-1]['metrics']
            final_m['variant'] = var_name
            final_m['seed'] = seed
            all_ablation_results.append(final_m)
            var_seed_metrics.append(final_m)

        df_v = pd.DataFrame(var_seed_metrics)
        summary_by_variant[var_name] = {
            'roc_auc_mean': float(df_v['roc_auc'].mean()),
            'roc_auc_std': float(df_v['roc_auc'].std()),
            'accuracy_mean': float(df_v['accuracy'].mean()),
            'accuracy_std': float(df_v['accuracy'].std()),
            'brier_mean': float(df_v['brier_score'].mean()),
            'brier_std': float(df_v['brier_score'].std()),
            'ece_mean': float(df_v['ece'].mean()),
            'ece_std': float(df_v['ece'].std()),
            'worst_client_mean': float(df_v['worst_client_performance'].mean())
        }

    results_dir = PROJECT_ROOT / 'experiments' / 'results'
    tables_dir = PROJECT_ROOT / 'experiments' / 'tables'
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(all_ablation_results)
    df_all.to_csv(results_dir / 'ablation_results.csv', index=False)

    with open(tables_dir / 'table7_ablation.json', 'w') as f:
        json.dump(summary_by_variant, f, indent=2)

    pd.DataFrame.from_dict(summary_by_variant, orient='index').to_csv(tables_dir / 'table7_ablation.csv')

    print("\n" + "="*80)
    print(" ABLATION STUDY RESULTS SUMMARY (5 SEEDS MEAN)")
    print("="*80)
    for v_name, stats in summary_by_variant.items():
        print(f"{v_name:<30} | AUC: {stats['roc_auc_mean']:.4f} | Acc: {stats['accuracy_mean']:.4f} | Brier: {stats['brier_mean']:.4f} | ECE: {stats['ece_mean']:.4f} | Worst Client: {stats['worst_client_mean']:.4f}")
    print("="*80 + "\n")

    return summary_by_variant

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CardioMind Ablation Runner")
    parser.add_argument('--config', default=None, help='Path to config JSON')
    args = parser.parse_args()
    run(args.config)

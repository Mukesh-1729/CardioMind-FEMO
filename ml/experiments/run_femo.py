import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.federated.client import FederatedClient
from ml.federated.server import FederatedServer
from ml.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / 'ml' / 'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = PROJECT_ROOT / 'ml' / 'data' / 'processed'
    test_path = data_dir / 'test_data.npz'
    hospitals_dir = data_dir / 'simulated_hospitals'

    if not test_path.exists() or not hospitals_dir.exists():
        logger.info("Preprocessed partitions not found. Running PreprocessingPipeline...")
        pipeline = PreprocessingPipeline()
        pipeline.run(
            dataset_path=str(PROJECT_ROOT / 'datasets' / 'raw' / 'cardio_data_processed.csv'),
            config_path=config_path,
            output_dir=str(data_dir)
        )

    test_data = np.load(test_path, allow_pickle=True)
    n_clients = config.get('n_clients', 5)
    n_rounds = config.get('federated', {}).get('n_rounds', 20)
    seeds = config.get('random_seeds', [42, 123, 456, 789, 1024])

    logger.info(f"=== Running Flagship FEMO across {n_clients} clients, {n_rounds} rounds, {len(seeds)} seeds ===")

    all_seed_results = []
    round_convergence = []
    per_client_final = []

    for seed in seeds:
        np.random.seed(seed)
        clients = [
            FederatedClient(
                client_id=f"Hospital_{i+1}",
                data_path=str(hospitals_dir / f"client_{i}" / "data.npz"),
                config=config
            )
            for i in range(n_clients)
        ]

        server = FederatedServer(config, n_clients)
        logs = server.run_simulation(n_rounds=n_rounds, strategy='femo', clients=clients, test_data=test_data)

        final_metrics = logs[-1]['metrics']
        final_metrics['seed'] = seed
        final_metrics['strategy'] = 'FEMO'
        all_seed_results.append(final_metrics)

        for log_entry in logs:
            r_metrics = log_entry['metrics'].copy()
            r_metrics['round'] = log_entry['round']
            r_metrics['seed'] = seed
            r_metrics['strategy'] = 'FEMO'
            round_convergence.append(r_metrics)

        # Track per-client performance for seed 42
        if seed == seeds[0]:
            for i, c in enumerate(clients):
                c_val = c.calculate_validation_metrics()
                per_client_final.append({
                    'client_id': f"Hospital_{i+1}",
                    'n_samples': c.n_samples,
                    'roc_auc': c_val['roc_auc'],
                    'accuracy': c_val['accuracy'],
                    'brier_score': c_val['brier_score']
                })

    results_dir = PROJECT_ROOT / 'experiments' / 'results'
    tables_dir = PROJECT_ROOT / 'experiments' / 'tables'
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_seeds = pd.DataFrame(all_seed_results)
    df_seeds.to_csv(results_dir / 'femo_results.csv', index=False)

    df_conv = pd.DataFrame(round_convergence)
    df_conv.to_csv(results_dir / 'femo_convergence.csv', index=False)

    # Save per-client table
    with open(tables_dir / 'table5_clients.json', 'w') as f:
        json.dump(per_client_final, f, indent=2)

    logger.info(f"Saved FEMO results to {results_dir / 'femo_results.csv'}")

    print("\n" + "="*60)
    print(" FEMO (FLAGSHIP) FINAL RESULTS (MEAN +/- STD)")
    print("="*60)
    print(f"ROC-AUC:      {df_seeds['roc_auc'].mean():.4f} +/- {df_seeds['roc_auc'].std():.4f}")
    print(f"Accuracy:     {df_seeds['accuracy'].mean():.4f} +/- {df_seeds['accuracy'].std():.4f}")
    print(f"Brier Score:  {df_seeds['brier_score'].mean():.4f} +/- {df_seeds['brier_score'].std():.4f}")
    print(f"ECE:          {df_seeds['ece'].mean():.4f} +/- {df_seeds['ece'].std():.4f}")
    print(f"Worst Client: {df_seeds['worst_client_performance'].mean():.4f} +/- {df_seeds['worst_client_performance'].std():.4f}")
    print(f"Client Std:   {df_seeds['client_variance'].mean():.4f} +/- {df_seeds['client_variance'].std():.4f}")
    print("="*60 + "\n")

    return all_seed_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CardioMind FEMO Runner")
    parser.add_argument('--config', default=None, help='Path to config JSON')
    args = parser.parse_args()
    run(args.config)

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiments import run_baselines
from ml.experiments import run_fedavg
from ml.experiments import run_fedprox
from ml.experiments import run_femo
from ml.experiments import run_ablation
from ml.evaluation.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_all(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / 'ml' / 'config.json')

    logger.info("===============================================================")
    logger.info(" STARTING MASTER CARDIOMIND EXPERIMENT & TRAINING PIPELINE")
    logger.info("===============================================================")

    # 1. Train Centralized Baselines
    logger.info("\n>>> Step 1/5: Training Centralized Baselines...")
    baselines_summary = run_baselines.run(config_path)

    # 2. Run FedAvg
    logger.info("\n>>> Step 2/5: Running Federated Averaging (FedAvg)...")
    fedavg_results = run_fedavg.run(config_path)

    # 3. Run FedProx
    logger.info("\n>>> Step 3/5: Running FedProx...")
    fedprox_results = run_fedprox.run(config_path)

    # 4. Run Flagship FEMO
    logger.info("\n>>> Step 4/5: Running Flagship FEMO (Multi-Objective)...")
    femo_results = run_femo.run(config_path)

    # 5. Run Ablation Study
    logger.info("\n>>> Step 5/5: Running Systematic Ablation Study (Variants A-E)...")
    ablation_summary = run_ablation.run(config_path)

    # Consolidated Federated Comparison Table 4
    tables_dir = PROJECT_ROOT / 'experiments' / 'tables'
    results_dir = PROJECT_ROOT / 'experiments' / 'results'

    df_fedavg = pd.read_csv(results_dir / 'fedavg_results.csv')
    df_fedprox = pd.read_csv(results_dir / 'fedprox_results.csv')
    df_femo = pd.read_csv(results_dir / 'femo_results.csv')

    table4_data = {
        'FedAvg': {
            'roc_auc_mean': float(df_fedavg['roc_auc'].mean()),
            'roc_auc_std': float(df_fedavg['roc_auc'].std()),
            'accuracy_mean': float(df_fedavg['accuracy'].mean()),
            'brier_mean': float(df_fedavg['brier_score'].mean()),
            'ece_mean': float(df_fedavg['ece'].mean()),
            'worst_client': float(df_fedavg['worst_client_performance'].mean())
        },
        'FedProx': {
            'roc_auc_mean': float(df_fedprox['roc_auc'].mean()),
            'roc_auc_std': float(df_fedprox['roc_auc'].std()),
            'accuracy_mean': float(df_fedprox['accuracy'].mean()),
            'brier_mean': float(df_fedprox['brier_score'].mean()),
            'ece_mean': float(df_fedprox['ece'].mean()),
            'worst_client': float(df_fedprox['worst_client_performance'].mean())
        },
        'FEMO (Ours)': {
            'roc_auc_mean': float(df_femo['roc_auc'].mean()),
            'roc_auc_std': float(df_femo['roc_auc'].std()),
            'accuracy_mean': float(df_femo['accuracy'].mean()),
            'brier_mean': float(df_femo['brier_score'].mean()),
            'ece_mean': float(df_femo['ece'].mean()),
            'worst_client': float(df_femo['worst_client_performance'].mean()),
            'client_variance': float(df_femo['client_variance'].mean())
        }
    }

    with open(tables_dir / 'table4_federated.json', 'w') as f:
        json.dump(table4_data, f, indent=2)

    pd.DataFrame.from_dict(table4_data, orient='index').to_csv(tables_dir / 'table4_federated.csv')

    # Generate Publication Figures
    logger.info("\n>>> Generating publication figures (300 DPI)...")
    rg = ReportGenerator(base_dir=str(PROJECT_ROOT / 'experiments'))
    rg.plot_fig2_client_distribution()
    rg.plot_fig3_convergence()
    rg.plot_fig10_ablation()

    print("\n" + "="*80)
    print(" ALL EXPERIMENTS COMPLETE SUCCESSFULLY!")
    print(" Results saved to:      experiments/results/")
    print(" Tables saved to:       experiments/tables/")
    print(" Figures saved to:      experiments/figures/")
    print("="*80 + "\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CardioMind Master Experiment Runner")
    parser.add_argument('--config', default=None, help='Path to config JSON')
    args = parser.parse_args()
    run_all(args.config)

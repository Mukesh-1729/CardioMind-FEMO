import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.random_forest import RandomForestModel
from ml.models.xgboost_model import XGBoostModel
from ml.models.model_registry import ModelRegistry
from ml.evaluation.metrics import MetricsCalculator
from ml.evaluation.calibration import CalibrationAnalyzer
from ml.preprocessing.pipeline import PreprocessingPipeline

# Optional gradient boosters
try:
    from ml.models.lightgbm_model import LightGBMModel
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from ml.models.catboost_model import CatBoostModel
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / 'ml' / 'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = PROJECT_ROOT / 'ml' / 'data' / 'processed'
    train_path = data_dir / 'train_data.npz'
    test_path = data_dir / 'test_data.npz'

    # Ensure preprocessed data exists
    if not train_path.exists() or not test_path.exists():
        logger.info("Preprocessed data not found. Running PreprocessingPipeline...")
        pipeline = PreprocessingPipeline()
        pipeline.run(
            dataset_path=str(PROJECT_ROOT / 'datasets' / 'raw' / 'cardio_data_processed.csv'),
            config_path=config_path,
            output_dir=str(data_dir)
        )

    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    feature_names = list(test_data['feature_names']) if 'feature_names' in test_data else [f'feat_{i}' for i in range(X_train.shape[1])]

    logger.info(f"Loaded training data: {X_train.shape}, test data: {X_test.shape}")

    models_to_train = [
        ('LogisticRegression', LogisticRegressionModel),
        ('RandomForest', RandomForestModel),
        ('XGBoost', XGBoostModel),
    ]

    if HAS_LIGHTGBM:
        models_to_train.append(('LightGBM', LightGBMModel))
    if HAS_CATBOOST:
        models_to_train.append(('CatBoost', CatBoostModel))

    metrics_calc = MetricsCalculator()
    cal_analyzer = CalibrationAnalyzer()
    registry = ModelRegistry(registry_dir=str(PROJECT_ROOT / 'ml' / 'models'))

    seeds = config.get('random_seeds', [42, 123, 456, 789, 1024])
    all_results = []
    summary_by_model = {}

    for model_name, model_cls in models_to_train:
        logger.info(f"=== Training {model_name} across {len(seeds)} random seeds ===")
        model_seed_metrics = []

        for seed in seeds:
            seed_config = json.loads(json.dumps(config))
            seed_config['random_seeds'] = [seed]

            model_instance = model_cls()
            try:
                model_instance.train(X_train, y_train, seed_config)
                y_pred_proba = model_instance.predict_proba(X_test)
                
                # If 2D proba returned, extract positive class
                if len(y_pred_proba.shape) == 2:
                    y_pred_proba = y_pred_proba[:, 1]

                metrics = metrics_calc.calculate(y_test, y_pred_proba)
                ece = metrics_calc.calculate_ece(y_test, y_pred_proba)
                metrics['ece'] = ece
                metrics['model'] = model_name
                metrics['seed'] = seed

                all_results.append(metrics)
                model_seed_metrics.append(metrics)

                # Register model artifact
                model_id = registry.register(
                    model=model_instance.model if hasattr(model_instance, 'model') and model_instance.model is not None else model_instance,
                    metrics=metrics,
                    config=seed_config,
                    features=feature_names,
                    dataset='CardioData'
                )

                logger.info(f"[{model_name}|Seed {seed}] AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Brier: {metrics['brier_score']:.4f} | ECE: {ece:.4f}")
            except Exception as e:
                logger.error(f"Failed to train {model_name} on seed {seed}: {e}")

        if model_seed_metrics:
            df_m = pd.DataFrame(model_seed_metrics)
            summary_by_model[model_name] = {
                'roc_auc_mean': float(df_m['roc_auc'].mean()),
                'roc_auc_std': float(df_m['roc_auc'].std()),
                'accuracy_mean': float(df_m['accuracy'].mean()),
                'accuracy_std': float(df_m['accuracy'].std()),
                'f1_mean': float(df_m['f1'].mean()),
                'f1_std': float(df_m['f1'].std()),
                'brier_mean': float(df_m['brier_score'].mean()),
                'brier_std': float(df_m['brier_score'].std()),
                'ece_mean': float(df_m['ece'].mean()),
                'ece_std': float(df_m['ece'].std())
            }

    # Save outputs to experiments/
    results_dir = PROJECT_ROOT / 'experiments' / 'results'
    tables_dir = PROJECT_ROOT / 'experiments' / 'tables'
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(all_results)
    csv_path = results_dir / 'centralized_results.csv'
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Saved all baseline seed results to {csv_path}")

    # Table 3 Baselines
    table3_json = tables_dir / 'table3_baselines.json'
    with open(table3_json, 'w') as f:
        json.dump(summary_by_model, f, indent=2)

    # Also save as readable markdown / CSV summary
    table3_df = pd.DataFrame.from_dict(summary_by_model, orient='index')
    table3_df.to_csv(tables_dir / 'table3_baselines.csv')
    logger.info(f"Saved baseline summary table to {table3_json}")

    print("\n" + "="*70)
    print(" CENTRALIZED BASELINE RESULTS SUMMARY (5 SEEDS MEAN +/- STD)")
    print("="*70)
    for m_name, stats in summary_by_model.items():
        print(f"{m_name:<20} | AUC: {stats['roc_auc_mean']:.4f} +/- {stats['roc_auc_std']:.4f} | Acc: {stats['accuracy_mean']:.4f} | F1: {stats['f1_mean']:.4f} | ECE: {stats['ece_mean']:.4f}")
    print("="*70 + "\n")

    return summary_by_model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CardioMind Centralized Baselines Runner")
    parser.add_argument('--config', default=None, help='Path to config JSON')
    args = parser.parse_args()
    run(args.config)

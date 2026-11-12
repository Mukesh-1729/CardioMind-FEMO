import os
import sys
import yaml
import time
import logging
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Add root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AutoDataLoader
from src.data.validator import DataValidator
from src.data.feature_engineering import FeatureEngineering
from src.data.preprocessing import DataPreprocessor
from src.data.partition import NonIIDPartitioner
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.evaluation.metrics import MetricsSuite
from src.evaluation.calibration import CalibrationEngine
from src.evaluation.interpretability import InterpretabilityEngine
from src.experiments.statistical_analysis import StatisticalAnalyzer
from src.experiments.plots_and_tables import ExperimentReporter

# Centralized model implementations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_research_config(config_path: str = 'research_config.yaml') -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_all_experiments():
    start_total_time = time.time()
    config = load_research_config()

    dataset_path = config.get('dataset', 'datasets/raw/cardio_data_processed.csv')
    seeds = config.get('seeds', [42, 52, 62, 72, 82])

    logger.info("==========================================================================")
    logger.info("   CARDIOMIND-FEMO FRAMEWORK — COMPREHENSIVE EXPERIMENT RE-BENCHMARKING   ")
    logger.info("==========================================================================")

    # Phase 2: Data Loading & Validation
    loader = AutoDataLoader()
    raw_df, target_col, inspection_report = loader.load_and_inspect(dataset_path)

    validator = DataValidator()
    clean_df = validator.validate_and_clean(raw_df, target_col)

    feat_eng = FeatureEngineering()
    df_engineered = feat_eng.transform(clean_df)

    metrics_suite = MetricsSuite()
    stat_analyzer = StatisticalAnalyzer()
    reporter = ExperimentReporter(results_dir='results', paper_dir='paper_results')

    all_seed_records = []
    all_ablation_records = []
    convergence_dict = {'FedAvg': [], 'FedProx': [], 'CardioMind-FEMO': []}
    roc_data = {}
    cal_data = {}
    last_femo_model = None
    last_test_X = None
    last_test_y = None
    last_feature_names = None
    client_stats_df = None

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n==========================================================================")
        logger.info(f"   EXECUTING EXPERIMENTS FOR SEED {seed} ({seed_idx + 1}/{len(seeds)})")
        logger.info(f"==========================================================================")

        # Preprocessing with strict test hold-out
        preprocessor = DataPreprocessor(
            train_ratio=config.get('train_ratio', 0.8),
            val_ratio=config.get('validation_ratio', 0.1),
            test_ratio=config.get('test_ratio', 0.1)
        )
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = preprocessor.split_and_transform(
            df_engineered, target_col, seed=seed
        )

        input_dim = X_train.shape[1]
        last_test_X, last_test_y, last_feature_names = X_test, y_test, feature_names

        # Phase 3: Simulated Hospital Non-IID Partitioning
        partitioner = NonIIDPartitioner(n_clients=config.get('clients', 5))
        client_partitions = partitioner.partition_training_data(X_train, y_train, feature_names, seed=seed)
        if seed_idx == 0:
            partitioner.save_distribution_report('results/client_distribution.csv')
            client_stats_df = partitioner.client_stats_df

        # --- A. Centralized Baselines ---
        centralized_models = [
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=seed)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)),
            ('XGBoost', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed, eval_metric='logloss', n_jobs=-1)),
            ('LightGBM', LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed, verbose=-1, n_jobs=-1)),
            ('CatBoost', CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=seed, verbose=0))
        ]

        for m_name, model in centralized_models:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            m_dict = metrics_suite.compute_all_metrics(y_test, y_pred_proba, training_time=train_time)
            m_dict['model'] = m_name
            m_dict['seed'] = seed
            m_dict['type'] = 'Centralized'
            all_seed_records.append(m_dict)

            if seed_idx == 0:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_data[m_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': m_dict['roc_auc']}

        # --- B. Federated Models ---
        fed_strategies = [
            ('FedAvg', 'fedavg'),
            ('FedProx', 'fedprox'),
            ('CardioMind-FEMO', 'femo')
        ]

        for display_name, strategy_code in fed_strategies:
            clients = [
                FederatedClient(
                    client_id=cp['client_id'],
                    hospital_name=cp['hospital_name'],
                    X=cp['X'],
                    y=cp['y'],
                    config=config
                )
                for cp in client_partitions
            ]

            server = FederatedServer(input_dim=input_dim, config=config)
            t0 = time.time()
            round_logs = server.run_simulation(
                clients, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, strategy=strategy_code
            )
            train_time = time.time() - t0

            # Final evaluation on locked test set using validation-selected best checkpoint
            y_pred_proba = server.global_model.predict_proba(X_test)
            final_m = round_logs[-1]
            m_dict = metrics_suite.compute_all_metrics(
                y_test, y_pred_proba, training_time=train_time,
                comm_cost_kb=final_m.get('communication_cost_kb', 0.0),
                n_rounds=config.get('rounds', 30),
                client_variance=final_m.get('client_variance', 0.0)
            )
            m_dict['model'] = display_name
            m_dict['seed'] = seed
            m_dict['type'] = 'Federated'
            m_dict['best_round'] = server.best_round
            all_seed_records.append(m_dict)

            if display_name in convergence_dict:
                df_r = pd.DataFrame(round_logs)
                df_r['seed'] = seed
                df_r['strategy_name'] = display_name
                convergence_dict[display_name].append(df_r)

            if seed_idx == 0:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_data[display_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': m_dict['roc_auc']}

                if display_name == 'CardioMind-FEMO':
                    last_femo_model = server.global_model
                    # Post-hoc Calibration Analysis
                    cal_engine = CalibrationEngine()
                    val_preds = server.global_model.predict_proba(X_val)
                    cal_engine.fit(val_preds, y_val)
                    cal_test_preds = cal_engine.calibrate(y_pred_proba)

                    uncal_cal = cal_engine.evaluate_calibration(y_test, y_pred_proba)
                    cal_cal = cal_engine.evaluate_calibration(y_test, cal_test_preds)

                    cal_data['Uncalibrated CardioMind-FEMO'] = uncal_cal
                    cal_data['Calibrated CardioMind-FEMO'] = cal_cal

                elif display_name == 'FedAvg':
                    cal_engine = CalibrationEngine()
                    cal_data['FedAvg'] = cal_engine.evaluate_calibration(y_test, y_pred_proba)

        # --- C. Phase 12: Systematic Ablation Study ---
        ablation_variants = [
            ('FEMO-Base (Pred only)', 'femo_base', '-', '-', '-'),
            ('FEMO-P (Pred + Prox)', 'femo_p', '\\checkmark', '-', '-'),
            ('FEMO-PC (Pred + Prox + Cal)', 'femo_pc', '\\checkmark', '\\checkmark', '-'),
            ('FEMO-PCR (Fixed Lambdas)', 'femo_pcr', '\\checkmark', '\\checkmark', '\\checkmark'),
            ('CardioMind-FEMO (Full Scheduled)', 'femo', '\\checkmark', '\\checkmark', '\\checkmark')
        ]

        for v_name, v_strat, has_p, has_c, has_r in ablation_variants:
            v_clients = [
                FederatedClient(
                    client_id=cp['client_id'],
                    hospital_name=cp['hospital_name'],
                    X=cp['X'],
                    y=cp['y'],
                    config=config
                )
                for cp in client_partitions
            ]
            v_server = FederatedServer(input_dim=input_dim, config=config)
            v_logs = v_server.run_simulation(
                v_clients, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, strategy=v_strat
            )
            v_preds = v_server.global_model.predict_proba(X_test)
            v_m = metrics_suite.compute_all_metrics(y_test, v_preds)
            v_m['variant'] = v_name
            v_m['has_prox'] = has_p
            v_m['has_cal'] = has_c
            v_m['has_risk'] = has_r
            v_m['seed'] = seed
            all_ablation_records.append(v_m)

    # --- Processing Results Across 5 Seeds ---
    seed_results_df = pd.DataFrame(all_seed_records)
    ablation_seed_df = pd.DataFrame(all_ablation_records)

    # Combine convergence curves across seeds
    conv_dfs = {}
    for k, v in convergence_dict.items():
        if v:
            conv_dfs[k] = pd.concat(v, ignore_index=True)

    # Group metrics for Summary Table
    metric_cols = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'brier_score', 'ece', 'training_time_sec', 'communication_cost_kb', 'client_variance']
    summary_rows = []

    model_order = [
        'Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost',
        'FedAvg', 'FedProx', 'CardioMind-FEMO'
    ]

    for m in model_order:
        sub = seed_results_df[seed_results_df['model'] == m]
        if not sub.empty:
            row = {'model': m, 'type': sub['type'].iloc[0]}
            for col in metric_cols:
                row[f'{col}_mean'] = float(sub[col].mean())
                row[f'{col}_std'] = float(sub[col].std())
            summary_rows.append(row)

    metrics_summary_df = pd.DataFrame(summary_rows)

    # Group Ablation
    ablation_summary_rows = []
    for v_name, _, has_p, has_c, has_r in ablation_variants:
        sub_v = ablation_seed_df[ablation_seed_df['variant'] == v_name]
        if not sub_v.empty:
            a_row = {
                'variant': v_name,
                'has_prox': has_p,
                'has_cal': has_c,
                'has_risk': has_r,
                'roc_auc_mean': float(sub_v['roc_auc'].mean()),
                'roc_auc_std': float(sub_v['roc_auc'].std()),
                'accuracy_mean': float(sub_v['accuracy'].mean()),
                'accuracy_std': float(sub_v['accuracy'].std()),
                'f1_mean': float(sub_v['f1'].mean()),
                'f1_std': float(sub_v['f1'].std()),
                'brier_score_mean': float(sub_v['brier_score'].mean()),
                'brier_score_std': float(sub_v['brier_score'].std()),
                'ece_mean': float(sub_v['ece'].mean()),
                'ece_std': float(sub_v['ece'].std())
            }
            ablation_summary_rows.append(a_row)

    ablation_summary_df = pd.DataFrame(ablation_summary_rows)

    # Phase 10: SHAP Interpretability
    logger.info("Executing post-hoc SHAP interpretability...")
    interp_engine = InterpretabilityEngine(last_femo_model, last_test_X, last_feature_names)
    shap_importance_df = interp_engine.compute_global_importance(last_test_X)

    # Phase 14: Statistical Analysis (Bootstrap CI, Paired t-tests, Cohen's d)
    stat_report = stat_analyzer.run_comparisons(seed_results_df)

    # Phase 15 & 16: Generate Plots, Summary CSVs, and LaTeX Tables
    reporter.generate_all_plots_and_tables(
        seed_results_df=seed_results_df,
        metrics_summary_df=metrics_summary_df,
        ablation_df=ablation_summary_df,
        convergence_dict=conv_dfs,
        calibration_data=cal_data,
        roc_data=roc_data,
        client_stats_df=client_stats_df if client_stats_df is not None else pd.DataFrame(),
        shap_importance_df=shap_importance_df,
        stat_report=stat_report
    )

    total_time_min = (time.time() - start_total_time) / 60.0

    print("\n" + "="*95)
    print("                 CARDIOMIND-FEMO MASTER BENCHMARK RE-RUN SUMMARY")
    print("                          (5 Random Seeds: 42, 52, 62, 72, 82)")
    print("="*95)
    print(f"{'Model':<22} | {'ROC-AUC':<17} | {'Accuracy':<17} | {'F1-Score':<17} | {'Brier':<10} | {'ECE':<10}")
    print("-"*95)
    for _, r in metrics_summary_df.iterrows():
        auc_str = f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}"
        acc_str = f"{r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}"
        f1_str = f"{r['f1_mean']:.4f} +/- {r['f1_std']:.4f}"
        print(f"{r['model']:<22} | {auc_str:<17} | {acc_str:<17} | {f1_str:<17} | {r['brier_score_mean']:.4f}     | {r['ece_mean']:.4f}")
    print("="*95)

    print("\n" + "="*95)
    print("                        SYSTEMATIC ABLATION STUDY RESULTS")
    print("="*95)
    for _, ar in ablation_summary_df.iterrows():
        auc_str = f"{ar['roc_auc_mean']:.4f} +/- {ar['roc_auc_std']:.4f}"
        print(f"{ar['variant']:<35} | AUC: {auc_str} | Acc: {ar['accuracy_mean']:.4f} | Brier: {ar['brier_score_mean']:.4f} | ECE: {ar['ece_mean']:.4f}")
    print("="*95)
    print(f"Total Execution Time: {total_time_min:.2f} minutes.\n")

if __name__ == '__main__':
    run_all_experiments()

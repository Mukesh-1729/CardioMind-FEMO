import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path("e:/CardioMind")
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

print("Starting smoke test on Seed 42...")
config = {
    'clients': 5,
    'rounds': 5,
    'local_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'model': {'hidden_dim_1': 64, 'hidden_dim_2': 32, 'dropout_rate': 0.2},
    'femo': {
        'lambda_cal_max': 0.05,
        'lambda_prox': 0.01,
        'lambda_risk_max': 0.01,
        'huber_delta': 0.1,
        'ema_decay': 0.90,
        'warmup_rounds': 2,
        'eta_quality': 0.5,
        'mu_stability': 0.5,
        'weight_clip_min': 0.05,
        'weight_clip_max': 0.50,
        'val_weights': {'auc': 1.0, 'brier': 0.5, 'ece': 0.5}
    }
}

loader = AutoDataLoader()
raw_df, target_col, _ = loader.load_and_inspect('datasets/raw/cardio_data_processed.csv')
validator = DataValidator()
clean_df = validator.validate_and_clean(raw_df, target_col)
feat_eng = FeatureEngineering()
df_eng = feat_eng.transform(clean_df)

preprocessor = DataPreprocessor(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
X_train, y_train, X_val, y_val, X_test, y_test, feat_names = preprocessor.split_and_transform(df_eng, target_col, seed=42)

partitioner = NonIIDPartitioner(n_clients=5)
client_partitions = partitioner.partition_training_data(X_train, y_train, feat_names, seed=42)

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

server = FederatedServer(input_dim=X_train.shape[1], config=config)
round_logs = server.run_simulation(clients, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, strategy='femo')

preds = server.global_model.predict_proba(X_test)
metrics = MetricsSuite().compute_all_metrics(y_test, preds)
print("\nSmoke Test Results on Seed 42:")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Brier: {metrics['brier_score']:.4f}, ECE: {metrics['ece']:.4f}")
print(f"Selected Checkpoint Round: {server.best_round} (Best Val Score: {server.best_val_score:.4f})")
print("Smoke Test Passed!")

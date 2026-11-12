import json
import logging
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    from .dataset_adapter import DatasetAdapter
    from .cleaning import DataCleaner
    from .feature_engineering import FeatureEngineer
    from .encoding import FeatureEncoder
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ml.preprocessing.dataset_adapter import DatasetAdapter
    from ml.preprocessing.cleaning import DataCleaner
    from ml.preprocessing.feature_engineering import FeatureEngineer
    from ml.preprocessing.encoding import FeatureEncoder

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Main pipeline orchestrating data preprocessing and partitioning."""
    
    def __init__(self):
        self.adapter = DatasetAdapter()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.encoder = FeatureEncoder()
        
    def run(self, dataset_path: str, config_path: str, output_dir: str):
        """Executes full preprocessing pipeline."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        np.random.seed(config.get('random_seeds', [42])[0])
        
        X_df, y_series, feature_names, _ = self.adapter.load(dataset_path)
        
        # Cleaning
        X_df['target'] = y_series
        X_df = self.cleaner.clean(X_df)
        y_series = X_df.pop('target')
        
        # Engineering
        X_df = self.engineer.transform(X_df, config)
        
        # Split out global test set FIRST (prevent data leakage)
        test_size = config.get('test_size', 0.15)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_df, y_series, test_size=test_size, stratify=y_series, random_state=config.get('random_seeds', [42])[0]
        )
        
        # Encoding (Fit on Train, transform test)
        X_train_val_enc = self.encoder.fit_transform(X_train_val)
        X_test_enc = self.encoder.transform(X_test)
        
        # Save pipeline and test set
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.encoder.save(os.path.join(output_dir, 'pipeline.pkl'))
        
        np.savez(os.path.join(output_dir, 'test_data.npz'), X=X_test_enc.values, y=y_test.values, feature_names=np.array(X_test_enc.columns))
        np.savez(os.path.join(output_dir, 'train_data.npz'), X=X_train_val_enc.values, y=y_train_val.values, feature_names=np.array(X_train_val_enc.columns))
        
        # Create partitions
        self.create_client_partitions(
            X_train_val_enc.values, y_train_val.values, 
            config.get('n_clients', 5), 
            partition_mode='iid', 
            output_dir=output_dir
        )
        
        logger.info("Preprocessing complete.")
        print(f"Preprocessing complete. Data saved to {output_dir}")
        
    def create_client_partitions(self, X_train: np.ndarray, y_train: np.ndarray, 
                               n_clients: int, partition_mode: str, output_dir: str):
        """Partitions data for federated clients."""
        clients_dir = Path(output_dir) / 'simulated_hospitals'
        clients_dir.mkdir(parents=True, exist_ok=True)
        
        indices = np.arange(len(y_train))
        
        if partition_mode == 'iid':
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=42)
            splits = [test_idx for _, test_idx in skf.split(X_train, y_train)]
        else:
            np.random.shuffle(indices)
            splits = np.array_split(indices, n_clients)
            
        for i, idx in enumerate(splits):
            client_path = clients_dir / f'client_{i}'
            client_path.mkdir(exist_ok=True)
            np.savez(
                client_path / 'data.npz', 
                X=X_train[idx], 
                y=y_train[idx]
            )
            logger.info(f"Saved client {i} partition with {len(idx)} samples.")
            print(f"Saved client {i} partition with {len(idx)} samples.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CardioMind Preprocessing Pipeline")
    parser.add_argument('--dataset', default='datasets/raw/cardio_data_processed.csv', help='Path to dataset CSV')
    parser.add_argument('--config', default='ml/experiments/configs/femo_config.json', help='Path to config JSON')
    parser.add_argument('--output', default='ml/data/processed', help='Output directory')
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline()
    pipeline.run(args.dataset, args.config, args.output)

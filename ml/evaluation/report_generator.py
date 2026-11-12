import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent / 'experiments'
        self.base_dir = Path(base_dir)
        self.tables_dir = self.base_dir / 'tables'
        self.figures_dir = self.base_dir / 'figures'
        self.results_dir = self.base_dir / 'results'
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set clean aesthetic style
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    def plot_fig2_client_distribution(self, client_stats=None):
        """Plot sample distribution across simulated hospital clients."""
        fig, ax = plt.subplots(figsize=(8, 5))
        if not client_stats:
            client_stats = [
                {'name': 'Hospital 1 (Metro General)', 'samples': 11591, 'pos_rate': 0.49},
                {'name': 'Hospital 2 (City Heart Clinic)', 'samples': 11591, 'pos_rate': 0.50},
                {'name': 'Hospital 3 (Regional Medical)', 'samples': 11591, 'pos_rate': 0.49},
                {'name': 'Hospital 4 (St. Jude Center)', 'samples': 11590, 'pos_rate': 0.50},
                {'name': 'Hospital 5 (Community Health)', 'samples': 11590, 'pos_rate': 0.49},
            ]
        names = [c['name'] for c in client_stats]
        samples = [c['samples'] for c in client_stats]
        
        bars = ax.bar(names, samples, color='#2b5c8f', width=0.55, edgecolor='black', alpha=0.85)
        ax.set_ylabel('Number of Patient Records')
        ax.set_title('Figure 2: Patient Sample Distribution Across Simulated Hospital Clients', fontweight='bold', pad=12)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 150, f"{yval:,}", ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        out_path = self.figures_dir / 'fig2_client_distribution.png'
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved Figure 2 to {out_path}")

    def plot_fig3_convergence(self):
        """Plot convergence curves: FedAvg vs FedProx vs FEMO."""
        fedavg_path = self.results_dir / 'fedavg_convergence.csv'
        fedprox_path = self.results_dir / 'fedprox_convergence.csv'
        femo_path = self.results_dir / 'femo_convergence.csv'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        data_plotted = False
        for path, label, color in [(fedavg_path, 'FedAvg', '#e74c3c'),
                                   (fedprox_path, 'FedProx', '#f39c12'),
                                   (femo_path, 'FEMO (Ours)', '#27ae60')]:
            if path.exists():
                df = pd.read_csv(path)
                grouped = df.groupby('round').agg({'roc_auc': ['mean', 'std'], 'brier_score': ['mean', 'std']})
                rounds = grouped.index
                
                # AUC
                auc_mean = grouped['roc_auc']['mean']
                auc_std = grouped['roc_auc']['std'].fillna(0)
                ax1.plot(rounds, auc_mean, label=label, color=color, linewidth=2.2, marker='o', markersize=4)
                ax1.fill_between(rounds, auc_mean - auc_std, auc_mean + auc_std, color=color, alpha=0.15)
                
                # Brier score
                brier_mean = grouped['brier_score']['mean']
                brier_std = grouped['brier_score']['std'].fillna(0)
                ax2.plot(rounds, brier_mean, label=label, color=color, linewidth=2.2, marker='s', markersize=4)
                ax2.fill_between(rounds, brier_mean - brier_std, brier_mean + brier_std, color=color, alpha=0.15)
                data_plotted = True

        if data_plotted:
            ax1.set_xlabel('Federated Round')
            ax1.set_ylabel('ROC-AUC Score')
            ax1.set_title('(A) Global Test ROC-AUC Convergence', fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(True, linestyle='--', alpha=0.6)

            ax2.set_xlabel('Federated Round')
            ax2.set_ylabel('Brier Calibration Loss (Lower is Better)')
            ax2.set_title('(B) Probability Calibration Convergence', fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.grid(True, linestyle='--', alpha=0.6)

            plt.suptitle('Figure 3: Multi-Round Convergence Across Federated Learning Frameworks', fontweight='bold', y=1.02)
            plt.tight_layout()
            out_path = self.figures_dir / 'fig3_convergence.png'
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved Figure 3 to {out_path}")

    def plot_fig10_ablation(self):
        """Plot Ablation Study comparison across metrics."""
        ablation_path = self.results_dir / 'ablation_results.csv'
        if not ablation_path.exists():
            return

        df = pd.read_csv(ablation_path)
        grouped = df.groupby('variant').agg({'roc_auc': 'mean', 'brier_score': 'mean', 'ece': 'mean'}).reset_index()

        fig, ax1 = plt.subplots(figsize=(10, 5.5))

        x = np.arange(len(grouped))
        width = 0.25

        bars1 = ax1.bar(x - width, grouped['roc_auc'], width, label='ROC-AUC (Higher is Better)', color='#2980b9')
        bars2 = ax1.bar(x, grouped['brier_score'], width, label='Brier Score (Lower is Better)', color='#e67e22')
        bars3 = ax1.bar(x + width, grouped['ece'], width, label='ECE (Lower is Better)', color='#27ae60')

        ax1.set_ylabel('Metric Value')
        ax1.set_title('Figure 10: Systematic Ablation Study (Variants A through E)', fontweight='bold', pad=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(grouped['variant'], rotation=15, ha='right')
        ax1.legend(loc='upper left')
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax1.set_ylim(0, 0.95)

        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha='center', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha='center', fontsize=8)
        for bar in bars3:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha='center', fontsize=8)

        plt.tight_layout()
        out_path = self.figures_dir / 'fig10_ablation.png'
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved Figure 10 to {out_path}")

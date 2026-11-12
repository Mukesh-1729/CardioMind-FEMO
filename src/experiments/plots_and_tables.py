import os
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ExperimentReporter:
    """
    Generates all mandatory publication figures (300 DPI), CSV summaries,
    and LaTeX tables for CardioMind-FEMO evaluation.
    """
    def __init__(self, results_dir: str = 'results', paper_dir: str = 'paper_results'):
        self.results_dir = results_dir
        self.paper_dir = paper_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.paper_dir, 'tables'), exist_ok=True)
        os.makedirs(os.path.join(self.paper_dir, 'figures'), exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    def generate_all_plots_and_tables(
        self,
        seed_results_df: pd.DataFrame,
        metrics_summary_df: pd.DataFrame,
        ablation_df: pd.DataFrame,
        convergence_dict: Dict[str, pd.DataFrame],
        calibration_data: Dict[str, Any],
        roc_data: Dict[str, Any],
        client_stats_df: pd.DataFrame,
        shap_importance_df: pd.DataFrame,
        stat_report: Dict[str, Any]
    ):
        logger.info("Generating publication figures and LaTeX outputs...")

        # 1. Save CSVs in results/ and paper_results/tables/
        seed_results_df.to_csv(os.path.join(self.results_dir, 'seed_results.csv'), index=False)
        metrics_summary_df.to_csv(os.path.join(self.results_dir, 'metrics_summary.csv'), index=False)
        ablation_df.to_csv(os.path.join(self.results_dir, 'ablation_results.csv'), index=False)
        client_stats_df.to_csv(os.path.join(self.results_dir, 'client_distribution.csv'), index=False)

        metrics_summary_df.to_csv(os.path.join(self.paper_dir, 'tables', 'table_main_benchmark.csv'), index=False)
        ablation_df.to_csv(os.path.join(self.paper_dir, 'tables', 'table_ablation.csv'), index=False)

        # 2. Metric Comparisons (Accuracy, AUC, F1)
        self._plot_metric_bar(metrics_summary_df, 'accuracy', 'Accuracy Comparison Across Centralized & Federated Models', 'accuracy_comparison.png', higher_is_better=True)
        self._plot_metric_bar(metrics_summary_df, 'roc_auc', 'ROC-AUC Comparison Across Centralized & Federated Models', 'auc_comparison.png', higher_is_better=True)
        self._plot_metric_bar(metrics_summary_df, 'f1', 'F1-Score Comparison Across Centralized & Federated Models', 'f1_comparison.png', higher_is_better=True)

        # 3. Calibration Curve
        self._plot_calibration_curve(calibration_data)

        # 4. ROC Curves
        self._plot_roc_curves(roc_data)

        # 5. Convergence Curve
        self._plot_convergence(convergence_dict)

        # 6. Client Performance
        self._plot_client_performance(client_stats_df, seed_results_df)

        # 7. Ablation Chart
        self._plot_ablation_chart(ablation_df)

        # 8. Communication Cost
        self._plot_communication_cost(convergence_dict)

        # 9. SHAP Summary
        self._plot_shap_summary(shap_importance_df)

        # 10. Copy figures to paper_results/figures/
        for fig_name in [
            'accuracy_comparison.png', 'auc_comparison.png', 'f1_comparison.png',
            'calibration_curve.png', 'roc_curves.png', 'convergence_curve.png',
            'client_performance.png', 'ablation_chart.png', 'communication_cost.png', 'shap_summary.png'
        ]:
            src_p = os.path.join(self.results_dir, fig_name)
            dst_p = os.path.join(self.paper_dir, 'figures', fig_name)
            if os.path.exists(src_p):
                with open(src_p, 'rb') as f_in, open(dst_p, 'wb') as f_out:
                    f_out.write(f_in.read())

        # 11. Generate LaTeX tables & macros
        self._generate_latex_outputs(metrics_summary_df, ablation_df, stat_report)
        logger.info("All plots and LaTeX tables generated successfully.")

    def _plot_metric_bar(self, df: pd.DataFrame, metric: str, title: str, filename: str, higher_is_better: bool = True):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        models = df['model'].tolist()
        means = df[f'{metric}_mean'].tolist()
        stds = df[f'{metric}_std'].tolist()

        colors = ['#34495e', '#7f8c8d', '#2980b9', '#16a085', '#8e44ad', '#e74c3c', '#f39c12', '#27ae60']
        if len(colors) < len(models):
            colors = colors * 2
        colors = colors[:len(models)]

        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=4, color=colors, width=0.55, edgecolor='black', alpha=0.88)
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_ylabel(metric.upper().replace('_', ' '))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=22, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        min_y = max(0.0, min(means) - 0.15) if higher_is_better else 0.0
        max_y = min(1.0, max(means) + 0.12)
        ax.set_ylim(min_y, max_y)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300)
        plt.close()

    def _plot_calibration_curve(self, cal_data: Dict[str, Any]):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")

        palette = {
            'Uncalibrated CardioMind-FEMO': '#e74c3c',
            'Calibrated CardioMind-FEMO': '#27ae60',
            'Uncalibrated FEMO': '#e74c3c',
            'Calibrated FEMO': '#27ae60',
            'FedAvg': '#3498db'
        }
        for name, data in cal_data.items():
            color = palette.get(name, '#9b59b6')
            ece = data.get('ece', 0.0)
            brier = data.get('brier_score', 0.0)
            ax.plot(data['prob_pred'], data['prob_true'], marker='s', label=f"{name} (ECE={ece:.4f}, Brier={brier:.4f})", color=color, linewidth=2.0)

        ax.set_xlabel("Mean Predicted Probability", fontweight='bold')
        ax.set_ylabel("Empirical Fraction of Positives", fontweight='bold')
        ax.set_title("Probability Calibration Reliability Diagram", fontweight='bold', pad=12)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'calibration_curve.png'), dpi=300)
        plt.close()

    def _plot_roc_curves(self, roc_data: Dict[str, Any]):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")

        for name, data in roc_data.items():
            auc = data.get('auc', 0.5)
            ax.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {auc:.4f})", linewidth=2.0)

        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight='bold')
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight='bold')
        ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontweight='bold', pad=12)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'roc_curves.png'), dpi=300)
        plt.close()

    def _plot_convergence(self, conv_dict: Dict[str, pd.DataFrame]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

        palette = {'FedAvg': '#e74c3c', 'FedProx': '#f39c12', 'FEMO': '#27ae60', 'CardioMind-FEMO': '#27ae60'}
        for name, df in conv_dict.items():
            color = palette.get(name, '#34495e')
            grouped = df.groupby('round').agg({'roc_auc': ['mean', 'std'], 'brier_score': ['mean', 'std']})
            rounds = grouped.index
            
            auc_m = grouped['roc_auc']['mean']
            auc_s = grouped['roc_auc']['std'].fillna(0)
            ax1.plot(rounds, auc_m, label=name, color=color, linewidth=2.0, marker='o', markersize=3.5)
            ax1.fill_between(rounds, auc_m - auc_s, auc_m + auc_s, color=color, alpha=0.12)

            brier_m = grouped['brier_score']['mean']
            brier_s = grouped['brier_score']['std'].fillna(0)
            ax2.plot(rounds, brier_m, label=name, color=color, linewidth=2.0, marker='s', markersize=3.5)
            ax2.fill_between(rounds, brier_m - brier_s, brier_m + brier_s, color=color, alpha=0.12)

        ax1.set_xlabel("Communication Round", fontweight='bold')
        ax1.set_ylabel("Global Test ROC-AUC", fontweight='bold')
        ax1.set_title("(A) ROC-AUC Convergence Across Communication Rounds", fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.set_xlabel("Communication Round", fontweight='bold')
        ax2.set_ylabel("Brier Calibration Loss", fontweight='bold')
        ax2.set_title("(B) Calibration Loss Convergence", fontweight='bold')
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.suptitle("Federated Multi-Round Optimization Convergence (5 Independent Seeds)", fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'convergence_curve.png'), dpi=300)
        plt.close()

    def _plot_client_performance(self, client_stats_df: pd.DataFrame, seed_results_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        names = client_stats_df['client_id'].tolist()
        samples = client_stats_df['sample_count'].tolist()
        pos_rates = (client_stats_df['positive_rate'] * 100).tolist()

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width/2, samples, width, label='Sample Count', color='#2b5c8f', edgecolor='black')
        ax2 = ax.twinx()
        ax2.plot(x + width/2, pos_rates, color='#e74c3c', marker='o', linewidth=2.2, label='CVD Positive Rate (%)')

        ax.set_ylabel('Number of Patient Records (Local Partition)', fontweight='bold')
        ax2.set_ylabel('CVD Positive Rate (%)', fontweight='bold', color='#e74c3c')
        ax.set_title('Non-IID Simulated Hospital Client Partitions', fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'client_performance.png'), dpi=300)
        plt.close()

    def _plot_ablation_chart(self, ablation_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        variants = ablation_df['variant'].tolist()
        auc_vals = ablation_df['roc_auc_mean'].tolist()
        brier_vals = ablation_df['brier_score_mean'].tolist()
        ece_vals = ablation_df['ece_mean'].tolist()

        x = np.arange(len(variants))
        width = 0.25

        b1 = ax.bar(x - width, auc_vals, width, label='ROC-AUC (Higher is Better)', color='#2980b9', edgecolor='black')
        b2 = ax.bar(x, brier_vals, width, label='Brier Score (Lower is Better)', color='#e67e22', edgecolor='black')
        b3 = ax.bar(x + width, ece_vals, width, label='ECE (Lower is Better)', color='#27ae60', edgecolor='black')

        ax.set_ylabel('Metric Value', fontweight='bold')
        ax.set_title('Systematic Component Ablation Study Across 5 Seeds', fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1.0)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha='center', fontsize=8, fontweight='bold')
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha='center', fontsize=8, fontweight='bold')
        for bar in b3:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'ablation_chart.png'), dpi=300)
        plt.close()

    def _plot_communication_cost(self, conv_dict: Dict[str, pd.DataFrame]):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in conv_dict.items():
            if 'communication_cost_kb' in df.columns:
                grouped = df.groupby('round').agg({'communication_cost_kb': 'mean', 'roc_auc': 'mean'})
                ax.plot(grouped['communication_cost_kb'] / 1024.0, grouped['roc_auc'], label=name, linewidth=2.0, marker='o', markersize=3.5)

        ax.set_xlabel("Cumulative Communication Overhead (MB)", fontweight='bold')
        ax.set_ylabel("Global Test ROC-AUC", fontweight='bold')
        ax.set_title("Communication Efficiency vs Model Performance", fontweight='bold', pad=12)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'communication_cost.png'), dpi=300)
        plt.close()

    def _plot_shap_summary(self, shap_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 6))
        top_10 = shap_df.head(10).iloc[::-1]

        ax.barh(top_10['feature'], top_10['importance'], color='#2c3e50', edgecolor='black', alpha=0.85)
        ax.set_xlabel('Mean Absolute SHAP Value (Impact on CVD Risk Probability)', fontweight='bold')
        ax.set_title('Top 10 Clinical Risk Drivers Identified by SHAP Explainer', fontweight='bold', pad=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'shap_summary.png'), dpi=300)
        plt.close()

    def _generate_latex_outputs(self, summary_df: pd.DataFrame, ablation_df: pd.DataFrame, stat_report: Dict[str, Any]):
        # 1. Main Benchmark Table
        tex_table1 = os.path.join(self.paper_dir, 'tables', 'table_main_benchmark.tex')
        with open(tex_table1, 'w') as f:
            f.write("% Auto-generated by CardioMind-FEMO Framework\n")
            f.write("\\begin{table*}[t]\n\\centering\n")
            f.write("\\caption{Comprehensive Performance Benchmark Across 5 Random Seeds (Mean $\\pm$ STD)}\n")
            f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
            f.write("Model & ROC-AUC & Accuracy & F1-Score & Brier Score & ECE & Comm (KB) \\\\ \\hline\n")
            for _, row in summary_df.iterrows():
                m_name = row['model']
                auc = f"{row['roc_auc_mean']:.4f} $\\pm$ {row['roc_auc_std']:.4f}"
                acc = f"{row['accuracy_mean']:.4f} $\\pm$ {row['accuracy_std']:.4f}"
                f1 = f"{row['f1_mean']:.4f} $\\pm$ {row['f1_std']:.4f}"
                brier = f"{row['brier_score_mean']:.4f} $\\pm$ {row['brier_score_std']:.4f}"
                ece = f"{row['ece_mean']:.4f} $\\pm$ {row['ece_std']:.4f}"
                comm = f"{row.get('communication_cost_kb_mean', 0.0):.1f}"
                f.write(f"{m_name} & {auc} & {acc} & {f1} & {brier} & {ece} & {comm} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\label{tab:main_benchmark}\n\\end{table*}\n")

        # 2. Ablation Table
        tex_table2 = os.path.join(self.paper_dir, 'tables', 'table_ablation.tex')
        with open(tex_table2, 'w') as f:
            f.write("% Auto-generated Ablation Table\n")
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write("\\caption{Systematic Ablation Study on Loss and Aggregation Components}\n")
            f.write("\\begin{tabular}{lccccc}\n\\hline\n")
            f.write("Variant & $\\mathcal{L}_{\\text{pred}}$ & $\\mathcal{L}_{\\text{prox}}$ & $\\mathcal{L}_{\\text{cal}}$ & $\\mathcal{L}_{\\text{risk}}$ & ROC-AUC \\\\ \\hline\n")
            for _, row in ablation_df.iterrows():
                f.write(f"{row['variant']} & \\checkmark & {row.get('has_prox', '-')} & {row.get('has_cal', '-')} & {row.get('has_risk', '-')} & {row['roc_auc_mean']:.4f} $\\pm$ {row['roc_auc_std']:.4f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\label{tab:ablation}\n\\end{table}\n")

        # 3. LaTeX Macros File
        tex_macros = os.path.join(self.paper_dir, 'latex_results.tex')
        with open(tex_macros, 'w') as f:
            f.write("% Auto-generated LaTeX Macros for CardioMind Paper\n")
            for _, row in summary_df.iterrows():
                clean_name = row['model'].replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
                f.write(f"\\newcommand{{\\val{clean_name}AUC}}{{{row['roc_auc_mean']:.4f}}}\n")
                f.write(f"\\newcommand{{\\val{clean_name}Acc}}{{{row['accuracy_mean']:.4f}}}\n")
                f.write(f"\\newcommand{{\\val{clean_name}FOne}}{{{row['f1_mean']:.4f}}}\n")
                f.write(f"\\newcommand{{\\val{clean_name}Brier}}{{{row['brier_score_mean']:.4f}}}\n")
                f.write(f"\\newcommand{{\\val{clean_name}ECE}}{{{row['ece_mean']:.4f}}}\n")

            if 'FEMO_vs_FedAvg' in stat_report:
                s = stat_report['FEMO_vs_FedAvg'].get('roc_auc', {})
                f.write(f"\\newcommand{{\\pValFemoVsFedAvgAUC}}{{{s.get('t_pvalue', 0.05):.5f}}}\n")
                f.write(f"\\newcommand{{\\cohenDFemoVsFedAvgAUC}}{{{s.get('cohens_d', 0.0):.4f}}}\n")

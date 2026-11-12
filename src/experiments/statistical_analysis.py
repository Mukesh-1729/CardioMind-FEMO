import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    Statistical hypothesis testing and bootstrap confidence interval engine for Phase 14:
    - 95% Bootstrap Confidence Intervals
    - Paired t-tests and Wilcoxon signed-rank tests (FEMO vs FedAvg, FEMO vs FedProx)
    - Cohen's d effect size calculation
    """
    def compute_bootstrap_ci(
        self, values: List[float], n_bootstraps: int = 1000, ci: float = 0.95
    ) -> Tuple[float, float, float]:
        arr = np.array(values)
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        
        boot_means = []
        np.random.seed(42)
        for _ in range(n_bootstraps):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            boot_means.append(np.mean(sample))

        alpha = (1.0 - ci) / 2.0
        low = float(np.percentile(boot_means, alpha * 100))
        high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        mean_val = float(np.mean(arr))
        return mean_val, low, high

    def compute_cohens_d(self, x1: np.ndarray, x2: np.ndarray) -> float:
        n1, n2 = len(x1), len(x2)
        s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        if pooled_std < 1e-8:
            return 0.0
        return float((np.mean(x1) - np.mean(x2)) / pooled_std)

    def run_comparisons(self, seed_results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs statistical tests comparing FEMO against FedAvg and FedProx across seeds.
        """
        femo_df = seed_results_df[seed_results_df['model'].str.contains('FEMO', case=False) & ~seed_results_df['model'].str.contains('Base|P$|PC$|PCR$', regex=True)]
        if femo_df.empty:
            femo_df = seed_results_df[seed_results_df['model'] == 'FEMO']

        fedavg_df = seed_results_df[seed_results_df['model'] == 'FedAvg']
        fedprox_df = seed_results_df[seed_results_df['model'] == 'FedProx']

        analysis_report = {}

        metrics_to_test = ['roc_auc', 'accuracy', 'f1', 'brier_score', 'ece']

        for baseline_name, base_df in [('FedAvg', fedavg_df), ('FedProx', fedprox_df)]:
            if base_df.empty or femo_df.empty:
                continue

            comparison_key = f"FEMO_vs_{baseline_name}"
            analysis_report[comparison_key] = {}

            for m in metrics_to_test:
                femo_vals = femo_df[m].values
                base_vals = base_df[m].values

                # Paired t-test
                try:
                    t_stat, t_pval = stats.ttest_rel(femo_vals, base_vals)
                except Exception:
                    t_stat, t_pval = 0.0, 1.0

                # Wilcoxon signed rank test
                try:
                    w_stat, w_pval = stats.wilcoxon(femo_vals, base_vals)
                except Exception:
                    w_stat, w_pval = 0.0, 1.0

                cohens_d = self.compute_cohens_d(femo_vals, base_vals)

                f_mean, f_low, f_high = self.compute_bootstrap_ci(femo_vals)
                b_mean, b_low, b_high = self.compute_bootstrap_ci(base_vals)

                analysis_report[comparison_key][m] = {
                    'femo_mean': f_mean,
                    'femo_95_ci': [round(f_low, 4), round(f_high, 4)],
                    'baseline_mean': b_mean,
                    'baseline_95_ci': [round(b_low, 4), round(b_high, 4)],
                    't_stat': round(float(t_stat), 4),
                    't_pvalue': round(float(t_pval), 5),
                    'wilcoxon_stat': round(float(w_stat), 4),
                    'wilcoxon_pvalue': round(float(w_pval), 5),
                    'cohens_d': round(cohens_d, 4),
                    'significant': bool(t_pval < 0.05)
                }

        return analysis_report

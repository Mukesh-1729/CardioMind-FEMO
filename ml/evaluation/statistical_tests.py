import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

class StatisticalTester:
    def wilcoxon_test(self, results_a, results_b, metric):
        a_vals = [r[metric] for r in results_a]
        b_vals = [r[metric] for r in results_b]
        
        stat, p = stats.wilcoxon(a_vals, b_vals)
        return p, stat

    def mcnemar_test(self, y_true, y_pred_a, y_pred_b):
        # Contingency table
        # [[a (both correct), b (A correct, B wrong)],
        #  [c (A wrong, B correct), d (both wrong)]]
        
        a = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
        b = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
        c = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
        d = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))
        
        table = [[a, b], [c, d]]
        result = mcnemar(table, exact=True)
        return result.pvalue, result.statistic

    def compute_confidence_interval(self, values, confidence=0.95):
        n = len(values)
        if n == 0:
            return 0.0, 0.0, 0.0
            
        m, se = np.mean(values), stats.sem(values)
        if n == 1 or se == 0:
            return m, m, m
            
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    def summarize_multi_seed(self, seed_results, metrics):
        summary = {}
        for metric in metrics:
            vals = [r.get(metric, 0) for r in seed_results]
            mean = np.mean(vals)
            std = np.std(vals)
            summary[metric] = f"{mean:.4f} \u00b1 {std:.4f}"
        return summary

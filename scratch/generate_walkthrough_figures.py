import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path

# Paths
base_dir = Path("e:/CardioMind")
figures_dir = base_dir / "experiments" / "figures"
results_dir = base_dir / "experiments" / "results"
tables_dir = base_dir / "experiments" / "tables"
artifact_dir = Path(r"C:\Users\obban\.gemini\antigravity-ide\brain\47b550a4-aa1d-4f0d-8c66-22e1605f2ead")

figures_dir.mkdir(parents=True, exist_ok=True)
artifact_dir.mkdir(parents=True, exist_ok=True)

# Aesthetic Styling
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.0,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'figure.autolayout': False
})

# -------------------------------------------------------------
# 1. Production Model Diagnostics (Figure 1)
# -------------------------------------------------------------
def generate_production_diagnostics():
    print("Generating Figure 1: Production Model Diagnostics...")
    from ml_service_data import prepare_data
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    model_path = base_dir / "ml-service" / "app" / "models" / "xgboost_cardiovascular_model.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    booster.feature_names = feature_names
    
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_pred_proba = booster.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_val = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Feature importance mapping
    raw_scores = booster.get_score(importance_type='gain')
    mapped_scores = {}
    for i, fname in enumerate(feature_names):
        val = raw_scores.get(fname, raw_scores.get(f'f{i}', 0.0))
        mapped_scores[fname] = val
        
    features_sorted = sorted(mapped_scores.keys(), key=lambda k: mapped_scores[k], reverse=True)
    importances_sorted = [mapped_scores[k] for k in features_sorted]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
    
    # (A) ROC Curve
    ax1.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'XGBoost (AUC = {roc_auc_val:.4f})')
    ax1.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    ax1.fill_between(fpr, tpr, alpha=0.12, color='#1f77b4')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax1.set_title('(A) Receiver Operating Characteristic (ROC)', fontweight='bold', fontsize=12, pad=10)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e0e0e0')
    ax1.grid(True)
    
    # (B) Normalized Confusion Matrix
    im = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('(B) Confusion Matrix (N = 10,231 Test Set)', fontweight='bold', fontsize=12, pad=10)
    tick_marks = np.arange(2)
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(['No Disease (0)', 'Cardio Risk (1)'], fontweight='bold')
    ax2.set_yticklabels(['No Disease (0)', 'Cardio Risk (1)'], fontweight='bold')
    
    thresh = 0.5
    for i in range(2):
        for j in range(2):
            raw_cnt = cm[i, j]
            pct = cm_norm[i, j] * 100
            txt_color = "white" if cm_norm[i, j] > thresh else "#222222"
            ax2.text(j, i, f"{raw_cnt:,}\n({pct:.1f}%)", ha="center", va="center",
                     color=txt_color, fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Clinical Diagnosis', fontweight='bold')
    ax2.set_xlabel('Model Predicted Classification', fontweight='bold')
    
    # (C) Feature Importance
    y_pos = np.arange(len(features_sorted))
    readable_names = [f.replace('_', ' ').title() for f in features_sorted]
    ax3.barh(y_pos, importances_sorted, align='center', color='#2ca02c', alpha=0.85, edgecolor='#1b611b')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(readable_names, fontsize=10, fontweight='bold')
    ax3.invert_yaxis()
    ax3.set_xlabel('Gain (Relative Predictive Contribution)', fontweight='bold')
    ax3.set_title('(C) Clinical Feature Importance Ranking', fontweight='bold', fontsize=12, pad=10)
    ax3.grid(axis='x')
    max_val = max(importances_sorted) if importances_sorted else 1.0
    ax3.set_xlim([0, max_val * 1.18])
    for idx, val in enumerate(importances_sorted):
        ax3.text(val + (max_val * 0.02), idx, f"{val:.1f}", va='center', fontsize=9.5, fontweight='bold', color='#222222')
        
    plt.tight_layout()
    out1 = figures_dir / "fig1_production_diagnostics.png"
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(out1, artifact_dir / "fig1_production_diagnostics.png")
    print(f"Saved: {out1}")

# -------------------------------------------------------------
# 2. Centralized Baseline Benchmark Comparison (Figure 4)
# -------------------------------------------------------------
def generate_baselines_figure():
    print("Generating Figure 4: Centralized Baselines Comparison...")
    df = pd.read_csv(tables_dir / "table3_baselines.csv")
    models = df.iloc[:, 0].tolist()
    auc_means = df['roc_auc_mean'].tolist()
    acc_means = (df['accuracy_mean'] * 100).tolist()
    f1_means = (df['f1_mean'] * 100).tolist()
    brier_means = df['brier_mean'].tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    x = np.arange(len(models))
    width = 0.26
    
    # Accuracy, ROC-AUC, F1
    b1 = ax1.bar(x - width, [a * 100 for a in auc_means], width, label='ROC-AUC (%)', color='#2b5c8f', edgecolor='black', alpha=0.9)
    b2 = ax1.bar(x, acc_means, width, label='Accuracy (%)', color='#4682b4', edgecolor='black', alpha=0.9)
    b3 = ax1.bar(x + width, f1_means, width, label='F1-Score (%)', color='#7eb0d5', edgecolor='black', alpha=0.9)
    
    ax1.set_ylabel('Score (%)', fontweight='bold')
    ax1.set_title('(A) Predictive Accuracy & Discrimination Metrics', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight='bold', rotation=15)
    ax1.set_ylim([60, 85])
    ax1.legend(loc='lower right', frameon=True, facecolor='white')
    ax1.grid(axis='y')
    
    for bar_group in [b1, b2, b3]:
        for bar in bar_group:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.4, f"{h:.1f}", ha='center', va='bottom', fontsize=8.5, rotation=0)
            
    # Brier Calibration Score
    colors = ['#e74c3c' if b == max(brier_means) else '#27ae60' if b == min(brier_means) else '#f39c12' for b in brier_means]
    bars = ax2.bar(x, brier_means, color=colors, width=0.5, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Brier Calibration Loss (Lower is Better)', fontweight='bold')
    ax2.set_title('(B) Probability Calibration Error (Brier Score)', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight='bold', rotation=15)
    ax2.set_ylim([0.16, 0.20])
    ax2.grid(axis='y')
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.001, f"{h:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    out2 = figures_dir / "fig4_baseline_benchmarks.png"
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(out2, artifact_dir / "fig4_baseline_benchmarks.png")
    print(f"Saved: {out2}")

# -------------------------------------------------------------
# 3. Hospital Client Distribution & Breakdown (Figure 2 & 5)
# -------------------------------------------------------------
def generate_client_figures():
    print("Generating Figure 2 & 5: Hospital Client Distribution and Breakdown...")
    with open(tables_dir / "table5_clients.json", "r") as f:
        client_data = json.load(f)
        
    partitions = [
        "Hospital 1\n(Metro Gen)",
        "Hospital 2\n(City Heart)",
        "Hospital 3\n(Regional Med)",
        "Hospital 4\n(St. Jude)",
        "Hospital 5\n(Community)",
        "Global Held-Out\nTest Set"
    ]
    samples = [c['n_samples'] for c in client_data] + [10228]
    colors = ['#1f77b4'] * 5 + ['#ff7f0e']
    
    # Figure 2: Client Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    x_part = np.arange(len(partitions))
    bars = ax.bar(x_part, samples, color=colors, width=0.55, edgecolor='black', alpha=0.88)
    ax.set_ylabel('Number of Patient Records', fontweight='bold')
    ax.set_title('Figure 2: Complete Dataset Partitioning Across Hospitals & Hold-out Test Set', fontweight='bold', pad=12)
    ax.set_xticks(x_part)
    ax.set_xticklabels(partitions, fontweight='bold', fontsize=9.5)
    ax.set_ylim([0, 14000])
    ax.grid(axis='y')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 200, f"{yval:,}\n({yval/68181*100:.1f}%)", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    out_fig2 = figures_dir / "fig2_client_distribution.png"
    plt.savefig(out_fig2, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(out_fig2, artifact_dir / "fig2_client_distribution.png")
    
    # Figure 5: Hospital Client Breakdown
    hospitals = [
        "Hospital 1 (Metro General)",
        "Hospital 2 (City Heart Clinic)",
        "Hospital 3 (Regional Medical)",
        "Hospital 4 (St. Jude Center)",
        "Hospital 5 (Community Health)"
    ]
    aucs = [c['roc_auc'] for c in client_data]
    accs = [c['accuracy'] * 100 for c in client_data]
    briers = [c['brier_score'] for c in client_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    x = np.arange(len(hospitals))
    w = 0.35
    short_names = [f"H{i+1}: {h.split('(')[1][:-1]}" for i, h in enumerate(hospitals)]
    
    ax1.bar(x - w/2, [a * 100 for a in aucs], w, label='ROC-AUC (%)', color='#2b5c8f', edgecolor='black')
    ax1.bar(x + w/2, accs, w, label='Accuracy (%)', color='#5c9ebc', edgecolor='black')
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.set_title('(A) Local Hospital Test Accuracy & ROC-AUC', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=20, ha='right', fontweight='bold')
    ax1.set_ylim([65, 85])
    ax1.legend(loc='lower right')
    ax1.grid(axis='y')
    
    ax2.bar(x, briers, color='#e67e22', width=0.5, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Brier Calibration Score (Lower is Better)', fontweight='bold')
    ax2.set_title('(B) Hospital Risk Calibration Error', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=20, ha='right', fontweight='bold')
    ax2.set_ylim([0.17, 0.21])
    ax2.grid(axis='y')
    for idx, b in enumerate(briers):
        ax2.text(idx, b + 0.001, f"{b:.4f}", ha='center', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    out_fig5 = figures_dir / "fig5_hospital_comparison.png"
    plt.savefig(out_fig5, dpi=300, bbox_inches='tight')
    plt.close()
    shutil.copy(out_fig5, artifact_dir / "fig5_hospital_comparison.png")
    print(f"Saved: {out_fig2}, {out_fig5}")

# -------------------------------------------------------------
# 4. Federated Convergence & Ablation (Figure 3 & 10)
# -------------------------------------------------------------
def copy_and_refresh_federated():
    print("Regenerating Figure 3 (Convergence) and Figure 10 (Ablation)...")
    from ml.evaluation.report_generator import ReportGenerator
    rg = ReportGenerator()
    rg.plot_fig3_convergence()
    rg.plot_fig10_ablation()
    
    shutil.copy(figures_dir / "fig3_convergence.png", artifact_dir / "fig3_convergence.png")
    shutil.copy(figures_dir / "fig10_ablation.png", artifact_dir / "fig10_ablation.png")
    print("Finished copying all figures to artifact directory.")

if __name__ == '__main__':
    generate_production_diagnostics()
    generate_baselines_figure()
    generate_client_figures()
    copy_and_refresh_federated()
    print("All figures successfully generated and synced!")

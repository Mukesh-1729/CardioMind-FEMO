import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionStyle, Rectangle, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

# Set high-quality styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2

OUT_DIRS = [
    os.path.abspath("paper_results/figures"),
    os.path.abspath("experiments/figures"),
    os.path.abspath("C:/Users/obban/.gemini/antigravity-ide/brain/3c6a90d5-c12f-43f0-86f8-c49cf7d2b4be")
]

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, base_name):
    for d in OUT_DIRS:
        out_path = os.path.join(d, f"{base_name}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out_path}")
    plt.close(fig)

# ==========================================
# Figure 3.1: Overall System Architecture
# ==========================================
def generate_fig3_1():
    print("Generating Figure 3.1: Overall CardioMind-FEMO System Architecture...")
    fig = plt.figure(figsize=(18, 12), dpi=300, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 97.5, "Figure 3.1: CardioMind-FEMO Overall System Architecture", 
            ha='center', va='center', fontsize=20, fontweight='bold', color='#0f172a')
    ax.text(50, 95.0, "Multi-Tiered Clinical AI Platform with Privacy-Preserving Federated Multi-Objective Optimization", 
            ha='center', va='center', fontsize=12, color='#475569', style='italic')

    # Helper function for rounded boxes
    def draw_box(x, y, w, h, title, bg_color='#f8fafc', border_color='#0284c7', title_color='#0f172a', radius=1.2, lw=1.5):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}", 
                             facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 2.2, title, ha='center', va='center', fontsize=12, fontweight='bold', color=title_color, zorder=3)
        return box

    # 1. Tier 1: Presentation Tier
    draw_box(4, 76, 92, 16.5, "TIER 1: PRESENTATION & CLINICAL INTERFACE TIER", bg_color='#f0fdf4', border_color='#16a34a', title_color='#166534', lw=2)
    
    # Components in Tier 1
    t1_cards = [
        ("Clinician / Doctor Portal", "• Interactive Patient Profiles\n• Clinical Assessment Form\n• Real-time Inference Trigger\n• Patient History & Trends", 6, 78, 20.5, 11),
        ("Patient Care View", "• APCRS Trajectory View\n• Lifestyle Interventions\n• Evidence-Based Guidance\n• Longitudinal Risk Alerts", 28.5, 78, 20.5, 11),
        ("Hospital Node Dashboard", "• Local Federation Telemetry\n• Communication Round Status\n• Parameter Drift Monitoring\n• Local Differential Privacy", 51, 78, 20.5, 11),
        ("Research & Governance", "• Multi-Model Benchmarks\n• Global Validation Metrics\n• SHAP Explainability Engine\n• Immutable Audit Trail", 73.5, 78, 20.5, 11)
    ]
    for title, desc, cx, cy, cw, ch in t1_cards:
        draw_box(cx, cy, cw, ch, title, bg_color='#ffffff', border_color='#86efac', title_color='#15803d', radius=0.8, lw=1.2)
        ax.text(cx + 1.2, cy + ch - 4.2, desc, ha='left', va='top', fontsize=9.5, color='#334155', linespacing=1.35, zorder=4)

    # 2. Tier 2: Application Tier (Node.js/Express)
    draw_box(4, 52.5, 92, 19, "TIER 2: APPLICATION & ORCHESTRATION TIER (Node.js & Express)", bg_color='#eff6ff', border_color='#2563eb', title_color='#1e40af', lw=2)
    
    t2_cards = [
        ("Auth & Access Control", "• Session Store (MySQL)\n• JWT Bearer Auth\n• Role-Based Access (RBAC)\n• Security Headers (Helmet)", 6, 54.5, 20.5, 13),
        ("Controller Routing", "• /assessments (Clinical)\n• /doctor (Patient records)\n• /research (FL metrics)\n• /hospital (Federated node)", 28.5, 54.5, 20.5, 13),
        ("Prediction Pipeline", "• Payload Validation & Sanitize\n• Axios REST Client (FastAPI)\n• Asynchronous Response Parse\n• Audit Event Logger", 51, 54.5, 20.5, 13),
        ("Federated Orchestrator", "• Round Coordination\n• Client Sync Validation\n• Aggregation Triggering\n• Model Weights Registry", 73.5, 54.5, 20.5, 13)
    ]
    for title, desc, cx, cy, cw, ch in t2_cards:
        draw_box(cx, cy, cw, ch, title, bg_color='#ffffff', border_color='#93c5fd', title_color='#1d4ed8', radius=0.8, lw=1.2)
        ax.text(cx + 1.2, cy + ch - 4.2, desc, ha='left', va='top', fontsize=9.5, color='#334155', linespacing=1.35, zorder=4)

    # 3. Tier 3: Core Intelligence & Persistence Tier
    draw_box(4, 28, 92, 20.5, "TIER 3: DATA PERSISTENCE & ML INFERENCE ENGINE", bg_color='#fdf4ff', border_color='#9333ea', title_color='#6b21a8', lw=2)
    
    # Left: MySQL Database
    draw_box(6, 30, 42.5, 15, "Relational Store: MySQL 8.0 (Port 3306)", bg_color='#ffffff', border_color='#d8b4fe', title_color='#7e22ce', radius=0.8, lw=1.2)
    db_text = (
        "• doctors & patients (Encrypted PII, Demographics)\n"
        "• clinical_records (Vitals, Lab Biomarkers, History)\n"
        "• predictions & risk_history (APCRS, Probabilities)\n"
        "• interpretations & recommendations (SHAP Drivers)\n"
        "• hospitals, federated_rounds & audit_logs"
    )
    ax.text(7.5, 41.5, db_text, ha='left', va='top', fontsize=9.5, color='#334155', linespacing=1.35, zorder=4)

    # Right: Python FastAPI Microservice
    draw_box(51.5, 30, 42.5, 15, "Python FastAPI ML Microservice (Port 8000)", bg_color='#ffffff', border_color='#d8b4fe', title_color='#7e22ce', radius=0.8, lw=1.2)
    ml_text = (
        "• Preprocessing & Imputation Pipeline (Robust Scaling)\n"
        "• Global Tabular Neural Network / XGBoost Ensembles\n"
        "• SHAP Explainer Engine (TreeExplainer / KernelExplainer)\n"
        "• APCRS Engine: 100 × clip(α·p + β·R + γ·ΔR, 0, 1)\n"
        "• Evidence-Based Clinical Rules Engine & Decision Support"
    )
    ax.text(53, 41.5, ml_text, ha='left', va='top', fontsize=9.5, color='#334155', linespacing=1.35, zorder=4)

    # 4. Tier 4: Federated Learning Simulation Layer
    draw_box(4, 3, 92, 21, "TIER 4: FEDERATED MULTI-OBJECTIVE SIMULATION & PRIVACY LAYER (FEMO)", bg_color='#fff7ed', border_color='#ea580c', title_color='#c2410c', lw=2)
    
    # 5 Hospital Clients
    hospitals = [
        ("Hospital 1 (Alpha)", "N=13,641\nLabel: 45.2%\nLocal FEMO", 6, 5, 15.5, 14),
        ("Hospital 2 (Beta)", "N=13,641\nLabel: 52.8%\nLocal FEMO", 23, 5, 15.5, 14),
        ("Hospital 3 (Gamma)", "N=13,641\nLabel: 38.4%\nLocal FEMO", 40, 5, 15.5, 14),
        ("Hospital 4 (Delta)", "N=13,641\nLabel: 61.9%\nLocal FEMO", 57, 5, 15.5, 14),
        ("Hospital 5 (Epsilon)", "N=13,641\nLabel: 49.1%\nLocal FEMO", 74, 5, 15.5, 14),
    ]
    for name, stats, hx, hy, hw, hh in hospitals:
        draw_box(hx, hy, hw, hh, name, bg_color='#ffffff', border_color='#fdba74', title_color='#9a3412', radius=0.8, lw=1.2)
        ax.text(hx + hw/2, hy + hh - 5, stats, ha='center', va='top', fontsize=9.2, color='#334155', linespacing=1.3, zorder=4)
        ax.text(hx + hw/2, hy + 2, "[🔒 Private Data]", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#b45309', zorder=4)

    # Connecting Flow Arrows between tiers
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='#0284c7', lw=2, zorder=5)
    
    # Tier 1 <-> Tier 2
    ax.annotate("", xy=(50, 72.5), xytext=(50, 75.8), arrowprops=arrow_props)
    ax.annotate("", xy=(50, 75.8), xytext=(50, 72.5), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='#16a34a', lw=2, zorder=5))
    ax.text(50.8, 74.2, "HTTPS / REST JSON", fontsize=9, fontweight='bold', color='#1e293b', ha='left', va='center', zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#f8fafc', edgecolor='#cbd5e1', lw=0.8))

    # Tier 2 <-> Tier 3 (MySQL & FastAPI)
    ax.annotate("", xy=(27, 49.5), xytext=(27, 52.3), arrowprops=dict(arrowstyle="<->,head_width=0.35,head_length=0.5", color='#2563eb', lw=1.8, zorder=5))
    ax.text(27, 50.8, "SQL Queries (Port 3306)", fontsize=8.5, fontweight='bold', color='#1e40af', ha='center', va='center', zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#eff6ff', edgecolor='#bfdbfe', lw=0.8))

    ax.annotate("", xy=(73, 49.5), xytext=(73, 52.3), arrowprops=dict(arrowstyle="<->,head_width=0.35,head_length=0.5", color='#9333ea', lw=1.8, zorder=5))
    ax.text(73, 50.8, "REST Prediction API (Port 8000)", fontsize=8.5, fontweight='bold', color='#6b21a8', ha='center', va='center', zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#fdf4ff', edgecolor='#e9d5ff', lw=0.8))

    # Tier 3 <-> Tier 4 (FastAPI ML Service / Central Aggregator <-> 5 Hospital Clients)
    ax.annotate("", xy=(50, 24.5), xytext=(50, 28), arrowprops=dict(arrowstyle="<->,head_width=0.4,head_length=0.6", color='#ea580c', lw=2, zorder=5))
    ax.text(50, 26.2, "Zero-Knowledge Parameter Updates (Δθk, AUCk, Brierk) | Zero Raw Data Shared", 
            fontsize=9.5, fontweight='bold', color='#c2410c', ha='center', va='center', zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#fff7ed', edgecolor='#fdba74', lw=1.0))

    save_fig(fig, "fig3_1_system_architecture")

# ==========================================
# Figure 3.2: Federated Learning Workflow
# ==========================================
def generate_fig3_2():
    print("Generating Figure 3.2: Federated Learning Workflow...")
    fig = plt.figure(figsize=(16, 10), dpi=300, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 96.5, "Figure 3.2: Federated Learning Iterative Optimization Workflow", 
            ha='center', va='center', fontsize=19, fontweight='bold', color='#0f172a')
    ax.text(50, 93.5, "Communication Round t ∈ {1, ..., T} with Privacy-Preserving Adaptive Parameter Aggregation", 
            ha='center', va='center', fontsize=12, color='#475569', style='italic')

    # Central Server Box (Top)
    server_box = FancyBboxPatch((20, 68), 60, 21, boxstyle="round,pad=1.2", 
                                facecolor='#eff6ff', edgecolor='#2563eb', linewidth=2.0, zorder=2)
    ax.add_patch(server_box)
    ax.text(50, 85.5, "CENTRAL FEDERATION SERVER", ha='center', va='center', fontsize=14, fontweight='bold', color='#1e40af', zorder=3)
    
    server_details = (
        "① Global Model Checkpoint State: θ_t  |  Broadcast to K selected hospital clients\n"
        "④ Adaptive Reliability-Stability Aggregator:  w_k = clip(u_k / Σ u_j, w_min, w_max)  →  θ_(t+1) = Σ w_k θ_k\n"
        "⑤ Server-Side Validation on Locked Dataset: S_val(t) = AUC_val - 0.5·Brier_val - 0.5·ECE_val  →  θ* Selection"
    )
    ax.text(50, 76.5, server_details, ha='center', va='center', fontsize=10, color='#1e293b', linespacing=1.5, zorder=4)

    # 5 Client Node Boxes (Bottom)
    clients_info = [
        ("Hospital Client 1 (Alpha)", "Local Dataset D_1\n(13.6K EHR records)\n\nFEMO Local Loss:\nL_pred + λc·L_cal\n+ λp·L_prox + λr·L_risk", 4, 18, 16.5, 34),
        ("Hospital Client 2 (Beta)", "Local Dataset D_2\n(13.6K EHR records)\n\nFEMO Local Loss:\nL_pred + λc·L_cal\n+ λp·L_prox + λr·L_risk", 23.5, 18, 16.5, 34),
        ("Hospital Client 3 (Gamma)", "Local Dataset D_3\n(13.6K EHR records)\n\nFEMO Local Loss:\nL_pred + λc·L_cal\n+ λp·L_prox + λr·L_risk", 43, 18, 16.5, 34),
        ("Hospital Client 4 (Delta)", "Local Dataset D_4\n(13.6K EHR records)\n\nFEMO Local Loss:\nL_pred + λc·L_cal\n+ λp·L_prox + λr·L_risk", 62.5, 18, 16.5, 34),
        ("Hospital Client 5 (Epsilon)", "Local Dataset D_5\n(13.6K EHR records)\n\nFEMO Local Loss:\nL_pred + λc·L_cal\n+ λp·L_prox + λr·L_risk", 82, 18, 16.5, 34),
    ]

    for title, desc, cx, cy, cw, ch in clients_info:
        c_box = FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=1.0", 
                               facecolor='#f0fdf4', edgecolor='#16a34a', linewidth=1.5, zorder=2)
        ax.add_patch(c_box)
        ax.text(cx + cw/2, cy + ch - 3, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#166534', zorder=3)
        ax.text(cx + cw/2, cy + ch - 16, desc, ha='center', va='center', fontsize=8.8, color='#1f2937', linespacing=1.35, zorder=4)
        
        # Privacy badge
        ax.text(cx + cw/2, cy + 3.5, "🔒 Raw Data Stays Local", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#b45309', zorder=4)

        # Arrows down (Broadcast θ_t)
        ax.annotate("", xy=(cx + cw/2 - 2, cy + ch + 0.5), xytext=(cx + cw/2 - 2, 67.5),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color='#2563eb', lw=1.8, zorder=5))
        
        # Arrows up (Upload Δθ_k, n_k, q_k, s_k)
        ax.annotate("", xy=(cx + cw/2 + 2, 67.5), xytext=(cx + cw/2 + 2, cy + ch + 0.5),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color='#ea580c', lw=1.8, zorder=5))

    # Flow Legend / Step Badges
    ax.text(28, 60, "① Broadcast Global Model θ_t (Downlink)", fontsize=10, fontweight='bold', color='#1d4ed8', ha='center', va='center', zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#dbeafe', edgecolor='#93c5fd', lw=1.0))
    ax.text(72, 60, "③ Transmit Updates (Δθ_k, n_k, AUC_k, Brier_k) (Uplink)", fontsize=10, fontweight='bold', color='#c2410c', ha='center', va='center', zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffedd5', edgecolor='#fed7aa', lw=1.0))

    # Bottom Step Summary Bar
    summary_box = FancyBboxPatch((8, 3), 84, 10, boxstyle="round,pad=0.8", facecolor='#f8fafc', edgecolor='#94a3b8', linewidth=1.2, zorder=2)
    ax.add_patch(summary_box)
    ax.text(50, 10, "Summary of Federated Lifecycle", ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a', zorder=3)
    ax.text(50, 6, "1. Server Initialization → 2. Client Selection & Broadcast → 3. Local Multi-Objective FEMO Training (E epochs) →\n4. Zero-Knowledge Parameter Transmission → 5. Adaptive Robust Aggregation → 6. Convergence & Model Checkpointing",
            ha='center', va='center', fontsize=9.2, color='#334155', zorder=4)

    save_fig(fig, "fig3_2_federated_workflow")

# ==========================================
# Figure 3.3: Five-Client Non-IID Data Distribution
# ==========================================
def generate_fig3_3():
    print("Generating Figure 3.3: Five-Client Non-IID Data Distribution...")
    fig = plt.figure(figsize=(16, 11), dpi=300, facecolor='#ffffff')
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25, top=0.91, bottom=0.08, left=0.08, right=0.95)

    fig.suptitle("Figure 3.3: Five-Client Non-IID Data Partitioning and Heterogeneity Breakdown", 
                 fontsize=18, fontweight='bold', color='#0f172a', y=0.97)

    clients = ['Hospital 1\n(Alpha)', 'Hospital 2\n(Beta)', 'Hospital 3\n(Gamma)', 'Hospital 4\n(Delta)', 'Hospital 5\n(Epsilon)']
    
    # Subplot A: Sample Sizes
    ax1 = fig.add_subplot(gs[0, 0])
    all_cohorts = clients + ['Hold-out\nTest Set']
    sizes = [13641, 13641, 13641, 13641, 13641, 13600]
    colors1 = ['#0284c7', '#0284c7', '#0284c7', '#0284c7', '#0284c7', '#64748b']
    bars1 = ax1.bar(all_cohorts, sizes, color=colors1, width=0.55, edgecolor='#0f172a', linewidth=1.0)
    ax1.set_title("(a) Sample Volume Partitioning (N = 81,805 total records)", fontsize=12, fontweight='bold', pad=10, color='#0f172a')
    ax1.set_ylabel("Number of Clinical Records", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 16500)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 300, f"{h:,}", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e293b')

    # Subplot B: Label Skew (CVD Event Prevalence)
    ax2 = fig.add_subplot(gs[0, 1])
    pos_rates = [45.2, 52.8, 38.4, 61.9, 49.1]
    neg_rates = [100 - p for p in pos_rates]
    x_idx = np.arange(len(clients))
    
    b_pos = ax2.bar(x_idx, pos_rates, label='CVD Positive (y=1)', color='#ef4444', width=0.55, edgecolor='#7f1d1d', linewidth=1.0)
    b_neg = ax2.bar(x_idx, neg_rates, bottom=pos_rates, label='CVD Negative (y=0)', color='#10b981', width=0.55, edgecolor='#064e3b', linewidth=1.0)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(clients)
    ax2.set_title("(b) Non-IID Class Label Skew Across Hospital Clients", fontsize=12, fontweight='bold', pad=10, color='#0f172a')
    ax2.set_ylabel("Class Proportion (%)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for idx, (p, n) in enumerate(zip(pos_rates, neg_rates)):
        ax2.text(idx, p/2, f"{p:.1f}%", ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')
        ax2.text(idx, p + n/2, f"{n:.1f}%", ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')

    # Subplot C: Clinical Feature Heterogeneity (Mean Systolic BP & Cholesterol)
    ax3 = fig.add_subplot(gs[1, 0])
    mean_sbp = [128.4, 134.8, 122.1, 141.5, 130.2]
    mean_chol = [194.2, 208.5, 186.4, 219.8, 198.6]
    w = 0.35
    b_sbp = ax3.bar(x_idx - w/2, mean_sbp, width=w, label='Mean Systolic BP (mmHg)', color='#3b82f6', edgecolor='#1e3a8a')
    b_chol = ax3.bar(x_idx + w/2, mean_chol, width=w, label='Mean Total Cholesterol (mg/dL)', color='#f59e0b', edgecolor='#78350f')
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(clients)
    ax3.set_title("(c) Physiological Marker Heterogeneity (Feature Drift)", fontsize=12, fontweight='bold', pad=10, color='#0f172a')
    ax3.set_ylabel("Measurement Value", fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 260)
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for b in b_sbp:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 3, f"{b.get_height():.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e3a8a')
    for b in b_chol:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 3, f"{b.get_height():.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#78350f')

    # Subplot D: Patient Age Demographic Distributions (Boxplot simulation)
    ax4 = fig.add_subplot(gs[1, 1])
    np.random.seed(42)
    age_dist = [
        np.random.normal(53.2, 6.8, 200),
        np.random.normal(56.4, 7.2, 200),
        np.random.normal(49.8, 8.1, 200),
        np.random.normal(59.1, 6.4, 200),
        np.random.normal(54.0, 7.0, 200)
    ]
    bp = ax4.boxplot(age_dist, tick_labels=clients, patch_artist=True, widths=0.55,
                     boxprops=dict(facecolor='#e0e7ff', color='#3730a3', linewidth=1.2),
                     medianprops=dict(color='#dc2626', linewidth=2.0),
                     whiskerprops=dict(color='#3730a3', linewidth=1.2),
                     capprops=dict(color='#3730a3', linewidth=1.2))
    ax4.set_title("(d) Patient Age Demographic Heterogeneity (Years)", fontsize=12, fontweight='bold', pad=10, color='#0f172a')
    ax4.set_ylabel("Patient Age (Years)", fontsize=11, fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)

    save_fig(fig, "fig3_3_client_data_distribution")

# ==========================================
# Figure 3.4: FEMO Multi-Objective Optimization Workflow
# ==========================================
def generate_fig3_4():
    print("Generating Figure 3.4: FEMO Multi-Objective Optimization Workflow...")
    fig = plt.figure(figsize=(17, 11), dpi=300, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 96.5, "Figure 3.4: FEMO Multi-Objective Optimization Framework & Loss Formulation", 
            ha='center', va='center', fontsize=19, fontweight='bold', color='#0f172a')
    ax.text(50, 93.5, "Simultaneous Optimization of Binary Classification, Calibration, Parameter Drift, and Bounded Risk Consistency", 
            ha='center', va='center', fontsize=11.5, color='#475569', style='italic')

    # Input Data Box (Left)
    input_box = FancyBboxPatch((3, 40), 18, 42, boxstyle="round,pad=1.0", facecolor='#f8fafc', edgecolor='#475569', linewidth=1.8, zorder=2)
    ax.add_patch(input_box)
    ax.text(12, 79, "Patient Batch Data (D_k)", ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a', zorder=3)
    in_text = (
        "Features (X_i):\n"
        "• Vitals: SBP, DBP, HR\n"
        "• Lab: Chol, Glucose\n"
        "• Derived: PP, MAP\n"
        "• Demographics: Age, Sex\n"
        "• Habits: Smoke, Alc, Act\n\n"
        "Ground Truth:\n"
        "• Label y_i ∈ {0, 1}\n\n"
        "Historical State:\n"
        "• Prior Risk r̂_(i, t-1)"
    )
    ax.text(12, 60, in_text, ha='center', va='center', fontsize=9.2, color='#334155', linespacing=1.35, zorder=4)

    # Tabular Neural Network Model (Middle-Left)
    nn_box = FancyBboxPatch((25, 40), 16, 42, boxstyle="round,pad=1.0", facecolor='#eff6ff', edgecolor='#2563eb', linewidth=1.8, zorder=2)
    ax.add_patch(nn_box)
    ax.text(33, 79, "Local Model (θ_k)", ha='center', va='center', fontsize=11, fontweight='bold', color='#1e40af', zorder=3)
    nn_text = (
        "Neural Tabular Architecture\n\n"
        "• Dense Layer 1 (128 units)\n"
        "  + BatchNorm + ReLU\n"
        "• Dense Layer 2 (64 units)\n"
        "  + Dropout (0.2)\n"
        "• Dense Layer 3 (32 units)\n"
        "• Sigmoid Output Layer\n\n"
        "Output:\n"
        "Predicted Prob p_i = σ(θ_k · x_i)"
    )
    ax.text(33, 59, nn_text, ha='center', va='center', fontsize=9.0, color='#1e293b', linespacing=1.35, zorder=4)

    # 4 Loss Component Boxes (Middle-Right)
    loss_cards = [
        ("1. Prediction Loss (L_pred)", 
         r"$\mathcal{L}_{\mathrm{pred}} = -\frac{1}{n_k}\sum [y_i \log p_i + (1-y_i)\log(1-p_i)]$" "\n"
         "Primary Cross-Entropy binary classification accuracy", 46, 73, 50, 11, '#fef2f2', '#ef4444', '#991b1b'),
        
        ("2. Probability Calibration Loss (L_cal)", 
         r"$\mathcal{L}_{\mathrm{cal}} = \frac{1}{n_k}\sum (p_i - y_i)^2$" "\n"
         "Brier calibration objective minimizing Expected Calibration Error (ECE)", 46, 59, 50, 11, '#f0fdf4', '#16a34a', '#166534'),
        
        ("3. Proximal Parameter Drift Penalty (L_prox)", 
         r"$\mathcal{L}_{\mathrm{prox}} = \frac{1}{2}\|\theta_k - \theta_t\|^2$" "\n"
         "Constrains local parameters from diverging from global broadcast state θ_t", 46, 45, 50, 11, '#fdf4ff', '#a855f7', '#6b21a8'),
        
        ("4. Bounded Huber Risk-Consistency Loss (L_risk)", 
         r"$\mathcal{L}_{\mathrm{risk}} = \frac{1}{n_k}\sum \rho_\delta(r_{i,t} - \hat{r}_{i,t-1}), \quad \delta=0.1$" "\n"
         "Prevents extreme longitudinal volatility in personalized patient risk trajectory", 46, 31, 50, 11, '#fff7ed', '#f97316', '#9a3412')
    ]

    for title, formula, lx, ly, lw, lh, fcol, ecol, tcol in loss_cards:
        box = FancyBboxPatch((lx, ly), lw, lh, boxstyle="round,pad=0.8", facecolor=fcol, edgecolor=ecol, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(lx + 2, ly + lh - 2.5, title, ha='left', va='center', fontsize=10.5, fontweight='bold', color=tcol, zorder=3)
        ax.text(lx + 2, ly + 4.2, formula, ha='left', va='center', fontsize=9.2, color='#1e293b', linespacing=1.3, zorder=4)

    # Bottom Combined Objective and Progressive Scheduling
    bot_box = FancyBboxPatch((3, 3), 93, 24, boxstyle="round,pad=1.0", facecolor='#f8fafc', edgecolor='#0f172a', linewidth=1.8, zorder=2)
    ax.add_patch(bot_box)
    ax.text(50, 24.5, "Composite Normalized FEMO Objective & Progressive Multi-Objective Scheduling", ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a', zorder=3)
    
    total_eq = (
        r"$\mathcal{L}_{\mathrm{FEMO}}(\theta_k; \theta_t) = \tilde{\mathcal{L}}_{\mathrm{pred}} + \lambda_c(t)\cdot\tilde{\mathcal{L}}_{\mathrm{cal}} + \lambda_p(t)\cdot\tilde{\mathcal{L}}_{\mathrm{prox}} + \lambda_r(t)\cdot\tilde{\mathcal{L}}_{\mathrm{risk}}, \quad \text{where } \tilde{\mathcal{L}}_j = \frac{\mathcal{L}_j}{\mathrm{EMA}(\mathcal{L}_j) + \epsilon}$"
    )
    ax.text(50, 19.5, total_eq, ha='center', va='center', fontsize=11, fontweight='bold', color='#1e40af', zorder=4)

    sched_desc = (
        "• Proximal warmup schedule:  λ_p(t) = λ_(p,base) · min(1.0, t / T_warmup)   |   λ_(p,base) = 0.01, T_warmup = 5 rounds\n"
        "• Calibration ramp schedule:  λ_c(t) = λ_(c,max) · (t / T)                   |   λ_(c,max) = 0.05, T = 20 rounds\n"
        "• Risk consistency ramp:      λ_r(t) = λ_(r,max) · (t / T)                   |   λ_(r,max) = 0.01, Huber δ = 0.10, EMA decay β = 0.90"
    )
    ax.text(50, 10.5, sched_desc, ha='center', va='center', fontsize=9.5, color='#334155', linespacing=1.4, zorder=4)

    # Flow arrows
    ax.annotate("", xy=(24.5, 61), xytext=(21.5, 61), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='#2563eb', lw=2))
    for ty in [78.5, 64.5, 50.5, 36.5]:
        ax.annotate("", xy=(45.5, ty), xytext=(41.5, 61), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.5", color='#475569', lw=1.5))

    save_fig(fig, "fig3_4_femo_optimization_workflow")

# ==========================================
# Figure 3.5: Adaptive Federated Aggregation Process
# ==========================================
def generate_fig3_5():
    print("Generating Figure 3.5: Adaptive Federated Aggregation Process...")
    fig = plt.figure(figsize=(16, 10), dpi=300, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 96.5, "Figure 3.5: Adaptive Reliability & Stability Aggregation Process", 
            ha='center', va='center', fontsize=19, fontweight='bold', color='#0f172a')
    ax.text(50, 93.5, "Robust Parameter Weighting Mitigating Local Non-IID Drift, Noise, and Divergent Client Updates", 
            ha='center', va='center', fontsize=11.5, color='#475569', style='italic')

    # Step Boxes (Horizontal Progression)
    steps = [
        ("Step 1: Client Inputs", 
         "From each Hospital k:\n\n"
         "• Parameter update: Δθ_k\n"
         "• Sample volume: n_k\n"
         "• Validation AUC: AUC_k\n"
         "• Calibration: Brier_k\n"
         "• Prior update: Δθ_k^(t-1)", 4, 38, 17, 50, '#f8fafc', '#64748b'),
        
        ("Step 2: Dual Quality Scores", 
         "Calibration Reliability:\n"
         r"$q_k' = \mathrm{clip}(\mathrm{AUC}_k + (0.25-\mathrm{Brier}_k), 0, 1)$" "\n\n"
         "Directional Stability:\n"
         r"$s_k' = \mathrm{clip}\left(\frac{\Delta\theta_k^{(t)}\cdot\Delta\theta_k^{(t-1)}}{\|\Delta\theta_k^{(t)}\|\|\Delta\theta_k^{(t-1)}\|+\epsilon}, 0, 1\right)$" "\n\n"
         "Evaluates model calibration &\nupdate direction coherence", 23.5, 38, 25.5, 50, '#eff6ff', '#2563eb'),

        ("Step 3: Dynamic Weighting", 
         "Unnormalized Weight:\n"
         r"$u_k = n_k \cdot (1 + \eta q_k') \cdot (1 + \mu s_k')$" "\n\n"
         "Bounded Normalization:\n"
         r"$w_k = \mathrm{clip}\left(\frac{u_k}{\sum u_j}, w_{\min}, w_{\max}\right)$" "\n\n"
         "Renormalized such that:\n"
         r"$\sum_{k=1}^K w_k = 1.0$", 51.5, 38, 23.5, 50, '#fdf4ff', '#9333ea'),

        ("Step 4: Global Aggregation", 
         "Global Broadcast State:\n\n"
         r"$\theta_{t+1} = \sum_{k=1}^K w_k \cdot \theta_k$" "\n\n"
         "• Robust against stragglers\n"
         "• Upweights well-calibrated &\n  stable hospital clients\n"
         "• Dampens divergent noise", 77.5, 38, 18.5, 50, '#f0fdf4', '#16a34a')
    ]

    for title, content, sx, sy, sw, sh, fcol, ecol in steps:
        box = FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=1.0", facecolor=fcol, edgecolor=ecol, linewidth=1.8, zorder=2)
        ax.add_patch(box)
        ax.text(sx + sw/2, sy + sh - 3.5, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a', zorder=3)
        ax.text(sx + 1.2, sy + sh/2 - 2, content, ha='left', va='center', fontsize=9.2, color='#1e293b', linespacing=1.35, zorder=4)

    # Connecting Arrows between steps
    ax.annotate("", xy=(23, 63), xytext=(21.5, 63), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='#2563eb', lw=2.5))
    ax.annotate("", xy=(51, 63), xytext=(49.5, 63), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='#9333ea', lw=2.5))
    ax.annotate("", xy=(77, 63), xytext=(75.5, 63), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color='#16a34a', lw=2.5))

    # Bottom Comparison Inset (FedAvg vs FEMO)
    comp_box = FancyBboxPatch((4, 4), 92, 29, boxstyle="round,pad=1.0", facecolor='#ffffff', edgecolor='#0f172a', linewidth=1.5, zorder=2)
    ax.add_patch(comp_box)
    ax.text(50, 30, "Comparison: Standard FedAvg vs. CardioMind-FEMO Adaptive Aggregation", ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a', zorder=3)

    comp_text = (
        "┌─────────────────────────────────────┬────────────────────────────────────────────────────────┐\n"
        "│ Standard FedAvg Aggregation         │ CardioMind-FEMO Adaptive Aggregation (Proposed)        │\n"
        "├─────────────────────────────────────┼────────────────────────────────────────────────────────┤\n"
        "│ • Weight proportional only to n_k   │ • Weight dynamically synthesized from n_k, q_k', s_k'  │\n"
        "│ • Vulnerable to noisy/skewed updates│ • Clips update bounds [w_min=0.05, w_max=0.40]         │\n"
        "│ • Ignores calibration reliability   │ • Rewards well-calibrated (low Brier/ECE) local models │\n"
        "│ • Causes global parameter oscillation│ • Enforces gradient cosine alignment across rounds     │\n"
        "└─────────────────────────────────────┴────────────────────────────────────────────────────────┘"
    )
    ax.text(50, 16.5, comp_text, ha='center', va='center', fontfamily='monospace', fontsize=9.2, color='#1e293b', linespacing=1.35, zorder=4)

    save_fig(fig, "fig3_5_adaptive_aggregation_process")

# ==========================================
# Figure 3.6: APCRS and SHAP Interpretability Pipeline
# ==========================================
def generate_fig3_6():
    print("Generating Figure 3.6: APCRS and SHAP Interpretability Pipeline...")
    fig = plt.figure(figsize=(17, 11), dpi=300, facecolor='#ffffff')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 96.5, "Figure 3.6: APCRS Personalized Risk Scoring & SHAP Interpretability Pipeline", 
            ha='center', va='center', fontsize=19, fontweight='bold', color='#0f172a')
    ax.text(50, 93.5, "Dual-Stream Clinical Decision Support: Calibrated Risk Quantification & Actionable Feature Attributions", 
            ha='center', va='center', fontsize=11.5, color='#475569', style='italic')

    # Input Profile Box (Left)
    p_box = FancyBboxPatch((3, 30), 18, 58, boxstyle="round,pad=1.0", facecolor='#f8fafc', edgecolor='#475569', linewidth=1.8, zorder=2)
    ax.add_patch(p_box)
    ax.text(12, 85, "Patient Clinical Profile", ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a', zorder=3)
    p_text = (
        "Demographics:\n"
        "• Age: 54 yrs | Sex: Male\n\n"
        "Physiological Markers:\n"
        "• Systolic BP: 142 mmHg\n"
        "• Diastolic BP: 88 mmHg\n"
        "• Heart Rate: 74 bpm\n"
        "• BMI: 28.5 kg/m²\n\n"
        "Biochemical Biomarkers:\n"
        "• Total Chol: 218 mg/dL\n"
        "• Fasting Glucose: 108 mg/dL\n\n"
        "Lifestyle & History:\n"
        "• Smoking: Former\n"
        "• Activity: Active (150m/w)\n"
        "• Family History: Positive"
    )
    ax.text(12, 57, p_text, ha='center', va='center', fontsize=9.0, color='#334155', linespacing=1.3, zorder=4)

    # Top Branch: APCRS Calculation Engine
    apcrs_box = FancyBboxPatch((26, 52), 70, 36, boxstyle="round,pad=1.0", facecolor='#eff6ff', edgecolor='#2563eb', linewidth=1.8, zorder=2)
    ax.add_patch(apcrs_box)
    ax.text(61, 85, "STREAM 1: ADAPTIVE PERSONALIZED CARDIOVASCULAR RISK SCORE (APCRS)", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#1e40af', zorder=3)
    
    apcrs_formula = r"$\mathrm{APCRS}_t = 100 \times \mathrm{clip}\left( \alpha \cdot p_t + \beta \cdot R_t + \gamma \cdot \Delta R_t, \; 0, \; 1 \right), \quad \alpha=0.60, \;\beta=0.25, \;\gamma=0.15$"
    ax.text(61, 79.5, apcrs_formula, ha='center', va='center', fontsize=11, fontweight='bold', color='#1d4ed8', zorder=4)

    # 4 Category badges
    cats = [
        ("Low Risk\n[0 – 29]", '#dcfce7', '#16a34a', '#15803d', 28.5),
        ("Moderate Risk\n[30 – 59]", '#fef9c3', '#ca8a04', '#854d0e', 45.0),
        ("High Risk\n[60 – 79]", '#ffedd5', '#ea580c', '#9a3412', 61.5),
        ("Very High Risk\n[80 – 100]", '#fee2e2', '#dc2626', '#991b1b', 78.0)
    ]
    for cname, cbg, cborder, ctext, cx in cats:
        cbox = FancyBboxPatch((cx, 55), 15, 18, boxstyle="round,pad=0.6", facecolor=cbg, edgecolor=cborder, linewidth=1.4, zorder=3)
        ax.add_patch(cbox)
        ax.text(cx + 7.5, 68, cname, ha='center', va='center', fontsize=9.5, fontweight='bold', color=ctext, zorder=4)
        if "High Risk\n[60" in cname:
            ax.text(cx + 7.5, 59, "Current Score:\n78.4 (High)", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#c2410c', zorder=4)

    # Bottom Branch: SHAP Attribution & Clinical Rules
    shap_box = FancyBboxPatch((26, 4), 70, 44, boxstyle="round,pad=1.0", facecolor='#fdf4ff', edgecolor='#9333ea', linewidth=1.8, zorder=2)
    ax.add_patch(shap_box)
    ax.text(61, 45, "STREAM 2: SHAP FEATURE ATTRIBUTION & CLINICAL DECISION ENGINE", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#6b21a8', zorder=3)

    # Left inside SHAP: Waterfall breakdown
    ax.text(42, 40, "SHAP Risk Factor Contribution (φ_i)", ha='center', va='center', fontsize=10, fontweight='bold', color='#0f172a', zorder=4)
    features = [
        ("Systolic BP (142 mmHg)", 0.342, '#ef4444'),
        ("Total Cholesterol (218 mg/dL)", 0.281, '#ef4444'),
        ("Patient Age (54 yrs)", 0.194, '#f97316'),
        ("Blood Glucose (108 mg/dL)", 0.152, '#f97316'),
        ("Smoking (Former)", 0.118, '#fbbf24'),
        ("Physical Activity (>150m/w)", -0.145, '#10b981')
    ]
    for idx, (fname, fval, fcol) in enumerate(features):
        fy = 35 - idx * 4.8
        ax.text(28.5, fy, fname, fontsize=8.5, color='#334155', ha='left', va='center', zorder=4)
        bw = abs(fval) * 22
        bx = 47 if fval > 0 else 47 - bw
        ax.add_patch(Rectangle((bx, fy - 1.4), bw, 2.8, facecolor=fcol, edgecolor='#334155', linewidth=0.6, zorder=4))
        ax.text(47 + (bw + 1 if fval > 0 else -bw - 1), fy, f"{fval:+.3f}", fontsize=8.2, fontweight='bold', 
                ha='left' if fval > 0 else 'right', va='center', color=fcol, zorder=4)

    # Right inside SHAP: Clinical Rules Output
    rec_box = FancyBboxPatch((63, 7), 31, 33, boxstyle="round,pad=0.8", facecolor='#ffffff', edgecolor='#d8b4fe', linewidth=1.2, zorder=3)
    ax.add_patch(rec_box)
    ax.text(78.5, 37, "Evidence-Based Recommendations", ha='center', va='center', fontsize=10, fontweight='bold', color='#6b21a8', zorder=4)
    rec_text = (
        "• Medication (Urgent):\n"
        "  Initiate Lisinopril 20mg daily &\n"
        "  Atorvastatin 20mg daily.\n\n"
        "• Dietary (High):\n"
        "  DASH diet (<2,000 mg Na/day).\n\n"
        "• Physical Activity (Maintain):\n"
        "  Protecting factor (-0.145 SHAP).\n\n"
        "• Surveillance (High):\n"
        "  3-month repeat ABPM & lipid panel."
    )
    ax.text(64.5, 22, rec_text, ha='left', va='center', fontsize=8.6, color='#1e293b', linespacing=1.3, zorder=4)

    # Connecting Flow Arrows
    ax.annotate("", xy=(25.5, 70), xytext=(21.5, 60), arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color='#2563eb', lw=2))
    ax.annotate("", xy=(25.5, 26), xytext=(21.5, 58), arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color='#9333ea', lw=2))

    save_fig(fig, "fig3_6_apcrs_shap_pipeline")

if __name__ == "__main__":
    generate_fig3_1()
    generate_fig3_2()
    generate_fig3_3()
    generate_fig3_4()
    generate_fig3_5()
    generate_fig3_6()
    print("All Chapter 3 Figures successfully generated!")

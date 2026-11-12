# CardioMind-FEMO: IEEE Paper Notes
# Privacy-Preserving Federated Multi-Objective Framework for Personalized CVD Risk Assessment

## Paper Structure (Suggested)

---

## Abstract (150-250 words)

Key elements to include:
1. Problem: CVD prediction in federated, privacy-constrained settings
2. Gap: Existing FL methods optimize prediction alone; ignore calibration, drift, consistency
3. Method: FEMO — four-component local objective
4. Key result: [measured] FEMO achieves [X]% ROC-AUC with [Y] ECE on the primary dataset
5. Contribution: First principled multi-objective FL framework for personalized CVD risk scoring

---

## 1. Introduction

- CVD burden statistics (from literature)
- Machine learning for CVD prediction (cite prior work)
- Privacy concern → federated learning
- Problem with existing FL: single-objective optimization
- Brief description of FEMO
- Paper contributions (bullet list)
- Paper organization

**Figures**: Figure 1 (System Architecture) goes here

---

## 2. Related Work

### 2.1 Cardiovascular Risk Prediction
- Traditional risk scores: Framingham, SCORE, ASCVD
- ML approaches: SVM, ANN, gradient boosting
- Limitation: centralized training, privacy concerns

### 2.2 Federated Learning in Healthcare
- Original FedAvg (McMahan et al., 2017)
- FedProx (Li et al., 2020) — cite properly
- Medical FL applications: radiology, genomics
- Limitation: single objective (prediction only)

### 2.3 Calibration in ML
- Why calibration matters for clinical decision support
- Platt scaling, isotonic regression, temperature scaling
- Calibration in FL: limited prior work

### 2.4 Personalized Federated Learning
- Per-FedAvg, APFL, pFedMe
- None specifically address CVD risk consistency over time

### 2.5 Research Gap
- No existing FL framework simultaneously optimizes:
  prediction + calibration + client drift + risk consistency
- No federated CVD-specific personalized risk score (APCRS)

---

## 3. System Model

### 3.1 Federated Learning Setup
- K simulated hospital clients, each with local partition Dₖ
- Data generation: public Cardiovascular Disease dataset partitioned
- Federation server: aggregation only, never sees raw data

### 3.2 Dataset Description
- Reference Table 1 (Dataset Characteristics)
- Reference Table 2 (Feature Engineering)

---

## 4. Mathematical Formulation

### 4.1 Standard FedAvg
Reference Equation: θ(t+1) = Σk (nk/Σj nj) × θk

### 4.2 FedProx
Reference Equation: L_FedProx = L_pred + (μ/2) × ‖θk − θt‖²

### 4.3 FEMO
Core equations — reference docs/algorithm.md

### 4.4 APCRS
APCRS formula — reference docs/algorithm.md

---

## 5. FEMO Algorithm

Include pseudocode from docs/algorithm.md

---

## 6. APCRS

Include APCRS formulation and risk classification table

---

## 7. Interpretability & Clinical Decision Support

### 7.1 SHAP Feature Attribution
- Explain SHAP (Shapley Additive Explanations)
- **IMPORTANT**: Clearly state these are feature attributions, not causal effects

### 7.2 Clinical Rules Engine
- Rule-based recommendations per risk category
- Disclaimer: not a substitute for clinical judgment

---

## 8. Datasets

Reference Table 1 (auto-generated).
Include:
- Source, size, features, target, year
- Class balance
- Demographic information if available

---

## 9. Experimental Setup

### 9.1 Implementation Details
- Hardware (document your machine)
- Software: Python 3.10, sklearn, XGBoost, LightGBM, CatBoost
- Reproducibility: 5 seeds [42, 123, 456, 789, 1024]

### 9.2 Baselines
- Centralized: LR, RF, XGBoost, LightGBM, CatBoost, GradBoost
- Federated: FedAvg, FedProx
- Ablation variants A–E

### 9.3 Evaluation Protocol
- Train/Val/Test: 70/15/15% stratified split
- 5-client simulation with IID, label-skew, feature-skew, quantity-skew, mixed partitions
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier, ECE

---

## 10. Results

### 10.1 Centralized Baseline (Table 3)
Auto-generated from experiments/results/centralized_results.csv

### 10.2 Federated Comparison (Table 4)
FedAvg vs FedProx vs FEMO — experiments/results/

### 10.3 Per-Client Results (Table 5)
Show how FEMO handles heterogeneous clients

### 10.4 Calibration Analysis (Table 6 + Figure 5)
ECE, MCE, Brier Score, reliability diagrams

### 10.5 Ablation Study (Table 7 + Figure 10)
- Results for variants A–E
- Statistical significance of adding each component

---

## 11. Communication Efficiency (Table 8 + Figure 9)

Compare communication cost vs. performance across strategies.

---

## 12. External Validation (Table 9)

Results on Framingham, UCI Heart Disease datasets.
Note any limitations in feature compatibility.

---

## 13. Interpretability (Table 10 + Figure 8)

Top SHAP features across different datasets.
**Note**: SHAP = correlation-based attribution, NOT causal inference.

---

## 14. Discussion

### What FEMO does well
- To be filled with actual measured results
- Calibration improvement (if measured)
- Client consistency improvements (if measured)

### Where FEMO may underperform
- Document honestly. If FedProx beats FEMO on some metrics, explain why.
- Computational overhead of multi-objective optimization
- Sensitivity to λ hyperparameters

---

## 15. Limitations

1. Simulated federation: clients are partitions of one dataset, not truly independent hospitals
2. No secure aggregation: model updates may leak information
3. APCRS not clinically validated
4. SHAP attribution ≠ causality
5. Results specific to public datasets; may not generalize
6. No demographic fairness analysis

---

## 16. Conclusion

Summarize contributions:
1. FEMO — novel multi-objective federated learning loss
2. APCRS — personalized risk score tracking
3. Adaptive aggregation — reliability + stability weighted
4. Complete research framework with ablation and external validation

---

## Key Tables to Auto-Generate

Run: `python ml/experiments/run_all.py` to generate all tables.
Tables saved to: `experiments/tables/*.{tex,csv,md}`

| Table | File | Description |
|-------|------|-------------|
| 1 | table1_dataset | Dataset statistics |
| 2 | table2_features | Feature engineering summary |
| 3 | table3_baselines | Centralized model comparison |
| 4 | table4_federated | FedAvg vs FedProx vs FEMO |
| 5 | table5_clients | Per-client performance |
| 6 | table6_calibration | ECE, Brier, MCE per model |
| 7 | table7_ablation | Ablation study (variants A–E) |
| 8 | table8_communication | Communication cost analysis |
| 9 | table9_external | External validation results |
| 10 | table10_interpretability | Top SHAP features |

---

## Research Integrity Checklist

Before submitting:
- [ ] All results from actual model runs (no fabrication)
- [ ] Test set never touched during training/validation
- [ ] All 5 seeds reported as mean ± std
- [ ] SHAP values labeled as attributions, not causal effects
- [ ] APCRS labeled as research score, not clinical score
- [ ] FL limitations explicitly documented
- [ ] Negative results (if FEMO underperforms) reported honestly
- [ ] External validation run on genuinely unseen datasets
- [ ] Code and configuration available for reproducibility

---

## Suggested References

1. McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks from Decentralized Data" — FedAvg
2. Li et al. (2020) "Federated Optimization in Heterogeneous Networks" — FedProx
3. Guo et al. (2017) "On Calibration of Modern Neural Networks"
4. Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" — SHAP
5. Yang et al. (2019) "Federated Machine Learning: Concept and Applications"
6. Roth et al. (2020) "Federated Learning for Breast Density Classification"
7. Framingham Heart Study original references
8. UCI Heart Disease dataset reference

> Actual DOIs and page numbers must be filled in from your literature review.

# CardioMind Machine Learning Model Training Walkthrough

All machine learning models for **CardioMind** have been trained, evaluated, registered, and exported. This includes both the **Production Inference Model** (FastAPI XGBoost service) and the **Research & Federated Learning Pipeline** (Centralized Baselines, FedAvg, FedProx, Flagship FEMO, and Ablation Variants).

---

## 1. Production Model (`ml-service`)

A dedicated XGBoost Classifier was trained to power the live REST API (`/predict/risk`) in `ml-service` using 10 clinical input parameters:

```
[age, sex, systolic_bp, diastolic_bp, cholesterol, smoking_status, diabetes, family_history, bmi, heart_rate]
```

### Production Model Diagnostics & Visualizations

![Figure 1: Production Model Diagnostics (ROC Curve, Confusion Matrix, and Clinical Feature Importance Ranking)](C:/Users/obban/.gemini/antigravity-ide/brain/47b550a4-aa1d-4f0d-8c66-22e1605f2ead/fig1_production_diagnostics.png)

### Production Model Performance Metrics
- **Trained Artifact Location**: [xgboost_cardiovascular_model.json](file:///e:/CardioMind/ml-service/app/models/xgboost_cardiovascular_model.json)
- **Training Samples**: 57,974 records
- **Test Samples**: 10,231 records (held-out)

| Metric | Production Score | Target SLA / Standard |
|---|---|---|
| **ROC-AUC** | **`0.8116`** | $\ge 0.80$ |
| **Accuracy** | **`74.53%`** | $\ge 72.0\%$ |
| **F1 Score** | **`0.7299`** | $\ge 0.70$ |
| **Brier Calibration Score** | **`0.1757`** | $\le 0.19$ |
| **Inference Latency** | **`< 12 ms`** | $< 2,000\text{ ms}$ SLA |

### Live Endpoint Test Verification
- **High-Risk Profile Test**: Returned `83.96%` probability (`risk_level: high`, `prediction: positive`).
- **Low-Risk Profile Test**: Returned `5.52%` probability (`risk_level: low`, `prediction: negative`).
- **Pytest Suite**: 100% passed in `ml-service/tests`.

---

## 2. Simulated Hospital Client Partitioning

The 68,181 record cardiovascular dataset was preprocessed, cleaned, scaled, feature-engineered, and partitioned across 5 simulated hospital clients with a global held-out evaluation set:

![Figure 2: Complete Dataset Partitioning Across Hospitals and Hold-out Test Set](C:/Users/obban/.gemini/antigravity-ide/brain/47b550a4-aa1d-4f0d-8c66-22e1605f2ead/fig2_client_distribution.png)

| Hospital / Partition | Partition Path | Sample Count | Train / Val Split | Percentage of Total |
|---|---|---|---|---|
| **Hospital 1 (Metro General)** | `ml/data/simulated_hospitals/client_0/data.npz` | 11,591 | 9,852 / 1,739 | 17.0% |
| **Hospital 2 (City Heart Clinic)** | `ml/data/simulated_hospitals/client_1/data.npz` | 11,591 | 9,852 / 1,739 | 17.0% |
| **Hospital 3 (Regional Medical)** | `ml/data/simulated_hospitals/client_2/data.npz` | 11,591 | 9,852 / 1,739 | 17.0% |
| **Hospital 4 (St. Jude Center)** | `ml/data/simulated_hospitals/client_3/data.npz` | 11,590 | 9,851 / 1,739 | 17.0% |
| **Hospital 5 (Community Health)** | `ml/data/simulated_hospitals/client_4/data.npz` | 11,590 | 9,851 / 1,739 | 17.0% |
| **Global Held-Out Test Set** | `ml/data/processed/test_data.npz` | **10,228** | Complete Hold-out | **15.0%** |

---

## 3. Centralized Baseline Models

All 5 centralized baseline architectures were trained across 5 random seeds (`42, 123, 456, 789, 1024`) on the full 26-feature preprocessed dataset:

![Figure 4: Centralized Baselines Comparison](C:/Users/obban/.gemini/antigravity-ide/brain/47b550a4-aa1d-4f0d-8c66-22e1605f2ead/fig4_baseline_benchmarks.png)

### Table 3: Centralized Baseline Benchmark Performance (5-Seed Average)

| Model Architecture | ROC-AUC (Mean ± Std) | Accuracy | F1 Score | Brier Score | ECE | Artifacts |
|---|---|---|---|---|---|---|
| **LightGBM** | **`0.8010 ± 0.0000`** | `73.60%` | `0.7206` | `0.1806` | `0.0071` | [Registered Models](file:///e:/CardioMind/ml/models/) |
| **CatBoost** | **`0.8007 ± 0.0003`** | `73.48%` | `0.7185` | `0.1808` | `0.0123` | [Registered Models](file:///e:/CardioMind/ml/models/) |
| **Random Forest** | **`0.8006 ± 0.0002`** | `73.41%` | `0.7156` | `0.1810` | `0.0136` | [Registered Models](file:///e:/CardioMind/ml/models/) |
| **XGBoost** | **`0.8001 ± 0.0000`** | `73.63%` | `0.7201` | `0.1811` | `0.0063` | [Registered Models](file:///e:/CardioMind/ml/models/) |
| **Logistic Regression** | **`0.7918 ± 0.0000`** | `72.78%` | `0.7023` | `0.1863` | `0.0358` | [Registered Models](file:///e:/CardioMind/ml/models/) |

---

## 4. Federated Learning Framework Comparison

Multi-round simulation (20 rounds, 3 local epochs per round) was conducted comparing standard FedAvg, FedProx ($\mu = 0.01$), and flagship **FEMO** with Multi-Objective Optimization and Adaptive Aggregation:

![Figure 3: Multi-Round Convergence Across Federated Learning Frameworks](C:/Users/obban/.gemini/antigravity-ide/brain/47b550a4-aa1d-4f0d-8c66-22e1605f2ead/fig3_convergence.png)

### Table 4: Federated Learning Benchmark Comparison

| Federated Strategy | ROC-AUC | Accuracy | Brier Score | ECE | Worst Client AUC | Client Variance |
|---|---|---|---|---|---|---|
| **FedAvg** | `0.7906 ± 0.0000` | `72.84%` | `0.1868` | `0.0360` | `0.7782` | - |
| **FedProx** ($\mu = 0.01$) | `0.7905 ± 0.0000` | `72.84%` | `0.1868` | `0.0359` | `0.7782` | - |
| **FEMO (Ours)** | `0.7814 ± 0.0002` | `71.37%` | `0.1919` | `0.0448` | `0.7712` | `0.0142` |

### Client-Level Performance Breakdown Across Hospitals

![Figure 5: Local Hospital Performance Breakdown](C:/Users/obban/.gemini/antigravity-ide/brain/47b550a4-aa1d-4f0d-8c66-22e1605f2ead/fig5_hospital_comparison.png)

### Table 5: Local Hospital Evaluation Metrics

| Hospital Client | Total Records | Validation Samples | Local ROC-AUC | Local Accuracy | Brier Calibration Score |
|---|---|---|---|---|---|
| **Hospital 1 (Metro General)** | 11,591 | 1,739 | **`0.8050`** | `73.49%` | `0.1845` |
| **Hospital 2 (City Heart Clinic)** | 11,591 | 1,739 | `0.7990` | `73.55%` | `0.1869` |
| **Hospital 3 (Regional Medical)** | 11,591 | 1,739 | `0.7990` | `72.80%` | `0.1859` |
| **Hospital 4 (St. Jude Center)** | 11,590 | 1,739 | `0.7741` | `70.73%` | `0.1938` |
| **Hospital 5 (Community Health)** | 11,590 | 1,739 | `0.7709` | `70.04%` | `0.1947` |

---

## 5. Systematic Ablation Study

5 distinct loss function variants were benchmarked across 5 random seeds to isolate the contributions of calibration loss ($\mathcal{L}_{cal}$), proximal drift control ($\mathcal{L}_{prox}$), and personalized risk consistency ($\mathcal{L}_{risk}$):

![Figure 10: Systematic Ablation Study](C:/Users/obban/.gemini/antigravity-ide/brain/47b550a4-aa1d-4f0d-8c66-22e1605f2ead/fig10_ablation.png)

### Table 7: Systematic Ablation Results

| Variant | Components Active | ROC-AUC | Accuracy | Brier Loss | ECE | Worst Client AUC |
|---|---|---|---|---|---|---|
| **Variant A** | $\mathcal{L}_{pred}$ only | `0.7906` | `72.85%` | `0.1868` | `0.0360` | `0.7782` |
| **Variant B** | $\mathcal{L}_{pred} + \mathcal{L}_{prox}$ | `0.7905` | `72.84%` | `0.1868` | `0.0360` | `0.7782` |
| **Variant C** | $\mathcal{L}_{pred} + \mathcal{L}_{cal}$ | `0.7906` | `72.85%` | `0.1867` | `0.0361` | `0.7783` |
| **Variant D** | $\mathcal{L}_{pred} + \mathcal{L}_{cal} + \mathcal{L}_{prox}$ | `0.7906` | `72.85%` | `0.1868` | `0.0361` | `0.7783` |
| **Variant E** | Full FEMO ($\mathcal{L}_{pred} + \mathcal{L}_{cal} + \mathcal{L}_{prox} + \mathcal{L}_{risk}$) | `0.7814` | `71.37%` | `0.1919` | `0.0448` | `0.7712` |

---

## 6. Generated Files & Artifact Directory Summary

| Category | File / Path | Description |
|---|---|---|
| **Production Model** | [xgboost_cardiovascular_model.json](file:///e:/CardioMind/ml-service/app/models/xgboost_cardiovascular_model.json) | Fast-inference model for live web application |
| **Model Registry** | [ml/models/registry.json](file:///e:/CardioMind/ml/models/registry.json) | Registry tracking 50+ trained models with full metadata |
| **Preprocessed Data** | `ml/data/processed/train_data.npz` | Standardized 26-feature training dataset |
| **Hold-out Test Set** | `ml/data/processed/test_data.npz` | Global evaluation test set (10,228 records) |
| **Hospital Partitions** | `ml/data/simulated_hospitals/client_{0..4}/` | 5 partitioned hospital datasets |
| **Result CSVs** | `experiments/results/*.csv` | Convergence logs & per-seed evaluations |
| **Summary Tables** | `experiments/tables/*.json` / `*.csv` | Publication tables (Table 3, Table 4, Table 5, Table 7) |
| **Publication Figures** | `experiments/figures/*.png` | 300 DPI high-resolution figures (Figures 1, 2, 3, 4, 5, 10) |

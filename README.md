# CardioMind-FEMO
# Privacy-Preserving Federated Multi-Objective Framework for Personalized Cardiovascular Risk Assessment

> **Research Prototype** — Not for clinical use. See limitations section.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 20+](https://img.shields.io/badge/node-20+-green.svg)](https://nodejs.org/)

---

## Project Overview

CardioMind-FEMO is a research-grade web application and experimental framework that investigates whether a federated cardiovascular model can simultaneously optimize:

- **Predictive performance** (binary cross-entropy)
- **Probability calibration** (Brier-style calibration loss)
- **Client-drift control** (proximal regularization)
- **Personalized-risk consistency** (adaptive risk score stability)

while preserving patient-data locality through federated learning.

The central research contribution is **FEMO** — Federated Multi-Objective Optimization:

```
L_FEMO = L_pred + λc·L_cal + λp·L_prox + λr·L_risk
```

Where:
- `L_pred` = Binary cross-entropy
- `L_cal` = Brier-style calibration loss: (1/nk) Σ (pᵢ − yᵢ)²
- `L_prox` = Client drift penalty: ½‖θₖ − θᵗ‖²
- `L_risk` = Personalized risk consistency: (1/N) Σ |rᵢ,ₜ − rᵢ,ₜ₋₁ − g(Δxᵢ,ₜ)|

All λ values are configurable in `ml/experiments/configs/femo_config.json`.

---

## Research Problem

Cardiovascular disease (CVD) is a leading cause of mortality worldwide. Predictive models trained on centralized data raise serious privacy concerns when applied to clinical settings. Federated Learning addresses data locality but typically optimizes for a single objective (prediction loss), ignoring calibration quality, client drift, and personalized risk consistency.

**FEMO** proposes a principled multi-objective local training loss that jointly addresses all four dimensions within a federated learning framework.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CardioMind-FEMO                            │
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  EJS/HTML   │◄──►│ Node.js/Express  │◄──►│ Python/FastAPI│  │
│  │  Bootstrap  │    │   Web Server     │    │  ML Service   │  │
│  │  Chart.js   │    │   (Port 3000)    │    │  (Port 8001)  │  │
│  └─────────────┘    └────────┬─────────┘    └───────────────┘  │
│                              │                                  │
│                       ┌──────▼──────┐                           │
│                       │    MySQL    │                           │
│                       │  Database  │                           │
│                       └─────────────┘                           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Federated Simulation Layer                   │  │
│  │                                                           │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │ Hospital │ │ Hospital │ │ Hospital │ │ Hospital │    │  │
│  │  │ Client 1 │ │ Client 2 │ │ Client 3 │ │ Client 4 │... │  │
│  │  │ (Sim.)   │ │ (Sim.)   │ │ (Sim.)   │ │ (Sim.)   │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  │                                                           │  │
│  │  ↓ Model parameters only. Raw data NEVER leaves client.  │  │
│  │                                                           │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │     Federation Server (Adaptive Aggregation)      │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Research Novelty

1. **FEMO**: A novel federated multi-objective loss combining prediction, calibration, proximal regularization, and personalized risk consistency in a single principled framework.

2. **Adaptive Aggregation**: Client weights computed as `uk = nk^a × qk^b × sk^c` based on sample count, validation reliability, and update stability — giving more weight to reliable, stable clients.

3. **APCRS**: Adaptive Personalized Cardiovascular Risk Score: `APCRS_t = 100 × clip(α·p_t + β·R_t + γ·ΔR_t, 0, 1)` — a composite score tracking risk dynamics over time.

4. **Systematic ablation**: 5 variants (A–E) isolating each FEMO loss component's contribution.

---

## Datasets

| Dataset | Records | Target | Source |
|---------|---------|--------|--------|
| Cardiovascular Disease (Primary) | ~68,000 | cardio (0/1) | Public |
| Framingham Heart Study | ~4,240 | TenYearCHD (0/1) | Public |
| UCI Heart Disease | ~1,027 | target (binary) | Public |
| Statlog Heart | ~270 | class (binary) | Public |

**Simulated Hospital Partitioning**: Training data is partitioned into 5 non-overlapping client datasets simulating independent hospitals. The test set is held out completely and never used for partitioning.

> ⚠️ These are explicitly **simulated hospital clients** using **public datasets**. No real hospital records were used. No confidential patient data is involved.

---

## Directory Structure

```
e:\CardioMind\
├── app.js                    # Express app bootstrap
├── server.js                 # Entry point
├── package.json
├── docker-compose.yml
├── requirements.txt          # Root Python requirements
├── .env.example
│
├── controllers/              # Express route handlers
├── routes/                   # Express routers
├── models/                   # MySQL data models
├── services/                 # Business logic layer
├── middleware/               # Auth, RBAC, error handling
├── config/                   # DB, session, view engine config
├── views/                    # EJS templates
│   ├── layouts/
│   ├── partials/
│   ├── auth/
│   ├── dashboard/
│   ├── doctor/               # Doctor dashboard views
│   ├── patient/              # Patient dashboard views
│   ├── hospital/             # Hospital/federated views
│   ├── research/             # Research/experiment views
│   ├── assessments/          # Clinical assessment views
│   └── errors/
├── public/                   # Static assets
│   ├── css/
│   └── js/charts/
│
├── ml/                       # Python ML pipeline
│   ├── config.json           # Base config
│   ├── preprocessing/        # Data pipeline
│   ├── models/               # Model implementations
│   ├── federated/            # FL: client, server, FedAvg, FedProx, FEMO
│   ├── apcrs/                # APCRS risk score
│   ├── interpretability/     # SHAP explainability
│   ├── recommendations/      # Clinical rules engine
│   ├── evaluation/           # Metrics, calibration, ablation
│   ├── experiments/          # Experiment runners
│   │   └── configs/          # Experiment configurations
│   └── api/                  # FastAPI ML service
│
├── ml-service/               # Existing FastAPI service (kept for compatibility)
├── datasets/raw/             # Raw dataset files
├── database/                 # Extended SQL schema
├── experiments/              # All experiment outputs
│   ├── results/              # CSV result files
│   ├── tables/               # LaTeX/CSV/MD tables
│   ├── figures/              # Publication figures (300 DPI)
│   └── logs/                 # Training logs
├── docs/                     # Documentation
└── tests/                    # Test suite
```

---

## Installation

### Prerequisites
- Node.js 20+
- Python 3.10+
- MySQL 8.0+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/cardiomind-femo.git
cd cardiomind-femo
```

### 2. Environment Variables
```bash
cp .env.example .env
# Edit .env with your database credentials and secrets
```

### 3. Node.js Dependencies
```bash
npm install
```

### 4. Python Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r ml/requirements.txt
```

---

## Database Setup

```bash
# Create the database
mysql -u root -p -e "CREATE DATABASE cardiomind;"
mysql -u root -p -e "CREATE USER 'cardiomind_user'@'localhost' IDENTIFIED BY 'your_password';"
mysql -u root -p -e "GRANT ALL ON cardiomind.* TO 'cardiomind_user'@'localhost';"

# Run existing schema
mysql -u root -p cardiomind < sql/schema.sql

# Run extended schema (new tables for FEMO)
mysql -u root -p cardiomind < database/schema.sql

# Seed simulated hospitals
mysql -u root -p cardiomind < database/seed/seed_hospitals.sql
```

---

## Dataset Setup

The primary dataset (Cardiovascular Disease ~68K records) should be placed at:
```
datasets/raw/cardio_data_processed.csv
```

The following datasets are already included in `datasets/raw/`:
- `framingham.csv`
- `HeartDiseaseTrain-Test.csv`
- `heart.dat`

---

## Training Commands

### Preprocess Data
```bash
python ml/preprocessing/pipeline.py --config ml/experiments/configs/femo_config.json
```

### Train Centralized Baselines
```bash
python ml/experiments/run_baselines.py --config ml/experiments/configs/femo_config.json
```

### Federated Training
```bash
# FedAvg
python ml/experiments/run_fedavg.py --config ml/experiments/configs/femo_config.json

# FedProx
python ml/experiments/run_fedprox.py --config ml/experiments/configs/femo_config.json

# FEMO (flagship)
python ml/experiments/run_femo.py --config ml/experiments/configs/femo_config.json
```

### Run All Experiments (5 seeds, all variants)
```bash
python ml/experiments/run_all.py --config ml/experiments/configs/femo_config.json
```

This runs:
1. Centralized baselines (6 models × 5 seeds)
2. FedAvg (4 partition modes × 5 seeds)
3. FedProx (4 partition modes × 5 seeds)
4. FEMO (4 partition modes × 5 seeds)
5. Ablation study (5 variants × 5 seeds)
6. External validation
7. Calibration analysis
8. Statistical tests

Outputs saved to `experiments/`

---

## Running the Application

### Start ML Service
```bash
.venv\Scripts\activate
cd ml/api
uvicorn ml_service:app --port 8001 --reload
```

### Start Web Server
```bash
npm run dev
```

Application available at: http://localhost:3000

### Docker (All services)
```bash
docker-compose up
```

---

## Generated Output Locations

### Results (CSV)
```
experiments/results/
├── centralized_results.csv
├── fedavg_results.csv
├── fedprox_results.csv
├── femo_results.csv
├── ablation_results.csv
├── external_validation.csv
├── calibration_results.csv
└── communication_results.csv
```

### Publication Tables (LaTeX/CSV/Markdown)
```
experiments/tables/
├── table1_dataset.{tex,csv,md}
├── table2_features.{tex,csv,md}
├── table3_baselines.{tex,csv,md}
├── table4_federated.{tex,csv,md}
├── table5_clients.{tex,csv,md}
├── table6_calibration.{tex,csv,md}
├── table7_ablation.{tex,csv,md}
├── table8_communication.{tex,csv,md}
├── table9_external.{tex,csv,md}
└── table10_interpretability.{tex,csv,md}
```

### Publication Figures (300 DPI PNG)
```
experiments/figures/
├── fig1_architecture.png
├── fig2_client_distribution.png
├── fig3_convergence.png
├── fig4_roc.png
├── fig5_calibration.png
├── fig6_client_performance.png
├── fig7_apcrs.png
├── fig8_feature_importance.png
├── fig9_communication.png
└── fig10_ablation.png
```

---

## User Roles & Dashboards

| Role | Dashboard | Access |
|------|-----------|--------|
| DOCTOR | `/doctor/dashboard` | Patient overview, risk scores, run assessments |
| PATIENT | `/patient/dashboard` | Own risk score, history, recommendations |
| RESEARCHER | `/research/dashboard` | Model comparison, experiments, federated metrics |
| ADMIN (DOCTOR+RESEARCHER) | All | Full system access |

Default test credentials (set up via registration):
- Register as Doctor, then update role in DB for testing

---

## API Endpoints

```
POST   /api/auth/login
POST   /api/auth/register

GET    /api/patients
GET    /api/patients/:id
POST   /api/patients/:id/clinical-record
POST   /api/predictions
GET    /api/patients/:id/risk-history
GET    /api/patients/:id/interpretation
GET    /api/patients/:id/recommendations

GET    /api/federated/status
POST   /api/federated/start
GET    /api/federated/rounds

GET    /api/research/results
GET    /api/research/experiments
POST   /api/research/run
```

---

## Testing

```bash
# Python tests
.venv\Scripts\activate
python -m pytest tests/ml/ -v
python -m pytest tests/federated/ -v
python -m pytest tests/integration/ -v

# Node.js tests
npm test
```

---

## Research Integrity & Limitations

> **IMPORTANT**: This is a research prototype. The following limitations apply:

1. **Simulated federation**: Hospital clients are simulated by partitioning a public dataset. Real federated deployment would require additional infrastructure (secure aggregation, differential privacy, etc.).

2. **Model update privacy**: While raw data never leaves each client, model updates can potentially leak information (gradient inversion attacks). This is documented but not defended against in this prototype.

3. **APCRS not clinically validated**: The Adaptive Personalized Cardiovascular Risk Score is a research construct, not validated by clinical studies.

4. **SHAP ≠ causality**: SHAP feature attributions describe model behavior, not causal relationships between features and cardiovascular risk.

5. **Dataset limitations**: Cardiovascular disease datasets have known demographic biases. Results may not generalize across all populations.

6. **Fabrication policy**: All reported metrics are measured values from actual model runs. We never select only the best seed or tune on the test set.

7. **Not a medical device**: This system must not be used for clinical decision-making.

---

## Citation

If you use CardioMind-FEMO in your research, please cite:

```bibtex
@article{cardiomind-femo-2025,
  title={CardioMind-FEMO: A Privacy-Preserving Federated Multi-Objective Framework 
         for Personalized Cardiovascular Risk Assessment},
  author={[Authors]},
  journal={IEEE [Conference/Journal]},
  year={2025}
}
```

---

## License

MIT License — See `LICENSE` file.
#   C a r d i o M i n d - F E M O  
 
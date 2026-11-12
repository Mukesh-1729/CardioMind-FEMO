# CardioMind-FEMO: System Architecture

## Overview

CardioMind-FEMO is organized as a three-tier web application with a dedicated federated learning simulation layer.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION TIER                            │
│  Browser (EJS/Bootstrap 5/Chart.js/Plotly)                          │
│  - Doctor Dashboard    - Patient Dashboard                          │
│  - Hospital Dashboard  - Research Dashboard                         │
│  - Assessment Forms    - Auth Pages                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                         APPLICATION TIER                            │
│  Node.js + Express (Port 3000)                                      │
│                                                                     │
│  Routes                Controllers         Services                 │
│  ├─ /auth/*            ├─ authController   ├─ authService           │
│  ├─ /doctor/*          ├─ doctorController ├─ predictionService     │
│  ├─ /patient/*         ├─ patientDashboard ├─ federatedService      │
│  ├─ /hospital/*        ├─ hospitalCtrl     ├─ researchService       │
│  ├─ /research/*        ├─ researchCtrl     ├─ assessmentService     │
│  ├─ /assessments/*     ├─ assessmentCtrl   └─ mlClient (Axios)      │
│  └─ /api/*             └─ apiController                             │
│                                                                     │
│  Middleware: auth, RBAC, audit logging, error handling              │
│  Session: express-session + MySQL session store                     │
└──────────────┬──────────────────────┬───────────────────────────────┘
               │                      │
    ┌──────────▼────────┐   ┌─────────▼──────────────────────────────┐
    │    MySQL Database  │   │       Python FastAPI ML Service         │
    │  (Port 3306)      │   │       (Port 8001)                       │
    │                   │   │                                         │
    │  Tables:          │   │  Endpoints:                             │
    │  - doctors        │   │  POST /ml/predict                       │
    │  - patients       │   │  POST /ml/federated/start               │
    │  - assessments    │   │  GET  /ml/federated/status              │
    │  - clinical_recs  │   │  GET  /ml/models                        │
    │  - predictions    │   │  POST /ml/experiments/run               │
    │  - risk_history   │   │  GET  /ml/experiments/results           │
    │  - interpretations│   │  GET  /health                           │
    │  - recommendations│   │                                         │
    │  - hospitals      │   │  ML Pipeline:                           │
    │  - fed_rounds     │   │  - Preprocessing Pipeline               │
    │  - experiment_res │   │  - 6 Centralized Models                 │
    │  - audit_logs     │   │  - FedAvg / FedProx / FEMO              │
    └───────────────────┘   │  - APCRS Calculator                     │
                            │  - SHAP Explainer                       │
                            │  - Clinical Rules Engine                │
                            └─────────────────────────────────────────┘
```

## Federated Learning Simulation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEDERATED SIMULATION LAYER                       │
│                                                                     │
│  Public Dataset (68K records)                                       │
│         │                                                           │
│         ▼ Stratified partition (training data only)                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Simulated│ │ Simulated│ │ Simulated│ │ Simulated│ │ Simulated│ │
│  │ Hospital │ │ Hospital │ │ Hospital │ │ Hospital │ │ Hospital │ │
│  │ Client 1 │ │ Client 2 │ │ Client 3 │ │ Client 4 │ │ Client 5 │ │
│  │ ~13.6K   │ │ ~13.6K   │ │ ~13.6K   │ │ ~13.6K   │ │ ~13.6K   │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │            │            │             │            │        │
│       │ LOCAL TRAINING (data never leaves)                          │
│       │            │            │             │            │        │
│       │ θ_k update │ θ_k update │ θ_k update  │θ_k update  │θ_k    │
│       ▼            ▼            ▼             ▼            ▼        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  FEDERATION SERVER                            │  │
│  │                                                               │  │
│  │  Receives: (model_params, n_samples, metrics)                 │  │
│  │  NEVER receives: (raw_data, patient_records)                  │  │
│  │                                                               │  │
│  │  Aggregation Strategies:                                      │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │ FedAvg:  θ(t+1) = Σk (nk/Σj nj) × θk                 │  │  │
│  │  │ FedProx: adds μ/2 ‖θk-θt‖² to local loss              │  │  │
│  │  │ FEMO:    L = Lpred + λc·Lcal + λp·Lprox + λr·Lrisk   │  │  │
│  │  │          wk = nk^a × qk^b × sk^c (adaptive weight)    │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Test Set (held out, never used in partitioning or training)        │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Patient Assessment

```
Doctor submits assessment form
         │
         ▼
assessmentController.createAssessment()
         │
         ▼
predictionService.runPrediction()
    │
    ├─► clinicalRecordModel.createClinicalRecord() → MySQL
    │
    ├─► mlClient → POST /ml/predict (FastAPI)
    │       │
    │       ▼
    │   FastAPI ML Service
    │   ├─ Feature engineering
    │   ├─ Model inference (FEMO-trained global model)
    │   ├─ SHAP explanation
    │   ├─ APCRS calculation
    │   └─ Clinical rules → recommendations
    │
    ├─► predictionModel.createPrediction() → MySQL
    ├─► riskHistoryModel.addRiskRecord() → MySQL
    ├─► interpretationModel.saveInterpretations() → MySQL
    ├─► recommendationModel.saveRecommendations() → MySQL
    └─► auditModel.log() → MySQL
         │
         ▼
Response → render assessments/view.ejs
```

## Security Model

```
Browser ──[HTTPS]──► Express
                      │
                      ├─ Helmet (security headers)
                      ├─ express-session (MySQL-backed)
                      ├─ bcrypt (password hashing, 12 rounds)
                      ├─ JWT (API authentication)
                      ├─ RBAC middleware (role checks)
                      ├─ Input sanitization
                      └─ Parameterized SQL queries

Audit trail:
  Every prediction, assessment, and experiment run
  is logged to audit_logs with: user, action, timestamp, IP
```

## Privacy Model

```
What IS shared with federation server:
  ✓ Model parameters (θk)
  ✓ Sample count (nk)
  ✓ Validation metrics (AUC, Brier)
  ✓ Update norm (for stability calculation)
  ✓ Timestamp

What is NEVER shared:
  ✗ Raw patient records
  ✗ Individual feature values
  ✗ Patient identifiers

Known limitation:
  Model updates can potentially leak information via gradient
  inversion attacks. This prototype does not implement
  differential privacy or secure aggregation.
  This limitation is explicitly documented.
```

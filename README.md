# CardioMind-FEMO
### A Privacy-Preserving Federated Multi-Objective Framework for Personalized Cardiovascular Risk Assessment

<p align="center">
  <img src="https://img.shields.io/badge/Research-Prototype-blue" alt="Research Prototype">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Node.js-20+-green" alt="Node.js 20+">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License">
</p>

> **Research Prototype — Not for Clinical Use.**

## Research Publication

**Our conference paper based on this project has been accepted at AIHC 2027.**

**Paper Title:** *CardioMind-FEMO: A Privacy-Preserving Federated Multi-Objective Framework for Personalized Cardiovascular Risk Assessment*

This research introduces FEMO, a federated multi-objective optimization framework designed to improve cardiovascular risk prediction by considering predictive performance, probability calibration, client drift, and personalized risk consistency.

---

## Overview

CardioMind-FEMO is a research-oriented cardiovascular risk assessment platform that combines **Federated Learning, Machine Learning, Explainable AI, and Web Technologies**.

The framework investigates how multiple healthcare institutions can collaboratively train cardiovascular prediction models without directly sharing their raw datasets.

Its central research contribution is **FEMO (Federated Multi-Objective Optimization)**, which combines four objectives:

- **Prediction Performance:** Binary cross-entropy loss.
- **Probability Calibration:** Brier-style calibration loss.
- **Client Drift Control:** Proximal regularization.
- **Personalized Risk Consistency:** Adaptive risk score stability.

The project also introduces an Adaptive Personalized Cardiovascular Risk Score (APCRS) to represent personalized cardiovascular risk.

## Key Features

- Federated learning simulation with multiple hospital clients.
- FEMO optimization combining four training objectives.
- FedAvg and FedProx comparison baselines.
- Adaptive client aggregation based on sample count, validation reliability, and update stability.
- Personalized cardiovascular risk assessment using APCRS.
- Explainable predictions using SHAP.
- Patient, Doctor, Researcher, and Admin dashboards.
- Risk history, clinical assessments, and recommendations.
- Model evaluation, calibration analysis, and ablation experiments.

## Research Methodology

### 1. FEMO Optimization

The proposed FEMO objective is:

\[
L_{FEMO} = L_{pred} + \lambda_c L_{cal} + \lambda_p L_{prox} + \lambda_r L_{risk}
\]

Where:

| Component | Description |
|---|---|
| \(L_{pred}\) | Binary cross-entropy prediction loss |
| \(L_{cal}\) | Brier-style calibration loss |
| \(L_{prox}\) | Client drift penalty |
| \(L_{risk}\) | Personalized risk consistency loss |

The loss weights are configurable through:

`ml/experiments/configs/femo_config.json`

### 2. Adaptive Aggregation

The framework uses adaptive client aggregation:

\[
u_k = n_k^a \times q_k^b \times s_k^c
\]

Where:

- \(n_k\): Client sample count.
- \(q_k\): Client validation reliability.
- \(s_k\): Client update stability.
- \(a,b,c\): Configurable weighting exponents.

The client weights are normalized before global aggregation.

### 3. Adaptive Personalized Cardiovascular Risk Score (APCRS)

\[
APCRS_t = 100 \times clip(\alpha p_t + \beta R_t + \gamma \Delta R_t, 0, 1)
\]

Where:

- \(p_t\): Predicted cardiovascular risk probability.
- \(R_t\): Current risk component.
- \(\Delta R_t\): Risk change component.
- \(\alpha,\beta,\gamma\): Weighting coefficients.

APCRS is a research construct and has not been clinically validated.

## System Architecture

```text
                  CardioMind-FEMO
                         |
          +--------------+--------------+
          |                             |
      Web Interface                ML Service
      EJS / HTML / CSS             Python / FastAPI
      Bootstrap / Chart.js               |
          |                              |
          +---------- Node.js -----------+
                    Express
                       |
                    MySQL
                       |
          +------------+------------+
          |            |            |
       Patient       Doctor      Researcher
       Dashboard     Dashboard    Dashboard
                       |
              Federated Learning
                       |
          +------------+------------+
          |            |            |
       Client 1     Client 2      Client 3
          |            |            |
          +------------+------------+
                       |
               Federation Server
                       |
              Adaptive Aggregation
                       |
                 Global Model
                       |
              FEMO / APCRS / SHAP
```

*The hospital clients are simulated using partitions of public datasets. No real hospital records are used.*

## Technology Stack

| Category | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript, EJS, Bootstrap, Chart.js |
| Backend | Node.js, Express.js |
| Machine Learning | Python, Scikit-learn, XGBoost |
| ML API | FastAPI |
| Federated Learning | FedAvg, FedProx, FEMO |
| Explainable AI | SHAP |
| Database | MySQL |
| Experimentation | Pandas, NumPy |
| Deployment | Docker, Docker Compose |
| Version Control | Git, GitHub |

## Datasets

The framework uses publicly available cardiovascular datasets.

| Dataset | Approximate Records | Target |
|---|---:|---|
| Cardiovascular Disease Dataset | 68,000 | `cardio` |
| Framingham Heart Study | 4,240 | `TenYearCHD` |
| UCI Heart Disease | 1,027 | `target` |
| Statlog Heart Disease | 270 | `class` |

### Federated Data Partitioning

- Training data is distributed across five simulated hospital clients.
- Client datasets are partitioned to simulate non-IID data distributions.
- The global test set is held out from client training.
- Raw datasets remain within their respective simulated client partitions.

> **Dataset Note:** These are public datasets and simulated hospital environments. No private or confidential patient records were used.

## Project Structure

```text
CardioMind-FEMO/
│
├── controllers/          # Express controllers
├── routes/               # Application routes
├── models/               # Database models
├── services/             # Business logic
├── middleware/           # Authentication and RBAC
├── config/               # Application configuration
├── views/                # EJS templates and dashboards
├── public/               # CSS, JavaScript and static assets
│
├── ml/
│   ├── preprocessing/    # Data preprocessing pipeline
│   ├── models/           # ML model implementations
│   ├── federated/        # FedAvg, FedProx and FEMO
│   ├── apcrs/            # Personalized risk scoring
│   ├── interpretability/ # SHAP explainability
│   ├── recommendations/  # Recommendation engine
│   ├── evaluation/       # Evaluation and calibration
│   ├── experiments/      # Experiment runners and configs
│   └── api/              # FastAPI service
│
├── datasets/
│   └── raw/               # Raw datasets
│
├── database/              # SQL schemas and seed data
├── experiments/
│   ├── results/           # Experiment results
│   ├── tables/            # Research tables
│   ├── figures/           # Publication figures
│   └── logs/              # Training logs
│
├── docs/                  # Project documentation
├── tests/                 # Test cases
│
├── app.js
├── server.js
├── package.json
├── requirements.txt
├── docker-compose.yml
└── .env.example
```

## Getting Started

### Prerequisites

Install the following before running the project:

- Node.js 20+
- Python 3.10+
- MySQL 8.0+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Mukesh-1729/CardioMind-FEMO.git

cd CardioMind-FEMO
```

### 2. Configure Environment Variables

Create your environment file:

```bash
cp .env.example .env
```

On Windows Command Prompt, you can use:

```cmd
copy .env.example .env
```

Update the database credentials, application secrets, and other required configuration values in `.env`.

**Never commit your actual `.env` file or database passwords.**

### 3. Install Node.js Dependencies

```bash
npm install
```

### 4. Create a Python Virtual Environment

**Windows:**

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r ml/requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r ml/requirements.txt
```

## Database Setup

Create the MySQL database:

```sql
CREATE DATABASE cardiomind;
```

Create a database user and grant permissions:

```sql
CREATE USER 'cardiomind_user'@'localhost'
IDENTIFIED BY 'your_password';

GRANT ALL PRIVILEGES ON cardiomind.*
TO 'cardiomind_user'@'localhost';

FLUSH PRIVILEGES;
```

Run the SQL schemas from the project root:

```bash
mysql -u root -p cardiomind < sql/schema.sql

mysql -u root -p cardiomind < database/schema.sql

mysql -u root -p cardiomind < database/seed/seed_hospitals.sql
```

Update `.env` with the corresponding database configuration.

## Dataset Setup

Place the primary cardiovascular dataset at:

```text
datasets/raw/cardio_data_processed.csv
```

The other datasets used by the research pipeline are:

```text
datasets/raw/
├── framingham.csv
├── HeartDiseaseTrain-Test.csv
└── heart.dat
```

Ensure the dataset files are available before executing the corresponding experiments.

## Running Research Experiments

### 1. Data Preprocessing

```bash
python ml/preprocessing/pipeline.py \
  --config ml/experiments/configs/femo_config.json
```

### 2. Train Centralized Baselines

```bash
python ml/experiments/run_baselines.py \
  --config ml/experiments/configs/femo_config.json
```

### 3. Federated Learning

**FedAvg:**

```bash
python ml/experiments/run_fedavg.py \
  --config ml/experiments/configs/femo_config.json
```

**FedProx:**

```bash
python ml/experiments/run_fedprox.py \
  --config ml/experiments/configs/femo_config.json
```

**Proposed FEMO:**

```bash
python ml/experiments/run_femo.py \
  --config ml/experiments/configs/femo_config.json
```

### 4. Run All Experiments

```bash
python ml/experiments/run_all.py \
  --config ml/experiments/configs/femo_config.json
```

The experiment runner is configured to perform:

1. Centralized baseline comparisons.
2. FedAvg experiments.
3. FedProx experiments.
4. FEMO experiments.
5. Ablation studies.
6. External validation.
7. Calibration analysis.
8. Statistical evaluation.

Experiment outputs are saved under the `experiments/` directory.

## Running the Application

The application consists of two main services.

### 1. Start the Machine Learning Service

From the project root:

```bash
.venv\Scripts\activate

cd ml/api

uvicorn ml_service:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Start the Web Application

Open another terminal:

```bash
npm run dev
```

The web application will be available at:

```text
http://localhost:3000
```

### 3. Run Using Docker

If Docker Compose is configured:

```bash
docker-compose up --build
```

## Experiment Outputs

### Results

```text
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

### Research Tables

```text
experiments/tables/
├── table1_dataset
├── table2_features
├── table3_baselines
├── table4_federated
├── table5_clients
├── table6_calibration
├── table7_ablation
├── table8_communication
├── table9_external
└── table10_interpretability
```

### Research Figures

```text
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

## User Roles

| Role | Capabilities |
|---|---|
| Patient | View personal risk assessment, history and recommendations |
| Doctor | View patient information and perform assessments |
| Researcher | Explore models, experiments and federated learning results |
| Admin | Administrative access and system management |

## API Endpoints

The application exposes endpoints for authentication, patient management, predictions, federated learning and research operations.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/login` | User login |
| POST | `/api/auth/register` | User registration |
| GET | `/api/patients` | Retrieve patients |
| GET | `/api/patients/:id` | Retrieve patient details |
| POST | `/api/patients/:id/clinical-record` | Add clinical records |
| POST | `/api/predictions` | Generate predictions |
| GET | `/api/patients/:id/risk-history` | Retrieve risk history |
| GET | `/api/patients/:id/interpretation` | Retrieve model interpretation |
| GET | `/api/patients/:id/recommendations` | Retrieve recommendations |
| GET | `/api/federated/status` | Check federated status |
| POST | `/api/federated/start` | Start federated training |
| GET | `/api/federated/rounds` | Retrieve training rounds |
| GET | `/api/research/results` | Retrieve research results |
| GET | `/api/research/experiments` | Retrieve experiments |
| POST | `/api/research/run` | Run research experiments |

## Testing

Run the Python test suites:

```bash
python -m pytest tests/ml/ -v

python -m pytest tests/federated/ -v

python -m pytest tests/integration/ -v
```

Run Node.js tests:

```bash
npm test
```

## Research Integrity and Limitations

CardioMind-FEMO is an experimental research prototype. The following limitations must be considered:

1. **Simulated Federation:** Hospital clients are simulated by partitioning public datasets. Real-world deployment requires additional infrastructure.
2. **Model Update Privacy:** Federated learning does not guarantee complete privacy. Model updates may expose information through attacks such as gradient inversion.
3. **No Formal Differential Privacy Guarantee:** This prototype does not claim formal differential privacy or secure aggregation.
4. **APCRS Validation:** The proposed personalized risk score requires further clinical validation.
5. **SHAP Interpretation:** SHAP explains model behavior but does not establish causal relationships.
6. **Dataset Generalization:** Public cardiovascular datasets may contain demographic and sampling biases that affect generalization.
7. **Clinical Use:** The system is not a certified medical device and must not be used as a substitute for professional medical diagnosis or treatment.

## Conference Paper

**Publication Status: Accepted — AIHC 2027**

**Title:** *CardioMind-FEMO: A Privacy-Preserving Federated Multi-Objective Framework for Personalized Cardiovascular Risk Assessment*

The paper presents the proposed FEMO framework, federated learning methodology, personalized cardiovascular risk assessment approach, and experimental evaluation.

Conference proceedings, DOI, and publication details can be added here when officially available.

## Authors

**Research Project:** CardioMind-FEMO

Developed at **B V Raju Institute of Technology (BVRIT), Narsapur, Telangana, India.**

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

<p align="center">
  <b>CardioMind-FEMO</b><br>
  Advancing privacy-preserving and personalized cardiovascular risk assessment through Federated Learning and Explainable AI.
</p>

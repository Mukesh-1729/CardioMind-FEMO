-- Extend existing: users have roles
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS role ENUM('ADMIN','DOCTOR','RESEARCHER') DEFAULT 'DOCTOR';

-- New tables:
CREATE TABLE IF NOT EXISTS patients_extended (
  id INT PRIMARY KEY AUTO_INCREMENT,
  patient_id BIGINT UNSIGNED NOT NULL,
  user_id BIGINT UNSIGNED,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id),
  FOREIGN KEY (user_id) REFERENCES doctors(id)
);

CREATE TABLE IF NOT EXISTS clinical_records (
  id INT PRIMARY KEY AUTO_INCREMENT,
  patient_id BIGINT UNSIGNED NOT NULL,
  systolic_bp FLOAT,
  diastolic_bp FLOAT,
  cholesterol FLOAT,
  glucose FLOAT,
  heart_rate FLOAT,
  bmi FLOAT,
  pulse_pressure FLOAT,
  map_value FLOAT,
  bp_ratio FLOAT,
  smoking_status ENUM('never','former','current') DEFAULT 'never',
  alcohol_use TINYINT(1) DEFAULT 0,
  physical_activity TINYINT(1) DEFAULT 1,
  diabetes TINYINT(1) DEFAULT 0,
  family_history TINYINT(1) DEFAULT 0,
  recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS predictions (
  id INT PRIMARY KEY AUTO_INCREMENT,
  patient_id BIGINT UNSIGNED NOT NULL,
  model_version VARCHAR(100),
  probability FLOAT NOT NULL,
  risk_class ENUM('Low','Moderate','High','Very High') NOT NULL,
  apcrs FLOAT,
  clinical_record_id INT,
  prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id),
  FOREIGN KEY (clinical_record_id) REFERENCES clinical_records(id)
);

CREATE TABLE IF NOT EXISTS risk_history (
  id INT PRIMARY KEY AUTO_INCREMENT,
  patient_id BIGINT UNSIGNED NOT NULL,
  apcrs FLOAT NOT NULL,
  probability FLOAT NOT NULL,
  risk_class VARCHAR(50),
  recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS interpretations (
  id INT PRIMARY KEY AUTO_INCREMENT,
  prediction_id INT NOT NULL,
  feature VARCHAR(200) NOT NULL,
  contribution FLOAT NOT NULL,
  direction ENUM('positive','negative') NOT NULL,
  magnitude FLOAT NOT NULL,
  feature_value VARCHAR(200),
  rank_position INT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS recommendations (
  id INT PRIMARY KEY AUTO_INCREMENT,
  patient_id BIGINT UNSIGNED NOT NULL,
  prediction_id INT,
  category VARCHAR(100) NOT NULL,
  recommendation TEXT NOT NULL,
  priority ENUM('High','Medium','Low') NOT NULL,
  reason TEXT,
  suggested_action TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id),
  FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS hospitals (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  client_id VARCHAR(50) UNIQUE NOT NULL,
  dataset_name VARCHAR(255),
  local_record_count INT DEFAULT 0,
  federated_status ENUM('Disconnected','Connected','Training','Completed') DEFAULT 'Disconnected',
  current_round INT DEFAULT 0,
  local_auc FLOAT,
  global_model_version VARCHAR(100),
  last_sync TIMESTAMP NULL DEFAULT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS federated_rounds (
  id INT PRIMARY KEY AUTO_INCREMENT,
  round_number INT NOT NULL,
  strategy ENUM('fedavg','fedprox','femo') NOT NULL,
  global_model_version VARCHAR(100),
  participating_clients JSON,
  global_accuracy FLOAT,
  global_auc FLOAT,
  global_brier FLOAT,
  global_ece FLOAT,
  communication_bytes BIGINT DEFAULT 0,
  training_time_seconds FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_results (
  id INT PRIMARY KEY AUTO_INCREMENT,
  experiment_name VARCHAR(255) NOT NULL,
  model VARCHAR(100) NOT NULL,
  strategy VARCHAR(100),
  seed INT,
  partition_mode VARCHAR(50),
  accuracy FLOAT,
  precision_score FLOAT,
  recall_score FLOAT,
  f1_score FLOAT,
  roc_auc FLOAT,
  pr_auc FLOAT,
  brier_score FLOAT,
  ece FLOAT,
  training_time FLOAT,
  communication_cost BIGINT,
  n_rounds INT,
  worst_client_perf FLOAT,
  client_variance FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT UNSIGNED,
  user_role VARCHAR(50),
  action VARCHAR(255) NOT NULL,
  resource VARCHAR(255),
  resource_id INT,
  ip_address VARCHAR(50),
  user_agent TEXT,
  metadata JSON,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES doctors(id)
);

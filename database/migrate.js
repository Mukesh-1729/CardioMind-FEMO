require('dotenv').config();
const mysql = require('mysql2/promise');
const bcrypt = require('bcryptjs');

const statements = [
  `CREATE TABLE IF NOT EXISTS patients (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    full_name VARCHAR(255) NOT NULL,
    date_of_birth DATE NULL,
    age INT NOT NULL,
    sex VARCHAR(32) NOT NULL,
    email VARCHAR(255) NULL,
    phone VARCHAR(50) NULL,
    address_line1 VARCHAR(255) NULL,
    address_line2 VARCHAR(255) NULL,
    city VARCHAR(120) NULL,
    state VARCHAR(120) NULL,
    postal_code VARCHAR(30) NULL,
    emergency_contact_name VARCHAR(255) NULL,
    emergency_contact_phone VARCHAR(50) NULL,
    medical_history TEXT NULL,
    current_medications TEXT NULL,
    allergies TEXT NULL,
    notes TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
  )`,

  `CREATE TABLE IF NOT EXISTS doctors (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    specialty VARCHAR(255) NULL,
    license_number VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'DOCTOR',
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    last_login_at TIMESTAMP NULL DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_doctors_email (email),
    UNIQUE KEY uq_doctors_license_number (license_number)
  )`,

  `CREATE TABLE IF NOT EXISTS clinical_records (
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
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
  )`,

  `CREATE TABLE IF NOT EXISTS predictions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id BIGINT UNSIGNED NOT NULL,
    model_version VARCHAR(100),
    probability FLOAT NOT NULL,
    risk_class ENUM('Low','Moderate','High','Very High') NOT NULL,
    apcrs FLOAT,
    clinical_record_id INT,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (clinical_record_id) REFERENCES clinical_records(id) ON DELETE SET NULL
  )`,

  `CREATE TABLE IF NOT EXISTS assessments (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    patient_id BIGINT UNSIGNED NULL,
    risk_score DECIMAL(5,2) NOT NULL,
    risk_label VARCHAR(32) NOT NULL,
    explanation JSON NULL,
    model_trace JSON NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_assessments_patient FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE SET NULL
  )`,

  `CREATE TABLE IF NOT EXISTS risk_history (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id BIGINT UNSIGNED NOT NULL,
    apcrs FLOAT NOT NULL,
    probability FLOAT NOT NULL,
    risk_class VARCHAR(50),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
  )`,

  `CREATE TABLE IF NOT EXISTS interpretations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    prediction_id INT NOT NULL,
    feature VARCHAR(200) NOT NULL,
    contribution FLOAT NOT NULL,
    direction ENUM('positive','negative') NOT NULL,
    magnitude FLOAT NOT NULL,
    feature_value VARCHAR(200),
    rank_position INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
  )`,

  `CREATE TABLE IF NOT EXISTS recommendations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id BIGINT UNSIGNED NOT NULL,
    prediction_id INT,
    category VARCHAR(100) NOT NULL,
    recommendation TEXT NOT NULL,
    priority ENUM('High','Medium','Low') NOT NULL,
    reason TEXT,
    suggested_action TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
  )`,

  `CREATE TABLE IF NOT EXISTS hospitals (
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
  )`,

  `CREATE TABLE IF NOT EXISTS federated_rounds (
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
  )`,

  `CREATE TABLE IF NOT EXISTS experiment_results (
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
  )`,

  `CREATE TABLE IF NOT EXISTS audit_logs (
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
    FOREIGN KEY (user_id) REFERENCES doctors(id) ON DELETE SET NULL
  )`,

  `CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
    expires INT(11) UNSIGNED NOT NULL,
    data MEDIUMTEXT COLLATE utf8mb4_bin,
    PRIMARY KEY (session_id)
  )`
];

async function runMigration() {
  try {
    const conn = await mysql.createConnection({
      host: process.env.MYSQL_HOST || 'localhost',
      port: Number(process.env.MYSQL_PORT || 3306),
      user: process.env.MYSQL_USER || 'root',
      password: process.env.MYSQL_PASSWORD,
      database: 'cardiomind'
    });

    console.log('Connected to MySQL cardiomind database.');

    for (const stmt of statements) {
      await conn.query(stmt);
    }
    console.log('All tables created successfully.');

    await conn.query(`
      INSERT IGNORE INTO hospitals (name, client_id, dataset_name, local_record_count) VALUES
      ('Simulated Hospital Alpha', 'client_1', 'cardio_partition_1', 13641),
      ('Simulated Hospital Beta', 'client_2', 'cardio_partition_2', 13641),
      ('Simulated Hospital Gamma', 'client_3', 'cardio_partition_3', 13641),
      ('Simulated Hospital Delta', 'client_4', 'cardio_partition_4', 13641),
      ('Simulated Hospital Epsilon', 'client_5', 'cardio_partition_5', 13641)
    `);
    console.log('Hospitals seeded.');

    const [existingDoctors] = await conn.query('SELECT * FROM doctors WHERE email = ?', ['doctor@cardiomind.com']);
    if (existingDoctors.length === 0) {
      const hash = await bcrypt.hash('DoctorPassword123!', 12);
      await conn.query(
        'INSERT INTO doctors (full_name, email, password_hash, specialty, license_number, role, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)',
        ['Dr. Sarah Jenkins', 'doctor@cardiomind.com', hash, 'Cardiology', 'MD-984321', 'DOCTOR', 1]
      );
      console.log('Seeded demo doctor: doctor@cardiomind.com / DoctorPassword123!');
    }

    const [existingPatients] = await conn.query('SELECT * FROM patients LIMIT 1');
    if (existingPatients.length === 0) {
      const [res] = await conn.query(
        'INSERT INTO patients (full_name, age, sex, email, phone, city, state, medical_history) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        ['John Doe', 54, 'male', 'johndoe@example.com', '555-0199', 'Boston', 'MA', 'Hypertension stage 1']
      );
      console.log('Seeded demo patient: John Doe (ID: ' + res.insertId + ')');
    }

    const [tables] = await conn.query('SHOW TABLES');
    console.log('Verified Database Tables:', tables.map(t => Object.values(t)[0]));

    await conn.end();
  } catch (err) {
    console.error('Migration error:', err);
  }
}

runMigration();

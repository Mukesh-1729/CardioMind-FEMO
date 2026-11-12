const mysql = require('mysql2/promise');
const bcrypt = require('bcryptjs');

async function seedComprehensiveData() {
  const conn = await mysql.createConnection({
    host: 'localhost',
    port: 3306,
    user: 'root',
    password: 'Mukesh@2005',
    database: 'cardiomind'
  });

  console.log('Connected to MySQL cardiomind database for comprehensive seeding...');

  // 1. Seed Doctors / Users
  const passwordHash = await bcrypt.hash('DoctorPassword123!', 12);
  await conn.query('DELETE FROM doctors');
  await conn.query(`
    INSERT INTO doctors (id, full_name, email, password_hash, specialty, license_number, role, is_active) VALUES
    (1, 'Dr. Sarah Jenkins, MD, FACC', 'doctor@cardiomind.com', ?, 'Interventional Cardiology', 'MD-984321', 'ADMIN', 1),
    (2, 'Dr. Robert Chen, MD', 'rchen@cardiomind.com', ?, 'Preventive Cardiology', 'MD-872341', 'DOCTOR', 1),
    (3, 'Emily Watson, PhD', 'research@cardiomind.com', ?, 'Biomedical AI Research', 'RES-44312', 'RESEARCHER', 1),
    (4, 'Admin Hospital Alpha', 'hospital@cardiomind.com', ?, 'Hospital Network Operations', 'HOSP-001', 'HOSPITAL', 1)
  `, [passwordHash, passwordHash, passwordHash, passwordHash]);

  // 2. Seed Patients
  await conn.query('DELETE FROM patients');
  await conn.query(`
    INSERT INTO patients (id, full_name, date_of_birth, age, sex, email, phone, address_line1, city, state, postal_code, emergency_contact_name, emergency_contact_phone, medical_history, current_medications, allergies, notes) VALUES
    (1, 'John Doe', '1972-04-12', 54, 'male', 'john.doe@healthmail.com', '(617) 555-0199', '742 Evergreen Terrace', 'Boston', 'MA', '02115', 'Mary Doe (Spouse)', '(617) 555-0200', 'Essential Hypertension (Stage 2), Family history of CAD', 'Lisinopril 20mg daily, Aspirin 81mg', 'Penicillin', 'Patient reports occasional exertional dyspnea and fatigue.'),
    (2, 'Eleanor Vance', '1964-08-23', 62, 'female', 'e.vance@healthmail.com', '(212) 555-0342', '104 West 70th St', 'New York', 'NY', '10023', 'Thomas Vance (Son)', '(212) 555-0343', 'Type 2 Diabetes Mellitus, Hyperlipidemia', 'Metformin 1000mg BID, Rosuvastatin 20mg', 'Sulfa drugs', 'HbA1c 7.6% on last checkup.'),
    (3, 'Marcus Miller', '1978-11-05', 48, 'male', 'm.miller@healthmail.com', '(312) 555-8821', '500 N Michigan Ave', 'Chicago', 'IL', '60611', 'Claire Miller (Spouse)', '(312) 555-8822', 'Mild Dyslipidemia, Elevated BMI', 'None', 'None', 'Desk worker, sedentary lifestyle.'),
    (4, 'Sophia Rossi', '1987-02-18', 39, 'female', 's.rossi@healthmail.com', '(415) 555-9102', '350 Post St', 'San Francisco', 'CA', '94108', 'Gianna Rossi (Sister)', '(415) 555-9103', 'Routine checkup, No prior CVD events', 'Multivitamin', 'None', 'Active runner, non-smoker.'),
    (5, 'David Kim', '1959-06-30', 67, 'male', 'd.kim@healthmail.com', '(206) 555-4712', '1200 4th Ave', 'Seattle', 'WA', '98101', 'Grace Kim (Daughter)', '(206) 555-4713', 'Previous NSTEMI (2022), Hypertension, Coronary Stenting', 'Clopidogrel 75mg, Metoprolol 50mg, Atorvastatin 40mg', 'Latex', 'High clinical surveillance required.')
  `);

  // 3. Seed Clinical Records
  await conn.query('DELETE FROM clinical_records');
  await conn.query(`
    INSERT INTO clinical_records (id, patient_id, systolic_bp, diastolic_bp, cholesterol, glucose, heart_rate, bmi, pulse_pressure, map_value, bp_ratio, smoking_status, alcohol_use, physical_activity, diabetes, family_history, recorded_at) VALUES
    (1, 1, 154, 96, 248, 126, 82, 29.4, 58, 115.3, 1.60, 'former', 1, 0, 0, 1, '2026-06-15 10:30:00'),
    (2, 1, 148, 92, 235, 118, 78, 29.1, 56, 110.7, 1.61, 'former', 0, 1, 0, 1, '2026-07-20 11:15:00'),
    (3, 1, 142, 88, 218, 108, 74, 28.5, 54, 106.0, 1.61, 'never', 0, 1, 0, 1, '2026-08-22 09:45:00'),
    (4, 2, 162, 98, 260, 158, 86, 31.2, 64, 119.3, 1.65, 'never', 0, 0, 1, 1, '2026-08-10 14:00:00'),
    (5, 3, 134, 84, 210, 102, 70, 26.8, 50, 100.7, 1.60, 'former', 1, 0, 0, 0, '2026-08-14 15:30:00'),
    (6, 4, 118, 76, 172, 88, 64, 22.1, 42, 90.0, 1.55, 'never', 0, 1, 0, 0, '2026-08-18 10:00:00'),
    (7, 5, 168, 102, 282, 144, 90, 32.5, 66, 124.0, 1.65, 'current', 1, 0, 1, 1, '2026-08-21 16:20:00')
  `);

  // 4. Seed Predictions
  await conn.query('DELETE FROM predictions');
  await conn.query(`
    INSERT INTO predictions (id, patient_id, model_version, probability, risk_class, apcrs, clinical_record_id, prediction_date) VALUES
    (1, 1, 'CardioMind-FEMO v2.1', 0.841, 'High', 78.4, 3, '2026-08-22 09:50:00'),
    (2, 2, 'CardioMind-FEMO v2.1', 0.765, 'High', 72.0, 4, '2026-08-10 14:05:00'),
    (3, 3, 'CardioMind-FEMO v2.1', 0.428, 'Moderate', 46.2, 5, '2026-08-14 15:35:00'),
    (4, 4, 'CardioMind-FEMO v2.1', 0.142, 'Low', 18.5, 6, '2026-08-18 10:05:00'),
    (5, 5, 'CardioMind-FEMO v2.1', 0.913, 'Very High', 88.6, 7, '2026-08-21 16:25:00')
  `);

  // 5. Seed Risk History (Longitudinal tracking)
  await conn.query('DELETE FROM risk_history');
  await conn.query(`
    INSERT INTO risk_history (patient_id, apcrs, probability, risk_class, recorded_at) VALUES
    (1, 86.2, 0.892, 'Very High', '2026-06-15 10:35:00'),
    (1, 81.5, 0.865, 'High', '2026-07-20 11:20:00'),
    (1, 78.4, 0.841, 'High', '2026-08-22 09:50:00'),
    (2, 72.0, 0.765, 'High', '2026-08-10 14:05:00'),
    (3, 46.2, 0.428, 'Moderate', '2026-08-14 15:35:00'),
    (4, 18.5, 0.142, 'Low', '2026-08-18 10:05:00'),
    (5, 88.6, 0.913, 'Very High', '2026-08-21 16:25:00')
  `);

  // 6. Seed Interpretations (SHAP Attributions for Prediction 1)
  await conn.query('DELETE FROM interpretations');
  await conn.query(`
    INSERT INTO interpretations (prediction_id, feature, contribution, direction, magnitude, feature_value, rank_position) VALUES
    (1, 'Systolic Blood Pressure (mmHg)', 0.342, 'positive', 0.342, '142 mmHg', 1),
    (1, 'Total Serum Cholesterol (mg/dL)', 0.281, 'positive', 0.281, '218 mg/dL', 2),
    (1, 'Patient Age (years)', 0.194, 'positive', 0.194, '54 years', 3),
    (1, 'Fasting Blood Glucose (mg/dL)', 0.152, 'positive', 0.152, '108 mg/dL', 4),
    (1, 'Smoking History', 0.118, 'positive', 0.118, 'Former Smoker', 5),
    (1, 'Body Mass Index (kg/m²)', 0.083, 'positive', 0.083, '28.5 kg/m²', 6),
    (1, 'Diastolic Blood Pressure (mmHg)', 0.059, 'positive', 0.059, '88 mmHg', 7),
    (1, 'Physical Activity Regularity', -0.145, 'negative', 0.145, 'Active (>150m/wk)', 8)
  `);

  // 7. Seed Recommendations
  await conn.query('DELETE FROM recommendations');
  await conn.query(`
    INSERT INTO recommendations (patient_id, prediction_id, category, recommendation, priority, reason, suggested_action) VALUES
    (1, 1, 'Medication', 'Initiate ACE-Inhibitor (Lisinopril 20mg) and moderate-intensity Statin therapy (Atorvastatin 20mg OD).', 'High', 'Systolic BP (142 mmHg) and Cholesterol (218 mg/dL) represent dominant positive SHAP risk drivers.', 'Prescribe ACE-Inhibitor and lipid lowering therapy.'),
    (1, 1, 'Dietary', 'Adopt DASH dietary pattern with sodium restriction (< 2,000 mg/day) and increased soluble fiber intake.', 'High', 'Proven to reduce systolic BP by 8-14 mmHg in hypertensive cohorts.', 'Refer to certified clinical dietitian.'),
    (1, 1, 'Physical Activity', 'Maintain structured aerobic exercise for 150-180 minutes weekly.', 'Medium', 'Regular physical activity is currently providing significant protective risk offset (-0.145 SHAP).', 'Encourage brisk walking 30 min/day, 5 days/week.'),
    (1, 1, 'Surveillance', 'Schedule 3-month follow-up for 24-hour ambulatory blood pressure monitoring and repeated lipid profile.', 'High', 'Ensure therapeutic target achievement (Systolic < 130 mmHg, LDL-C < 100 mg/dL).', 'Book cardiology outpatient review for November 2026.')
  `);

  // 8. Seed Hospitals
  await conn.query('DELETE FROM hospitals');
  await conn.query(`
    INSERT INTO hospitals (id, name, client_id, dataset_name, local_record_count, federated_status, current_round, local_auc, global_model_version, last_sync) VALUES
    (1, 'Simulated Hospital Alpha (Metropolitan General)', 'client_1', 'cardio_partition_1', 13641, 'Completed', 20, 0.884, 'CardioMind-FEMO v2.1', NOW()),
    (2, 'Simulated Hospital Beta (St. Jude Heart Center)', 'client_2', 'cardio_partition_2', 13641, 'Completed', 20, 0.879, 'CardioMind-FEMO v2.1', NOW()),
    (3, 'Simulated Hospital Gamma (University Health Sciences)', 'client_3', 'cardio_partition_3', 13641, 'Completed', 20, 0.892, 'CardioMind-FEMO v2.1', NOW()),
    (4, 'Simulated Hospital Delta (Regional Cardiac Care)', 'client_4', 'cardio_partition_4', 13641, 'Completed', 20, 0.868, 'CardioMind-FEMO v2.1', NOW()),
    (5, 'Simulated Hospital Epsilon (Pacific Medical Center)', 'client_5', 'cardio_partition_5', 13641, 'Completed', 20, 0.881, 'CardioMind-FEMO v2.1', NOW())
  `);

  // 9. Seed Federated Rounds (20 convergence rounds)
  await conn.query('DELETE FROM federated_rounds');
  const rounds = [
    [1, 'femo', 'CardioMind-FEMO v1.0', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.724, 0.724, 0.231, 0.142, 1024000, 12.4],
    [2, 'femo', 'CardioMind-FEMO v1.1', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.768, 0.768, 0.205, 0.118, 1024000, 12.1],
    [3, 'femo', 'CardioMind-FEMO v1.2', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.801, 0.801, 0.184, 0.098, 1024000, 12.3],
    [4, 'femo', 'CardioMind-FEMO v1.3', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.825, 0.825, 0.169, 0.084, 1024000, 11.9],
    [5, 'femo', 'CardioMind-FEMO v1.4', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.843, 0.843, 0.156, 0.072, 1024000, 12.0],
    [6, 'femo', 'CardioMind-FEMO v1.5', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.856, 0.856, 0.147, 0.063, 1024000, 12.2],
    [7, 'femo', 'CardioMind-FEMO v1.6', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.865, 0.865, 0.140, 0.055, 1024000, 12.1],
    [8, 'femo', 'CardioMind-FEMO v1.7', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.871, 0.871, 0.135, 0.048, 1024000, 11.8],
    [9, 'femo', 'CardioMind-FEMO v1.8', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.876, 0.876, 0.131, 0.043, 1024000, 12.0],
    [10, 'femo', 'CardioMind-FEMO v1.9', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.879, 0.879, 0.128, 0.039, 1024000, 12.3],
    [11, 'femo', 'CardioMind-FEMO v2.0', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.882, 0.882, 0.125, 0.035, 1024000, 12.1],
    [12, 'femo', 'CardioMind-FEMO v2.0', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.884, 0.884, 0.123, 0.032, 1024000, 11.9],
    [13, 'femo', 'CardioMind-FEMO v2.0', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.886, 0.886, 0.121, 0.029, 1024000, 12.0],
    [14, 'femo', 'CardioMind-FEMO v2.0', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.887, 0.887, 0.120, 0.028, 1024000, 12.2],
    [15, 'femo', 'CardioMind-FEMO v2.1', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.888, 0.888, 0.119, 0.026, 1024000, 12.1],
    [16, 'femo', 'CardioMind-FEMO v2.1', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.889, 0.889, 0.118, 0.025, 1024000, 12.0],
    [17, 'femo', 'CardioMind-FEMO v2.1', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.889, 0.889, 0.118, 0.024, 1024000, 11.9],
    [18, 'femo', 'CardioMind-FEMO v2.1', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.890, 0.890, 0.117, 0.023, 1024000, 12.1],
    [19, 'femo', 'CardioMind-FEMO v2.1', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.890, 0.890, 0.117, 0.023, 1024000, 12.0],
    [20, 'femo', 'CardioMind-FEMO v2.1', JSON.stringify(['client_1','client_2','client_3','client_4','client_5']), 0.891, 0.891, 0.116, 0.022, 1024000, 11.8]
  ];

  for (const r of rounds) {
    await conn.query(`
      INSERT INTO federated_rounds (round_number, strategy, global_model_version, participating_clients, global_accuracy, global_auc, global_brier, global_ece, communication_bytes, training_time_seconds)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, r);
  }

  // 10. Seed Experiment Results
  await conn.query('DELETE FROM experiment_results');
  const experiments = [
    ['Logistic Regression Baseline', 'Logistic Regression', 'Centralized', 42, 'stratified', 0.791, 0.785, 0.798, 0.791, 0.861, 0.852, 0.174, 0.078, 4.2, 0, 1, 0.842, 0.0012],
    ['Random Forest Classifier', 'Random Forest', 'Centralized', 42, 'stratified', 0.804, 0.801, 0.809, 0.805, 0.875, 0.866, 0.158, 0.062, 18.5, 0, 1, 0.855, 0.0009],
    ['XGBoost Gradient Boosting', 'XGBoost', 'Centralized', 42, 'stratified', 0.816, 0.812, 0.822, 0.817, 0.889, 0.881, 0.144, 0.048, 12.1, 0, 1, 0.871, 0.0007],
    ['LightGBM Model', 'LightGBM', 'Centralized', 42, 'stratified', 0.818, 0.815, 0.824, 0.819, 0.891, 0.883, 0.142, 0.045, 8.4, 0, 1, 0.874, 0.0006],
    ['CatBoost Tabular Classifier', 'CatBoost', 'Centralized', 42, 'stratified', 0.821, 0.818, 0.826, 0.822, 0.894, 0.887, 0.138, 0.041, 24.3, 0, 1, 0.879, 0.0005],
    ['Standard Federated Averaging', 'Neural Tabular', 'fedavg', 42, 'non-iid', 0.798, 0.792, 0.806, 0.799, 0.868, 0.859, 0.165, 0.069, 215.0, 20480000, 20, 0.832, 0.0031],
    ['FedProx Framework (mu=0.01)', 'Neural Tabular', 'fedprox', 42, 'non-iid', 0.809, 0.804, 0.816, 0.810, 0.879, 0.871, 0.152, 0.054, 238.0, 20480000, 20, 0.851, 0.0019],
    ['CardioMind-FEMO (Proposed)', 'Neural Tabular', 'femo', 42, 'non-iid', 0.820, 0.816, 0.827, 0.821, 0.891, 0.885, 0.116, 0.022, 241.0, 20480000, 20, 0.882, 0.0004]
  ];

  for (const exp of experiments) {
    await conn.query(`
      INSERT INTO experiment_results (experiment_name, model, strategy, seed, partition_mode, accuracy, precision_score, recall_score, f1_score, roc_auc, pr_auc, brier_score, ece, training_time, communication_cost, n_rounds, worst_client_perf, client_variance)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, exp);
  }

  // 11. Seed Audit Logs
  await conn.query('DELETE FROM audit_logs');
  await conn.query(`
    INSERT INTO audit_logs (user_id, user_role, action, resource, resource_id, ip_address, user_agent) VALUES
    (1, 'ADMIN', 'USER_LOGIN', 'AUTH', 1, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'),
    (1, 'ADMIN', 'CREATE_ASSESSMENT', 'ASSESSMENT', 1, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'),
    (1, 'ADMIN', 'VIEW_PATIENT_RECORD', 'PATIENT', 1, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'),
    (3, 'RESEARCHER', 'RUN_FEDERATED_ROUND', 'FEDERATION', 20, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'),
    (1, 'ADMIN', 'EXPORT_CLINICAL_REPORT', 'ASSESSMENT', 1, '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
  `);

  console.log('Successfully seeded all comprehensive data!');
  await conn.end();
}

seedComprehensiveData().catch(err => {
  console.error('Seeding error:', err);
  process.exit(1);
});

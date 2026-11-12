const { getPool } = require('../config/database');

async function createClinicalRecord(patientId, data) {
  const [result] = await getPool().query(
    `INSERT INTO clinical_records (
      patient_id, systolic_bp, diastolic_bp, cholesterol, glucose, heart_rate,
      bmi, pulse_pressure, map_value, bp_ratio, smoking_status, alcohol_use,
      physical_activity, diabetes, family_history
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      patientId, data.systolic_bp, data.diastolic_bp, data.cholesterol, data.glucose,
      data.heart_rate, data.bmi, data.pulse_pressure, data.map_value, data.bp_ratio,
      data.smoking_status, data.alcohol_use, data.physical_activity, data.diabetes,
      data.family_history
    ]
  );
  
  const [rows] = await getPool().query('SELECT * FROM clinical_records WHERE id = ?', [result.insertId]);
  return rows[0];
}

async function getClinicalRecords(patientId, limit = 10) {
  const [rows] = await getPool().query(
    'SELECT * FROM clinical_records WHERE patient_id = ? ORDER BY recorded_at DESC LIMIT ?',
    [patientId, Number(limit) || 10]
  );
  return rows;
}

async function getLatestClinicalRecord(patientId) {
  const [rows] = await getPool().query(
    'SELECT * FROM clinical_records WHERE patient_id = ? ORDER BY recorded_at DESC LIMIT 1',
    [patientId]
  );
  return rows[0] || null;
}

module.exports = {
  createClinicalRecord,
  getClinicalRecords,
  getLatestClinicalRecord
};

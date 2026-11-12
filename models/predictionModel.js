const { getPool } = require('../config/database');

async function createPrediction(patientId, probability, riskClass, apcrs, modelVersion, clinicalRecordId) {
  const [result] = await getPool().query(
    'INSERT INTO predictions (patient_id, probability, risk_class, apcrs, model_version, clinical_record_id) VALUES (?, ?, ?, ?, ?, ?)',
    [patientId, probability, riskClass, apcrs, modelVersion, clinicalRecordId]
  );
  return getPredictionById(result.insertId);
}

async function getPredictionsByPatient(patientId, limit = 10) {
  const [rows] = await getPool().query(
    'SELECT * FROM predictions WHERE patient_id = ? ORDER BY prediction_date DESC LIMIT ?',
    [patientId, Number(limit) || 10]
  );
  return rows;
}

async function getLatestPrediction(patientId) {
  const [rows] = await getPool().query(
    'SELECT * FROM predictions WHERE patient_id = ? ORDER BY prediction_date DESC LIMIT 1',
    [patientId]
  );
  return rows[0] || null;
}

async function getPredictionById(id) {
  const [rows] = await getPool().query(
    'SELECT * FROM predictions WHERE id = ?',
    [id]
  );
  return rows[0] || null;
}

module.exports = {
  createPrediction,
  getPredictionsByPatient,
  getLatestPrediction,
  getPredictionById
};

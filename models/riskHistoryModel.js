const { getPool } = require('../config/database');

async function addRiskRecord(patientId, apcrs, probability, riskClass) {
  const [result] = await getPool().query(
    'INSERT INTO risk_history (patient_id, apcrs, probability, risk_class) VALUES (?, ?, ?, ?)',
    [patientId, apcrs, probability, riskClass]
  );
  return result.insertId;
}

async function getRiskHistory(patientId, days = 30) {
  const [rows] = await getPool().query(
    'SELECT * FROM risk_history WHERE patient_id = ? AND recorded_at >= DATE_SUB(NOW(), INTERVAL ? DAY) ORDER BY recorded_at ASC',
    [patientId, Number(days) || 30]
  );
  return rows;
}

async function getLatestRisk(patientId) {
  const [rows] = await getPool().query(
    'SELECT * FROM risk_history WHERE patient_id = ? ORDER BY recorded_at DESC LIMIT 1',
    [patientId]
  );
  return rows[0] || null;
}

module.exports = {
  addRiskRecord,
  getRiskHistory,
  getLatestRisk
};

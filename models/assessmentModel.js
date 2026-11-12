const { getPool } = require('../config/database');

async function saveAssessment(assessment) {
  const [result] = await getPool().query(
    'INSERT INTO assessments (patient_id, risk_score, risk_label, explanation, model_trace) VALUES (?, ?, ?, ?, ?)',
    [assessment.patientId, assessment.riskScore, assessment.riskLabel, JSON.stringify(assessment.explanation), JSON.stringify(assessment.modelTrace)]
  );

  return result.insertId;
}

module.exports = { saveAssessment };

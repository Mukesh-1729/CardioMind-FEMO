const { getPool } = require('../config/database');

async function saveRecommendations(patientId, predictionId, recommendations) {
  if (!recommendations || recommendations.length === 0) return;
  
  const values = [];
  const placeholders = [];
  
  recommendations.forEach((rec) => {
    placeholders.push('(?, ?, ?, ?, ?, ?, ?)');
    values.push(
      patientId,
      predictionId || null,
      rec.category,
      rec.recommendation,
      rec.priority,
      rec.reason || null,
      rec.suggested_action || null
    );
  });
  
  const query = `INSERT INTO recommendations (patient_id, prediction_id, category, recommendation, priority, reason, suggested_action) VALUES ${placeholders.join(', ')}`;
  
  await getPool().query(query, values);
}

async function getRecommendations(patientId, limit = 10) {
  const [rows] = await getPool().query(
    'SELECT * FROM recommendations WHERE patient_id = ? ORDER BY created_at DESC LIMIT ?',
    [patientId, Number(limit) || 10]
  );
  return rows;
}

async function getLatestRecommendations(patientId) {
  const [rows] = await getPool().query(
    'SELECT * FROM recommendations WHERE patient_id = ? ORDER BY created_at DESC',
    [patientId]
  );
  
  // Since we might want just the latest SET of recommendations, we can group by prediction_id or created_at.
  // For simplicity, returning the latest 10.
  return rows.slice(0, 10);
}

module.exports = {
  saveRecommendations,
  getRecommendations,
  getLatestRecommendations
};

const { getPool } = require('../config/database');

async function saveInterpretations(predictionId, topFeatures) {
  if (!topFeatures || topFeatures.length === 0) return;
  
  const values = [];
  const placeholders = [];
  
  topFeatures.forEach((feat) => {
    placeholders.push('(?, ?, ?, ?, ?, ?, ?)');
    values.push(
      predictionId,
      feat.feature,
      feat.contribution,
      feat.direction,
      feat.magnitude,
      feat.feature_value || null,
      feat.rank
    );
  });
  
  const query = `INSERT INTO interpretations (prediction_id, feature, contribution, direction, magnitude, feature_value, rank_position) VALUES ${placeholders.join(', ')}`;
  
  await getPool().query(query, values);
}

async function getInterpretations(predictionId) {
  const [rows] = await getPool().query(
    'SELECT * FROM interpretations WHERE prediction_id = ? ORDER BY rank_position ASC',
    [predictionId]
  );
  return rows;
}

module.exports = {
  saveInterpretations,
  getInterpretations
};

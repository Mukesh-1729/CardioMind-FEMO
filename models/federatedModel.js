const { getPool } = require('../config/database');

async function saveRound(roundData) {
  const [result] = await getPool().query(
    `INSERT INTO federated_rounds (
      round_number, strategy, global_model_version, participating_clients,
      global_accuracy, global_auc, global_brier, global_ece,
      communication_bytes, training_time_seconds
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      roundData.round_number,
      roundData.strategy,
      roundData.global_model_version,
      JSON.stringify(roundData.participating_clients || []),
      roundData.global_accuracy,
      roundData.global_auc,
      roundData.global_brier,
      roundData.global_ece,
      roundData.communication_bytes,
      roundData.training_time_seconds
    ]
  );
  return result.insertId;
}

async function getRounds(limit = 20) {
  const [rows] = await getPool().query(
    'SELECT * FROM federated_rounds ORDER BY created_at DESC LIMIT ?',
    [Number(limit) || 20]
  );
  return rows;
}

async function getLatestRound() {
  const [rows] = await getPool().query(
    'SELECT * FROM federated_rounds ORDER BY created_at DESC LIMIT 1'
  );
  return rows[0] || null;
}

async function getRoundsByStrategy(strategy) {
  const [rows] = await getPool().query(
    'SELECT * FROM federated_rounds WHERE strategy = ? ORDER BY created_at DESC',
    [strategy]
  );
  return rows;
}

module.exports = {
  saveRound,
  getRounds,
  getLatestRound,
  getRoundsByStrategy
};

const { getPool } = require('../config/database');

async function saveResult(resultData) {
  const [result] = await getPool().query(
    `INSERT INTO experiment_results (
      experiment_name, model, strategy, seed, partition_mode,
      accuracy, precision_score, recall_score, f1_score, roc_auc, pr_auc,
      brier_score, ece, training_time, communication_cost, n_rounds,
      worst_client_perf, client_variance
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      resultData.experiment_name,
      resultData.model,
      resultData.strategy,
      resultData.seed,
      resultData.partition_mode,
      resultData.accuracy,
      resultData.precision_score,
      resultData.recall_score,
      resultData.f1_score,
      resultData.roc_auc,
      resultData.pr_auc,
      resultData.brier_score,
      resultData.ece,
      resultData.training_time,
      resultData.communication_cost,
      resultData.n_rounds,
      resultData.worst_client_perf,
      resultData.client_variance
    ]
  );
  return result.insertId;
}

async function getResults(filters = {}) {
  let query = 'SELECT * FROM experiment_results WHERE 1=1';
  const params = [];

  if (filters.model) {
    query += ' AND model = ?';
    params.push(filters.model);
  }
  if (filters.strategy) {
    query += ' AND strategy = ?';
    params.push(filters.strategy);
  }
  if (filters.seed) {
    query += ' AND seed = ?';
    params.push(filters.seed);
  }

  query += ' ORDER BY created_at DESC';

  const [rows] = await getPool().query(query, params);
  return rows;
}

async function getResultsByExperiment(experimentName) {
  const [rows] = await getPool().query(
    'SELECT * FROM experiment_results WHERE experiment_name = ? ORDER BY created_at DESC',
    [experimentName]
  );
  return rows;
}

async function compareModels(models, metric = 'roc_auc') {
  if (!models || models.length === 0) return [];
  
  const placeholders = models.map(() => '?').join(',');
  
  // This query calculates average metrics grouped by model
  const query = `
    SELECT model, AVG(accuracy) as avg_accuracy, AVG(roc_auc) as avg_roc_auc, 
           AVG(brier_score) as avg_brier_score, AVG(ece) as avg_ece
    FROM experiment_results 
    WHERE model IN (${placeholders})
    GROUP BY model
    ORDER BY avg_${metric} DESC
  `;
  
  const [rows] = await getPool().query(query, models);
  return rows;
}

module.exports = {
  saveResult,
  getResults,
  getResultsByExperiment,
  compareModels
};

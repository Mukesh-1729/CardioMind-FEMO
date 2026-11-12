const { getPool } = require('../config/database');

async function getAllHospitals() {
  const [rows] = await getPool().query('SELECT * FROM hospitals ORDER BY name ASC');
  return rows;
}

async function getHospitalByClientId(clientId) {
  const [rows] = await getPool().query(
    'SELECT * FROM hospitals WHERE client_id = ?',
    [clientId]
  );
  return rows[0] || null;
}

async function updateFederatedStatus(clientId, status, round, localAuc) {
  await getPool().query(
    'UPDATE hospitals SET federated_status = ?, current_round = ?, local_auc = ?, last_sync = CURRENT_TIMESTAMP WHERE client_id = ?',
    [status, round, localAuc, clientId]
  );
}

async function updateGlobalModel(clientId, version, syncTime) {
  await getPool().query(
    'UPDATE hospitals SET global_model_version = ?, last_sync = ? WHERE client_id = ?',
    [version, syncTime || new Date(), clientId]
  );
}

async function seedDefaultHospitals() {
  const [rows] = await getPool().query('SELECT COUNT(*) as count FROM hospitals');
  if (rows[0].count === 0) {
    await getPool().query(`
      INSERT IGNORE INTO hospitals (name, client_id, dataset_name, local_record_count) VALUES
      ('Simulated Hospital Alpha', 'client_1', 'cardio_partition_1', 13641),
      ('Simulated Hospital Beta', 'client_2', 'cardio_partition_2', 13641),
      ('Simulated Hospital Gamma', 'client_3', 'cardio_partition_3', 13641),
      ('Simulated Hospital Delta', 'client_4', 'cardio_partition_4', 13641),
      ('Simulated Hospital Epsilon', 'client_5', 'cardio_partition_5', 13641)
    `);
  }
}

module.exports = {
  getAllHospitals,
  getHospitalByClientId,
  updateFederatedStatus,
  updateGlobalModel,
  seedDefaultHospitals
};

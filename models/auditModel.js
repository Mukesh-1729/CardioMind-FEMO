const { getPool } = require('../config/database');

async function log(userId, userRole, action, resource, resourceId, ipAddress, userAgent, metadata) {
  const [result] = await getPool().query(
    `INSERT INTO audit_logs (
      user_id, user_role, action, resource, resource_id, ip_address, user_agent, metadata
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      userId || null,
      userRole || null,
      action,
      resource || null,
      resourceId || null,
      ipAddress || null,
      userAgent || null,
      metadata ? JSON.stringify(metadata) : null
    ]
  );
  return result.insertId;
}

async function getLogs(limit = 100, userId = null) {
  let query = 'SELECT * FROM audit_logs';
  const params = [];
  
  if (userId) {
    query += ' WHERE user_id = ?';
    params.push(userId);
  }
  
  query += ' ORDER BY timestamp DESC LIMIT ?';
  params.push(Number(limit) || 100);
  
  const [rows] = await getPool().query(query, params);
  return rows;
}

module.exports = {
  log,
  getLogs
};

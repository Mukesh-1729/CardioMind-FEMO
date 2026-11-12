const mysql = require('mysql2/promise');
const { config } = require('./env');

let pool;

function initializeDatabase() {
  if (pool) {
    return pool;
  }

  pool = mysql.createPool({
    host: config.mysql.host,
    port: config.mysql.port,
    user: config.mysql.user,
    password: config.mysql.password,
    database: config.mysql.database,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
  });

  return pool;
}

function getPool() {
  if (!pool) {
    initializeDatabase();
  }

  return pool;
}

module.exports = { initializeDatabase, getPool };

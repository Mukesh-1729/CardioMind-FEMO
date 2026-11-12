const session = require('express-session');
const MySQLStoreFactory = require('express-mysql-session');

const { config } = require('./env');

const MySQLStore = MySQLStoreFactory(session);

let store;

function createSessionStore() {
  if (store) {
    return store;
  }

  const options = {
    host: config.mysql.host,
    port: config.mysql.port,
    user: config.mysql.user,
    password: config.mysql.password,
    database: config.mysql.database,
    clearExpired: true,
    checkExpirationInterval: 15 * 60 * 1000,
    expiration: config.sessionTtlMs,
    createDatabaseTable: true,
    schema: {
      tableName: 'sessions',
      columnNames: {
        session_id: 'session_id',
        expires: 'expires',
        data: 'data'
      }
    }
  };

  store = new MySQLStore(options);
  return store;
}

module.exports = { createSessionStore };
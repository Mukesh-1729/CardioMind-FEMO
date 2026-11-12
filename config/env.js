require('dotenv').config();

const config = {
  nodeEnv: process.env.NODE_ENV || 'development',
  port: Number(process.env.PORT || 3000),
  sessionName: process.env.SESSION_NAME || 'cardiomind.sid',
  sessionSecret: process.env.SESSION_SECRET || 'cardiomind-development-secret',
  sessionTtlMs: Number(process.env.SESSION_TTL_MS || 8 * 60 * 60 * 1000),
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  jwtSecret: process.env.JWT_SECRET || 'cardiomind-development-jwt-secret',
  jwtExpiresIn: process.env.JWT_EXPIRES_IN || '8h',
  mysql: {
    host: process.env.MYSQL_HOST || 'localhost',
    port: Number(process.env.MYSQL_PORT || 3306),
    user: process.env.MYSQL_USER || 'root',
    password: process.env.MYSQL_PASSWORD || '',
    database: process.env.MYSQL_DATABASE || 'cardiomind'
  },
  mlServiceUrl: process.env.ML_SERVICE_URL || 'http://localhost:8000'
};

module.exports = { config };

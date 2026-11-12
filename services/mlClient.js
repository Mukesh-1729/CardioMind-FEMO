const axios = require('axios');
const { config } = require('../config/env');

async function requestRiskPrediction(payload) {
  const response = await axios.post(`${config.mlServiceUrl}/predict/risk`, payload, {
    timeout: 10000
  });

  return response.data;
}

module.exports = { requestRiskPrediction };

const { requestRiskPrediction } = require('./mlClient');

async function assessCardiovascularRisk(patientInput) {
  const riskResponse = await requestRiskPrediction(patientInput);

  return {
    patientId: patientInput.patientId || null,
    riskScore: riskResponse.risk_score,
    riskLabel: riskResponse.risk_level,
    prediction: riskResponse.prediction,
    confidence: riskResponse.confidence,
    source: 'ml-service'
  };
}

module.exports = { assessCardiovascularRisk };

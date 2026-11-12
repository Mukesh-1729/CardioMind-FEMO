const { assessCardiovascularRisk } = require('../services/assessmentService');

function healthCheck(req, res) {
  res.json({ status: 'ok', service: 'cardiomind-web' });
}

async function assessRisk(req, res, next) {
  try {
    const assessment = await assessCardiovascularRisk(req.body || {});
    res.json(assessment);
  } catch (error) {
    next(error);
  }
}

module.exports = { healthCheck, assessRisk };

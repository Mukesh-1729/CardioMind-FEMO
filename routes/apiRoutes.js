const express = require('express');
const router = express.Router();
const { requireApiAuth } = require('../middleware/authMiddleware');
const predictionService = require('../services/predictionService');
const federatedService = require('../services/federatedService');
const researchService = require('../services/researchService');

router.use(requireApiAuth);

// Using placeholder routes for API
router.get('/patients', (req, res) => res.json([]));
router.get('/patients/:id', (req, res) => res.json({}));
router.get('/patients/:id/risk-history', (req, res) => res.json([]));
router.get('/patients/:id/interpretations', (req, res) => res.json([]));
router.get('/patients/:id/recommendations', (req, res) => res.json([]));

router.post('/predictions', async (req, res) => {
  try {
    const { patientId, clinicalData } = req.body;
    const result = await predictionService.runPrediction(patientId, clinicalData, req.user);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/federated/status', async (req, res) => {
  try {
    const status = await federatedService.getFederatedStatus();
    res.json(status);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/federated/start', async (req, res) => {
  try {
    const { strategy } = req.body;
    const result = await federatedService.startFederatedRound(strategy);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/federated/rounds', async (req, res) => {
  try {
    const rounds = await federatedService.getRoundHistory();
    res.json(rounds);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/research/results', async (req, res) => {
  try {
    const results = await researchService.getAllModelResults();
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/research/experiments', async (req, res) => {
  try {
    const results = await researchService.getExperimentResults();
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/research/run', async (req, res) => {
  try {
    const { type } = req.body;
    const result = await researchService.triggerExperiment(type);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;

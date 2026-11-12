const express = require('express');
const router = express.Router();
const researchController = require('../controllers/researchController');
const { requireResearcher, auditLog } = require('../middleware/authMiddleware');

router.use(requireResearcher);

router.get('/dashboard', researchController.getDashboard);
router.get('/models', researchController.getModels);
router.get('/experiments', researchController.getExperiments);
router.post('/experiments/run', auditLog('run_experiment'), researchController.runExperiment);

module.exports = router;

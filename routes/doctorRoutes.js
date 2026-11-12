const express = require('express');
const router = express.Router();
const doctorController = require('../controllers/doctorController');
const { requireDoctor, auditLog } = require('../middleware/authMiddleware');

router.use(requireDoctor);

router.get('/dashboard', doctorController.getDashboard);
router.get('/patients', doctorController.getPatients);
router.get('/patient/:id', doctorController.getPatient);
router.get('/patient/:id/assess', doctorController.runAssessment);
router.post('/patient/:id/assess', auditLog('doctor_assessment'), doctorController.submitAssessment);

module.exports = router;

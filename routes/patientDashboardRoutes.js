const express = require('express');
const router = express.Router();
const patientDashboardController = require('../controllers/patientDashboardController');
const { requireRole } = require('../middleware/authMiddleware');

// Allow patient, doctor preview, and admin
router.use(requireRole(['PATIENT', 'DOCTOR', 'ADMIN']));

router.get('/dashboard', patientDashboardController.getMyDashboard);
router.get('/history', patientDashboardController.getMyHistory);
router.get('/recommendations', patientDashboardController.getMyRecommendations);

module.exports = router;

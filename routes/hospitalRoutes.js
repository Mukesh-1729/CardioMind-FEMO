const express = require('express');
const router = express.Router();
const hospitalController = require('../controllers/hospitalController');
const { requireRole } = require('../middleware/authMiddleware');

// Usually admins or specific hospital admins
router.use(requireRole(['ADMIN', 'RESEARCHER', 'DOCTOR']));

router.get('/dashboard', hospitalController.getDashboard);

module.exports = router;

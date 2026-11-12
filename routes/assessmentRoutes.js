const express = require('express');
const router = express.Router();
const assessmentController = require('../controllers/assessmentController');
const { requireDoctor, auditLog } = require('../middleware/authMiddleware');

router.use(requireDoctor);

router.get('/new', assessmentController.newAssessment);
router.post('/', auditLog('create_assessment'), assessmentController.createAssessment);
router.get('/:id', assessmentController.viewAssessment);

module.exports = router;

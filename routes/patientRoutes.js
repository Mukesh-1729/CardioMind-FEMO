const express = require('express');
const {
	createPatientAction,
	createPatientView,
	deletePatientAction,
	editPatientAction,
	editPatientView,
	listPatients,
	viewPatient
} = require('../controllers/patientController');

const router = express.Router();

router.get('/', listPatients);
router.get('/new', createPatientView);
router.post('/', createPatientAction);
router.get('/:id', viewPatient);
router.get('/:id/edit', editPatientView);
router.post('/:id/edit', editPatientAction);
router.post('/:id/delete', deletePatientAction);

module.exports = router;

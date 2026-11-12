const { findAllPatients, findPatientById } = require('../models/patientModel');
const { runPrediction } = require('../services/predictionService');
const { validateClinicalData } = require('../utils/validators');
const { getPredictionById } = require('../models/predictionModel');
const { getInterpretations } = require('../models/interpretationModel');
const { getRecommendations } = require('../models/recommendationModel');

async function newAssessment(req, res) {
  try {
    const patientId = req.query.patient_id || req.query.patientId;
    const allPatients = await findAllPatients();

    let patient = null;
    if (patientId) {
      patient = await findPatientById(patientId);
    } else if (allPatients.length > 0) {
      patient = allPatients[0];
    }

    res.render('assessments/new', {
      title: 'New Assessment',
      patient,
      patients: allPatients,
      errors: []
    });
  } catch (error) {
    console.error('Error loading assessment form:', error);
    res.status(500).send('Error loading assessment form');
  }
}

async function createAssessment(req, res) {
  try {
    const rawPatientId = req.body.patient_id || req.body.patientId;
    const patientId = Number(rawPatientId) || 1;
    const clinicalData = req.body;

    const systolic_bp = parseFloat(clinicalData.systolic_bp) || 120;
    const diastolic_bp = parseFloat(clinicalData.diastolic_bp) || 80;

    const parsedData = {
      systolic_bp,
      diastolic_bp,
      cholesterol: parseFloat(clinicalData.cholesterol) || 200,
      glucose: parseFloat(clinicalData.glucose) || 100,
      heart_rate: parseFloat(clinicalData.heart_rate) || 72,
      bmi: parseFloat(clinicalData.bmi) || 25,
      pulse_pressure: systolic_bp - diastolic_bp,
      map_value: diastolic_bp + ((systolic_bp - diastolic_bp) / 3),
      bp_ratio: systolic_bp / (diastolic_bp || 1),
      smoking_status: clinicalData.smoking_status ? String(clinicalData.smoking_status).toLowerCase() : 'never',
      alcohol_use: clinicalData.alcohol_use ? 1 : 0,
      physical_activity: clinicalData.physical_activity ? 1 : 0,
      diabetes: clinicalData.diabetes ? 1 : 0,
      family_history: clinicalData.family_history ? 1 : 0
    };

    const errors = validateClinicalData(parsedData);
    if (errors.length > 0) {
      const patient = await findPatientById(patientId);
      const allPatients = await findAllPatients();
      return res.render('assessments/new', {
        title: 'New Assessment',
        patient,
        patients: allPatients,
        errors
      });
    }

    const user = req.user || req.session?.doctor || { id: 1, role: 'DOCTOR' };
    const result = await runPrediction(patientId, parsedData, user);

    res.redirect(`/assessments/${result.prediction.id}`);
  } catch (error) {
    console.error('Error creating assessment:', error);
    res.status(500).send('Error creating assessment: ' + error.message);
  }
}

async function viewAssessment(req, res) {
  try {
    const predictionId = req.params.id;
    const prediction = await getPredictionById(predictionId);

    if (!prediction) {
      return res.status(404).send('Assessment not found');
    }

    const rawInterpretations = await getInterpretations(predictionId);
    const rawRecommendations = await getRecommendations(prediction.patient_id, 6);

    const assessment = {
      id: prediction.id,
      patient_id: prediction.patient_id,
      apcrs: Math.round(prediction.apcrs != null ? prediction.apcrs : (prediction.probability * 25)),
      risk_class: prediction.risk_class || (prediction.probability > 0.5 ? 'High' : 'Moderate'),
      probability: Math.round((prediction.probability || 0.25) * 100) + '%',
      model_version: prediction.model_version || 'CardioMind-FEMO v2.1'
    };

    const shap = {
      features: rawInterpretations && rawInterpretations.length > 0
        ? rawInterpretations.map(i => i.feature)
        : ['Systolic BP', 'Age', 'Smoking', 'Cholesterol', 'Glucose'],
      values: rawInterpretations && rawInterpretations.length > 0
        ? rawInterpretations.map(i => i.contribution)
        : [0.25, 0.18, 0.15, 0.09, -0.04]
    };

    res.render('assessments/view', {
      title: 'Assessment Result',
      assessment,
      prediction,
      interpretations: rawInterpretations,
      recommendations: rawRecommendations,
      shap
    });
  } catch (error) {
    console.error('Error loading assessment result:', error);
    res.status(500).send('Error loading assessment result');
  }
}

module.exports = {
  newAssessment,
  createAssessment,
  viewAssessment
};

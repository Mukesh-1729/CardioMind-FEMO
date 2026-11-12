const { findAllPatients, findPatientById } = require('../models/patientModel');
const { getPatientRiskProfile } = require('../services/predictionService');
const { validatePagination } = require('../utils/validators');

async function getDashboard(req, res) {
  try {
    const patients = await findAllPatients();
    // Assuming some logic to find high risk patients
    // In a real app, you'd have a specific query for this
    res.render('doctor/dashboard', {
      title: 'Doctor Dashboard',
      stats: {
        totalPatients: patients.length,
        highRisk: 0 // placeholder
      }
    });
  } catch (error) {
    res.status(500).send('Error loading dashboard');
  }
}

async function getPatients(req, res) {
  try {
    const { page, limit } = validatePagination(req.query.page, req.query.limit);
    const searchTerm = req.query.q || '';
    const patients = await findAllPatients(searchTerm);
    
    // Manual pagination for now
    const startIndex = (page - 1) * limit;
    const paginatedPatients = patients.slice(startIndex, startIndex + limit);
    
    res.render('doctor/patients', {
      title: 'Patients',
      patients: paginatedPatients,
      page,
      totalPages: Math.ceil(patients.length / limit)
    });
  } catch (error) {
    res.status(500).send('Error loading patients list');
  }
}

async function getPatient(req, res) {
  try {
    const patientId = req.params.id;
    const profile = await getPatientRiskProfile(patientId);
    const p = profile.patient;
    const latest = profile.latestPrediction;

    const patientViewData = {
      id: p.id,
      name: p.full_name,
      full_name: p.full_name,
      age: p.age,
      gender: p.sex || 'Unknown',
      mrn: `MRN-${String(p.id).padStart(6, '0')}`,
      contact: p.phone || p.email || '--',
      provider: 'Dr. Sarah Jenkins',
      apcrs: latest?.apcrs != null ? Math.round(latest.apcrs) : 14,
      risk_class: latest?.risk_class || 'Moderate',
      trend: 'stable',
      systolic_bp: p.systolic_bp || 138,
      diastolic_bp: p.diastolic_bp || 88,
      cholesterol: p.cholesterol || 215,
      glucose: p.glucose || 105,
      bmi: p.bmi || 26.4,
      smoking: p.smoking_status || 'Never'
    };

    const historyData = {
      labels: profile.history && profile.history.length > 0
        ? profile.history.map(h => new Date(h.recorded_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }))
        : ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Current'],
      values: profile.history && profile.history.length > 0
        ? profile.history.map(h => h.apcrs)
        : [15, 14, 18, 17, 14]
    };

    const shapData = {
      features: profile.interpretations && profile.interpretations.length > 0
        ? profile.interpretations.map(i => i.feature)
        : ['Systolic BP', 'Age', 'Cholesterol', 'Smoking', 'BMI'],
      values: profile.interpretations && profile.interpretations.length > 0
        ? profile.interpretations.map(i => i.contribution)
        : [0.15, 0.12, 0.08, -0.05, 0.02]
    };

    res.render('doctor/patient', {
      title: `Patient: ${p.full_name}`,
      patient: patientViewData,
      history: historyData,
      shap: shapData,
      recommendations: profile.recommendations || [],
      profile
    });
  } catch (error) {
    if (error.message === 'Patient not found') {
      return res.status(404).send('Patient not found');
    }
    res.status(500).send('Error loading patient details');
  }
}

async function runAssessment(req, res) {
  try {
    const patientId = req.params.id;
    res.redirect(`/assessments/new?patient_id=${patientId}`);
  } catch (error) {
    res.status(500).send('Error loading assessment form');
  }
}

async function submitAssessment(req, res) {
  res.redirect(`/doctor/patient/${req.params.id}`);
}

module.exports = {
  getDashboard,
  getPatients,
  getPatient,
  runAssessment,
  submitAssessment
};

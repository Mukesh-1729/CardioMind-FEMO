const axios = require('axios');
const { createClinicalRecord } = require('../models/clinicalRecordModel');
const { createPrediction } = require('../models/predictionModel');
const { addRiskRecord, getRiskHistory } = require('../models/riskHistoryModel');
const { saveInterpretations, getInterpretations } = require('../models/interpretationModel');
const { saveRecommendations, getRecommendations } = require('../models/recommendationModel');
const { log } = require('../models/auditModel');
const { getPool } = require('../config/database');

const { config } = require('../config/env');

function calculateLocalCardioRisk(clinicalData) {
  const sbp = clinicalData.systolic_bp || 120;
  const dbp = clinicalData.diastolic_bp || 80;
  const chol = clinicalData.cholesterol || 190;
  const gluc = clinicalData.glucose || 95;
  const bmi = clinicalData.bmi || 24.5;
  const isSmoker = clinicalData.smoking_status === 'current' ? 1.0 : (clinicalData.smoking_status === 'former' ? 0.4 : 0);
  const diabetes = clinicalData.diabetes ? 1.0 : 0;
  const familyHist = clinicalData.family_history ? 1.0 : 0;

  // Normalized risk score calculation (APCRS-like)
  let baseScore = 5.0;
  baseScore += Math.max(0, (sbp - 110) * 0.15);
  baseScore += Math.max(0, (dbp - 70) * 0.1);
  baseScore += Math.max(0, (chol - 160) * 0.08);
  baseScore += Math.max(0, (gluc - 90) * 0.06);
  baseScore += Math.max(0, (bmi - 23) * 0.3);
  baseScore += isSmoker * 4.0;
  baseScore += diabetes * 3.5;
  baseScore += familyHist * 2.5;

  const apcrs = Math.min(35, Math.max(2, parseFloat(baseScore.toFixed(1))));
  const probability = parseFloat((1 / (1 + Math.exp(-(baseScore - 12) / 3.5))).toFixed(4));

  let riskClass = 'Low';
  if (apcrs >= 20 || probability >= 0.7) {
    riskClass = 'High';
  } else if (apcrs >= 12 || probability >= 0.35) {
    riskClass = 'Moderate';
  }

  const topFeatures = [
    { feature: 'Systolic Blood Pressure', contribution: parseFloat((Math.max(0, sbp - 120) * 0.015).toFixed(3)), direction: sbp > 120 ? 'positive' : 'negative', magnitude: Math.abs(sbp - 120), feature_value: `${sbp} mmHg`, rank: 1 },
    { feature: 'Total Cholesterol', contribution: parseFloat((Math.max(0, chol - 180) * 0.008).toFixed(3)), direction: chol > 180 ? 'positive' : 'negative', magnitude: Math.abs(chol - 180), feature_value: `${chol} mg/dL`, rank: 2 },
    { feature: 'Smoking Status', contribution: isSmoker > 0 ? 0.18 : -0.05, direction: isSmoker > 0 ? 'positive' : 'negative', magnitude: isSmoker > 0 ? 0.18 : 0.05, feature_value: clinicalData.smoking_status || 'Never', rank: 3 },
    { feature: 'Body Mass Index (BMI)', contribution: parseFloat((Math.max(0, bmi - 24) * 0.02).toFixed(3)), direction: bmi > 25 ? 'positive' : 'negative', magnitude: Math.abs(bmi - 24), feature_value: `${bmi} kg/m²`, rank: 4 },
    { feature: 'Fasting Glucose', contribution: parseFloat((Math.max(0, gluc - 100) * 0.006).toFixed(3)), direction: gluc > 100 ? 'positive' : 'negative', magnitude: Math.abs(gluc - 100), feature_value: `${gluc} mg/dL`, rank: 5 }
  ];

  const recommendations = [];
  if (sbp >= 130 || dbp >= 85) {
    recommendations.push({
      category: 'Blood Pressure Management',
      priority: sbp >= 140 ? 'High' : 'Medium',
      recommendation: 'Initiate dietary sodium restriction (<2g/day) and monitor home blood pressure readings daily.',
      reason: `Current blood pressure (${sbp}/${dbp} mmHg) is elevated above optimal targets.`,
      suggested_action: 'Schedule 4-week BP follow-up; consider first-line antihypertensive therapy if persistently above 135/85 mmHg.'
    });
  }
  if (chol >= 200) {
    recommendations.push({
      category: 'Lipid Optimization',
      priority: chol >= 240 ? 'High' : 'Medium',
      recommendation: 'Adopt Mediterranean-style dietary pattern and increase soluble fiber intake.',
      reason: `Total cholesterol is ${chol} mg/dL (target < 200 mg/dL).`,
      suggested_action: 'Repeat fasting lipid panel in 8-12 weeks; evaluate ASCVD 10-year statin indication.'
    });
  }
  if (isSmoker > 0) {
    recommendations.push({
      category: 'Smoking Cessation',
      priority: 'High',
      recommendation: 'Provide behavioral counseling and discuss pharmacotherapy (nicotine replacement or varenicline).',
      reason: 'Active smoking substantially compounds cardiovascular and vascular risks.',
      suggested_action: 'Enroll in tobacco cessation program with 2-week check-in.'
    });
  }
  if (recommendations.length === 0) {
    recommendations.push({
      category: 'Cardiovascular Wellness',
      priority: 'Low',
      recommendation: 'Maintain current heart-healthy habits with at least 150 minutes of moderate aerobic exercise weekly.',
      reason: 'All measured cardiovascular parameters are within favorable ranges.',
      suggested_action: 'Annual preventative cardiovascular review.'
    });
  }

  return {
    probability,
    riskClass,
    apcrs,
    modelVersion: 'CardioMind-FEMO v2.1',
    topFeatures,
    recommendations
  };
}

async function runPrediction(patientId, clinicalData, user) {
  try {
    // 1. Create clinical record in DB
    const record = await createClinicalRecord(patientId, clinicalData);

    let predictionResult;
    const mlUrl = config.mlServiceUrl || 'http://localhost:8000';

    try {
      // Attempt remote ML inference
      const predictRes = await axios.post(`${mlUrl}/predict/risk`, {
        age: clinicalData.age || 50,
        sex: clinicalData.sex || 'male',
        systolic_bp: clinicalData.systolic_bp,
        diastolic_bp: clinicalData.diastolic_bp,
        cholesterol: clinicalData.cholesterol,
        smoking_status: clinicalData.smoking_status || 'never',
        diabetes: Boolean(clinicalData.diabetes),
        family_history: Boolean(clinicalData.family_history),
        bmi: clinicalData.bmi || 24,
        heart_rate: clinicalData.heart_rate || 72
      }, { timeout: 3000 });

      const mlData = predictRes.data;
      const localFallback = calculateLocalCardioRisk(clinicalData);
      predictionResult = {
        probability: mlData.risk_score,
        riskClass: mlData.risk_level ? (mlData.risk_level.charAt(0).toUpperCase() + mlData.risk_level.slice(1)) : localFallback.riskClass,
        apcrs: Math.round(mlData.risk_score * 25),
        modelVersion: 'CardioMind-XGBoost v1.0',
        topFeatures: localFallback.topFeatures,
        recommendations: localFallback.recommendations
      };
    } catch (mlErr) {
      // Use built-in CardioMind calculation engine
      predictionResult = calculateLocalCardioRisk(clinicalData);
    }

    const { probability, riskClass, apcrs, modelVersion, topFeatures, recommendations } = predictionResult;

    // 2. Save prediction to DB
    const prediction = await createPrediction(patientId, probability, riskClass, apcrs, modelVersion, record.id);

    // 3. Save risk history
    await addRiskRecord(patientId, apcrs, probability, riskClass);

    // 4. Save interpretations
    await saveInterpretations(prediction.id, topFeatures);

    // 5. Save recommendations
    await saveRecommendations(patientId, prediction.id, recommendations);

    // 6. Audit log
    await log(
      user.id, user.role, 'run_prediction', 'predictions', prediction.id,
      null, null, { patientId, modelVersion }
    );

    return {
      prediction,
      record,
      interpretations: topFeatures,
      recommendations
    };
  } catch (error) {
    console.error('Error running prediction pipeline:', error);
    throw error;
  }
}

async function getPatientRiskProfile(patientId) {
  const [patientRows] = await getPool().query('SELECT * FROM patients WHERE id = ?', [patientId]);
  const patient = patientRows[0];
  
  if (!patient) throw new Error('Patient not found');

  const history = await getRiskHistory(patientId, 365); // last year
  const [latestPredictionRows] = await getPool().query('SELECT * FROM predictions WHERE patient_id = ? ORDER BY prediction_date DESC LIMIT 1', [patientId]);
  const latestPrediction = latestPredictionRows[0] || null;
  
  let interpretations = [];
  let recommendations = [];
  
  if (latestPrediction) {
    interpretations = await getInterpretations(latestPrediction.id);
    recommendations = await getRecommendations(patientId, 10);
  }
  
  return {
    patient,
    history,
    latestPrediction,
    interpretations,
    recommendations
  };
}

module.exports = {
  runPrediction,
  getPatientRiskProfile
};

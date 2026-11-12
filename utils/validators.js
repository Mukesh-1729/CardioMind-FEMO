function validateClinicalData(data) {
  const errors = [];
  
  if (data.systolic_bp < 70 || data.systolic_bp > 250) errors.push('Systolic BP out of valid range (70-250)');
  if (data.diastolic_bp < 40 || data.diastolic_bp > 150) errors.push('Diastolic BP out of valid range (40-150)');
  if (data.cholesterol < 100 || data.cholesterol > 600) errors.push('Cholesterol out of valid range (100-600)');
  if (data.glucose < 50 || data.glucose > 500) errors.push('Glucose out of valid range (50-500)');
  if (data.heart_rate < 30 || data.heart_rate > 220) errors.push('Heart rate out of valid range (30-220)');
  if (data.bmi < 10 || data.bmi > 70) errors.push('BMI out of valid range (10-70)');
  
  return errors;
}

function sanitizePatientInput(data) {
  const sanitized = { ...data };
  for (const key in sanitized) {
    if (typeof sanitized[key] === 'string') {
      // Basic sanitization
      sanitized[key] = sanitized[key].trim().replace(/[<>]/g, '');
    }
  }
  return sanitized;
}

function validatePagination(page, limit) {
  const parsedPage = parseInt(page, 10);
  const parsedLimit = parseInt(limit, 10);
  
  return {
    page: isNaN(parsedPage) || parsedPage < 1 ? 1 : parsedPage,
    limit: isNaN(parsedLimit) || parsedLimit < 1 ? 10 : parsedLimit > 100 ? 100 : parsedLimit
  };
}

module.exports = {
  validateClinicalData,
  sanitizePatientInput,
  validatePagination
};

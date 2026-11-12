document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('assessmentForm');
  const weightInput = document.getElementById('weight');
  const heightInput = document.getElementById('height');
  const bmiInput = document.getElementById('bmi');
  const systolicInput = document.getElementById('systolic_bp');
  const diastolicInput = document.getElementById('diastolic_bp');
  const pulsePressureInput = document.getElementById('pulse_pressure'); // optional display field
  const submitBtn = document.getElementById('submitBtn');

  // Auto-calculate BMI
  function calculateBMI() {
    if (!weightInput || !heightInput || !bmiInput) return;
    const weight = parseFloat(weightInput.value);
    const heightCm = parseFloat(heightInput.value);
    
    if (weight > 0 && heightCm > 0) {
      const heightM = heightCm / 100;
      const bmi = weight / (heightM * heightM);
      bmiInput.value = bmi.toFixed(1);
    } else {
      bmiInput.value = '';
    }
  }

  if (weightInput) weightInput.addEventListener('input', calculateBMI);
  if (heightInput) heightInput.addEventListener('input', calculateBMI);

  // Auto-calculate Pulse Pressure
  function calculatePulsePressure() {
    if (!systolicInput || !diastolicInput) return;
    const systolic = parseFloat(systolicInput.value);
    const diastolic = parseFloat(diastolicInput.value);
    
    if (systolic > 0 && diastolic > 0 && pulsePressureInput) {
      pulsePressureInput.value = systolic - diastolic;
    }
  }

  if (systolicInput) systolicInput.addEventListener('input', calculatePulsePressure);
  if (diastolicInput) diastolicInput.addEventListener('input', calculatePulsePressure);

  // Real-time validation & loading state
  if (form) {
    form.addEventListener('submit', function(event) {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      } else {
        if (submitBtn) {
          submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
          submitBtn.disabled = true;
        }
      }
      form.classList.add('was-validated');
    }, false);
  }
});

const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = fs.existsSync('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe')
  ? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
  : 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

const OUT_DIRS = [
  path.resolve('paper_results/figures'),
  path.resolve('experiments/figures'),
  path.resolve('C:/Users/obban/.gemini/antigravity-ide/brain/3c6a90d5-c12f-43f0-86f8-c49cf7d2b4be')
];

for (const dir of OUT_DIRS) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

async function saveScreenshot(page, baseName) {
  // Wait a bit for Chart.js / Plotly / CSS animations to settle
  await new Promise(r => setTimeout(r, 1200));
  const buf = await page.screenshot({ fullPage: false });
  for (const dir of OUT_DIRS) {
    const p = path.join(dir, `${baseName}.png`);
    fs.writeFileSync(p, buf);
    console.log(`Saved screenshot: ${p}`);
  }
}

async function run() {
  console.log('Launching browser with:', CHROME_PATH);
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--window-size=1600,1050', '--disable-dev-shm-usage']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 2 });

  // 1. Figure 4.1: Project Home / Landing Page
  console.log('Capturing Figure 4.1: Project Home / Landing Page...');
  await page.goto('http://localhost:3000/auth/login', { waitUntil: 'networkidle2' });
  // Visit home / public preview
  await page.goto('http://localhost:3000/dashboard/about', { waitUntil: 'networkidle0' }).catch(() => {});
  // Or render the hero page
  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });
  // If redirected to login, capture login/home or capture authenticated home
  if (page.url().includes('login')) {
    // Log in first to show full platform home
    await page.type('#email', 'doctor@cardiomind.com');
    await page.type('#password', 'DoctorPassword123!');
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle0' }),
      page.click('button[type="submit"]')
    ]);
  }
  
  // Now we are logged in, capture Project Home / Landing Page
  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });
  await saveScreenshot(page, 'fig4_1_landing_page');

  // 2. Figure 4.2: Login and Registration Page
  console.log('Capturing Figure 4.2: Login and Registration Page...');
  const incognitoContext = await browser.createBrowserContext();
  const authPage = await incognitoContext.newPage();
  await authPage.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 2 });
  await authPage.goto('http://localhost:3000/auth/login', { waitUntil: 'networkidle0' });
  await saveScreenshot(authPage, 'fig4_2_login_registration');
  await incognitoContext.close();

  // 3. Figure 4.3: User Dashboard (Doctor Dashboard)
  console.log('Capturing Figure 4.3: User Dashboard...');
  await page.goto('http://localhost:3000/doctor/dashboard', { waitUntil: 'networkidle0' });
  await saveScreenshot(page, 'fig4_3_user_dashboard');

  // 4. Figure 4.4: Cardiovascular Risk Assessment Form
  console.log('Capturing Figure 4.4: Cardiovascular Risk Assessment Form...');
  await page.goto('http://localhost:3000/assessments/new?patient_id=1', { waitUntil: 'networkidle0' });
  // Pre-fill realistic form values for clear display
  await page.evaluate(() => {
    const sbp = document.getElementById('systolic_bp');
    if (sbp) sbp.value = '142';
    const dbp = document.getElementById('diastolic_bp');
    if (dbp) dbp.value = '88';
    const chol = document.getElementById('cholesterol');
    if (chol) chol.value = '218';
    const gluc = document.getElementById('glucose');
    if (gluc) gluc.value = '108';
    const hr = document.getElementById('heart_rate');
    if (hr) hr.value = '74';
    const bmi = document.getElementById('bmi');
    if (bmi) bmi.value = '28.5';
  });
  await saveScreenshot(page, 'fig4_4_risk_assessment_form');

  // 5. Figure 4.5: Cardiovascular Risk Prediction Result
  console.log('Capturing Figure 4.5: Cardiovascular Risk Prediction Result...');
  await page.goto('http://localhost:3000/assessments/1', { waitUntil: 'networkidle0' });
  await saveScreenshot(page, 'fig4_5_risk_prediction_result');

  // 6. Figure 4.6: APCRS Personalized Risk Dashboard
  console.log('Capturing Figure 4.6: APCRS Personalized Risk Dashboard...');
  await page.goto('http://localhost:3000/patient/dashboard', { waitUntil: 'networkidle0' });
  await saveScreenshot(page, 'fig4_6_apcrs_personalized_dashboard');

  // 7. Figure 4.7: SHAP Feature Explanation
  console.log('Capturing Figure 4.7: SHAP Feature Explanation...');
  // Navigate to assessment 1 and scroll / focus on SHAP panel or full assessment
  await page.goto('http://localhost:3000/assessments/1', { waitUntil: 'networkidle0' });
  await page.evaluate(() => {
    window.scrollTo(0, 100);
  });
  await saveScreenshot(page, 'fig4_7_shap_feature_explanation');

  // 8. Figure 4.8: Federated Learning / Hospital Client Dashboard
  console.log('Capturing Figure 4.8: Federated Learning / Hospital Client Dashboard...');
  await page.goto('http://localhost:3000/hospital/dashboard', { waitUntil: 'networkidle0' });
  await saveScreenshot(page, 'fig4_8_hospital_client_dashboard');

  // 9. Figure 4.9: Model Performance and Results Dashboard
  console.log('Capturing Figure 4.9: Model Performance and Results Dashboard...');
  await page.goto('http://localhost:3000/research/dashboard', { waitUntil: 'networkidle0' });
  await saveScreenshot(page, 'fig4_9_model_performance_dashboard');

  // 10. Figure 4.10: Admin / System Dashboard
  console.log('Capturing Figure 4.10: Admin / System Dashboard...');
  await page.goto('http://localhost:3000/research/experiments', { waitUntil: 'networkidle0' });
  await saveScreenshot(page, 'fig4_10_admin_system_dashboard');

  console.log('All Chapter 4 figures captured successfully!');
  await browser.close();
}

run().catch(err => {
  console.error('Error capturing figures:', err);
  process.exit(1);
});

const authRoutes = require('./authRoutes');
const apiAuthRoutes = require('./apiAuthRoutes');
const dashboardRoutes = require('./dashboardRoutes');
const apiRoutes = require('./apiRoutes');
const patientRoutes = require('./patientRoutes');
const doctorRoutes = require('./doctorRoutes');
const patientDashboardRoutes = require('./patientDashboardRoutes');
const hospitalRoutes = require('./hospitalRoutes');
const researchRoutes = require('./researchRoutes');
const assessmentRoutes = require('./assessmentRoutes');
const { requireWebAuth } = require('../middleware/authMiddleware');

function attachRoutes(app) {
  app.use('/auth', authRoutes);
  app.use('/api/auth', apiAuthRoutes);
  app.use('/', requireWebAuth, dashboardRoutes);
  app.use('/patients', requireWebAuth, patientRoutes);
  app.use('/doctor', requireWebAuth, doctorRoutes);
  app.use('/patient', requireWebAuth, patientDashboardRoutes);
  app.use('/hospital', requireWebAuth, hospitalRoutes);
  app.use('/research', requireWebAuth, researchRoutes);
  app.use('/assessments', requireWebAuth, assessmentRoutes);
  app.use('/api', apiRoutes);
}

module.exports = { attachRoutes };

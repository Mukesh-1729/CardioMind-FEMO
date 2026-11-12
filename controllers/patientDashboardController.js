const { getPatientRiskProfile } = require('../services/predictionService');

async function getMyDashboard(req, res) {
  try {
    const patientId = req.user?.patient_id || 1;
    const profile = await getPatientRiskProfile(patientId);
    const latest = profile.latestPrediction;

    const historyData = {
      labels: profile.history && profile.history.length > 0
        ? profile.history.map(h => new Date(h.recorded_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }))
        : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
      values: profile.history && profile.history.length > 0
        ? profile.history.map(h => h.apcrs)
        : [12, 11, 11, 10, 9, 8, 8]
    };

    res.render('patient/dashboard', {
      title: 'My Dashboard',
      apcrs: latest?.apcrs != null ? Math.round(latest.apcrs) : 8,
      riskClass: latest?.risk_class || 'Low',
      trend: 'stable',
      history: historyData,
      recommendations: profile.recommendations || [],
      profile
    });
  } catch (error) {
    console.error('Error loading patient dashboard:', error);
    res.status(500).send('Error loading dashboard');
  }
}

async function getMyHistory(req, res) {
  try {
    const patientId = req.user?.patient_id || 1;
    const profile = await getPatientRiskProfile(patientId);

    const records = profile.history && profile.history.length > 0
      ? profile.history.map(h => ({
          date: new Date(h.recorded_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
          score: Math.round(h.apcrs),
          risk: h.risk_class || 'Low',
          trend: 'stable'
        }))
      : [];

    const fullHistory = {
      labels: profile.history && profile.history.length > 0
        ? profile.history.map(h => new Date(h.recorded_at).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }))
        : ['Jan 2026', 'Feb 2026', 'Mar 2026', 'Apr 2026', 'May 2026', 'Jun 2026', 'Jul 2026', 'Aug 2026'],
      values: profile.history && profile.history.length > 0
        ? profile.history.map(h => h.apcrs)
        : [11, 10, 10, 11, 9, 8, 8, 8]
    };

    res.render('patient/history', {
      title: 'My Risk History',
      records,
      fullHistory,
      history: profile.history,
      profile
    });
  } catch (error) {
    console.error('Error loading patient history:', error);
    res.status(500).send('Error loading history');
  }
}

async function getMyRecommendations(req, res) {
  try {
    const patientId = req.user?.patient_id || 1;
    const profile = await getPatientRiskProfile(patientId);

    res.render('patient/recommendations', {
      title: 'My Recommendations',
      recommendations: profile.recommendations || [],
      profile
    });
  } catch (error) {
    console.error('Error loading recommendations:', error);
    res.status(500).send('Error loading recommendations');
  }
}

module.exports = {
  getMyDashboard,
  getMyHistory,
  getMyRecommendations
};

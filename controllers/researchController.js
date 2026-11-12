const { getAllModelResults, getAblationResults, triggerExperiment, getExperimentResults } = require('../services/researchService');
const { getRoundHistory } = require('../services/federatedService');

async function getDashboard(req, res) {
  try {
    const results = await getAllModelResults();
    const federatedHistory = await getRoundHistory();
    
    res.render('research/dashboard', {
      title: 'Research Dashboard',
      results: results.slice(0, 10),
      federatedHistory
    });
  } catch (error) {
    res.status(500).send('Error loading research dashboard');
  }
}

async function getModels(req, res) {
  try {
    const results = await getAllModelResults();
    res.render('research/models', {
      title: 'Model Comparison',
      results
    });
  } catch (error) {
    res.status(500).send('Error loading models');
  }
}

async function getExperiments(req, res) {
  try {
    const { results } = await getExperimentResults();
    res.render('research/experiments', {
      title: 'Experiments Runner',
      results
    });
  } catch (error) {
    res.status(500).send('Error loading experiments');
  }
}

async function runExperiment(req, res) {
  try {
    const { type } = req.body;
    await triggerExperiment(type);
    res.redirect('/research/experiments');
  } catch (error) {
    res.status(500).send('Error running experiment');
  }
}

async function getResults(req, res) {
  try {
    const { results } = await getExperimentResults();
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: 'Error fetching results JSON' });
  }
}

module.exports = {
  getDashboard,
  getModels,
  getExperiments,
  runExperiment,
  getResults
};

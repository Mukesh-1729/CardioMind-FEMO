function renderDashboard(req, res) {
  res.render('dashboard/index', {
    title: 'Clinical Intelligence Dashboard'
  });
}

function renderAbout(req, res) {
  res.render('dashboard/about', {
    title: 'About CardioMind'
  });
}

module.exports = { renderDashboard, renderAbout };

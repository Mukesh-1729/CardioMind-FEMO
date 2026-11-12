const express = require('express');
const { renderDashboard, renderAbout } = require('../controllers/dashboardController');

const router = express.Router();

router.get('/', renderDashboard);
router.get('/about', renderAbout);

module.exports = router;

const express = require('express');

const {
  handleLogin,
  handleLogout,
  handleRegister,
  renderLoginView,
  renderRegisterView
} = require('../controllers/authController');

const router = express.Router();

router.get('/login', renderLoginView);
router.post('/login', handleLogin);
router.get('/register', renderRegisterView);
router.post('/register', handleRegister);
router.post('/logout', handleLogout);

module.exports = router;
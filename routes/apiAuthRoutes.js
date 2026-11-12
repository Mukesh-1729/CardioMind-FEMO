const express = require('express');

const { apiLogin, apiLogout, apiRegister } = require('../controllers/authController');

const router = express.Router();

router.post('/login', apiLogin);
router.post('/register', apiRegister);
router.post('/logout', apiLogout);

module.exports = router;
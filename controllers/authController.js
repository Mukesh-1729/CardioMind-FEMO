const {
  authenticateDoctor,
  registerDoctorAccount
} = require('../services/authService');
const { config } = require('../config/env');

function renderLoginView(req, res) {
  res.render('auth/login', {
    title: 'Doctor Login',
    form: {
      email: ''
    },
    error: null
  });
}

function renderRegisterView(req, res) {
  res.render('auth/register', {
    title: 'Doctor Registration',
    form: {
      fullName: '',
      email: '',
      specialty: '',
      licenseNumber: ''
    },
    error: null
  });
}

function saveSession(session) {
  return new Promise((resolve, reject) => {
    session.save((error) => {
      if (error) {
        reject(error);
        return;
      }

      resolve();
    });
  });
}

async function handleLogin(req, res, next) {
  try {
    const { doctor, token } = await authenticateDoctor(req.body || {});
    req.session.doctor = doctor;
    req.session.jwt = token;

    await saveSession(req.session);

    req.flash('success', `Welcome back, ${doctor.full_name}.`);
    res.redirect('/');
  } catch (error) {
    const statusCode = error.statusCode || 500;
    res.status(statusCode).render('auth/login', {
      title: 'Doctor Login',
      form: {
        email: req.body?.email || ''
      },
      error: error.message
    });
  }
}

async function handleRegister(req, res, next) {
  try {
    const { doctor, token } = await registerDoctorAccount(req.body || {});
    req.session.doctor = doctor;
    req.session.jwt = token;

    await saveSession(req.session);

    req.flash('success', 'Doctor account created successfully.');
    res.redirect('/');
  } catch (error) {
    const statusCode = error.statusCode || 500;
    res.status(statusCode).render('auth/register', {
      title: 'Doctor Registration',
      form: {
        fullName: req.body?.fullName || '',
        email: req.body?.email || '',
        specialty: req.body?.specialty || '',
        licenseNumber: req.body?.licenseNumber || ''
      },
      error: error.message
    });
  }
}

async function handleLogout(req, res, next) {
  try {
    const sessionCookieName = config.sessionName;

    req.session.destroy((error) => {
      if (error) {
        next(error);
        return;
      }

      res.clearCookie(sessionCookieName);
      res.redirect('/auth/login');
    });
  } catch (error) {
    next(error);
  }
}

async function apiLogin(req, res, next) {
  try {
    const { doctor, token } = await authenticateDoctor(req.body || {});
    req.session.doctor = doctor;
    req.session.jwt = token;

    await saveSession(req.session);

    res.json({
      message: 'Login successful.',
      doctor,
      token
    });
  } catch (error) {
    const statusCode = error.statusCode || 500;
    res.status(statusCode).json({ error: error.message });
  }
}

async function apiRegister(req, res, next) {
  try {
    const { doctor, token } = await registerDoctorAccount(req.body || {});
    req.session.doctor = doctor;
    req.session.jwt = token;

    await saveSession(req.session);

    res.status(201).json({
      message: 'Doctor account created successfully.',
      doctor,
      token
    });
  } catch (error) {
    const statusCode = error.statusCode || 500;
    res.status(statusCode).json({ error: error.message });
  }
}

async function apiLogout(req, res, next) {
  try {
    if (!req.session) {
      res.status(204).send();
      return;
    }

    req.session.destroy((error) => {
      if (error) {
        next(error);
        return;
      }

      res.status(204).send();
    });
  } catch (error) {
    next(error);
  }
}

module.exports = {
  apiLogin,
  apiLogout,
  apiRegister,
  handleLogin,
  handleLogout,
  handleRegister,
  renderLoginView,
  renderRegisterView
};
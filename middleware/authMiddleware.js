const { createAuthError, extractBearerToken, sanitizeDoctor, verifyJwtToken } = require('../services/authService');
const { findDoctorById } = require('../models/doctorModel');

function syncCurrentUser(req, res, next) {
  res.locals.currentUser = req.session?.doctor || null;
  next();
}

function requireWebAuth(req, res, next) {
  if (req.session?.doctor) {
    return next();
  }

  req.flash('error', 'Please sign in to continue.');
  return res.redirect('/auth/login');
}

async function requireApiAuth(req, res, next) {
  try {
    if (req.session?.doctor) {
      req.user = req.session.doctor;
      return next();
    }

    const token = extractBearerToken(req);
    if (!token) {
      throw createAuthError(401, 'Authentication required.');
    }

    const decodedToken = verifyJwtToken(token);
    const doctorId = Number(decodedToken.sub);
    if (!Number.isFinite(doctorId)) {
      throw createAuthError(401, 'Invalid authentication token.');
    }

    const doctor = await findDoctorById(doctorId);
    if (!doctor || !doctor.is_active) {
      throw createAuthError(401, 'Invalid authentication token.');
    }

    req.user = sanitizeDoctor(doctor);
    next();
  } catch (error) {
    const statusCode = error.statusCode || 401;
    res.status(statusCode).json({
      error: error.message || 'Authentication required.'
    });
  }
}

function requireRole(roles) {
  return (req, res, next) => {
    const userRole = (req.user?.role || req.session?.doctor?.role || '').toUpperCase();
    if (roles.includes(userRole) || userRole === 'ADMIN') {
      return next();
    }
    req.flash('error', 'You do not have permission to access this resource.');
    return res.redirect('/');
  };
}

const requireDoctor = requireRole(['DOCTOR', 'ADMIN']);
const requireResearcher = requireRole(['DOCTOR', 'RESEARCHER', 'ADMIN']);

function auditLog(action) {
  return async (req, res, next) => {
    try {
      const { log } = require('../models/auditModel');
      const user = req.user || req.session?.doctor;
      await log(
        user?.id,
        user?.role,
        action,
        req.originalUrl,
        null,
        req.ip,
        req.get('User-Agent'),
        req.body ? { bodyKeys: Object.keys(req.body) } : null
      );
    } catch (err) {
      console.error('Audit log error', err);
    }
    next();
  };
}

module.exports = {
  requireApiAuth,
  requireWebAuth,
  syncCurrentUser,
  requireRole,
  requireDoctor,
  requireResearcher,
  auditLog
};
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const { config } = require('../config/env');
const {
  createDoctor,
  findDoctorByEmail,
  findDoctorByLicenseNumber,
  updateDoctorLastLogin
} = require('../models/doctorModel');

const PASSWORD_MIN_LENGTH = 12;
const BCRYPT_ROUNDS = 12;

function sanitizeDoctor(doctor) {
  if (!doctor) {
    return null;
  }

  const { password_hash: passwordHash, ...safeDoctor } = doctor;
  return safeDoctor;
}

function createAuthError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

function validateRegistrationInput(payload) {
  const fullName = typeof payload.fullName === 'string' ? payload.fullName.trim() : '';
  const email = typeof payload.email === 'string' ? payload.email.trim().toLowerCase() : '';
  const password = typeof payload.password === 'string' ? payload.password : '';
  const confirmPassword = typeof payload.confirmPassword === 'string' ? payload.confirmPassword : '';
  const licenseNumber = typeof payload.licenseNumber === 'string' ? payload.licenseNumber.trim() : '';
  const specialty = typeof payload.specialty === 'string' ? payload.specialty.trim() : '';

  if (!fullName) {
    throw createAuthError(400, 'Full name is required.');
  }

  if (!email) {
    throw createAuthError(400, 'Email is required.');
  }

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    throw createAuthError(400, 'A valid email address is required.');
  }

  if (!licenseNumber) {
    throw createAuthError(400, 'Medical license number is required.');
  }

  if (!password || password.length < PASSWORD_MIN_LENGTH) {
    throw createAuthError(400, `Password must be at least ${PASSWORD_MIN_LENGTH} characters long.`);
  }

  if (password !== confirmPassword) {
    throw createAuthError(400, 'Passwords do not match.');
  }

  return {
    fullName,
    email,
    password,
    specialty,
    licenseNumber
  };
}

function validateLoginInput(payload) {
  const email = typeof payload.email === 'string' ? payload.email.trim().toLowerCase() : '';
  const password = typeof payload.password === 'string' ? payload.password : '';

  if (!email) {
    throw createAuthError(400, 'Email is required.');
  }

  if (!password) {
    throw createAuthError(400, 'Password is required.');
  }

  return { email, password };
}

function issueJwtToken(doctor) {
  return jwt.sign(
    {
      sub: String(doctor.id),
      email: doctor.email,
      role: doctor.role,
      fullName: doctor.full_name
    },
    config.jwtSecret,
    {
      expiresIn: config.jwtExpiresIn,
      issuer: 'cardiomind',
      audience: 'cardiomind-api'
    }
  );
}

async function registerDoctorAccount(payload) {
  const registration = validateRegistrationInput(payload);

  const existingDoctor = await findDoctorByEmail(registration.email);
  if (existingDoctor) {
    throw createAuthError(409, 'A doctor account with that email already exists.');
  }

  const existingLicense = await findDoctorByLicenseNumber(registration.licenseNumber);
  if (existingLicense) {
    throw createAuthError(409, 'That medical license number is already registered.');
  }

  const passwordHash = await bcrypt.hash(registration.password, BCRYPT_ROUNDS);
  const doctor = await createDoctor({
    ...registration,
    passwordHash
  });

  return {
    doctor: sanitizeDoctor(doctor),
    token: issueJwtToken(doctor)
  };
}

async function authenticateDoctor(payload) {
  const credentials = validateLoginInput(payload);
  const doctor = await findDoctorByEmail(credentials.email);

  if (!doctor || !doctor.password_hash) {
    throw createAuthError(401, 'Invalid email or password.');
  }

  if (!doctor.is_active) {
    throw createAuthError(403, 'This account is disabled.');
  }

  const passwordMatches = await bcrypt.compare(credentials.password, doctor.password_hash);
  if (!passwordMatches) {
    throw createAuthError(401, 'Invalid email or password.');
  }

  await updateDoctorLastLogin(doctor.id);
  const refreshedDoctor = await findDoctorByEmail(credentials.email);

  return {
    doctor: sanitizeDoctor(refreshedDoctor),
    token: issueJwtToken(refreshedDoctor)
  };
}

function extractBearerToken(request) {
  const authorizationHeader = request.headers.authorization || '';
  const [scheme, token] = authorizationHeader.split(' ');

  if (scheme !== 'Bearer' || !token) {
    return null;
  }

  return token.trim();
}

function verifyJwtToken(token) {
  return jwt.verify(token, config.jwtSecret, {
    issuer: 'cardiomind',
    audience: 'cardiomind-api'
  });
}

module.exports = {
  authenticateDoctor,
  createAuthError,
  extractBearerToken,
  issueJwtToken,
  registerDoctorAccount,
  sanitizeDoctor,
  validateLoginInput,
  validateRegistrationInput,
  verifyJwtToken
};
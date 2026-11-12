const { getPool } = require('../config/database');

async function findDoctorById(id) {
  const [rows] = await getPool().query(
    `SELECT id, full_name, email, specialty, license_number, role, is_active, created_at, updated_at, last_login_at, password_hash
     FROM doctors
     WHERE id = ?
     LIMIT 1`,
    [id]
  );

  return rows[0] || null;
}

async function findDoctorByEmail(email) {
  const [rows] = await getPool().query(
    `SELECT id, full_name, email, specialty, license_number, role, is_active, created_at, updated_at, last_login_at, password_hash
     FROM doctors
     WHERE email = ?
     LIMIT 1`,
    [email.trim().toLowerCase()]
  );

  return rows[0] || null;
}

async function findDoctorByLicenseNumber(licenseNumber) {
  const [rows] = await getPool().query(
    `SELECT id, full_name, email, specialty, license_number, role, is_active, created_at, updated_at, last_login_at, password_hash
     FROM doctors
     WHERE license_number = ?
     LIMIT 1`,
    [licenseNumber.trim()]
  );

  return rows[0] || null;
}

async function createDoctor(doctorData) {
  const normalizedEmail = doctorData.email.trim().toLowerCase();
  const normalizedLicenseNumber = doctorData.licenseNumber.trim();
  const normalizedSpecialty = doctorData.specialty ? doctorData.specialty.trim() : null;

  const [result] = await getPool().query(
    `INSERT INTO doctors (
      full_name,
      email,
      password_hash,
      specialty,
      license_number,
      role,
      is_active,
      created_at,
      updated_at
    ) VALUES (?, ?, ?, ?, ?, 'doctor', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
    [
      doctorData.fullName.trim(),
      normalizedEmail,
      doctorData.passwordHash,
      normalizedSpecialty,
      normalizedLicenseNumber
    ]
  );

  return findDoctorById(result.insertId);
}

async function updateDoctorLastLogin(id) {
  await getPool().query('UPDATE doctors SET last_login_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = ?', [id]);
}

module.exports = {
  createDoctor,
  findDoctorByEmail,
  findDoctorById,
  findDoctorByLicenseNumber,
  updateDoctorLastLogin
};
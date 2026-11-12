const { getPool } = require('../config/database');

async function findAllPatients(searchTerm = '') {
  const normalizedSearch = (searchTerm || '').trim();

  if (!normalizedSearch) {
    const [rows] = await getPool().query(
      `SELECT id, full_name, age, sex, email, phone, city, state, created_at, updated_at
       FROM patients
       ORDER BY created_at DESC`
    );
    return rows;
  }

  const pattern = `%${normalizedSearch}%`;
  const [rows] = await getPool().query(
    `SELECT id, full_name, age, sex, email, phone, city, state, created_at, updated_at
     FROM patients
     WHERE full_name LIKE ?
       OR email LIKE ?
       OR phone LIKE ?
       OR city LIKE ?
       OR state LIKE ?
     ORDER BY created_at DESC`,
    [pattern, pattern, pattern, pattern, pattern]
  );

  return rows;
}

async function findPatientById(id) {
  const [rows] = await getPool().query(
    `SELECT id,
            full_name,
            date_of_birth,
            age,
            sex,
            email,
            phone,
            address_line1,
            address_line2,
            city,
            state,
            postal_code,
            emergency_contact_name,
            emergency_contact_phone,
            medical_history,
            current_medications,
            allergies,
            notes,
            created_at,
            updated_at
     FROM patients
     WHERE id = ?
     LIMIT 1`,
    [id]
  );

  return rows[0] || null;
}

async function createPatient(patient) {
  const [result] = await getPool().query(
    `INSERT INTO patients (
      full_name,
      date_of_birth,
      age,
      sex,
      email,
      phone,
      address_line1,
      address_line2,
      city,
      state,
      postal_code,
      emergency_contact_name,
      emergency_contact_phone,
      medical_history,
      current_medications,
      allergies,
      notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      patient.fullName,
      patient.dateOfBirth,
      patient.age,
      patient.sex,
      patient.email,
      patient.phone,
      patient.addressLine1,
      patient.addressLine2,
      patient.city,
      patient.state,
      patient.postalCode,
      patient.emergencyContactName,
      patient.emergencyContactPhone,
      patient.medicalHistory,
      patient.currentMedications,
      patient.allergies,
      patient.notes
    ]
  );

  return findPatientById(result.insertId);
}

async function updatePatient(id, patient) {
  const [result] = await getPool().query(
    `UPDATE patients
     SET full_name = ?,
         date_of_birth = ?,
         age = ?,
         sex = ?,
         email = ?,
         phone = ?,
         address_line1 = ?,
         address_line2 = ?,
         city = ?,
         state = ?,
         postal_code = ?,
         emergency_contact_name = ?,
         emergency_contact_phone = ?,
         medical_history = ?,
         current_medications = ?,
         allergies = ?,
         notes = ?,
         updated_at = CURRENT_TIMESTAMP
     WHERE id = ?`,
    [
      patient.fullName,
      patient.dateOfBirth,
      patient.age,
      patient.sex,
      patient.email,
      patient.phone,
      patient.addressLine1,
      patient.addressLine2,
      patient.city,
      patient.state,
      patient.postalCode,
      patient.emergencyContactName,
      patient.emergencyContactPhone,
      patient.medicalHistory,
      patient.currentMedications,
      patient.allergies,
      patient.notes,
      id
    ]
  );

  return result.affectedRows > 0;
}

async function deletePatient(id) {
  const [result] = await getPool().query('DELETE FROM patients WHERE id = ?', [id]);
  return result.affectedRows > 0;
}

module.exports = {
  createPatient,
  deletePatient,
  findAllPatients,
  findPatientById,
  updatePatient
};

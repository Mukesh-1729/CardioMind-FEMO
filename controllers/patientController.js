const {
  createPatient,
  deletePatient,
  findAllPatients,
  findPatientById,
  updatePatient
} = require('../models/patientModel');

function normalizePatientPayload(body) {
  const asText = (value) => {
    if (typeof value !== 'string') {
      return null;
    }

    const trimmed = value.trim();
    return trimmed ? trimmed : null;
  };

  const age = Number.parseInt(body.age, 10);

  return {
    fullName: asText(body.fullName),
    dateOfBirth: asText(body.dateOfBirth),
    age: Number.isNaN(age) ? null : age,
    sex: asText(body.sex),
    email: asText(body.email),
    phone: asText(body.phone),
    addressLine1: asText(body.addressLine1),
    addressLine2: asText(body.addressLine2),
    city: asText(body.city),
    state: asText(body.state),
    postalCode: asText(body.postalCode),
    emergencyContactName: asText(body.emergencyContactName),
    emergencyContactPhone: asText(body.emergencyContactPhone),
    medicalHistory: asText(body.medicalHistory),
    currentMedications: asText(body.currentMedications),
    allergies: asText(body.allergies),
    notes: asText(body.notes)
  };
}

function toDateInputValue(value) {
  if (!value) {
    return '';
  }

  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return value.toISOString().slice(0, 10);
  }

  if (typeof value === 'string') {
    return value.slice(0, 10);
  }

  return '';
}

function validatePatientPayload(payload) {
  if (!payload.fullName) {
    return 'Full name is required.';
  }

  if (!payload.sex) {
    return 'Sex is required.';
  }

  if (payload.age === null || payload.age < 0 || payload.age > 130) {
    return 'Age must be between 0 and 130.';
  }

  return null;
}

function mapPatientForm(payload = {}) {
  return {
    fullName: payload.fullName || '',
    dateOfBirth: toDateInputValue(payload.dateOfBirth),
    age: payload.age ?? '',
    sex: payload.sex || '',
    email: payload.email || '',
    phone: payload.phone || '',
    addressLine1: payload.addressLine1 || '',
    addressLine2: payload.addressLine2 || '',
    city: payload.city || '',
    state: payload.state || '',
    postalCode: payload.postalCode || '',
    emergencyContactName: payload.emergencyContactName || '',
    emergencyContactPhone: payload.emergencyContactPhone || '',
    medicalHistory: payload.medicalHistory || '',
    currentMedications: payload.currentMedications || '',
    allergies: payload.allergies || '',
    notes: payload.notes || ''
  };
}

async function listPatients(req, res, next) {
  try {
    const query = typeof req.query.q === 'string' ? req.query.q.trim() : '';
    const patients = await findAllPatients(query);

    res.render('patients/list', {
      title: 'Patient Registry',
      patients,
      query
    });
  } catch (error) {
    next(error);
  }
}

function createPatientView(req, res) {
  res.render('patients/new', {
    title: 'Add Patient',
    form: mapPatientForm(),
    error: null
  });
}

async function createPatientAction(req, res, next) {
  try {
    const payload = normalizePatientPayload(req.body || {});
    const validationError = validatePatientPayload(payload);

    if (validationError) {
      res.status(400).render('patients/new', {
        title: 'Add Patient',
        form: mapPatientForm(payload),
        error: validationError
      });
      return;
    }

    const created = await createPatient(payload);
    req.flash('success', 'Patient was added successfully.');
    res.redirect(`/patients/${created.id}`);
  } catch (error) {
    next(error);
  }
}

async function viewPatient(req, res, next) {
  try {
    const patient = await findPatientById(req.params.id);

    if (!patient) {
      res.status(404).render('error', {
        title: 'Patient not found',
        message: 'The requested patient record could not be found.'
      });
      return;
    }

    res.render('patients/view', {
      title: patient.full_name,
      patient
    });
  } catch (error) {
    next(error);
  }
}

async function editPatientView(req, res, next) {
  try {
    const patient = await findPatientById(req.params.id);

    if (!patient) {
      res.status(404).render('error', {
        title: 'Patient not found',
        message: 'The requested patient record could not be found.'
      });
      return;
    }

    res.render('patients/edit', {
      title: `Edit ${patient.full_name}`,
      patient,
      form: mapPatientForm({
        fullName: patient.full_name,
        dateOfBirth: patient.date_of_birth,
        age: patient.age,
        sex: patient.sex,
        email: patient.email,
        phone: patient.phone,
        addressLine1: patient.address_line1,
        addressLine2: patient.address_line2,
        city: patient.city,
        state: patient.state,
        postalCode: patient.postal_code,
        emergencyContactName: patient.emergency_contact_name,
        emergencyContactPhone: patient.emergency_contact_phone,
        medicalHistory: patient.medical_history,
        currentMedications: patient.current_medications,
        allergies: patient.allergies,
        notes: patient.notes
      }),
      error: null
    });
  } catch (error) {
    next(error);
  }
}

async function editPatientAction(req, res, next) {
  try {
    const payload = normalizePatientPayload(req.body || {});
    const validationError = validatePatientPayload(payload);

    if (validationError) {
      res.status(400).render('patients/edit', {
        title: 'Edit Patient',
        patient: { id: req.params.id, full_name: payload.fullName || 'Patient' },
        form: mapPatientForm(payload),
        error: validationError
      });
      return;
    }

    const updated = await updatePatient(req.params.id, payload);
    if (!updated) {
      res.status(404).render('error', {
        title: 'Patient not found',
        message: 'The requested patient record could not be found.'
      });
      return;
    }

    req.flash('success', 'Patient details were updated successfully.');
    res.redirect(`/patients/${req.params.id}`);
  } catch (error) {
    next(error);
  }
}

async function deletePatientAction(req, res, next) {
  try {
    const deleted = await deletePatient(req.params.id);

    if (!deleted) {
      res.status(404).render('error', {
        title: 'Patient not found',
        message: 'The requested patient record could not be found.'
      });
      return;
    }

    req.flash('success', 'Patient record deleted successfully.');
    res.redirect('/patients');
  } catch (error) {
    next(error);
  }
}

module.exports = {
  createPatientAction,
  createPatientView,
  deletePatientAction,
  editPatientAction,
  editPatientView,
  listPatients,
  viewPatient
};

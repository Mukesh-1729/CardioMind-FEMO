const { getHospitalByClientId } = require('../models/hospitalModel');

async function getDashboard(req, res) {
  try {
    const clientId = req.user?.client_id || 'client_1';
    let hospital = await getHospitalByClientId(clientId);
    
    if (!hospital) {
      hospital = {
        id: 1,
        name: 'Simulated Hospital Alpha',
        client_id: 'client_1',
        dataset_name: 'cardio_partition_1',
        local_record_count: 13641,
        federated_status: 'Training',
        current_round: 12,
        local_auc: 0.884,
        global_model_version: 'CardioMind-FEMO-v2.1'
      };
    }
    
    res.render('hospital/dashboard', {
      title: `${hospital.name} Dashboard`,
      hospital,
      fedStatus: { state: hospital.federated_status || 'Training' },
      localStats: {
        records: (hospital.local_record_count || 13641).toLocaleString(),
        auc: (hospital.local_auc || 0.884).toFixed(3),
        partition: 'Non-IID (Dirichlet α=0.5)'
      },
      fedInfo: {
        round: `${hospital.current_round || 12} / 50`,
        strategy: 'FEMO',
        modelVersion: hospital.global_model_version || 'CardioMind-FEMO-v2.1',
        progress: '45%'
      },
      netStats: {
        sent: '45.2 MB',
        received: '43.8 MB',
        participated: hospital.current_round || 12
      }
    });
  } catch (error) {
    console.error('Error loading hospital dashboard:', error);
    res.status(500).send('Error loading hospital dashboard');
  }
}

module.exports = {
  getDashboard
};

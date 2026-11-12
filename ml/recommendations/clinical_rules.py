import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ClinicalRulesEngine:
    """
    Rule-based recommendations (NOT ML-generated)
    NEVER prescribes medication automatically
    ALWAYS includes 'consult qualified healthcare professional' disclaimer
    """
    
    def generate_recommendations(self, patient_features: Dict[str, float], risk_score: float, top_shap_features: List[Dict]) -> List[Dict]:
        recs = []
        
        # Blood Pressure
        sbp = patient_features.get('systolic_bp', 0)
        dbp = patient_features.get('diastolic_bp', 0)
        if sbp > 140 or dbp > 90:
            recs.append({
                'category': 'BloodPressure',
                'priority': 'High',
                'reason': f'Blood pressure is elevated ({sbp}/{dbp})',
                'suggested_action': 'Monitor blood pressure regularly and discuss lifestyle modifications or treatments with your doctor.'
            })
            
        # Smoking
        smoke = patient_features.get('smoke', 0)
        if smoke > 0:
            recs.append({
                'category': 'Smoking',
                'priority': 'High',
                'reason': 'Current smoker',
                'suggested_action': 'Consider enrolling in a smoking cessation program.'
            })
            
        # Activity
        active = patient_features.get('active', 1)
        if active == 0:
            recs.append({
                'category': 'PhysicalActivity',
                'priority': 'Medium',
                'reason': 'Reported low physical activity',
                'suggested_action': 'Aim for at least 150 minutes of moderate aerobic activity per week, as approved by your doctor.'
            })
            
        if not recs:
            recs.append({
                'category': 'General',
                'priority': 'Low',
                'reason': 'No critical risk factors identified',
                'suggested_action': 'Maintain a healthy lifestyle and attend regular check-ups.'
            })
            
        return recs

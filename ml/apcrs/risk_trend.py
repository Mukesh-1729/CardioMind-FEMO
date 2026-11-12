import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class RiskTrendAnalyzer:
    def analyze_trend(self, apcrs_history: List[float]) -> Dict:
        if len(apcrs_history) < 2:
            return {'direction': 'Stable', 'slope': 0.0, 'recent_change': 0.0}
            
        y = np.array(apcrs_history)
        x = np.arange(len(y))
        
        # Fit line
        slope, _ = np.polyfit(x, y, 1)
        recent_change = y[-1] - y[-2]
        
        direction = 'Stable'
        if slope > 2.0:
            direction = 'Worsening'
        elif slope < -2.0:
            direction = 'Improving'
            
        return {
            'direction': direction,
            'slope': float(slope),
            'recent_change': float(recent_change)
        }

    def compute_moving_average(self, history: List[float], window: int = 3) -> List[float]:
        if len(history) < window:
            return history
            
        ret = np.cumsum(history, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        ma = ret[window - 1:] / window
        
        # Pad beginning
        res = [history[0]] * (window - 1) + ma.tolist()
        return res

    def generate_trend_data_for_chart(self, history: List[float]) -> Dict:
        return {
            'labels': [f"T-{len(history)-1-i}" for i in range(len(history))],
            'datasets': [
                {
                    'label': 'APCRS',
                    'data': history,
                    'borderColor': 'rgb(75, 192, 192)'
                },
                {
                    'label': 'Moving Average',
                    'data': self.compute_moving_average(history),
                    'borderColor': 'rgb(255, 99, 132)',
                    'borderDash': [5, 5]
                }
            ]
        }

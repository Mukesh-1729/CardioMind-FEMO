// Reusable Chart.js builders for Risk visualizations
const RiskCharts = {
  getChartTheme() {
    return {
      color: '#8b949e',
      gridColor: '#1f2937',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    };
  },

  renderAPCRSTrend(canvasId, labels, values) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const theme = this.getChartTheme();
    
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels || [],
        datasets: [{
          label: 'APCRS Score',
          data: values || [],
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          borderWidth: 2,
          tension: 0.4,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.color } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.color }, beginAtZero: true }
        }
      }
    });
  },

  renderRiskDistribution(canvasId, counts) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    return new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
        datasets: [{
          data: counts || [0, 0, 0, 0],
          backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#991b1b'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '70%',
        plugins: {
          legend: { position: 'bottom', labels: { color: '#c9d1d9', usePointStyle: true } }
        }
      }
    });
  },

  renderSHAPBars(canvasId, features, contributions) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const theme = this.getChartTheme();
    
    const colors = (contributions || []).map(val => val > 0 ? '#06b6d4' : '#8b5cf6');
    
    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: features || [],
        datasets: [{
          label: 'Impact on Risk',
          data: contributions || [],
          backgroundColor: colors,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.color } },
          y: { grid: { display: false }, ticks: { color: theme.color } }
        }
      }
    });
  },
  
  renderRiskHistory(canvasId, labels, values) {
    return this.renderAPCRSTrend(canvasId, labels, values); // Alias for now
  }
};

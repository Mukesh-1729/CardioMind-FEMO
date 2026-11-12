const FederatedCharts = {
  getTheme() {
    return { color: '#8b949e', gridColor: '#1f2937' };
  },

  renderConvergence(canvasId, roundData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const theme = this.getTheme();
    
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: (roundData || []).map((_, i) => `Round ${i+1}`),
        datasets: [{
          label: 'Global AUC',
          data: roundData || [],
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          borderWidth: 2,
          tension: 0.2,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#c9d1d9' } } },
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.color } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.color }, min: 0.5, max: 1.0 }
        }
      }
    });
  },

  renderClientPerformance(canvasId, clientData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const theme = this.getTheme();
    
    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: (clientData || []).map(c => c.name),
        datasets: [{
          label: 'Client AUC',
          data: (clientData || []).map(c => c.auc),
          backgroundColor: '#06b6d4',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: theme.color } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.color } }
        }
      }
    });
  },

  renderCommunicationCost(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const theme = this.getTheme();
    
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: (data || []).map((_, i) => `R${i+1}`),
        datasets: [{
          label: 'Cumulative Bytes (MB)',
          data: data || [],
          borderColor: '#f59e0b',
          borderWidth: 2,
          stepped: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: theme.gridColor }, ticks: { color: theme.color } },
          y: { grid: { color: theme.gridColor }, ticks: { color: theme.color } }
        }
      }
    });
  }
};

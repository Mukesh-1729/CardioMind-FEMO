const ResearchCharts = {
  getLayoutTemplate() {
    return {
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      font: { color: '#c9d1d9' },
      xaxis: { gridcolor: '#1f2937', zerolinecolor: '#1f2937' },
      yaxis: { gridcolor: '#1f2937', zerolinecolor: '#1f2937' },
      margin: { t: 30, b: 40, l: 40, r: 20 }
    };
  },

  renderROCCurves(divId, rocData) {
    const el = document.getElementById(divId);
    if (!el || typeof Plotly === 'undefined') return;
    
    const traces = (rocData || []).map(model => ({
      x: model.fpr,
      y: model.tpr,
      mode: 'lines',
      name: `${model.name} (AUC=${model.auc.toFixed(3)})`
    }));
    
    // Add diagonal reference line
    traces.push({
      x: [0, 1], y: [0, 1],
      mode: 'lines',
      name: 'Random',
      line: { dash: 'dash', color: '#8b949e' },
      showlegend: false
    });
    
    const layout = {
      ...this.getLayoutTemplate(),
      title: 'ROC Curves',
      xaxis: { ...this.getLayoutTemplate().xaxis, title: 'False Positive Rate' },
      yaxis: { ...this.getLayoutTemplate().yaxis, title: 'True Positive Rate' }
    };
    
    Plotly.newPlot(divId, traces, layout, {responsive: true, displayModeBar: false});
  },

  renderCalibrationCurves(divId, calData) {
    const el = document.getElementById(divId);
    if (!el || typeof Plotly === 'undefined') return;
    
    const traces = (calData || []).map(model => ({
      x: model.mean_predicted_value,
      y: model.fraction_of_positives,
      mode: 'lines+markers',
      name: model.name
    }));
    
    // Add perfectly calibrated line
    traces.push({
      x: [0, 1], y: [0, 1],
      mode: 'lines',
      name: 'Perfectly Calibrated',
      line: { dash: 'dot', color: '#8b949e' }
    });
    
    const layout = {
      ...this.getLayoutTemplate(),
      title: 'Calibration Curves',
      xaxis: { ...this.getLayoutTemplate().xaxis, title: 'Mean Predicted Value' },
      yaxis: { ...this.getLayoutTemplate().yaxis, title: 'Fraction of Positives' }
    };
    
    Plotly.newPlot(divId, traces, layout, {responsive: true, displayModeBar: false});
  },

  renderAblationComparison(canvasId, ablationData) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Full Model', 'No Labs', 'No Vitals', 'No History'],
        datasets: [{
          label: 'AUC Score',
          data: ablationData || [0.85, 0.78, 0.81, 0.83],
          backgroundColor: '#06b6d4'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#c9d1d9' } },
          y: { grid: { color: '#1f2937' }, ticks: { color: '#c9d1d9' } }
        }
      }
    });
  },

  renderCommVsPerformance(divId, data) {
    const el = document.getElementById(divId);
    if (!el || typeof Plotly === 'undefined') return;
    
    const trace = {
      x: (data || []).map(d => d.bytes),
      y: (data || []).map(d => d.auc),
      mode: 'markers+text',
      type: 'scatter',
      text: (data || []).map(d => d.strategy),
      textposition: 'top center',
      marker: { size: 12, color: '#8b5cf6' }
    };
    
    const layout = {
      ...this.getLayoutTemplate(),
      title: 'Comm Cost vs AUC',
      xaxis: { ...this.getLayoutTemplate().xaxis, title: 'Total Communication (MB)' },
      yaxis: { ...this.getLayoutTemplate().yaxis, title: 'Global AUC' }
    };
    
    Plotly.newPlot(divId, [trace], layout, {responsive: true, displayModeBar: false});
  }
};

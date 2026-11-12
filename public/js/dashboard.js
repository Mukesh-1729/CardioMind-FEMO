document.addEventListener('DOMContentLoaded', () => {
  const chartElement = document.getElementById('riskChart');

  if (chartElement && window.Chart) {
    new Chart(chartElement, {
      type: 'doughnut',
      data: {
        labels: ['Low', 'Moderate', 'High'],
        datasets: [
          {
            data: [54, 31, 15],
            backgroundColor: ['#56cfe1', '#4d7cff', '#f97316'],
            borderWidth: 0
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'bottom'
          }
        }
      }
    });
  }
});

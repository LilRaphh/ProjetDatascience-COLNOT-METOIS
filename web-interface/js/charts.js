const charts = {}; // Stocke les instances de graphiques

function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
        delete charts[id];
    }
}

function renderChart(canvasId, type, data, options = {}) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId).getContext('2d');
    // Thème sombre
    Chart.defaults.color = '#c8d8f0';
    Chart.defaults.borderColor = '#1e2d47';

    charts[canvasId] = new Chart(ctx, {
        type: type,
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { font: { family: 'IBM Plex Mono' } } },
                tooltip: { titleFont: { family: 'IBM Plex Mono' }, bodyFont: { family: 'IBM Plex Mono' } }
            },
            scales: {
                x: { ticks: { font: { family: 'IBM Plex Mono' } } },
                y: { ticks: { font: { family: 'IBM Plex Mono' }, beginAtZero: false } }
            },
            ...options
        }
    });
}

function renderEDACharts(report) {
    document.getElementById('edaChartsArea').style.display = 'block';

    // 1. Returns Distribution (Histogramme)
    if (report.returns && report.returns.histogram) {
        const hist = report.returns.histogram;
        // hist.bins a 51 éléments pour 50 barres. On prend le milieu ou la borne inf.
        const labels = hist.bins.slice(0, -1).map(b => b.toFixed(5));

        renderChart('chartReturns', 'bar', {
            labels: labels,
            datasets: [{
                label: 'Distribution des Rendements',
                data: hist.counts,
                backgroundColor: 'rgba(0, 212, 255, 0.5)',
                borderColor: '#00d4ff',
                borderWidth: 1,
                barPercentage: 1.0,
                categoryPercentage: 1.0
            }]
        }, {
            scales: {
                x: { ticks: { maxTicksLimit: 10 } }
            }
        });
    }

    // 2. Volatilité (Line Chart)
    if (report.volatility && report.volatility.monthly_volatility) {
        const monthly = report.volatility.monthly_volatility;
        const labels = Object.keys(monthly).sort();
        const values = labels.map(k => monthly[k]);

        renderChart('chartVolatility', 'line', {
            labels: labels,
            datasets: [{
                label: 'Volatilité Mensuelle (Rolling window)',
                data: values,
                borderColor: '#ffcc00',
                backgroundColor: 'rgba(255, 204, 0, 0.1)',
                tension: 0.2,
                fill: true
            }]
        });
    }

    // 3. Hourly Analysis
    if (report.hourly && report.hourly.hourly_stats) {
        const stats = report.hourly.hourly_stats;
        stats.sort((a, b) => a.hour - b.hour);

        const hours = stats.map(s => `${s.hour}h`);
        const means = stats.map(s => s.mean_return);
        const ranges = stats.map(s => s.mean_range);

        renderChart('chartHourly', 'bar', {
            labels: hours,
            datasets: [
                {
                    label: 'Rendement Moyen',
                    data: means,
                    backgroundColor: 'rgba(0, 255, 136, 0.5)',
                    yAxisID: 'y'
                },
                {
                    label: 'Range Moyen',
                    data: ranges,
                    type: 'line',
                    borderColor: '#ff4757',
                    yAxisID: 'y1'
                }
            ]
        }, {
            scales: {
                y: { position: 'left', title: { display: true, text: 'Rendement' } },
                y1: { position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Range (High-Low)' } }
            }
        });
    }

    // 4. Autocorrelation
    if (report.autocorrelation && report.autocorrelation.acf) {
        const acf = report.autocorrelation.acf;
        const lags = report.autocorrelation.lags || acf.map((_, i) => i + 1);
        const labels = lags.map(l => `Lag ${l}`);

        renderChart('chartACF', 'bar', {
            labels: labels,
            datasets: [{
                label: 'Autocorrélation',
                data: acf,
                backgroundColor: 'rgba(168, 196, 255, 0.5)',
                borderColor: '#a8c4ff',
                borderWidth: 1
            }]
        }, {
            scales: { y: { min: -0.5, max: 0.5 } }
        });
    }
}

function renderEvalChart(strategies) {
    if (!strategies) return;
    document.getElementById('evalChartContainer').style.display = 'block';

    const labels = Object.keys(strategies);
    const metrics = ['sharpe', 'total_return_pct', 'max_drawdown_pct'];
    const datasets = [];

    const colors = {
        'sharpe': 'rgba(0, 255, 136, 0.7)',
        'total_return_pct': 'rgba(0, 212, 255, 0.7)',
        'max_drawdown_pct': 'rgba(255, 71, 87, 0.7)'
    };

    metrics.forEach(m => {
        const data = labels.map(l => strategies[l][m] || 0);
        datasets.push({
            label: m,
            data: data,
            backgroundColor: colors[m],
            borderColor: colors[m].replace('0.7', '1'),
            borderWidth: 1
        });
    });

    renderChart('chartEval', 'bar', {
        labels: labels,
        datasets: datasets
    }, {
        indexAxis: 'y',
        scales: {
            x: { beginAtZero: true }
        }
    });
}

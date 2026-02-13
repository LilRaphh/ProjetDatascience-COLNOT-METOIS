// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const data = await callAPI('GET', '/health');
        document.getElementById('healthStatus').textContent = 'API connectée';
        document.getElementById('healthBadge').style.borderColor = 'rgba(0, 255, 136, 0.3)';
        document.getElementById('healthBadge').style.background = 'rgba(0, 255, 136, 0.1)';
    } catch (e) {
        document.getElementById('healthStatus').textContent = 'API hors ligne';
        document.getElementById('healthBadge').style.background = 'rgba(255,71,87,0.1)';
        document.getElementById('healthBadge').style.borderColor = 'rgba(255,71,87,0.3)';
    }
}

// ── Dataset Tracker ───────────────────────────────────────────────────────────

// Style injecté pour les headers d'année
const style = document.createElement('style');
style.innerHTML = `
  .dataset-year-group { margin-bottom: 12px; }
  .dataset-year-header { 
    font-size: 11px; font-weight: 600; color: var(--text-dim); 
    background: rgba(255,255,255,0.03); padding: 4px 8px; border-radius: 4px;
    margin-bottom: 6px; font-family: 'IBM Plex Mono', monospace;
    display: flex; align-items: center; gap: 6px;
  }
  .dataset-year-header::before { content: '📂'; font-size: 10px; opacity: 0.7; }
  .dataset-list { display: flex; flex-direction: column; gap: 6px; padding-left: 6px; border-left: 1px solid var(--border); margin-left: 6px; }
`;
document.head.appendChild(style);

function getPhaseScore(id) {
    if (id.includes('trading_')) return 5;
    if (id.includes('_features')) return 4;
    if (id.includes('_clean')) return 3;
    if (id.includes('_m15')) return 2;
    if (id.startsWith('m1_')) return 1;
    return 0;
}

function getYearFromId(id) {
    const match = id.match(/_(\d{4})_/);
    return match ? match[1] : 'Autres';
}

async function refreshDatasets() {
    try {
        const data = await callAPI('GET', '/dataset/list');
        const datasets = data.result?.datasets || [];
        const el = document.getElementById('datasetTracker');

        if (datasets.length === 0) {
            el.innerHTML = '<div style="color:var(--text-dim);font-size:12px;">Aucun dataset en mémoire.</div>';
            return;
        }

        // Grouper par année
        const groups = {};
        datasets.forEach(d => {
            const year = getYearFromId(d.dataset_id);
            if (!groups[year]) groups[year] = [];
            groups[year].push(d);
        });

        // Trier les groupes (années) et les datasets (phase)
        const sortedYears = Object.keys(groups).sort();

        let html = '';
        sortedYears.forEach(year => {
            // Trier les datasets de l'année par phase
            groups[year].sort((a, b) => getPhaseScore(a.dataset_id) - getPhaseScore(b.dataset_id));

            html += `
        <div class="dataset-year-group">
          <div class="dataset-year-header">${year}</div>
          <div class="dataset-list">
            ${groups[year].map(d => `
              <div style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 12px;display:flex;justify-content:space-between;align-items:center;">
                <div style="overflow:hidden;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--accent);margin-bottom:2px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                       onclick="navigator.clipboard.writeText('${d.dataset_id}');showToast('ID copié !')" title="${d.dataset_id}">
                    ${d.dataset_id}
                  </div>
                  <div style="font-size:10px;color:var(--text-dim);">
                    ${d.phase || '—'} <span style="opacity:0.5;">|</span> ${d.shape?.[0]?.toLocaleString() || '?'} × ${d.shape?.[1] || '?'}
                  </div>
                </div>
                <div style="font-size:14px;cursor:pointer;opacity:0.5;margin-left:8px;" onclick="navigator.clipboard.writeText('${d.dataset_id}');showToast('ID copié !')">📋</div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
        });

        el.innerHTML = html;
    } catch (e) { console.error(e); }
}

// ── PHASE 1: Import M1 ────────────────────────────────────────────────────────
async function importM1() {
    setLoading('btnImport', true);
    try {
        const year = document.getElementById('importYear').value;
        const data = await callAPI('POST', '/dataset/load_m1', { year });
        setOutput('out-import', data);
        showToast(`Dataset M1 ${year} chargé : ${data.result?.shape?.[0]?.toLocaleString()} lignes`);
        refreshDatasets();
    } catch (e) {
        document.getElementById('out-import').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnImport', false);
    }
}

// ── PHASE 2: Agrégation M15 ───────────────────────────────────────────────────
async function aggregateM15() {
    setLoading('btnAgg', true);
    try {
        const id = document.getElementById('aggDatasetId').value.trim();
        if (!id) throw new Error('Entrez un Dataset ID.');
        const data = await callAPI('POST', '/m15/aggregate', { dataset_id: id });
        setOutput('out-agg', data);
        showToast(`Agrégation réussie : ${data.n_rows?.toLocaleString()} bougies M15`);
        refreshDatasets();
    } catch (e) {
        document.getElementById('out-agg').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnAgg', false);
    }
}

// ── PHASE 3: Nettoyage M15 ────────────────────────────────────────────────────
async function cleanM15() {
    setLoading('btnClean15', true);
    try {
        const id = document.getElementById('clean15DatasetId').value.trim();
        const threshold = parseFloat(document.getElementById('gapThreshold').value);
        if (!id) throw new Error('Entrez un Dataset ID.');
        const data = await callAPI('POST', '/m15/clean', {
            dataset_id: id,
            gap_return_threshold: threshold,
            drop_gaps: true
        });
        setOutput('out-clean15', data);
        showToast(`Nettoyage réussi : ${data.report?.dropped_total || 0} lignes supprimées`);
        refreshDatasets();
    } catch (e) {
        document.getElementById('out-clean15').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnClean15', false);
    }
}

// ── PHASE 4: Features ─────────────────────────────────────────────────────────
async function computeFeatures() {
    setLoading('btnFeat', true);
    try {
        const id = document.getElementById('featDatasetId').value.trim();
        const addTarget = document.getElementById('addTarget').value === 'true';
        if (!id) throw new Error('Entrez un Dataset ID.');
        const data = await callAPI('POST', '/features/compute', {
            dataset_id: id, drop_na: true, add_target: addTarget
        });
        setOutput('out-features', data);
        showToast(`Features calculées : ${data.n_features} features, ${data.n_rows?.toLocaleString()} lignes`);
        refreshDatasets();
    } catch (e) {
        document.getElementById('out-features').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnFeat', false);
    }
}

// ── PHASE 5: EDA ──────────────────────────────────────────────────────────────
async function runEDA() {
    setLoading('btnEDA', true);
    document.getElementById('edaChartsArea').style.display = 'none';
    try {
        const id = document.getElementById('edaDatasetId').value.trim();
        const type = document.getElementById('edaType').value;
        if (!id) throw new Error('Entrez un Dataset ID.');

        let data;

        if (type === 'full_report') {
            data = await callAPI('GET', `/eda/full_report/${id}`);
            renderEDACharts(data.eda_report);
        } else {
            data = await callAPI('GET', `/eda/${type}/${id}`);
            // Hack pour simuler structure
            if (type === 'returns' && data.returns) renderEDACharts({ returns: data.returns });
            if (type === 'volatility' && data.volatility) renderEDACharts({ volatility: data.volatility });
            if (type === 'hourly' && data.hourly) renderEDACharts({ hourly: data.hourly });
            if (type === 'autocorrelation' && data.autocorrelation) renderEDACharts({ autocorrelation: data.autocorrelation });
        }

        setOutput('out-eda', data);
        showToast('Analyse EDA terminée');
    } catch (e) {
        document.getElementById('out-eda').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnEDA', false);
    }
}

// ── PHASE 6: Baseline ─────────────────────────────────────────────────────────
async function runBaseline() {
    setLoading('btnBaseline', true);
    try {
        const id = document.getElementById('baselineDatasetId').value.trim();
        const seed = document.getElementById('baselineSeed').value;
        if (!id) throw new Error('Entrez un Dataset ID.');
        const data = await callAPI('GET', `/baseline/compare/${id}`, { seed });
        renderBaselineTable(data.baselines);
        setOutput('out-baseline', data);
        showToast('Comparaison baseline terminée');
    } catch (e) {
        document.getElementById('out-baseline').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnBaseline', false);
    }
}

function renderBaselineTable(baselines) {
    if (!baselines) return;
    const rows = Object.entries(baselines).map(([name, m]) => ({
        name, sharpe: m.sharpe, ret: m.total_return_pct, mdd: m.max_drawdown_pct, pf: m.profit_factor
    })).sort((a, b) => (b.sharpe || -999) - (a.sharpe || -999));

    document.getElementById('baselineTable').innerHTML = `
<table class="compare-table" style="margin-bottom:20px;">
  <thead><tr>
    <th>#</th><th>Stratégie</th><th>Sharpe</th><th>Return %</th><th>Max DD %</th><th>Profit Factor</th>
  </tr></thead>
  <tbody>
    ${rows.map((r, i) => `
      <tr class="${i === 0 ? 'rank-1' : ''}">
        <td>${i + 1}</td>
        <td>${r.name}</td>
        <td class="${r.sharpe > 0 ? 'positive' : 'negative'}">${r.sharpe?.toFixed(2) ?? '—'}</td>
        <td class="${r.ret > 0 ? 'positive' : 'negative'}">${r.ret?.toFixed(2) ?? '—'}%</td>
        <td class="negative">${r.mdd?.toFixed(2) ?? '—'}%</td>
        <td>${r.pf?.toFixed(2) ?? '—'}</td>
      </tr>
    `).join('')}
  </tbody>
</table>
`;
}

// ── PHASE 7: ML ───────────────────────────────────────────────────────────────
async function trainML() {
    setLoading('btnML', true);
    try {
        const trainId = document.getElementById('mlTrainId').value.trim();
        const valId = document.getElementById('mlValId').value.trim();
        const testId = document.getElementById('mlTestId').value.trim();
        const modelType = document.getElementById('mlModelType').value;
        if (!trainId || !valId) throw new Error('Train et Val Dataset IDs requis.');
        const params = { dataset_train_id: trainId, dataset_val_id: valId, model_type: modelType };
        if (testId) params.dataset_test_id = testId;
        const data = await callAPI('POST', '/trading_ml/train', params);
        setOutput('out-ml', data);
        showToast(`Modèle ${data.model_type} (${data.version}) entraîné — Sharpe val: ${data.metrics?.val?.sharpe}`);
    } catch (e) {
        document.getElementById('out-ml').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnML', false);
    }
}

// ── PHASE 8: RL ───────────────────────────────────────────────────────────────
async function trainRL() {
    setLoading('btnRL', true);
    const out = document.getElementById('out-rl');
    out.textContent = '🚀 Démarrage de l\'entraînement...\n';

    // Reset chart data
    const episodesData = [];
    const rewardsData = [];
    const equitiesData = [];
    document.getElementById('rlChartContainer').style.display = 'block';

    try {
        const trainId = document.getElementById('rlTrainId').value.trim();
        const valId = document.getElementById('rlValId').value.trim();
        const episodes = parseInt(document.getElementById('rlEpisodes').value);
        const seed = parseInt(document.getElementById('rlSeed').value);

        if (!trainId || !valId) throw new Error('Train et Val Dataset IDs requis.');

        const response = await fetch(`${API}/rl/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_train_id: trainId,
                dataset_val_id: valId,
                n_episodes: episodes,
                seed
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Garder le reste incomplete

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.replace('data: ', '');
                    try {
                        const event = JSON.parse(jsonStr);

                        if (event.type === 'progress') {
                            const pct = Math.round((event.episode / event.total_episodes) * 100);

                            // Update Chart Data
                            episodesData.push(event.episode);
                            rewardsData.push(event.reward);
                            equitiesData.push(event.equity);
                            renderRLTrainingChart(episodesData, rewardsData, equitiesData);

                            out.innerHTML = `⏳ Épisode ${event.episode}/${event.total_episodes} (${pct}%)\n` +
                                `Reward: <span class="num">${event.reward}</span> | ` +
                                `Equity: <span class="num">${event.equity}</span> | ` +
                                `Epsilon: <span class="num">${event.epsilon}</span>`;
                        } else if (event.type === 'result') {
                            const data = event.payload;
                            out.innerHTML += `\n\n✅ <span class="key">Entraînement terminé !</span>\n` +
                                renderJSON(data);
                            showToast(`Agent Q-Learning entraîné — ${data.agent_info?.n_states_visited} états visités`);
                        } else if (event.type === 'error') {
                            throw new Error(event.message);
                        }
                    } catch (e) {
                        console.error('SSE Parse Error', e);
                    }
                }
            }
        }
    } catch (e) {
        out.textContent += '\n\n❌ Erreur: ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnRL', false);
    }
}

async function getRLDesign() {
    setLoading('btnRLDesign', true);
    try {
        const data = await callAPI('GET', '/rl/design');
        setOutput('out-rl-design', data);
    } catch (e) {
        document.getElementById('out-rl-design').textContent = '❌ ' + e.message;
    } finally {
        setLoading('btnRLDesign', false);
    }
}

// ── PHASE 9: Évaluation ───────────────────────────────────────────────────────
async function runEvaluation() {
    setLoading('btnEval', true);
    try {
        const id = document.getElementById('evalDatasetId').value.trim();
        const mlId = document.getElementById('evalMLId').value.trim();
        if (!id) throw new Error('Dataset ID requis.');
        const params = {};
        if (mlId) params.ml_model_id = mlId;
        const data = await callAPI('GET', `/evaluate/compare/${id}`, params);

        renderEvalTable(data.ranking_by_sharpe);
        renderEvalChart(data.strategies);

        setOutput('out-eval', data);
        showToast('Évaluation finale terminée');
    } catch (e) {
        document.getElementById('out-eval').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    } finally {
        setLoading('btnEval', false);
    }
}

async function runStressTest() {
    const id = document.getElementById('evalDatasetId').value.trim();
    if (!id) { showToast('Entrez un Dataset ID.', 'error'); return; }
    try {
        const data = await callAPI('GET', `/evaluate/stress_test/${id}`);
        setOutput('out-eval', data);
        showToast('Stress test trimestriel terminé');
    } catch (e) {
        document.getElementById('out-eval').textContent = '❌ ' + e.message;
        showToast(e.message, 'error');
    }
}

function renderEvalTable(ranking) {
    if (!ranking?.length) return;
    document.getElementById('evalTable').innerHTML = `
<table class="compare-table" style="margin-bottom:20px;">
  <thead><tr>
    <th>#</th><th>Stratégie</th><th>Sharpe</th><th>Return %</th><th>Max DD %</th><th>Profit Factor</th>
  </tr></thead>
  <tbody>
    ${ranking.map(r => `
      <tr class="${r.rank === 1 ? 'rank-1' : ''}">
        <td>${r.rank}</td>
        <td>${r.strategy}</td>
        <td class="${r.sharpe > 0 ? 'positive' : 'negative'}">${r.sharpe?.toFixed(2) ?? '—'}</td>
        <td class="${r.total_return_pct > 0 ? 'positive' : 'negative'}">${r.total_return_pct?.toFixed(2) ?? '—'}%</td>
        <td class="negative">${r.max_drawdown_pct?.toFixed(2) ?? '—'}%</td>
        <td>${r.profit_factor?.toFixed(2) ?? '—'}</td>
      </tr>
    `).join('')}
  </tbody>
</table>
`;
}

// ── Init ──────────────────────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000);
showPanel('import');

function showToast(msg, type = 'success') {
    const t = document.getElementById('toast');
    document.getElementById('toastMsg').textContent = msg;
    document.getElementById('toastIcon').textContent = type === 'success' ? '✅' : '❌';
    t.className = `toast show ${type}`;
    setTimeout(() => t.className = 'toast', 4000);
}

function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    btn.classList.toggle('loading', loading);
    btn.disabled = loading;
}

function renderJSON(data) {
    return JSON.stringify(data, null, 2)
        .replace(/"([^"]+)":/g, '<span class="key">"$1"</span>:')
        .replace(/: "(.*?)"/g, ': <span class="str">"$1"</span>')
        .replace(/: (-?\d+\.?\d*)/g, ': <span class="num">$1</span>')
        .replace(/: (true|false)/g, ': <span class="bool">$1</span>');
}

function setOutput(id, data) {
    document.getElementById(id).innerHTML = renderJSON(data);
}

function setTab(el, name) {
    el.closest('.panel-body').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    const siblings = el.closest('.panel-body').querySelectorAll('[id^="tab-"]');
    siblings.forEach(s => s.style.display = 'none');
    const target = el.closest('.panel-body').querySelector('#tab-' + name);
    if (target) target.style.display = 'block';
}

function showPanel(name) {
    document.querySelectorAll('.panel-section').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.pipeline-step').forEach(s => s.classList.remove('active'));
    const panel = document.getElementById('panel-' + name);
    if (panel) {
        panel.style.display = 'block';
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    const steps = {
        'import': 0, 'm15': 1, 'features': 2, 'eda': 3,
        'baseline': 4, 'ml': 5, 'rl': 6, 'evaluate': 7
    };
    const idx = steps[name];
    if (idx !== undefined) {
        document.querySelectorAll('.pipeline-step')[idx]?.classList.add('active');
    }
}

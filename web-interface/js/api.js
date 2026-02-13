const API = 'http://localhost:8000';

async function callAPI(method, path, params = {}) {
    let url = `${API}${path}`;
    const opts = { method };

    // Pour GET/DELETE : paramètres dans l'URL (query string)
    if (method === 'GET' || method === 'DELETE') {
        const qs = new URLSearchParams(params).toString();
        if (qs) url += '?' + qs;
    }
    // Pour POST/PUT/PATCH : paramètres dans le body JSON
    else {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(params);
    }

    const res = await fetch(url, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    return data;
}

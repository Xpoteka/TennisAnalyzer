'use strict';

/* ---------- tiny helpers ---------- */

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (sel) => document.querySelector(sel);
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
// Node.append() turns a null into the text "null", so drop the empty slots first.
const mount = (el, ...kids) => { el.append(...kids.filter(Boolean)); return el; };

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (res.status === 401 && S.meta && S.meta.auth) {
    location.href = '/login';  // the login expired, or the password changed
    throw new Error('log in again');
  }
  const text = await res.text();
  let body = {};
  try { body = text ? JSON.parse(text) : {}; } catch { body = { error: text.slice(0, 400) }; }
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
  return body;
}
const postJSON = (path, data) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(data) });

function toast(message, kind) {
  const el = h('div', { class: `toast ${kind || ''}` }, message);
  $('#toasts').append(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 400); },
             kind === 'bad' ? 9000 : 4500);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied to the clipboard', 'ok');
  } catch {
    toast('Could not reach the clipboard; select the command and copy it', 'bad');
  }
}

const clock = (s) => {
  if (s === null || s === undefined || !isFinite(s)) return '–';
  const t = Math.max(0, s);
  const m = Math.floor(t / 60), sec = t % 60;
  return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
};
const dur = (s) => {
  if (!s && s !== 0) return '–';
  const m = Math.round(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
};
const num = (v, d = 2) =>
  v === null || v === undefined || !isFinite(v) ? '–' : Number(v).toFixed(d);
const bytes = (n) => {
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};

/* ---------- state ---------- */

const S = {
  meta: null, sessions: [], sel: null, detail: null, swings: null, metricMeta: [],
  jobs: [], watching: null, since: 0, tab: 'overview', view: 'analyze',
  swingSort: { key: 't_contact', dir: 1 }, swingSel: null, processOpts: {},
};

// Two letters are not enough: clean, classify and clips would all read "cl".
const STAGE_ABBR = {
  ingest: 'ing', contacts: 'con', pose: 'pos', clean: 'cln', classify: 'cls',
  metrics: 'met', labels: 'lab', clips: 'clp', report: 'rep',
};

const VIEWS = [
  ['analyze', 'Analyse'], ['videos', 'Videos'], ['sessions', 'Sessions'],
  ['commands', 'All commands'], ['config', 'Config'],
];

function setView(name) {
  S.view = name;
  for (const [id] of VIEWS) $(`#view-${id}`).classList.toggle('on', id === name);
  for (const b of $('#nav').children) b.classList.toggle('on', b.dataset.view === name);
  location.hash = name === 'sessions' && S.sel ? `sessions/${S.sel}` : name;
  if (name === 'videos') refreshVideos();
}

/* ---------- boot ---------- */

async function boot() {
  S.meta = await api('/api/meta');
  mount(clear($('#topmeta')), `v${S.meta.version} · ${S.meta.data_root}`,
    S.meta.auth ? h('button', { class: 'ghost small', onclick: logout }, 'Log out') : null);
  $('#uploads-dir').textContent = S.meta.uploads_dir;

  clear($('#nav')).append(...VIEWS.map(([id, label]) =>
    h('button', { 'data-view': id, onclick: () => setView(id) }, label)));

  clear($('#stage-list')).append(...S.meta.stages.map((s) =>
    h('li', {}, h('b', {}, s.name), s.optional ? ' (optional)' : '',
      s.implemented ? '' : ` — not built yet (${s.milestone})`)));

  renderProcessOptions();
  renderCommands();
  wireDropzone();
  wireConsole();
  wireConfig();

  await refreshSessions();
  await loadConfig();
  await pollJobs();

  const hash = location.hash.slice(1);
  const [view, sid] = hash.split('/');
  if (VIEWS.some(([id]) => id === view)) {
    if (sid) await selectSession(decodeURIComponent(sid));
    setView(view);
  } else setView('analyze');

  setInterval(pollJobs, 1500);
  setInterval(() => {
    if (S.view === 'sessions' || S.view === 'analyze') refreshSessions();
    if (S.view === 'videos') refreshVideos();
  }, 8000);
}

async function logout() {
  try { await postJSON('/api/logout', {}); } catch { /* going to the login page anyway */ }
  location.href = '/login';
}

/* ---------- forms built from the command catalogue ---------- */

function fieldInput(field, value, onChange) {
  const id = `f-${Math.random().toString(36).slice(2, 9)}`;
  const set = (v) => onChange(v);
  let input;
  if (field.kind === 'bool') {
    input = h('input', { type: 'checkbox', id, onchange: (e) => set(e.target.checked) });
    if (value) input.checked = true;
    return h('div', { class: 'field bool' }, input, h('label', { for: id }, field.label),
      field.help ? h('span', { class: 'help' }, field.help) : null);
  }
  if (field.kind === 'choice') {
    input = h('select', { id, onchange: (e) => set(e.target.value) },
      ...field.choices.map((c) => h('option', { value: c, selected: c === value }, c)));
  } else if (field.kind === 'session') {
    input = h('select', { id, 'data-picker': 'session', onchange: (e) => set(e.target.value) },
      h('option', { value: '' }, '— pick a session —'),
      ...S.sessions.map((s) => h('option', { value: s.id, selected: s.id === value }, s.id)));
  } else if (field.kind === 'labels') {
    const sel = h('select', { id, 'data-picker': 'labels', onchange: (e) => set(e.target.value) },
      h('option', { value: '' }, '— none —'),
      ...(S.labelFiles || []).map((f) =>
        h('option', { value: f.path, selected: f.path === value }, f.name)));
    const up = h('button', { class: 'ghost small', type: 'button',
      onclick: () => uploadLabelFile(sel, set) }, 'Upload CSV…');
    input = h('div', { class: 'with-btn' }, sel, up);
  } else if (field.kind === 'video') {
    const text = h('input', { type: 'text', id, value: value || '',
      placeholder: field.placeholder || '/path/to/session.mp4',
      oninput: (e) => set(e.target.value) });
    input = h('div', { class: 'with-btn' }, text,
      h('button', { class: 'ghost small', type: 'button',
        onclick: () => pickAndUpload((p) => { text.value = p; set(p); }) }, 'Upload…'));
  } else {
    input = h('input', {
      type: field.kind === 'number' ? 'number' : 'text', id, value: value || '',
      step: 'any', placeholder: field.placeholder || field.default || '',
      oninput: (e) => set(e.target.value),
    });
  }
  return h('div', { class: 'field' },
    h('label', { for: id }, field.label, field.required ? ' *' : ''),
    input,
    field.help ? h('div', { class: 'help' }, field.help) : null);
}

function renderProcessOptions() {
  const cmd = S.meta.commands.find((c) => c.name === 'process');
  const box = clear($('#process-fields'));
  for (const f of cmd.fields) {
    if (f.name === 'video_path') continue;  // the drop zone supplies it
    box.append(fieldInput(f, S.processOpts[f.name], (v) => { S.processOpts[f.name] = v; }));
  }
}

function renderCommands() {
  const root = clear($('#command-groups'));
  for (const group of S.meta.groups) {
    const commands = S.meta.commands.filter((c) => c.group === group);
    if (!commands.length) continue;
    root.append(h('div', { class: 'group-title' }, group));
    for (const cmd of commands) root.append(commandCard(cmd));
  }
}

function commandCard(cmd) {
  const values = {};
  for (const f of cmd.fields) if (f.default) values[f.name] = f.default;
  const fields = h('div', { class: 'fields' });
  const rebuild = () => {
    clear(fields).append(...cmd.fields.map((f) =>
      fieldInput(f, values[f.name], (v) => { values[f.name] = v; })));
  };
  rebuild();
  const card = h('div', { class: 'cmd-card' },
    h('h3', {}, cmd.title, cmd.slow ? h('span', { class: 'slow' }, 'slow') : null,
      h('span', { class: 'cli' }, `tennis ${cmd.name}`)),
    h('p', { class: 'hint' }, cmd.summary),
    fields,
    h('div', { class: 'row' },
      h('button', { class: 'primary', onclick: () => startJob(cmd.name, values) }, 'Run')));
  card._rebuild = rebuild;
  return card;
}

/* ---------- uploads ---------- */

/* Files go up in pieces, and the server keeps what arrived. A dropped connection, a proxy
   timeout or a closed tab only costs the piece in flight: the upload asks the server how
   much it has and carries on from there, now or when the same file is dropped again. */
const PIECE = 32 * 1024 * 1024;  // under the 100 MB request limit of tunnels like Cloudflare's
const MAX_RETRIES = 8;
let uploading = 0;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function upload(file, kind, onProgress) {
  const q = `kind=${kind}&name=${encodeURIComponent(file.name)}&size=${file.size}`;
  uploading++;
  try {
    let state = null;
    let failures = 0;
    while (!state || !state.done) {
      try {
        if (!state) state = await api(`/api/upload?${q}`);
        if (state.done) break;
        const offset = state.received;
        onProgress(offset / file.size);
        const piece = file.slice(offset, Math.min(offset + PIECE, file.size));
        state = await sendPiece(`/api/upload?${q}&offset=${offset}`, piece,
          (loaded) => onProgress((offset + loaded) / file.size));
        failures = 0;
      } catch (e) {
        if (e.received !== undefined) {  // the server has a different amount: go on from it
          state = { done: false, received: e.received };
          continue;
        }
        if (e.fatal || ++failures > MAX_RETRIES) throw e;
        state = null;  // ask again how much arrived
        await sleep(Math.min(30000, 1000 * 2 ** failures));
      }
    }
    onProgress(1);
    return state;
  } finally {
    uploading--;
  }
}

function sendPiece(url, piece, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded); };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { body = {}; }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      if (xhr.status === 401) { location.href = '/login'; }
      const err = new Error(body.error || `upload failed (${xhr.status})`);
      if (xhr.status === 409 && typeof body.received === 'number') err.received = body.received;
      // Proxies answer 502-504 when the server is busy or restarting: worth another try.
      else if (xhr.status < 500 || xhr.status === 507) err.fatal = true;
      reject(err);
    };
    xhr.onerror = () => reject(new Error('upload failed: the connection dropped'));
    xhr.send(piece);
  });
}

window.addEventListener('beforeunload', (e) => {
  if (uploading) { e.preventDefault(); e.returnValue = ''; }
});

function pickAndUpload(then) {
  const input = h('input', { type: 'file' });
  input.onchange = async () => {
    const file = input.files[0];
    if (!file) return;
    try {
      toast(`Uploading ${file.name}…`);
      const r = await upload(file, 'video', () => {});
      then(r.path);
      toast(`Uploaded to ${r.path}`, 'ok');
    } catch (e) { toast(e.message, 'bad'); }
  };
  input.click();
}

function uploadLabelFile(select, set) {
  const input = h('input', { type: 'file', accept: '.csv,.txt' });
  input.onchange = async () => {
    const file = input.files[0];
    if (!file) return;
    try {
      const r = await upload(file, 'labels', () => {});
      await refreshLabels();
      refreshPickers();
      select.value = r.path;
      set(r.path);
      toast(`Saved ${r.name} in the labels folder`, 'ok');
    } catch (e) { toast(e.message, 'bad'); }
  };
  input.click();
}

function wireDropzone() {
  const zone = $('#dropzone');
  const input = $('#file-input');
  $('#pick-file').onclick = () => input.click();
  input.onchange = () => { if (input.files[0]) handleVideo(input.files[0]); input.value = ''; };

  for (const type of ['dragenter', 'dragover']) {
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add('over'); });
  }
  for (const type of ['dragleave', 'dragend']) {
    zone.addEventListener(type, () => zone.classList.remove('over'));
  }
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('over');
    const file = e.dataTransfer.files[0];
    if (file) handleVideo(file);
  });
  // A file dropped anywhere else should not navigate the page away.
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());

  $('#path-go').onclick = () => {
    const path = $('#path-input').value.trim();
    if (!path) return toast('Give the path of a video on the server', 'bad');
    startJob('process', { ...S.processOpts, video_path: path });
  };
}

async function handleVideo(file) {
  const zone = $('#dropzone');
  const suffixes = S.meta.video_suffixes;
  const suffix = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if (!suffixes.includes(suffix)) {
    return toast(`${file.name} is not a video: expected ${suffixes.join(', ')}`, 'bad');
  }
  zone.classList.add('busy');
  $('#dz-progress').hidden = false;
  const fill = $('#dz-fill'), pct = $('#dz-pct');
  const show = (f) => { fill.style.width = `${f * 100}%`; pct.textContent = `${Math.round(f * 100)}%`; };
  show(0);
  try {
    const r = await upload(file, 'video', show);
    show(1);
    pct.textContent = r.reused ? 'already on the server' : 'uploaded';
    refreshVideos();
    await startJob('process', { ...S.processOpts, video_path: r.path });
  } catch (e) {
    toast(e.message, 'bad');
  } finally {
    zone.classList.remove('busy');
    setTimeout(() => { $('#dz-progress').hidden = true; }, 2500);
  }
}

/* ---------- jobs ---------- */

async function startJob(command, values) {
  try {
    const job = await postJSON('/api/jobs', { command, values });
    S.watching = job.id;
    S.since = 0;
    clear($('#console-out'));
    $('#console').dataset.open = 'true';
    toast(`Started: ${job.title}`, 'ok');
    await pollJobs();
  } catch (e) {
    toast(e.message, 'bad');
  }
}

function wireConsole() {
  const box = $('#console');
  $('#console-head').onclick = (e) => {
    if (e.target.id === 'console-cancel') return;
    box.dataset.open = box.dataset.open === 'true' ? 'false' : 'true';
  };
  $('#console-cancel').onclick = async (e) => {
    e.stopPropagation();
    if (!S.watching) return;
    try { await postJSON(`/api/jobs/${S.watching}/cancel`, {}); toast('Stopping…'); }
    catch (err) { toast(err.message, 'bad'); }
  };
}

async function pollJobs() {
  let jobs;
  try { jobs = (await api('/api/jobs')).jobs; } catch { return; }
  const wasRunning = new Map(S.jobs.map((j) => [j.id, j.state]));
  S.jobs = jobs;
  if (!S.watching && jobs.length) S.watching = jobs[0].id;

  clear($('#job-tabs')).append(...jobs.slice(0, 12).map((j) =>
    h('button', { class: j.id === S.watching ? 'on' : '',
      onclick: () => { S.watching = j.id; S.since = 0; clear($('#console-out')); pollJobs(); } },
      `${stateIcon(j.state)} ${j.title}`)));

  for (const j of jobs) {
    const before = wasRunning.get(j.id);
    if (before && before !== j.state && (j.state === 'done' || j.state === 'failed')) {
      toast(j.state === 'done' ? `Finished: ${j.title}` : `Failed: ${j.title} (exit ${j.returncode})`,
            j.state === 'done' ? 'ok' : 'bad');
      refreshSessions(true);
      if (S.view === 'videos') refreshVideos();
      if (S.sel && (j.session_id === S.sel || !j.session_id)) selectSession(S.sel, true);
    }
  }
  if (S.watching) await pollJobLines(S.watching);
}

const stateIcon = (s) =>
  ({ queued: '◷', running: '●', done: '✓', failed: '✕', cancelled: '⊘' })[s] || '·';

async function pollJobLines(id) {
  let job;
  try { job = await api(`/api/jobs/${id}?since=${S.since}`); } catch { return; }
  const out = $('#console-out');
  const stuck = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
  if (job.lines.length) {
    out.append(document.createTextNode(job.lines.join('\n') + '\n'));
    S.since = job.next_since;
    if (stuck) out.scrollTop = out.scrollHeight;
  }
  $('#console-dot').className = `dot ${job.state}`;
  $('#console-title').textContent =
    `${job.title} — ${job.state}${job.state === 'running' ? ` (${job.elapsed_s}s)` : ''}`;
  $('#console-cancel').hidden = job.state !== 'running' && job.state !== 'queued';
  mount(clear($('#console-foot')),
    h('code', {}, `tennis ${job.argv.join(' ')}`),
    job.state === 'done' && job.session_id
      ? h('button', { class: 'ghost small',
          onclick: () => { selectSession(job.session_id); setView('sessions'); } },
          'Open session')
      : null,
    job.state === 'done' && job.command === 'process'
      ? h('button', { class: 'ghost small', onclick: () => { refreshSessions(); setView('sessions'); } },
          'See sessions')
      : null);
}

/* ---------- videos ---------- */

async function refreshVideos() {
  let lib;
  try { lib = await api('/api/videos'); } catch (e) { return; }
  const signature = JSON.stringify(lib);
  if (signature === S.videoSignature) return;  // keep a click or a scroll position intact
  S.videoSignature = signature;
  renderVideos(lib);
}

function renderVideos(lib) {
  const root = clear($('#video-library'));
  const disk = lib.disk;
  root.append(h('div', { class: 'panel' },
    h('h2', {}, 'Videos on the server ',
      h('span', { class: 'muted' }, `${lib.videos.length} · ${bytes(disk.free)} free of ${bytes(disk.total)}`)),
    h('p', { class: 'hint' },
      'Every uploaded video is kept, so a session can be rerun without sending the file again. ',
      'Files copied into ', h('code', {}, lib.dir),
      ' by other means, such as a network share, show up here too.')));

  if (lib.partial.length) {
    root.append(h('div', { class: 'notice' },
      h('b', {}, 'Unfinished uploads. '),
      'Drop the same file on the Analyse page to carry on where it stopped.',
      h('table', {}, h('tbody', {}, ...lib.partial.map((p) => h('tr', {},
        h('td', {}, p.name),
        h('td', { class: 'num' }, `${bytes(p.received)} of ${bytes(p.size)}`),
        h('td', { class: 'num' }, `${Math.floor((100 * p.received) / p.size)}%`),
        h('td', { class: 'num' }, h('button', { class: 'ghost small',
          onclick: () => discardUpload(p) }, 'Discard'))))))));
  }

  if (!lib.videos.length) {
    root.append(h('div', { class: 'panel' }, h('p', { class: 'empty' },
      'No videos yet. Drop one on the Analyse page.')));
    return;
  }
  root.append(h('div', { class: 'panel' }, h('div', { class: 'scroll' }, h('table', {},
    h('thead', {}, h('tr', {}, h('th', {}, 'video'), h('th', { class: 'num' }, 'size'),
      h('th', {}, 'added'), h('th', {}, 'sessions'), h('th', {}, ''))),
    h('tbody', {}, ...lib.videos.map((v) => h('tr', {},
      h('td', {}, v.name),
      h('td', { class: 'num' }, bytes(v.size)),
      h('td', { class: 'muted' }, new Date(v.modified * 1000).toLocaleString()),
      h('td', {}, v.sessions.length
        ? v.sessions.map((sid) => h('button', { class: 'link session-link',
            onclick: () => { selectSession(sid); setView('sessions'); } }, sid))
        : h('span', { class: 'muted' }, 'not analysed')),
      h('td', { class: 'actions' },
        h('button', { class: v.sessions.length ? 'ghost small' : 'primary small',
          title: `tennis process ${v.path}`,
          onclick: () => startJob('process', { ...S.processOpts, video_path: v.path }) },
          v.sessions.length ? 'Run again' : 'Analyse'),
        h('a', { class: 'button ghost small', href: v.url, download: v.name }, 'Download'),
        h('button', { class: 'ghost small danger', onclick: () => deleteVideo(v) },
          'Delete')))))))));
}

async function deleteVideo(v) {
  const used = v.sessions.length
    ? `\n\nIt is used by ${v.sessions.join(', ')}. Their results stay, but they cannot be `
      + 'rerun until the video is uploaded again.'
    : '';
  if (!confirm(`Delete ${v.name} (${bytes(v.size)}) from the server for good?${used}`)) return;
  try {
    await postJSON('/api/videos/delete', { name: v.name });
    toast(`Deleted ${v.name}`, 'ok');
    refreshVideos();
  } catch (e) { toast(e.message, 'bad'); }
}

async function discardUpload(p) {
  if (!confirm(`Throw away the ${bytes(p.received)} already uploaded of ${p.name}?`)) return;
  try {
    await postJSON(`/api/upload/discard?kind=video&name=${encodeURIComponent(p.name)}&size=${p.size}`, {});
    refreshVideos();
  } catch (e) { toast(e.message, 'bad'); }
}

/* ---------- sessions ---------- */

async function refreshSessions(force) {
  let sessions;
  try {
    sessions = (await api('/api/sessions')).sessions;
  } catch (e) { return; }
  const signature = JSON.stringify(sessions.map((s) =>
    [s.id, s.stages, s.n_swings, s.has_report]));
  S.sessions = sessions;
  await refreshLabels();
  refreshPickers();
  // Redrawing the list would swallow a click landing at that moment and reset its scroll,
  // so only redraw when something actually changed.
  if (!force && signature === S.sessionSignature) return;
  S.sessionSignature = signature;
  const list = clear($('#session-list'));
  if (!S.sessions.length) {
    list.append(h('li', {}, h('div', { class: 'empty' }, 'No sessions yet')));
  }
  for (const s of S.sessions) {
    list.append(h('li', {}, h('button', {
      class: s.id === S.sel ? 'on' : '', onclick: () => selectSession(s.id),
    },
      h('div', { class: 'sid' }, s.id),
      h('div', { class: 'sub' },
        `${dur(s.duration_s)} · ${s.n_swings} swings${s.has_report ? ' · report' : ''}`),
      h('div', { class: 'pills' }, ...S.meta.stages.map((st) =>
        h('span', {
          class: `pill ${cls(s.stages[st.name])}`,
          title: `${st.number}. ${st.name}: ${s.stages[st.name]}`,
        }, STAGE_ABBR[st.name] || st.name.slice(0, 3)))))));
  }
}

/* Session and label dropdowns live inside forms the user may be filling in, so they are
   refreshed in place: rebuilding the form would throw away what is typed in it. */
function refreshPickers() {
  for (const sel of document.querySelectorAll('select[data-picker="session"]')) {
    setOptions(sel, S.sessions.map((s) => [s.id, s.id]), '— pick a session —');
  }
  for (const sel of document.querySelectorAll('select[data-picker="labels"]')) {
    setOptions(sel, (S.labelFiles || []).map((f) => [f.path, f.name]), '— none —');
  }
}

function setOptions(select, pairs, blank) {
  const wanted = select.value;
  const same = [...select.options].slice(1).map((o) => o.value).join('\u0000')
    === pairs.map(([v]) => v).join('\u0000');
  if (same) return;
  clear(select).append(h('option', { value: '' }, blank),
    ...pairs.map(([value, label]) => h('option', { value }, label)));
  if (pairs.some(([value]) => value === wanted)) select.value = wanted;
}

const cls = (status) => (status === 'ok' ? 'ok' : status === 'stale' ? 'stale' : 'none');

async function refreshLabels() {
  try { S.labelFiles = (await api('/api/labels')).files; } catch { S.labelFiles = []; }
}

async function selectSession(id, keepTab) {
  S.sel = id;
  if (!keepTab) { S.swingSel = null; }
  for (const b of $('#session-list').querySelectorAll('button')) {
    b.classList.toggle('on', b.querySelector('.sid').textContent === id);
  }
  try {
    S.detail = await api(`/api/sessions/${encodeURIComponent(id)}`);
  } catch (e) {
    clear($('#session-detail')).append(h('p', { class: 'empty' }, e.message));
    return;
  }
  S.swings = null;
  renderDetail();
  if (S.view === 'sessions') location.hash = `sessions/${id}`;
}

const TABS = [
  ['overview', 'Overview'], ['swings', 'Swings'], ['report', 'Report'],
  ['review', 'Review'], ['log', 'Log'],
];

function renderDetail() {
  const d = S.detail;
  const root = clear($('#session-detail'));
  root.append(h('h2', {}, d.id, ' ', h('span', { class: 'muted' }, d.dir)));
  root.append(h('div', { class: 'tabs' }, ...TABS.map(([id, label]) =>
    h('button', { class: S.tab === id ? 'on' : '', onclick: () => { S.tab = id; renderDetail(); } },
      label))));
  const body = h('div');
  root.append(body);
  ({ overview: tabOverview, swings: tabSwings, report: tabReport, review: tabReview, log: tabLog }
    [S.tab] || tabOverview)(body, d);
}

/* Which stage each action reads from. A session that stopped early has none of them, and
   the command would fail on a missing input, so the button says so instead of running. */
const ACTION_NEEDS = {
  report: 'metrics', players: 'clean', 'pose-preview': 'pose', 'swing-plots': 'clean',
};

const ranStages = (d) =>
  new Set(d.stages.filter((s) => s.status === 'ok' || s.status === 'stale').map((s) => s.name));

const sourceProblem = (d) => {
  if (!d.source_path) return 'this session has no video linked into it';
  if (d.source_missing) return `the video is gone: ${d.source_path}`;
  if (d.source_offline) return 'the video is in the cloud, not on this disk';
  return null;
};

function runPipeline(d, extra) {
  const problem = sourceProblem(d);
  if (problem) return toast(`Cannot process ${d.id}: ${problem}`, 'bad');
  startJob('process', { video_path: d.source_path, session_id: d.id, ...(extra || {}) });
}

function actionRow(d) {
  const done = ranStages(d);
  const complete = done.has('metrics');
  const action = (cmd, label, extra, primary) => {
    const needs = ACTION_NEEDS[cmd];
    const blocked = needs && !done.has(needs);
    return h('button', {
      class: primary && !blocked ? 'primary' : 'ghost',
      disabled: blocked || undefined,
      title: blocked
        ? `needs the ${needs} stage; run the pipeline on this session first`
        : `tennis ${cmd} ${d.id}`,
      onclick: () => startJob(cmd, { session_id: d.id, ...(extra || {}) }),
    }, label);
  };
  return h('div', { class: 'row' },
    h('button', {
      class: complete ? 'ghost' : 'primary',
      disabled: sourceProblem(d) || undefined,
      title: sourceProblem(d) || `tennis process ${d.source_path}`,
      onclick: () => runPipeline(d),
    },
      complete ? 'Rerun the pipeline' : 'Run the pipeline'),
    action('report', 'Rebuild report', {}, complete),
    action('players', 'Who is who'),
    action('pose-preview', 'Pose preview', { count: '20', speed: '0.5' }),
    action('swing-plots', 'Swing plots', { count: '6' }),
    h('button', { class: 'ghost', onclick: () => startJob('trends', {}) }, 'Trends'));
}

/* A session that stopped part-way is the normal case after an interrupted run, or after a
   stage that did not exist yet when it was first processed. Say so, and offer the fix. */
function unfinishedNotice(d) {
  const done = ranStages(d);
  const problem = sourceProblem(d);
  if (done.has('metrics') && !problem) return null;
  const pending = d.stages.filter((s) => s.implemented && !s.optional && !done.has(s.name));
  if (d.source_offline) {
    // ffmpeg would block on the download with nothing on screen, so say it up front.
    return h('div', { class: 'notice bad' },
      h('b', {}, 'This session\u2019s video is in the cloud, not on this disk. '),
      'macOS has evicted its contents ("Optimize Mac Storage"), so anything that reads it '
      + 'stalls, with no output, until all of it downloads again. Download it first:',
      h('pre', { class: 'cmd' }, `brctl download '${d.source_path}'`),
      h('div', { class: 'row' },
        h('button', { class: 'ghost',
          onclick: () => copyText(`brctl download '${d.source_path}'`) }, 'Copy command'),
        h('button', { class: 'ghost', onclick: () => selectSession(d.id, true) },
          'Check again')));
  }
  if (problem) {
    return h('div', { class: 'notice bad' },
      h('b', {}, 'The video for this session is missing. '),
      d.source_path
        ? `${d.source_path} cannot be read; nothing can be reprocessed until it is back. `
          + 'If the data folder was moved here from another machine, put the video in '
          + `${S.meta.uploads_dir} and relink it.`
        : 'The session has no source.* link, so there is nothing to process.',
      d.source_path
        ? h('div', { class: 'row' },
            h('button', { class: 'ghost', onclick: () => startJob('relink', {}) },
              'Relink moved videos'))
        : null);
  }
  if (!pending.length) return null;
  const last = [...d.stages].reverse().find((s) => done.has(s.name));
  return h('div', { class: 'notice' },
    h('b', {}, `This session stopped after ${last ? last.name : 'nothing'}. `),
    `${pending.map((s) => s.name).join(', ')} ${pending.length === 1 ? 'has' : 'have'} not run. `,
    'Running the pipeline picks up where it left off — finished stages are skipped.',
    h('div', { class: 'row' },
      h('button', { class: 'primary', onclick: () => runPipeline(d) }, 'Run the pipeline'),
      h('button', {
        class: 'ghost', title: 'Ignore the cache and run every stage again',
        onclick: () => runPipeline(d, { force: true }),
      }, 'Rerun everything')));
}

function tabOverview(root, d) {
  const m = d.metadata || {}, c = d.counts, p = d.players || {};
  mount(root, unfinishedNotice(d));
  root.append(h('div', { class: 'panel' },
    h('div', { class: 'stats' },
      stat(dur(m.duration_s), 'duration'),
      stat(m.fps ? `${num(m.fps, 0)}${m.is_vfr ? '*' : ''}` : '–', m.is_vfr ? 'fps (variable)' : 'fps'),
      stat(m.resolution ? m.resolution.join('×') : '–', 'resolution'),
      stat(c.contacts, 'sounds found'),
      stat(c.self_confirmed, 'your hits'),
      stat(c.measured, 'measured swings'),
      stat(c.clips, 'clips'),
      stat(c.labels, 'voice labels')),
    actionRow(d)));

  root.append(h('div', { class: 'panel' }, h('h2', {}, 'Stages'),
    h('table', {}, h('tbody', {}, ...d.stages.map((s) =>
      h('tr', {},
        h('td', {}, `${s.number}. ${s.name}`, s.optional ? h('span', { class: 'tag' }, 'optional') : null),
        h('td', {}, h('span', { class: `pill ${cls(s.status)}` }, s.status)),
        h('td', { class: 'num' }, s.elapsed_s ? `${num(s.elapsed_s, 1)} s` : ''),
        h('td', { class: 'muted' }, s.finished_at || '')))))));

  if (p.me) {
    const hand = p.handedness || {};
    root.append(h('div', { class: 'panel' }, h('h2', {}, 'Players'),
      h('p', {}, `You are player ${p.me} (${p.me_reason}). Racket hand: ${hand.hand || '–'}`,
        hand.reason ? ` — ${hand.reason}.` : '.'),
      (p.sides || []).length
        ? h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'from'), h('th', {}, 'to'),
            h('th', {}, 'your side'), h('th', { class: 'num' }, 'windows'))),
            h('tbody', {}, ...p.sides.map((s) => h('tr', {},
              h('td', {}, clock(s.start)), h('td', {}, clock(s.end)),
              h('td', {}, s.side), h('td', { class: 'num' }, s.windows)))))
        : null,
      d.debug_files.includes('players.jpg')
        ? h('img', { class: 'thumb', src: fileURL(d.id, 'debug/players.jpg'), alt: 'player thumbnails' })
        : null));
  }

  if (Object.keys(d.strokes).length) {
    root.append(h('div', { class: 'panel' }, h('h2', {}, 'Strokes'),
      h('div', { class: 'stats' },
        ...Object.entries(d.strokes).map(([k, v]) => stat(v, k)))));
  }
  if (d.summary.length) {
    const byStroke = {};
    for (const r of d.summary) (byStroke[r.stroke_type] ||= []).push(r);
    root.append(h('div', { class: 'panel' }, h('h2', {}, 'Metric summary'),
      ...Object.entries(byStroke).map(([stroke, rows]) => h('div', {},
        h('h3', {}, stroke),
        h('div', { class: 'scroll' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'metric'),
          h('th', { class: 'num' }, 'n'), h('th', { class: 'num' }, 'mean'),
          h('th', { class: 'num' }, 'median'), h('th', { class: 'num' }, 'p10'),
          h('th', { class: 'num' }, 'p90'), h('th', {}, 'unit'))),
          h('tbody', {}, ...rows.map((r) => h('tr', {},
            h('td', {}, r.metric), h('td', { class: 'num' }, r.count),
            h('td', { class: 'num' }, num(r.mean)), h('td', { class: 'num' }, num(r.median)),
            h('td', { class: 'num' }, num(r.p10)), h('td', { class: 'num' }, num(r.p90)),
            h('td', { class: 'muted' }, r.unit || '')))))))))); 
  }
}

const stat = (v, k) => h('div', { class: 'stat' }, h('div', { class: 'v' }, String(v)),
  h('div', { class: 'k' }, k));

const fileURL = (id, rel) => `/files/data/sessions/${encodeURIComponent(id)}/${rel}`;

/* --- swings --- */

const SWING_COLUMNS = [
  ['swing_id', '#', true], ['t_contact', 'time', true], ['stroke_type', 'stroke', false],
  ['peak_wrist_speed', 'wrist speed', true], ['contact_height', 'height', true],
  ['contact_forward', 'forward', true], ['knee_flex_min', 'knee', true],
  ['elbow_angle_contact', 'elbow', true], ['outlier_score', 'outlier', true],
];

async function tabSwings(root, d) {
  if (!S.swings) {
    root.append(h('p', { class: 'empty' }, 'Loading…'));
    try {
      const r = await api(`/api/sessions/${encodeURIComponent(d.id)}/swings`);
      S.swings = r.swings; S.metricMeta = r.metrics;
    } catch (e) { S.swings = []; toast(e.message, 'bad'); }
    if (S.sel === d.id && S.tab === 'swings') renderDetail();
    return;
  }
  if (!S.swings.length) {
    root.append(h('p', { class: 'empty' },
      'No measured swings yet. Run the pipeline through the metrics stage.'));
    return;
  }
  const filters = { stroke: '', outliers: false, clips: false };
  const table = h('div');
  const detail = h('div');

  const draw = () => {
    let rows = S.swings.filter((s) =>
      (!filters.stroke || s.stroke_type === filters.stroke) &&
      (!filters.outliers || s.is_outlier) && (!filters.clips || s.clip));
    const { key, dir } = S.swingSort;
    rows = rows.slice().sort((a, b) => {
      const x = a[key], y = b[key];
      if (typeof x === 'string' || typeof y === 'string') {
        return dir * String(x).localeCompare(String(y));
      }
      return dir * ((x ?? -Infinity) - (y ?? -Infinity));
    });
    clear(table).append(
      h('div', { class: 'scroll' },
        h('table', {},
          h('thead', {}, h('tr', {},
            ...SWING_COLUMNS.map(([k, label, isNum]) => h('th', {
              class: isNum ? 'num' : '',
              onclick: () => {
                S.swingSort = { key: k, dir: S.swingSort.key === k ? -S.swingSort.dir : 1 };
                draw();
              },
            }, label, S.swingSort.key === k ? (S.swingSort.dir > 0 ? ' ▲' : ' ▼') : '')),
            h('th', {}, 'flags'))),
          h('tbody', {}, ...rows.map((s) => h('tr', {
            class: s.swing_id === S.swingSel ? 'on' : '',
            onclick: () => { S.swingSel = s.swing_id; draw(); showSwing(detail, s, d); },
          },
            h('td', { class: 'num' }, s.swing_id),
            h('td', { class: 'num' }, clock(s.t_contact)),
            h('td', {}, s.stroke_type, s.two_handed ? h('span', { class: 'tag' }, '2h') : null),
            ...SWING_COLUMNS.slice(3).map(([k]) => h('td', { class: 'num' }, num(s[k]))),
            h('td', {},
              s.is_outlier ? h('span', { class: 'tag outlier' }, 'outlier') : null,
              s.clip ? h('span', { class: 'tag' }, 'clip') : null,
              ...(s.labels || []).map((l) => h('span', { class: 'tag label' }, l))))))),
      ),
      h('p', { class: 'hint' }, `${rows.length} of ${S.swings.length} swings. Click one to see it.`));
  };

  const strokes = [...new Set(S.swings.map((s) => s.stroke_type))].sort();
  root.append(h('div', { class: 'panel' },
    h('div', { class: 'row' },
      h('select', { onchange: (e) => { filters.stroke = e.target.value; draw(); } },
        h('option', { value: '' }, 'all strokes'),
        ...strokes.map((s) => h('option', { value: s }, s))),
      h('label', { class: 'field bool' },
        h('input', { type: 'checkbox', onchange: (e) => { filters.outliers = e.target.checked; draw(); } }),
        ' outliers only'),
      h('label', { class: 'field bool' },
        h('input', { type: 'checkbox', onchange: (e) => { filters.clips = e.target.checked; draw(); } }),
        ' with a clip')),
    table), detail);
  draw();
  const chosen = S.swings.find((s) => s.swing_id === S.swingSel);
  if (chosen) showSwing(detail, chosen, d);
}

function showSwing(root, swing, d) {
  const peers = S.swings.filter((s) => s.stroke_type === swing.stroke_type);
  const median = (key) => {
    const xs = peers.map((s) => s[key]).filter((v) => v !== null && isFinite(v)).sort((a, b) => a - b);
    if (!xs.length) return null;
    const mid = xs.length >> 1;
    return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
  };
  clear(root).append(h('div', { class: 'panel' },
    h('h2', {}, `Swing ${swing.swing_id}`, ' ',
      h('span', { class: 'muted' },
        `${swing.stroke_type}${swing.two_handed ? ', two-handed' : ''} at ${clock(swing.t_contact)}`)),
    (swing.clip_reasons || []).length
      ? h('p', { class: 'hint' }, `Clipped because: ${swing.clip_reasons.join(', ')}`) : null,
    swing.clip
      ? h('video', { src: fileURL(d.id, `clips/${swing.clip}`), controls: true, loop: true,
                     preload: 'metadata', style: 'max-height:46vh' })
      : h('p', { class: 'hint' },
          'No clip for this swing. Rebuilding the report renders clips for outliers, the most '
          + 'typical swing of each stroke type, and labelled swings.'),
    h('div', { class: 'scroll' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'metric'), h('th', { class: 'num' }, 'value'),
        h('th', {}, 'unit'), h('th', { class: 'num' }, `median ${swing.stroke_type}`),
        h('th', { class: 'num' }, 'delta'))),
      h('tbody', {}, ...S.metricMeta.filter((m) => swing[m.name] !== null
          && swing[m.name] !== undefined).map((m) => {
        const med = median(m.name), delta = med === null ? null : swing[m.name] - med;
        return h('tr', { title: m.description },
          h('td', {}, m.name), h('td', { class: 'num' }, num(swing[m.name])),
          h('td', { class: 'muted' }, m.unit || ''), h('td', { class: 'num' }, num(med)),
          h('td', { class: 'num' }, delta === null ? '–' : (delta >= 0 ? '+' : '') + num(delta)));
      }))))));
}

/* --- report / review / log --- */

function tabReport(root, d) {
  if (!d.has_report) {
    const ready = ranStages(d).has('metrics');
    mount(root, unfinishedNotice(d), h('div', { class: 'panel' },
      h('p', {}, ready
        ? 'This session has no report yet.'
        : 'A report needs the metrics stage, which has not run for this session yet.'),
      h('div', { class: 'row' },
        ready
          ? h('button', { class: 'primary',
              onclick: () => startJob('report', { session_id: d.id }) }, 'Build it')
          : h('button', { class: 'primary', onclick: () => runPipeline(d) },
              'Run the pipeline'))));
    return;
  }
  const url = fileURL(d.id, 'report.html');
  root.append(h('div', { class: 'row' },
    h('a', { href: url, target: '_blank', class: 'ghost',
      style: 'text-decoration:none;padding:.45rem .8rem;border:1px solid var(--line);border-radius:8px' },
      'Open in a new tab ↗'),
    h('button', { class: 'ghost', onclick: () => startJob('report', { session_id: d.id }) },
      'Rebuild'),
    h('button', { class: 'ghost', onclick: () => startJob('report', { session_id: d.id, no_clips: true }) },
      'Rebuild HTML only')));
  root.append(h('iframe', { class: 'report', src: url }));
}

function tabReview(root, d) {
  const has = (name) => d.debug_files.includes(name);
  root.append(h('div', { class: 'panel' }, h('h2', {}, 'Tracking review'),
    h('p', { class: 'hint' },
      'The preview video draws the skeleton on sampled swings, with the racket side in orange and '
      + 'a red border on contact frames.'),
    h('div', { class: 'row' },
      h('button', { class: 'primary',
        onclick: () => startJob('pose-preview', { session_id: d.id, count: '20', speed: '0.5' }) },
        'Render pose preview'),
      h('button', { class: 'ghost',
        onclick: () => startJob('swing-plots', { session_id: d.id, count: '6' }) },
        'Plot trajectories'),
      h('button', { class: 'ghost',
        onclick: () => startJob('players', { session_id: d.id }) }, 'Refresh players')),
    has('pose_preview.mp4')
      ? h('video', { src: fileURL(d.id, 'debug/pose_preview.mp4'), controls: true,
                     preload: 'metadata', style: 'margin-top:1rem;max-height:60vh' })
      : h('p', { class: 'hint' }, 'No preview rendered yet.'),
    has('swing_trajectories.html')
      ? h('p', {}, h('a', { href: fileURL(d.id, 'debug/swing_trajectories.html'), target: '_blank' },
          'Open the swing trajectory plots ↗'))
      : null));

  root.append(h('div', { class: 'panel' }, h('h2', {}, 'Files'),
    h('ul', {}, ...d.debug_files.map((f) =>
      h('li', {}, h('a', { href: fileURL(d.id, `debug/${f}`), target: '_blank' }, `debug/${f}`))),
      d.has_report
        ? h('li', {}, h('a', { href: fileURL(d.id, 'report.html'), target: '_blank' }, 'report.html'))
        : null)));
}

function tabLog(root, d) {
  root.append(h('div', { class: 'panel' },
    h('h2', {}, 'pipeline.log', ' ', h('span', { class: 'muted' }, 'last 200 lines')),
    h('pre', { class: 'console-out', style: 'height:60vh;border-radius:8px' },
      d.log_lines.join('\n') || 'nothing logged yet')));
}

/* ---------- config ---------- */

async function loadConfig() {
  let cfg;
  try { cfg = await api('/api/config'); } catch (e) { return; }
  $('#config-path').textContent = cfg.exists
    ? cfg.path
    : `no config file yet — the defaults are in use, and saving creates ${cfg.would_create}`;
  $('#config-text').value = cfg.text;
  $('#config-save').textContent = cfg.exists ? 'Save' : 'Create config.yaml';
  const status = $('#config-status');
  status.className = `status ${cfg.error ? 'bad' : ''}`;
  status.textContent = cfg.error || '';
}

function wireConfig() {
  $('#config-reload').onclick = loadConfig;
  $('#config-save').onclick = async () => {
    const status = $('#config-status');
    try {
      await postJSON('/api/config', { text: $('#config-text').value });
      status.className = 'status ok';
      status.textContent = 'saved';
      refreshSessions();  // stage staleness depends on the config
    } catch (e) {
      status.className = 'status bad';
      status.textContent = e.message;
    }
  };
}

boot().catch((e) => {
  document.body.prepend(h('div', { class: 'panel', style: 'margin:1rem;border-color:var(--bad)' },
    h('h2', {}, 'The UI could not start'), h('pre', {}, String(e))));
});

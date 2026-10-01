import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  MODEL_URL, VESSELS, VESSEL_GROUPS, NEUTRAL_VESSEL_GROUPS, VESSEL_INFO, GROUPS, REGIONS, prettyName,
} from '/anatomy.js';

/* ---------- helpers ---------- */
const $ = (s) => document.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) e.append(kid instanceof Node ? kid : document.createTextNode(kid ?? ''));
  return e;
};
const svgEl = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  kids.flat().forEach((k) => e.append(k instanceof Node ? k : document.createTextNode(k)));
  return e;
};
const riskColor = (p) => new THREE.Color().setHSL((120 * (1 - p)) / 360, 0.7, 0.45, THREE.SRGBColorSpace);   // same as CSS hsl(), so cards, legend and 3D agree
const riskCss = (p) => `#${riskColor(p).getHexString()}`;
const pct = (p) => `${(p * 100).toFixed(0)}%`;
const fmt = (v) => (v == null ? 'blank' : typeof v === 'number' ? String(+v.toFixed(2)) : String(v));
const nice = (s) => s.replace(/_/g, ' ');
const TARGET_COLORS = { CAD: '#0f6b7a', LAD: '#c2410c', LCX: '#7c3aed', RCA: '#1d6fb8' };

/* ---------- state ---------- */
const S = { schema: null, values: {}, result: null, target: 'CAD', region: null, showAll: false, seq: 0, timer: null,
  samples: [], importance: null, impTarget: 'CAD' };
const vessels = Object.fromEntries(VESSELS.map((n) => [n, { meshes: [], mat: null, label: null, anchor: new THREE.Vector3() }]));
const regionMeshes = {};              // region key -> meshes
const pickables = [];                 // meshes that respond to hover / click
let heartPivot = null; const heartCentre = new THREE.Vector3(0.25, 0.6, 0.3);
let dirty = true, frames = 0, modelReady = false;

/* ================= 3D scene ================= */
const viewer = $('#viewer');
/* Lite mode (software WebGL, or ?lite=1): no antialiasing, lower resolution, no environment map, no idle animation. */
function softwareGL() {
  try {
    const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl');
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    return /swiftshader|llvmpipe|softpipe|software/i.test(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '');
  } catch { return false; }
}
const LITE = softwareGL() || new URLSearchParams(location.search).has('lite');
const renderer = new THREE.WebGLRenderer({ antialias: !LITE, alpha: true, powerPreference: 'default' });
const BASE_RATIO = LITE ? 0.7 : Math.min(window.devicePixelRatio, 2);
function setRatio(r) { renderer.setPixelRatio(r); renderer.setSize(viewer.clientWidth, viewer.clientHeight, false); dirty = true; }
renderer.setPixelRatio(BASE_RATIO);
viewer.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(36, 1, 0.05, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 0.9;
controls.maxDistance = 9;
controls.addEventListener('change', () => { dirty = true; });
let restoreTimer = null;                      // render at lower resolution while the user drags, then sharpen
controls.addEventListener('start', () => { clearTimeout(restoreTimer); setRatio(BASE_RATIO * 0.6); });
controls.addEventListener('end', () => { restoreTimer = setTimeout(() => setRatio(BASE_RATIO), 180); });
scene.add(new THREE.HemisphereLight(0xdfeaff, 0x1a2430, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.7); key.position.set(2, 3, 5); scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 0.7); rim.position.set(-4, 2, -4); scene.add(rim);
const pmrem = new THREE.PMREMGenerator(renderer);
if (!LITE) scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const skeleton = new THREE.Group(); scene.add(skeleton);
const skelMats = {};
const heartMat = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.05, emissive: 0x000000, envMapIntensity: 0.8 });
vessels.LAD.mat = heartMat(0x9aa7b2); vessels.LCX.mat = heartMat(0x9aa7b2); vessels.RCA.mat = heartMat(0x9aa7b2);
const neutralMat = heartMat(0xb9c2ca);

function parseNode(mesh) {
  const raw = mesh.name.includes('|') ? mesh.name : (mesh.parent?.name || '');
  const [group, id] = raw.split('|');
  return { group, id, raw };
}

function buildModel(gltf) {
  const root = gltf.scene;
  const heartParts = [], meshes = [];
  root.updateMatrixWorld(true);
  root.traverse((o) => { if (o.isMesh) meshes.push(o); });     // collect first: we re-parent below
  scene.add(root);
  for (const o of meshes) {
    const { group, id, raw } = parseNode(o);
    o.userData = { group, id, raw };
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();    // the GLB ships without normals
    const cfg = GROUPS[group];
    if (cfg?.kind === 'skeleton') {
      o.material = skelMats[group] ||= new THREE.MeshStandardMaterial({ color: cfg.color, roughness: 0.65, transparent: true,
        opacity: cfg.opacity, depthWrite: false, envMapIntensity: 0.4 });
      o.renderOrder = 2; skeleton.attach(o); continue;
    }
    if (VESSEL_GROUPS[group]) {
      const v = vessels[VESSEL_GROUPS[group]];
      o.material = v.mat; v.meshes.push(o); heartParts.push(o); pickables.push(o); continue;
    }
    if (NEUTRAL_VESSEL_GROUPS.includes(group)) { o.material = neutralMat; heartParts.push(o); pickables.push(o); continue; }
    if (cfg?.kind === 'heart') {
      o.material = heartMat(cfg.color);
      const rk = REGIONS[`${group}|${id}`] ? `${group}|${id}` : group;
      (regionMeshes[rk] ||= []).push(o); o.userData.region = rk;
      heartParts.push(o); pickables.push(o);
    }
  }
  // Heartbeat pivots about the heart's own centre, so the beat never moves it across the chest.
  const box = new THREE.Box3(); heartParts.forEach((m) => box.expandByObject(m));
  box.getCenter(heartCentre);
  heartPivot = new THREE.Group(); heartPivot.position.copy(heartCentre); scene.add(heartPivot); heartPivot.updateMatrixWorld(true);
  heartParts.forEach((m) => heartPivot.attach(m));
  // Label anchor = the vertex of the artery closest to its centroid, so labels and clicks land on the vessel itself.
  for (const n of VESSELS) {
    const pts = [];
    vessels[n].meshes.forEach((m) => { const pos = m.geometry.attributes.position; m.updateWorldMatrix(true, false);
      for (let i = 0; i < pos.count; i += 5) pts.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld)); });
    const mean = pts.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(pts.length);
    vessels[n].anchor.copy(pts.reduce((best, p) => (p.distanceToSquared(mean) < best.distanceToSquared(mean) ? p : best)));
  }
  setView('front', true);
  modelReady = true; dirty = true;
  $('#loading').hidden = true;
  if (S.result) render();
}

new GLTFLoader().load(MODEL_URL, buildModel, undefined, (e) => {
  $('#loading').textContent = `Could not load the 3D anatomy (${e?.message || 'network error'}).`;
  $('#loading').classList.add('error');
});

/* ---------- labels, tooltip ---------- */
const labels = $('#labels');
VESSELS.forEach((n) => { vessels[n].label = el('div', { class: 'vlabel' }, n); labels.append(vessels[n].label); });
const tip = $('#tooltip');

/* ---------- colouring ---------- */
function paintVessel(name, p, selected) {
  const m = vessels[name].mat, c = riskColor(p);
  m.color.copy(c); m.emissive.copy(c).multiplyScalar(selected ? 0.5 : 0.1);
  m.needsUpdate = true; dirty = true;
}
function highlightRegion(key) {
  for (const [k, meshes] of Object.entries(regionMeshes)) {
    meshes.forEach((m) => m.material.emissive.setHex(k === key ? 0x2a4a6a : 0x000000));
  }
  dirty = true;
}

/* ---------- views & toggles ---------- */
function setView(name, instant = false) {
  const d = 3.1, c = heartCentre;
  const pos = { front: [0.15, 0.2, d], back: [0, 0.2, -d], left: [d, 0.2, 0.3] }[name] || [0.15, 0.2, d];
  camera.position.set(c.x + pos[0], c.y + pos[1], c.z + pos[2]);
  controls.target.copy(c); controls.update(); dirty = true;
}
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
const togg = (id, fn) => { const b = $(id); b.addEventListener('click', () => { const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', on); fn(on); }); };
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let beat = !reduceMotion && !LITE; $('#tg-beat').setAttribute('aria-pressed', String(beat));
togg('#tg-skel', (on) => { skeleton.visible = on; dirty = true; });
togg('#tg-beat', (on) => { beat = on; if (!on && heartPivot) heartPivot.scale.setScalar(1); dirty = true; });

/* ---------- picking ---------- */
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function hit(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const h = ray.intersectObjects(pickables, false)[0];
  return h ? h.object : null;
}
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) { down = null; return; }   // a drag, not a click
  down = null;
  const o = hit(e);
  if (!o) { S.region = null; selectTarget('CAD'); return; }
  const { group, region } = o.userData;
  if (VESSEL_GROUPS[group]) { S.region = null; selectTarget(VESSEL_GROUPS[group]); }
  else if (region) { S.region = region; render(); }
});
let moveQueued = false;
renderer.domElement.addEventListener('pointermove', (e) => {
  if (moveQueued || e.buttons) { if (e.buttons) tip.hidden = true; return; }
  moveQueued = true;
  requestAnimationFrame(() => {
    moveQueued = false;
    const o = hit(e), P = S.result?.predictions;
    if (!o) { tip.hidden = true; renderer.domElement.style.cursor = 'grab'; return; }
    const { group, region, raw } = o.userData;
    let text = prettyName(raw);
    if (VESSEL_GROUPS[group] && P) text += `\n${VESSEL_GROUPS[group]} segment · predicted stenosis ${pct(P[VESSEL_GROUPS[group]].probability)}`;
    else if (NEUTRAL_VESSEL_GROUPS.includes(group)) text += '\nNot a prediction target';
    else if (region) text = `${REGIONS[region].label}\nClick for details`;
    const r = viewer.getBoundingClientRect();
    tip.textContent = text; tip.style.left = `${e.clientX - r.left + 14}px`; tip.style.top = `${e.clientY - r.top + 14}px`;
    tip.hidden = false; renderer.domElement.style.cursor = 'pointer';
  });
});
renderer.domElement.addEventListener('pointerleave', () => { tip.hidden = true; });

/* ---------- resize & loop ---------- */
function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
}
new ResizeObserver(resize).observe(viewer); resize();
const tmp = new THREE.Vector3(), toCam = new THREE.Vector3(), out = new THREE.Vector3();
function placeLabels() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  for (const n of VESSELS) {
    const v = vessels[n];
    tmp.copy(v.anchor).project(camera);
    v.label.style.left = `${(tmp.x * 0.5 + 0.5) * w}px`; v.label.style.top = `${(-tmp.y * 0.5 + 0.5) * h}px`;
    toCam.copy(camera.position).sub(heartCentre).normalize(); out.copy(v.anchor).sub(heartCentre).normalize();
    v.label.style.opacity = modelReady ? (toCam.dot(out) > -0.15 ? 1 : 0.25) : 0;      // fade when the vessel faces away
  }
}
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  if (controls.update()) dirty = true;
  if (beat && heartPivot) {
    const t = clock.getElapsedTime(), ph = (t * 1.2) % 1;
    heartPivot.scale.setScalar(1 + 0.022 * Math.pow(Math.max(0, Math.sin(ph * Math.PI * 2)), 3)); dirty = true;
  }
  if (!dirty) return;                        // draw only when something changed: keeps idle CPU/GPU use near zero
  dirty = false; renderer.render(scene, camera); placeLabels(); frames++;
});

/* ================= dashboard ================= */
function selectTarget(t) { S.target = t; S.showAll = false; render(); }

function buildForm() {
  const form = $('#form'); form.replaceChildren();
  const groups = {};
  S.schema.features.forEach((f) => (groups[f.group] ||= []).push(f));
  Object.entries(groups).forEach(([g, feats], gi) => {
    const d = el('details', { open: gi === 0 }, el('summary', {}, g));
    feats.forEach((f) => {
      const id = `f-${f.key}`;
      let input;
      if (f.type === 'number') {
        input = el('input', { id, type: 'number', step: 'any', min: f.min, max: f.max, value: S.values[f.name] ?? '' });
        input.addEventListener('input', () => { S.values[f.name] = input.value === '' ? null : Number(input.value); schedule(); });
      } else {
        input = el('select', { id }, f.options.map((o) => el('option', { value: o }, String(o))));
        input.value = String(S.values[f.name]);
        input.addEventListener('change', () => { S.values[f.name] = typeof f.options[0] === 'number' ? Number(input.value) : input.value; schedule(); });
      }
      d.append(el('div', { class: 'field', title: `${f.name}: ${f.about}` },
        el('label', { for: id }, f.label, f.unit ? el('span', { class: 'unit' }, ` ${f.unit}`) : ''), input));
    });
    form.append(d);
  });
}
function setValues(vals) {
  S.values = Object.fromEntries(S.schema.features.map((f) => [f.name, vals[f.name] ?? f.default]));
  buildForm(); schedule(0);
}
const schedule = (ms = 350) => { clearTimeout(S.timer); S.timer = setTimeout(predict, ms); };

async function predict() {
  const seq = ++S.seq;
  try {
    const r = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: S.values }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    if (seq !== S.seq) return;          // a newer request superseded this one
    S.result = data; render();
  } catch (e) {
    $('#explain').replaceChildren(el('p', { class: 'warn' }, `Prediction failed: ${e.message}`));
  }
}

const featList = (items) => items.length
  ? items.map((x) => `${x.label} (${fmt(x.value)}${x.unit ? ' ' + x.unit : ''})`).join(', ') : 'nothing notable';

function regionCard(P) {
  const card = $('#region-card');
  if (!S.region) { card.hidden = true; return; }
  const r = REGIONS[S.region];
  const chips = r.vessels.map((v) => el('button', { type: 'button', class: 'chip btn', style: `--c:${riskCss(P[v].probability)}`,
    onclick: () => { S.region = null; selectTarget(v); } }, `${v} ${pct(P[v].probability)}`));
  const extra = (r.features || []).map((f) => {
    const c = P[S.target].explanation.contributions.find((x) => x.feature === f);
    return c ? el('li', {}, `${c.label}: ${fmt(c.value)}${c.unit ? ' ' + c.unit : ''} (${c.share >= 0 ? '+' : '\u2212'}${Math.abs(c.share).toFixed(1)}% of the ${S.target} explanation)`) : null;
  }).filter(Boolean);
  card.hidden = false;
  card.replaceChildren(
    el('div', { class: 'rc-head' }, el('h2', {}, r.label), el('button', { type: 'button', class: 'ghost', onclick: () => { S.region = null; render(); } }, 'Close')),
    el('p', { class: 'small' }, r.text),
    r.vessels.length ? el('div', { class: 'rc-chips' }, el('span', { class: 'muted small' }, 'Supplying arteries (predicted stenosis):'), chips) : '',
    extra.length ? el('ul', { class: 'small rc-list' }, extra) : '');
}

function render() {
  const R = S.result; if (!R) return;
  const P = R.predictions;
  if (modelReady) { VESSELS.forEach((n) => { paintVessel(n, P[n].probability, S.target === n); vessels[n].label.classList.toggle('sel', S.target === n); }); highlightRegion(S.region); }
  regionCard(P);

  $('#vessel-cards').replaceChildren(...VESSELS.map((n) => el('button', { class: 'vcard', type: 'button',
    'aria-pressed': String(S.target === n), style: `--c:${riskCss(P[n].probability)}`, title: VESSEL_INFO[n],
    onclick: () => { S.region = null; selectTarget(n); } },
  el('span', {}, n), el('b', {}, pct(P[n].probability)),
    el('span', {}, `${P[n].label ? 'Stenosis predicted' : 'No stenosis predicted'} (cut-off ${pct(P[n].threshold)})`),
    el('span', {}, `Display band: ${P[n].risk_band}`))));

  const c = P.CAD;
  $('#overall').replaceChildren(el('div', { class: 'overall' },
    el('div', { class: 'pct' }, pct(c.probability)),
    el('div', {}, el('h2', {}, 'Overall CAD probability'),
      el('span', { class: 'chip', style: `--c:${riskCss(c.probability)}` }, `${c.risk_band} model risk`),
      el('div', { class: 'muted small' }, `${c.label ? 'CAD-positive' : 'CAD-negative'} at this model\u2019s ${pct(c.threshold)} decision threshold`))));
  $('#warnings').replaceChildren(...R.warnings.map((w) => el('div', { class: 'warn' }, w)));

  const t = P[S.target], ex = t.explanation;
  const rows = S.showAll ? ex.contributions : ex.contributions.slice(0, 12);
  const max = Math.max(...ex.contributions.map((x) => Math.abs(x.contribution)), 1e-9);
  const title = S.target === 'CAD' ? 'Overall CAD' : `${S.target} stenosis`;
  $('#explain').replaceChildren(
    el('div', { class: 'seg' }, ['CAD', ...VESSELS].map((n) =>
      el('button', { type: 'button', 'aria-pressed': String(S.target === n), onclick: () => { S.region = null; selectTarget(n); } }, n))),
    el('h2', {}, `Why the model predicts ${pct(t.probability)} for ${title}`),
    el('p', { class: 'summary' },
      el('b', {}, 'Pushing the risk up: '), featList(t.summary.raises), '. ',
      el('b', {}, 'Pulling it down: '), featList(t.summary.lowers), '.'),
    el('p', { class: 'muted small' }, S.target !== 'CAD' ? `${VESSEL_INFO[S.target]}. ` : '',
      `Bars are SHAP contributions of the ${nice(t.model)} model, in ${ex.units} (average output ${ex.base_value.toFixed(2)}). `,
      'The percentage is each measurement\u2019s share of the total absolute contribution. The probability shown is calibrated; the bars explain the underlying score.'),
    el('div', { class: 'key' }, el('span', {}, '\u2190 lowers risk'), el('span', {}, 'raises risk \u2192')),
    ...rows.map((x) => {
      const w = (Math.abs(x.contribution) / max) * 50;
      const side = x.contribution >= 0 ? `left:50%;width:${w}%;background:var(--pos)` : `right:50%;width:${w}%;background:var(--neg)`;
      return el('div', { class: 'contrib', title: `${x.feature}: ${x.about}` },
        el('div', { class: 'name' }, x.label, el('div', { class: 'val' }, `${fmt(x.value)}${x.unit ? ' ' + x.unit : ''}`)),
        el('div', { class: 'bar' }, el('i', { style: side })),
        el('div', { class: 'pc' }, `${x.share >= 0 ? '+' : '\u2212'}${Math.abs(x.share).toFixed(1)}%`));
    }),
    el('button', { class: 'more', type: 'button', onclick: () => { S.showAll = !S.showAll; render(); } },
      S.showAll ? 'Show top 12 only' : `Show all ${ex.contributions.length} measurements`));
}

/* ---------- performance tab ---------- */
const f3 = (v) => (v == null ? '\u2013' : v.toFixed(3));
function chart(title, W, curves, diag, xl, yl) {
  const m = { l: 34, r: 8, t: 8, b: 30 }, w = W - m.l - m.r, h = W - m.t - m.b;
  const X = (v) => m.l + v * w, Y = (v) => m.t + (1 - v) * h;
  const g = svgEl('svg', { viewBox: `0 0 ${W} ${W}`, class: 'chart', role: 'img', 'aria-label': title });
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    g.append(svgEl('line', { x1: X(0), x2: X(1), y1: Y(t), y2: Y(t), class: 'grid' }), svgEl('line', { x1: X(t), x2: X(t), y1: Y(0), y2: Y(1), class: 'grid' }));
    g.append(svgEl('text', { x: X(0) - 4, y: Y(t) + 3, class: 'tick', 'text-anchor': 'end' }, t.toFixed(t % 0.5 ? 2 : 1)));
    g.append(svgEl('text', { x: X(t), y: Y(0) + 12, class: 'tick', 'text-anchor': 'middle' }, t.toFixed(t % 0.5 ? 2 : 1)));
  });
  if (diag) g.append(svgEl('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), class: 'diag' }));
  curves.forEach((c) => {
    const d = c.x.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(c.y[i]).toFixed(1)}`).join('');
    g.append(svgEl('path', { d, fill: 'none', stroke: c.color, 'stroke-width': 2 }));
    if (c.dots) c.x.forEach((x, i) => g.append(svgEl('circle', { cx: X(x), cy: Y(c.y[i]), r: 3, fill: c.color })));
  });
  g.append(svgEl('text', { x: X(0.5), y: W - 4, class: 'axis', 'text-anchor': 'middle' }, xl));
  g.append(svgEl('text', { x: 9, y: Y(0.5), class: 'axis', 'text-anchor': 'middle', transform: `rotate(-90 9 ${Y(0.5)})` }, yl));
  return g;
}
async function loadMetrics() {
  const [{ metrics, thresholds, protocol }, info] = await Promise.all([fetch('/api/metrics').then((r) => r.json()), fetch('/api/model-info').then((r) => r.json())]);
  const T = Object.keys(metrics);
  const ROWS = [['roc_auc', 'ROC-AUC'], ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['specificity', 'Specificity'], ['f1', 'F1'], ['brier', 'Brier (lower is better)'], ['ece', 'Calibration error']];
  const legend = el('div', { class: 'legend2' }, T.map((t) => el('span', {}, el('i', { style: `background:${TARGET_COLORS[t]}` }), t)));
  $('#view-perf').replaceChildren(
    el('h2', {}, 'Cross-validated performance'),
    el('p', { class: 'muted small' }, `All ${metrics.CAD.n} patients; every prediction below comes from a model that never saw that patient. ${protocol.outer} outer CV, ${protocol.inner} for model selection, ${protocol.calibration}. Values are means with 95% bootstrap intervals. Each target has its own decision threshold (${T.map((t) => `${t} ${pct(thresholds[t])}`).join(', ')}), chosen on other patients\u2019 out-of-fold predictions.`),
    el('div', { class: 'tscroll' }, el('table', { class: 'metrics' },
      el('thead', {}, el('tr', {}, el('th', {}, ''), T.map((t) => el('th', {}, t)))),
      el('tbody', {},
        el('tr', { class: 'modelrow' }, el('td', {}, 'Model'), T.map((t) => el('td', {}, nice(metrics[t].model)))),
        ROWS.map(([k, label]) => el('tr', {}, el('td', {}, label), T.map((t) => { const m = metrics[t].metrics[k];
          return el('td', {}, f3(m.mean), el('div', { class: 'ci' }, `${m.ci[0].toFixed(2)}\u2013${m.ci[1].toFixed(2)}`)); })))))),
    el('div', { class: 'charts' },
      el('figure', {}, chart('ROC curves', 230, T.map((t) => ({ x: metrics[t].roc.fpr, y: metrics[t].roc.tpr, color: TARGET_COLORS[t] })), true, 'False positive rate', 'True positive rate'), el('figcaption', {}, 'ROC curves')),
      el('figure', {}, chart('Calibration', 230, T.map((t) => ({ x: metrics[t].calibration.mean_pred, y: metrics[t].calibration.frac_pos, color: TARGET_COLORS[t], dots: true })), true, 'Predicted probability', 'Observed frequency'), el('figcaption', {}, 'Calibration (closer to the diagonal is better)'))),
    legend,
    el('h2', {}, 'Choosing a threshold'),
    el('p', { class: 'muted small' }, 'Lower thresholds find more disease but raise false alarms. Recall / specificity at each threshold:'),
    el('div', { class: 'tscroll' }, el('table', { class: 'metrics' },
      el('thead', {}, el('tr', {}, el('th', {}, 'Target'), ...[0.3, 0.4, 0.5, 0.6, 0.7].map((x) => el('th', {}, x)))),
      el('tbody', {}, T.map((t) => el('tr', {}, el('td', {}, t), ...metrics[t].thresholds.map((r) => el('td', {}, `${(r.recall * 100).toFixed(0)} / ${(r.specificity * 100).toFixed(0)}`))))))),
    el('h2', {}, 'Confusion matrices (average over repeats)'),
    ...T.map((t) => { const [[tn, fp], [fn, tp]] = metrics[t].confusion_matrix; return el('p', { class: 'small' }, `${t}: TN ${tn} \u00b7 FP ${fp} \u00b7 FN ${fn} \u00b7 TP ${tp} (n=${metrics[t].n})`); }),
    el('p', { class: 'muted small' }, `LAD, LCX, RCA, Cath and CAD are never used as inputs. LCX and RCA are only moderately predictable from these measurements. Model: scikit-learn ${info.versions['scikit-learn']}, trained ${info.created.slice(0, 10)} on ${info.data.file} (SHA-256 ${info.data.sha256.slice(0, 10)}\u2026).`),
    info.version_note ? el('div', { class: 'warn' }, info.version_note) : '');
}

/* ---------- importance tab ---------- */
function renderImportance() {
  const rows = S.importance[S.impTarget].slice(0, 15), max = rows[0].share;
  const about = Object.fromEntries(S.schema.features.map((f) => [f.name, f.about]));
  $('#view-imp').replaceChildren(
    el('h2', {}, 'What matters most overall'),
    el('p', { class: 'muted small' }, 'Average absolute SHAP contribution across 150 patients, as a share of the total. It shows which measurements the model leans on in general; the Prediction tab shows what drove one patient\u2019s result.'),
    el('div', { class: 'seg' }, ['CAD', ...VESSELS].map((n) => el('button', { type: 'button', 'aria-pressed': String(S.impTarget === n), onclick: () => { S.impTarget = n; renderImportance(); } }, n))),
    ...rows.map((r) => el('div', { class: 'imp', title: about[r.feature] || '' },
      el('div', { class: 'name' }, r.label), el('div', { class: 'ibar' }, el('i', { style: `width:${(r.share / max) * 100}%` })), el('div', { class: 'pc' }, `${r.share.toFixed(1)}%`))));
}

/* ---------- tabs ---------- */
const TABS = ['pred', 'perf', 'imp'];
const tab = (on) => TABS.forEach((t) => { $(`#tab-${t}`).setAttribute('aria-selected', String(t === on)); $(`#view-${t}`).hidden = t !== on; });
TABS.forEach((t) => { $(`#tab-${t}`).onclick = () => tab(t); });

/* ---------- boot ---------- */
const ready = (async function init() {
  try {
    const [schema, samples] = await Promise.all([fetch('/api/schema'), fetch('/api/samples')]);
    if (!schema.ok) throw new Error((await schema.json()).detail);
    S.schema = await schema.json();
    S.samples = await samples.json();
    $('#synthetic-banner').hidden = !S.schema.is_synthetic;
    const sel = $('#sample-select');
    sel.append(el('option', { value: '' }, 'Load an example patient from the dataset'));
    S.samples.forEach((s, i) => sel.append(el('option', { value: i }, `Patient #${s.id} \u00b7 recorded CAD: ${s.truth.CAD ? 'yes' : 'no'}`)));
    sel.onchange = () => {
      if (sel.value === '') return;
      const s = S.samples[+sel.value], yn = (v) => (v == null ? '?' : v ? 'yes' : 'no');
      $('#sample-truth').textContent = 'Recorded: ' + ['CAD', ...VESSELS].map((t) => `${t} ${yn(s.truth[t])}`).join(' \u00b7 ') +
        '. Out-of-fold estimate (model that never saw this patient): ' + ['CAD', ...VESSELS].map((t) => `${t} ${pct(s.oof[t])}`).join(' \u00b7 ') +
        '. The live prediction is in-sample, so it can look better.';
      setValues(s.features);
    };
    $('#reset-btn').onclick = () => { sel.value = ''; $('#sample-truth').textContent = ''; setValues({}); };
    setValues({});
    loadMetrics();
    S.importance = await (await fetch('/api/importance')).json(); renderImportance();
  } catch (e) {
    $('#explain').replaceChildren(el('p', { class: 'warn' }, `Could not load the model: ${e.message}. Train it first: python -m cardio.train --data data/<file>`));
  }
})();

/* Small hook used by the automated browser test (tests/e2e). */
window.__cardio = {
  ready, state: S, tab,
  select: (t) => { S.region = null; selectTarget(t); },
  selectRegion: (k) => { S.region = k; render(); },
  vesselColours: () => Object.fromEntries(VESSELS.map((n) => [n, `#${vessels[n].mat.color.getHexString()}`])),
  info: () => ({ modelReady, frames, triangles: renderer.info.render.triangles, vesselMeshes: Object.fromEntries(VESSELS.map((n) => [n, vessels[n].meshes.length])), regions: Object.keys(regionMeshes) }),
  pickScreen: (name) => { const v = vessels[name]; tmp.copy(v.anchor).project(camera); const r = renderer.domElement.getBoundingClientRect();
    return [r.left + (tmp.x * 0.5 + 0.5) * r.width, r.top + (-tmp.y * 0.5 + 0.5) * r.height]; },
  setView, lite: LITE,
  /* Render n frames while orbiting; returns milliseconds per frame, with a pixel readback so the GPU work is included. */
  benchmark: (n = 12) => {
    const gl = renderer.getContext(), px = new Uint8Array(4), t0 = performance.now(), off = camera.position.clone().sub(heartCentre);
    for (let i = 0; i < n; i++) {
      off.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.05); camera.position.copy(heartCentre).add(off); camera.lookAt(heartCentre);
      renderer.render(scene, camera); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);   // readback forces the GPU to really finish
    }
    const ms = (performance.now() - t0) / n; setView('front'); return ms;
  },
};

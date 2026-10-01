// End-to-end test of the 3D viewer in headless Chromium with SOFTWARE WebGL (no GPU needed).
//   1. start the app:  uvicorn cardio.serve:app --port 8000
//   2. cd tests/e2e && npm install && npm test        (BASE_URL=http://localhost:8000 by default)
// Set CHROME_PATH to use an installed Chrome/Chromium instead of the bundled one.
import { mkdirSync } from 'node:fs';
import puppeteer from 'puppeteer-core';

const BASE = process.env.BASE_URL || 'http://localhost:8000';
const SHOTS = new URL('./shots/', import.meta.url).pathname;
mkdirSync(SHOTS, { recursive: true });
let failures = 0;
const check = (ok, msg, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}${extra ? '  ' + extra : ''}`); if (!ok) failures++; };

let exe = process.env.CHROME_PATH, args = [];
if (!exe) { const c = (await import('@sparticuz/chromium')).default; exe = await c.executablePath(); args = c.args; }
const browser = await puppeteer.launch({ executablePath: exe, headless: 'shell',
  args: [...args, '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 900 });
const errors = [], hosts = new Set();
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));
page.on('request', (r) => { const u = new URL(r.url()); if (u.protocol.startsWith('http')) hosts.add(u.host); });

const hsl = (h, s, l) => { const a = s * Math.min(l, 1 - l), f = (n) => { const k = (n + h / 30) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)]; };
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const expectedHex = (p) => hsl(120 * (1 - p), 0.7, 0.45).map(srgbToLinear).map((v) => v);   // Three stores linear values
const hexToLinear = (hex) => [1, 3, 5].map((i) => srgbToLinear(parseInt(hex.slice(i, i + 2), 16) / 255));

await page.goto(BASE, { waitUntil: 'load' });
await page.evaluate(() => window.__cardio.ready);
await page.waitForFunction(() => window.__cardio.info().modelReady && window.__cardio.state.result, { timeout: 90000 });
const info = await page.evaluate(() => window.__cardio.info());
check(info.vesselMeshes.LAD > 1 && info.vesselMeshes.LCX > 1 && info.vesselMeshes.RCA > 1, 'GLB loaded; each artery has several segments', JSON.stringify(info.vesselMeshes));
check(info.regions.includes('vent') && info.regions.includes('atria|FJ2439'), 'heart regions registered', info.regions.join(', '));
check([...hosts].every((h) => new URL(BASE).host === h), 'no requests to external hosts', [...hosts].join(', '));

// Colour <-> probability correspondence for two different patients.
const pick = async (i) => { await page.select('#sample-select', String(i)); await new Promise((r) => setTimeout(r, 1500));
  await page.waitForFunction(() => window.__cardio.state.result); await new Promise((r) => setTimeout(r, 600)); };
const snap = () => page.evaluate(() => ({ p: Object.fromEntries(['LAD', 'LCX', 'RCA'].map((v) => [v, window.__cardio.state.result.predictions[v].probability])), c: window.__cardio.vesselColours() }));
await pick(0); const a = await snap(); await page.screenshot({ path: `${SHOTS}01_patient0.png` });
await pick(1); const b = await snap(); await page.screenshot({ path: `${SHOTS}02_patient1.png` });
for (const [tag, s] of [['patient 0', a], ['patient 1', b]]) for (const v of ['LAD', 'LCX', 'RCA']) {
  const e = expectedHex(s.p[v]), g = hexToLinear(s.c[v]);
  check(e.every((x, i) => Math.abs(x - g[i]) < 0.02), `${tag}: ${v} colour matches ${(s.p[v] * 100).toFixed(0)}% probability`, s.c[v]);
}
check(['LAD', 'LCX', 'RCA'].some((v) => a.c[v] !== b.c[v]), 'colours change when the patient changes');

// Typing a value updates the prediction and the colours (real-time link).
const before = await snap();
await page.evaluate(() => { const i = document.querySelector('#f-age'); i.value = '85'; i.dispatchEvent(new Event('input', { bubbles: true })); });
await new Promise((r) => setTimeout(r, 1500)); const after = await snap();
check(JSON.stringify(before.p) !== JSON.stringify(after.p), 'editing Age re-runs the model and repaints', `LAD ${before.p.LAD.toFixed(2)} -> ${after.p.LAD.toFixed(2)}`);

// Canvas picking: scan a grid around each artery's anchor until a click selects it.
const state = (k) => page.evaluate((key) => window.__cardio.state[key], k);
// `test` must be a real function (evaluated in the page with `arg`); returns the screen point of the first click that passes.
async function findClick(test, arg, centre, radius = 90, step = 10) {
  for (let r = 0; r <= radius; r += step) for (let a = 0; a < 360; a += r ? 30 : 360) {
    const x = centre[0] + r * Math.cos((a * Math.PI) / 180), y = centre[1] + r * Math.sin((a * Math.PI) / 180);
    await page.mouse.click(x, y); await new Promise((res) => setTimeout(res, 60));
    if (await page.evaluate(test, arg)) return [x, y];
  } return null;
}
await page.evaluate(() => window.__cardio.setView('front'));
for (const v of ['LAD', 'RCA']) {
  await page.evaluate(() => window.__cardio.select('CAD'));
  const c = await page.evaluate((n) => window.__cardio.pickScreen(n), v);
  const hit = await findClick((n) => window.__cardio.state.target === n, v, c);
  check(!!hit, `clicking the ${v} artery in the 3D view selects it`, hit ? `at ${hit.map(Math.round)}` : '');
}
await page.screenshot({ path: `${SHOTS}03_selected_vessel.png` });
await page.evaluate(() => window.__cardio.select('CAD'));
const heart = await page.evaluate(() => window.__cardio.pickScreen('LAD'));
const rh = await findClick(() => window.__cardio.state.region !== null, null, [heart[0] - 60, heart[1] + 40], 140, 10);
check(!!rh, 'clicking the heart wall selects a region and shows its card', rh ? await state('region') : '');
await new Promise((r) => setTimeout(r, 400));
check(await page.evaluate(() => window.__cardio.state.region !== null && !document.querySelector('#region-card').hidden), 'region card is visible and stays visible', JSON.stringify(await page.evaluate(() => ({ region: window.__cardio.state.region, target: window.__cardio.state.target }))));
await page.screenshot({ path: `${SHOTS}04_region.png` });

// Other views and tabs.
for (const v of ['back', 'left']) { await page.evaluate((n) => window.__cardio.setView(n), v); await new Promise((r) => setTimeout(r, 500)); await page.screenshot({ path: `${SHOTS}05_view_${v}.png` }); }
await page.evaluate(() => window.__cardio.setView('front'));
await page.click('#tab-perf'); await new Promise((r) => setTimeout(r, 400));
check(await page.evaluate(() => document.querySelectorAll('#view-perf svg.chart').length === 2), 'performance tab shows ROC and calibration charts');
await page.screenshot({ path: `${SHOTS}06_performance.png` });
await page.click('#tab-imp'); await new Promise((r) => setTimeout(r, 400));
check(await page.evaluate(() => document.querySelectorAll('#view-imp .imp').length === 15), 'importance tab lists the top 15 measurements');
await page.screenshot({ path: `${SHOTS}07_importance.png` });
await page.click('#tab-pred');

// Rendering cost with software WebGL (CPU only): the worst case for "no dedicated GPU".
await page.evaluate(() => window.__cardio.select('CAD'));
const lite = await page.evaluate(() => window.__cardio.lite);
const ms = await page.evaluate(() => window.__cardio.benchmark(12));
const tris = await page.evaluate(() => window.__cardio.info().triangles);
console.log(`INFO  lite mode: ${lite}; ${ms.toFixed(0)} ms/frame (${(1000 / ms).toFixed(1)} fps) with software WebGL, ${tris.toLocaleString()} triangles per frame`);
check(ms < 400, 'scene stays interactive with CPU-only WebGL (< 400 ms/frame)', `${ms.toFixed(0)} ms`);

check(errors.length === 0, 'no console or page errors', errors.slice(0, 3).join(' | '));
await browser.close();
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll checks passed');
process.exit(failures ? 1 : 0);

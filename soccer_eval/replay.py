"""Writes a self-contained HTML replay for one match result."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Frame, MatchResult


def replay_data(result: MatchResult, labels: dict[str, str] | None = None) -> dict[str, Any]:
    frames: list[Frame] = []
    ticks: list[dict[str, Any]] = []
    for rec in result["log"]:
        ticks.append({
            "tick": rec["tick"], "f0": len(frames), "score": rec["score"],
            "events": rec["events"], "actions": rec["actions"],
        })
        frames.extend(rec["frames"])
    return {
        "config": result["config"], "teams": result["teams"], "labels": labels or {"A": "A", "B": "B"},
        "score": result["score"], "frames": frames, "ticks": ticks,
    }


def write_replay(result: MatchResult, path: Path, labels: dict[str, str] | None = None) -> None:
    data = json.dumps(replay_data(result, labels), separators=(",", ":")).replace("</", "<\\/")
    Path(path).write_text(TEMPLATE.replace("__DATA__", data))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soccer-Bench Replay</title>
<style>
/* A broadcast-style player: score banner, the pitch with video-player controls, and a live feed
   of every shout and key event beside it. Colours follow the site (light, or dark with the system).
   Every box has a fixed size, so nothing moves while a game plays. */
:root {
  --bg: #ffffff; --surface: #f6f7f4; --raised: #eceee9; --border: #dde2da; --text: #1b1e22; --muted: #5f6670;
  --banner: #2f8550; --pitch: #3f7d4f; --pitch-line: rgba(255,255,255,.75); --goal: rgba(255,255,255,.35);
  --team-a: #2f6fdb; --team-b: #e0692d; --ball: #ffffff; --ball-edge: #1b1e22; --pass: rgba(255,255,255,.9);
  --goal-row: rgba(47,133,80,.14);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171a; --surface: #1c2024; --raised: #262b30; --border: #2e3532; --text: #eceeea; --muted: #9aa19b;
    --team-a: #6c9cf0; --team-b: #f08a52; --ball-edge: #000000; --goal-row: rgba(47,133,80,.3);
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.embedded body { background: transparent; } .embedded main { padding: 0; max-width: none; }  /* inside the site's page */
.banner { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 12px;
  background: var(--banner); color: #fff; border-radius: 10px; padding: 10px 16px; }
.banner .team { display: flex; align-items: center; gap: 8px; min-width: 0; font-weight: 600; }
.banner .team span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.banner .team.b { justify-content: flex-end; text-align: right; }
.banner .dot { width: 12px; height: 12px; border-radius: 50%; flex: none; border: 2px solid rgba(255,255,255,.8); }
.banner .mid { text-align: center; font-variant-numeric: tabular-nums; }
.banner .score { font-size: 26px; font-weight: 700; line-height: 1.1; }
.banner .clock { font-size: 12px; opacity: .85; }
.stage { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 12px; }
.screen { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
canvas { display: block; width: 100%; height: auto; border-radius: 8px; }
.controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
button, select { font: inherit; color: var(--text); background: var(--raised); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer; }
button:hover, select:hover { border-color: var(--muted); }
#play { min-width: 64px; font-weight: 600; }
.timeline { position: relative; flex: 1 1 240px; height: 28px; display: flex; align-items: center; }
.timeline input { width: 100%; margin: 0; accent-color: var(--banner); cursor: pointer; }
.marker { position: absolute; top: calc(50% - 7px); width: 14px; height: 14px; margin-left: -7px; border-radius: 50%; padding: 0; border: 3px solid var(--banner); background: var(--bg); cursor: pointer; z-index: 1; }
.marker.minor { top: calc(50% + 7px); width: 4px; height: 6px; margin-left: -2px; border: 0; border-radius: 1px; background: var(--muted); z-index: 0; }
label { display: flex; align-items: center; gap: 5px; color: var(--muted); font-size: 13px; cursor: pointer; }
.feed { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; display: flex; flex-direction: column; min-height: 0; min-width: 0; }
@media (min-width: 901px) { .feed { contain: size; } }
.feed-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px 12px; border-bottom: 1px solid var(--border); }
.feed-head h2 { margin: 0; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
.filters { display: flex; gap: 10px; }
.msgs { flex: 1 1 0; min-height: 0; overflow-y: auto; padding: 6px 0; }
.msg { display: grid; grid-template-columns: 34px 26px minmax(0, 1fr); gap: 4px; padding: 3px 12px; font-size: 13px; cursor: pointer; }
.msg:hover { background: var(--raised); }
.msg .t { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; padding-top: 1px; }
.msg .who { font-weight: 700; }
.msg.a .who { color: var(--team-a); } .msg.b .who { color: var(--team-b); }
.msg .text { overflow-wrap: anywhere; }
.msg.ev .text { color: var(--muted); font-style: italic; }
.msg.goal { background: var(--goal-row); }
.msg.goal .text { color: var(--text); font-style: normal; font-weight: 700; }
.msg.now .t { color: var(--text); }
.msg.later { display: none; }
.hideA .msg.a, .hideB .msg.b, .hideEv .msg.ev { display: none; }
.empty { color: var(--muted); padding: 8px 12px; font-size: 13px; }
details { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
summary { padding: 8px 12px; cursor: pointer; color: var(--muted); font-size: 13px; }
.actions { height: 260px; overflow-y: auto; border-top: 1px solid var(--border); }
.row { display: grid; grid-template-columns: 44px minmax(0, 1fr); gap: 8px; padding: 6px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
.row .say { font-style: italic; overflow-wrap: anywhere; }
.row .err { color: var(--team-b); }
.tag { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.muted { color: var(--muted); }
@media (max-width: 900px) { .stage { grid-template-columns: 1fr; } .feed { height: 340px; } }
@media (max-width: 520px) { .banner { padding: 8px 10px; } .banner .score { font-size: 22px; } .banner .team { font-size: 13px; } }
</style>
</head>
<body>
<main>
  <div class="banner">
    <div class="team a"><span class="dot" style="background:var(--team-a)"></span><span id="nameA"></span></div>
    <div class="mid"><div class="score" id="score">0 – 0</div><div class="clock" id="clock"></div></div>
    <div class="team b"><span id="nameB"></span><span class="dot" style="background:var(--team-b)"></span></div>
  </div>
  <div class="stage">
    <div class="screen">
      <canvas id="pitch" aria-label="Match replay"></canvas>
      <div class="controls">
        <button id="play" title="Play or pause (space)">Play</button>
        <button id="prev" title="Previous tick (←)" aria-label="Previous tick">&#8592;</button>
        <button id="next" title="Next tick (→)" aria-label="Next tick">&#8594;</button>
        <div class="timeline" id="timeline" title="White dots are goals, grey ticks are ball wins"><input type="range" id="scrub" min="0" value="0" aria-label="Timeline"></div>
        <select id="speed" aria-label="Speed"><option value="0.5">0.5x</option><option value="1">1x</option><option value="2" selected>2x</option><option value="4">4x</option><option value="8">8x</option></select>
      </div>
      <div class="controls">
        <label><input type="checkbox" id="targets" checked> move targets</label>
        <label><input type="checkbox" id="shouts"> shouts on pitch</label>
      </div>
    </div>
    <div class="feed">
      <div class="feed-head">
        <h2>Live feed</h2>
        <div class="filters">
          <label><input type="checkbox" data-hide="hideA" checked><span class="tag" style="background:var(--team-a)"></span>A</label>
          <label><input type="checkbox" data-hide="hideB" checked><span class="tag" style="background:var(--team-b)"></span>B</label>
          <label><input type="checkbox" data-hide="hideEv" checked>events</label>
        </div>
      </div>
      <div class="msgs" id="feed" aria-live="polite"></div>
    </div>
  </div>
  <details>
    <summary id="actionsTitle">Every player's action this tick</summary>
    <div class="actions" id="actions"></div>
  </details>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
if (window.self !== window.top) document.documentElement.classList.add('embedded');
const D = JSON.parse(document.getElementById('data').textContent);
const C = D.config, F = D.frames, T = D.ticks, N = C.n_per_team;
// Per-frame lookups: which tick a frame belongs to, and the score shown on it (a goal
// only counts from the frame where the ball actually went in).
const frameTick = new Array(F.length), frameScore = new Array(F.length);
// Frames where the ball crosses the line: playback holds there so a goal doesn't flash
// straight into the kickoff reset.
const goalFrames = new Map();
const GOAL_HOLD_MS = 500;
T.forEach((t, k) => {
  const end = k + 1 < T.length ? T[k + 1].f0 : F.length;
  const before = k > 0 ? T[k - 1].score : [0, 0];
  const goal = t.events.find(e => e.type === 'goal');
  const goalAt = goal ? t.f0 + (goal.frame ?? end - t.f0 - 1) : Infinity;
  if (goal) goalFrames.set(Math.min(goalAt, end - 1), goal);
  for (let i = t.f0; i < end; i++) { frameTick[i] = k; frameScore[i] = i >= goalAt ? t.score : before; }
});
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const names = [...Array(2 * N)].map((_, g) => (g < N ? 'A' : 'B') + (g % N));
for (const s of ['A', 'B']) {
  const specs = D.teams[s];
  $('name' + s).textContent = D.labels[s] || s;
  $('name' + s).title = specs.every(x => x === specs[0]) ? specs[0] : specs.join(', ');
}
const canvas = $('pitch'), ctx = canvas.getContext('2d');
const scrub = $('scrub'); scrub.max = F.length - 1;
// `pos` is the fractional playback position; draw() always renders the whole frame at idx().
let pos = 0, playing = false, last = 0, holdUntil = 0, restart = false;
const idx = () => Math.max(0, Math.min(F.length - 1, Math.floor(pos)));
const css = () => { const s = getComputedStyle(document.documentElement); return k => s.getPropertyValue(k).trim(); };
const jump = k => { pause(); pos = T[k].f0; draw(); };

function resize() {
  const w = canvas.clientWidth, pad = 4, h = w * (C.height + 2 * pad) / (C.width + 2 * pad);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = h + 'px';
  draw();
}
function draw() {
  const v = css(), dpr = window.devicePixelRatio || 1, pad = 4;
  const s = canvas.width / (C.width + 2 * pad);
  const X = x => (x + pad) * s, Y = y => (y + pad) * s;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = v('--pitch'); ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = v('--pitch-line'); ctx.lineWidth = Math.max(1, 1.2 * dpr);
  ctx.strokeRect(X(0), Y(0), C.width * s, C.height * s);
  ctx.beginPath(); ctx.moveTo(X(C.width / 2), Y(0)); ctx.lineTo(X(C.width / 2), Y(C.height)); ctx.stroke();
  ctx.beginPath(); ctx.arc(X(C.width / 2), Y(C.height / 2), 9 * s, 0, 2 * Math.PI); ctx.stroke();
  const by0 = (C.height - C.box_width) / 2, gy0 = (C.height - C.goal_width) / 2;
  ctx.strokeRect(X(0), Y(by0), C.box_depth * s, C.box_width * s);
  ctx.strokeRect(X(C.width - C.box_depth), Y(by0), C.box_depth * s, C.box_width * s);
  ctx.fillStyle = v('--goal');
  ctx.fillRect(X(-3), Y(gy0), 3 * s, C.goal_width * s);
  ctx.fillRect(X(C.width), Y(gy0), 3 * s, C.goal_width * s);
  ctx.lineWidth = Math.max(2, 3 * dpr);
  ctx.beginPath(); ctx.moveTo(X(0), Y(gy0)); ctx.lineTo(X(0), Y(gy0 + C.goal_width)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(X(C.width), Y(gy0)); ctx.lineTo(X(C.width), Y(gy0 + C.goal_width)); ctx.stroke();

  const fi = idx(), f = F[fi], k = frameTick[fi], tk = T[k];
  if ($('targets').checked && tk.actions.length) {
    const start = F[Math.max(0, tk.f0 - 1)];
    tk.actions.forEach((a, g) => {
      if (!a.target || a.action === 'stay') return;
      const [sx, sy] = start.p[g];
      ctx.save();
      ctx.strokeStyle = a.action === 'pass' ? v('--pass') : (g < N ? v('--team-a') : v('--team-b'));
      ctx.lineWidth = Math.max(1, (a.action === 'pass' ? 1.6 : 1.2) * dpr);
      if (a.action !== 'pass') ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.globalAlpha = 0.85;
      ctx.beginPath(); ctx.moveTo(X(sx), Y(sy)); ctx.lineTo(X(a.target[0]), Y(a.target[1])); ctx.stroke();
      ctx.restore();
    });
  }
  const pr = Math.max(C.player_radius * s, 7 * dpr);
  f.p.forEach(([x, y], g) => {
    ctx.beginPath(); ctx.arc(X(x), Y(y), pr, 0, 2 * Math.PI);
    ctx.fillStyle = g < N ? v('--team-a') : v('--team-b'); ctx.fill();
    ctx.lineWidth = Math.max(1, (g % N === 0 ? 2.5 : 1) * dpr);
    ctx.strokeStyle = g % N === 0 ? v('--ball') : 'rgba(0,0,0,.35)'; ctx.stroke();
    if (f.h === g) { ctx.beginPath(); ctx.arc(X(x), Y(y), pr + 3 * dpr, 0, 2 * Math.PI); ctx.strokeStyle = v('--ball'); ctx.lineWidth = 1.5 * dpr; ctx.stroke(); }
    ctx.fillStyle = '#fff'; ctx.font = `600 ${Math.round(pr * 1.05)}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(g % N), X(x), Y(y) + 0.5);
  });
  const br = Math.max(C.ball_radius * s, 3.5 * dpr);
  ctx.beginPath(); ctx.arc(X(f.b[0]), Y(f.b[1]), br, 0, 2 * Math.PI);
  ctx.fillStyle = v('--ball'); ctx.fill(); ctx.lineWidth = dpr; ctx.strokeStyle = v('--ball-edge'); ctx.stroke();

  // Speech labels for shouts made this tick (teammates hear them at the start of the next tick).
  if ($('shouts').checked) {
    ctx.font = `500 ${Math.round(11 * dpr)}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const placed = [], h = 16 * dpr, gap = 2 * dpr;
    const hits = (cx, cy, w) => placed.some(r => Math.abs(r.cx - cx) < (r.w + w) / 2 + gap && Math.abs(r.cy - cy) < h + gap);
    tk.actions.forEach((a, g) => {
      if (!a.say) return;
      const text = a.say.length > 42 ? a.say.slice(0, 41) + '…' : a.say;
      const [x, y] = f.p[g], w = ctx.measureText(text).width + 10 * dpr;
      const cx = Math.min(Math.max(X(x), w / 2 + 2), canvas.width - w / 2 - 2);
      // Start just above the speaker and step away (up first, then down) until the label is clear.
      const base = Y(y) - pr - h;
      let cy = base;
      for (let k = 1; k < 40 && (hits(cx, cy, w) || cy < h / 2 + 2 || cy > canvas.height - h / 2 - 2); k++) {
        cy = base + (k % 2 ? -1 : 1) * Math.ceil(k / 2) * (h + gap);
      }
      placed.push({ cx, cy, w });
      ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.lineWidth = dpr;
      ctx.beginPath(); ctx.moveTo(X(x), Y(y) - pr); ctx.lineTo(Math.min(Math.max(X(x), cx - w / 2), cx + w / 2), cy + (cy < Y(y) ? h / 2 : -h / 2)); ctx.stroke();
      ctx.fillStyle = 'rgba(0,0,0,.72)';
      ctx.beginPath(); ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 4 * dpr); ctx.fill();
      ctx.fillStyle = g < N ? v('--team-a') : v('--team-b'); ctx.fillRect(cx - w / 2, cy - h / 2, 3 * dpr, h);
      ctx.fillStyle = '#fff'; ctx.fillText(text, cx + 1.5 * dpr, cy + 0.5);
    });
  }

  $('score').textContent = `${frameScore[fi][0]} – ${frameScore[fi][1]}`;
  $('clock').textContent = `tick ${Math.max(0, tk.tick)} / ${C.max_ticks}`;
  scrub.value = fi;
  showTick(k);
}

// ----- live feed: every shout and key event, revealed up to the tick on screen -----
const KEEP = new Set(['goal', 'tackle', 'interception']);
const feed = $('feed'), feedRows = [];
T.forEach((tk, k) => {
  if (tk.tick < 0) return;
  const add = (cls, who, text) => {
    const r = document.createElement('div');
    r.className = `msg later ${cls}`;
    r.innerHTML = `<span class="t">${tk.tick}′</span><span class="who">${who}</span><span class="text">${text}</span>`;
    r.onclick = () => jump(k);
    feed.appendChild(r); feedRows.push({ k, r });
  };
  tk.actions.forEach((a, g) => { if (a.say) add(g < N ? 'a' : 'b', names[g], esc(a.say)); });
  tk.events.forEach(e => {
    if (!KEEP.has(e.type)) return;
    if (e.type === 'goal') add('ev goal ' + (e.team === 'A' ? 'a' : 'b'), e.team,
      `GOAL${e.own_goal ? ' (own goal by ' + esc(e.player) + ')' : e.player ? ' · ' + esc(e.player) : ''} · ${e.score[0]}–${e.score[1]}`);
    else add('ev ' + ((e.player || 'A')[0] === 'A' ? 'a' : 'b'), (e.player || '')[0] || '', `${e.type} by ${esc(e.player)}${e.from ? ' from ' + esc(e.from) : ''}`);
  });
});
if (!feedRows.length) feed.innerHTML = '<div class="empty">No shouts in this game.</div>';
document.querySelectorAll('[data-hide]').forEach(cb => cb.onchange = () => feed.classList.toggle(cb.dataset.hide, !cb.checked));

let shownTick = -2, revealed = 0;
function showTick(k) {
  if (k === shownTick) return;
  const follow = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 40;
  while (revealed < feedRows.length && feedRows[revealed].k <= k) feedRows[revealed++].r.classList.remove('later');
  while (revealed > 0 && feedRows[revealed - 1].k > k) feedRows[--revealed].r.classList.add('later');
  feedRows.forEach(({ k: rk, r }) => r.classList.toggle('now', rk === k));
  if (follow || k < shownTick) feed.scrollTop = feed.scrollHeight;
  shownTick = k;
  renderActions(k);
}
function renderActions(k) {
  const tk = T[k], el = $('actions');
  $('actionsTitle').textContent = tk.tick < 0 ? "Every player's action this tick" : `Every player's action at tick ${tk.tick}`;
  if (!tk.actions.length) { el.innerHTML = '<div class="row"><span></span><span class="muted">Kickoff</span></div>'; return; }
  el.innerHTML = tk.actions.map((a, g) => {
    const what = a.error ? `<span class="err">${esc(a.error)}</span>`
      : `${a.action}${a.target ? ` → (${a.target[0].toFixed(1)}, ${a.target[1].toFixed(1)})` : ''}`;
    const said = a.say ? `<br><span class="say">“${esc(a.say)}”</span>` : '';
    return `<div class="row"><span><span class="tag" style="background:var(${g < N ? '--team-a' : '--team-b'})"></span>${names[g]}</span><span>${what}${said}<br><span class="muted" style="font-size:11px">${esc(a.agent || '')}</span></span></div>`;
  }).join('');
}

// ----- timeline markers: goals (white) and ball wins (grey); click to jump -----
const tl = $('timeline'), span = Math.max(1, F.length - 1);
T.forEach((tk, k) => tk.events.forEach(e => {
  if (!KEEP.has(e.type)) return;
  const m = document.createElement('button');
  m.className = 'marker' + (e.type === 'goal' ? '' : ' minor');
  m.style.left = `calc(8px + (100% - 16px) * ${tk.f0 / span})`;
  m.title = e.type === 'goal' ? `Goal ${e.team} · tick ${tk.tick}` : `${e.type} · tick ${tk.tick}`;
  m.setAttribute('aria-label', m.title);
  m.onclick = () => jump(k);
  tl.appendChild(m);
}));

function frame(ts) {
  if (!playing) return;
  // rAF timestamps can precede the performance.now() taken in play(), so clamp at zero.
  const dt = Math.max(0, (ts - last) / 1000); last = ts;
  if (ts < holdUntil) { requestAnimationFrame(frame); return; }
  if (restart) { restart = false; pos = 0; }
  const from = idx();
  pos = Math.min(F.length - 1, pos + dt * C.substeps * Number($('speed').value));
  // Stop on a goal frame the playback just passed, and hold it for a moment in real time.
  for (let i = from + 1; i <= idx(); i++) {
    if (goalFrames.has(i)) { pos = i; holdUntil = ts + GOAL_HOLD_MS; break; }
  }
  draw();
  // Loop: hold the final frame briefly, then start the game again.
  if (pos >= F.length - 1) { holdUntil = ts + 1500; restart = true; }
  requestAnimationFrame(frame);
}
function play() {
  if (idx() >= F.length - 1) pos = 0;
  playing = true; $('play').textContent = 'Pause';
  last = performance.now(); requestAnimationFrame(frame);
}
function pause() { playing = false; pos = idx(); $('play').textContent = 'Play'; }
function stepTick(d) { pause(); const k = Math.min(T.length - 1, Math.max(0, frameTick[idx()] + d)); pos = T[k].f0; draw(); }
$('play').onclick = () => playing ? pause() : play();
$('prev').onclick = () => stepTick(-1);
$('next').onclick = () => stepTick(1);
// Keep focus off the buttons so space/arrows always reach the shortcuts below.
document.querySelectorAll('button').forEach(b => b.addEventListener('mousedown', e => e.preventDefault()));
scrub.oninput = () => { pause(); pos = Number(scrub.value); draw(); };
$('targets').onchange = draw;
$('shouts').onchange = draw;
document.addEventListener('keydown', e => {
  const tag = e.target.tagName;
  if (tag === 'SELECT' || (tag === 'INPUT' && e.target.type === 'range')) return;
  if (e.key === ' ') { e.preventDefault(); playing ? pause() : play(); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); stepTick(-1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); stepTick(1); }
});
window.addEventListener('resize', resize);
resize();
// ?t=N opens the replay at tick N (links from the writeup); ?autoplay starts playing.
const params = new URLSearchParams(location.search);
const startTick = T.findIndex(tk => tk.tick === Number(params.get('t')));
if (params.has('t') && startTick >= 0) { pos = T[startTick].f0; draw(); }
if (params.has('autoplay')) play();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        src = Path(arg)
        write_replay(json.loads(src.read_text()), src.with_suffix(".html"))
        print("wrote", src.with_suffix(".html"))

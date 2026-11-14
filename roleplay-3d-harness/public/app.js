import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const $ = (s) => document.querySelector(s);
const EMOTES = ["wave", "nod", "shake_head", "bow", "jump", "dance", "sit", "stand", "point", "laugh", "cry", "shrug", "cheer", "think"];

// ------------------------------------------------------------------ renderer

const viewport = $("#viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
viewport.appendChild(renderer.domElement);

const labels = new CSS2DRenderer();
labels.domElement.className = "label-layer";
Object.assign(labels.domElement.style, { position: "absolute", top: "0", left: "0" });
viewport.appendChild(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
camera.position.set(0, 15, 24);
camera.lookAt(0, 0, 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 3);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 4;
controls.maxDistance = 90;

const hemi = new THREE.HemisphereLight(0xffffff, 0x444433, 1.1);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(18, 30, 12);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -35, right: 35, top: 35, bottom: -35, near: 1, far: 100 });
scene.add(sun);

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  renderer.setSize(w, h);
  labels.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

// ------------------------------------------------------------------ world objects

let world = null; // scene info
const sceneRoot = new THREE.Group();
scene.add(sceneRoot);
let ground = null;
const chars = new Map(); // id -> view
const items = new Map(); // id -> view
const landmarkMeshes = [];
let selectedId = null;
let targetName = null; // what the player last clicked on
let placing = null; // pending spawn
const state = { characters: [], items: [] };

const mat = (color, opts = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.8, ...opts });
const shadow = (m) => { m.castShadow = true; m.receiveShadow = true; return m; };

function geometryFor(kind, [w, h, d]) {
  switch (kind) {
    case "cylinder": return new THREE.CylinderGeometry(w / 2, w / 2, h, 24);
    case "cone": return new THREE.ConeGeometry(w / 2, h, 20);
    case "sphere": return new THREE.SphereGeometry(w / 2, 20, 14);
    default: return new THREE.BoxGeometry(w, h, d);
  }
}

function label(className, html) {
  const el = document.createElement("div");
  el.className = className;
  el.innerHTML = html;
  const obj = new CSS2DObject(el);
  obj.center.set(0.5, 1);
  return obj;
}

function clearScene() {
  for (const v of chars.values()) v.group.traverse((o) => o.isCSS2DObject && o.element.remove());
  sceneRoot.traverse((o) => o.isCSS2DObject && o.element.remove());
  sceneRoot.clear();
  chars.clear();
  items.clear();
  landmarkMeshes.length = 0;
}

function buildScene(info) {
  const fresh = !world || world.id !== info.id || info.characters.some((c) => !chars.has(c.id));
  if (fresh) {
    clearScene();
    scene.background = new THREE.Color(info.skyColor);
    scene.fog = new THREE.Fog(info.skyColor, info.worldSize * 1.2, info.worldSize * 3);
    const g = new THREE.Mesh(new THREE.PlaneGeometry(info.worldSize, info.worldSize), mat(info.groundColor));
    g.rotation.x = -Math.PI / 2;
    g.receiveShadow = true;
    g.userData = { kind: "ground" };
    sceneRoot.add(g);
    ground = g;
    const grid = new THREE.GridHelper(info.worldSize, info.worldSize / 2, 0x000000, 0x000000);
    grid.material.opacity = 0.08;
    grid.material.transparent = true;
    grid.position.y = 0.01;
    sceneRoot.add(grid);

    for (const l of info.landmarks) {
      const shape = l.shape ?? { kind: "cylinder", size: [l.radius * 2, 0.2, l.radius * 2], color: "#777" };
      const m = shadow(new THREE.Mesh(geometryFor(shape.kind, shape.size), mat(shape.color)));
      m.position.set(l.position[0], shape.size[1] / 2, l.position[1]);
      m.userData = { kind: "landmark", name: l.name };
      sceneRoot.add(m);
      landmarkMeshes.push(m);
      const tag = label("placetag", l.name);
      tag.position.set(0, shape.size[1] / 2 + 0.5, 0);
      m.add(tag);
    }
    for (const c of info.characters) chars.set(c.id, makeCharacter(c));
    $("#title").textContent = info.title;
    $("#director-notes").textContent = info.directorNotes ? `Director notes: ${info.directorNotes}` : "";
  }
  for (const c of info.characters) {
    const v = chars.get(c.id);
    v.info = c;
    v.ctl.textContent = c.control === "player" ? "(you)" : "";
  }
  for (const it of info.items) if (!items.has(it.id)) items.set(it.id, makeItem(it));
  world = info;
  refreshPlayerBar();
}

function makeCharacter(c) {
  const h = c.height ?? 1;
  const group = new THREE.Group();
  const rig = new THREE.Group(); // animated part
  group.add(rig);
  const bodyMat = mat(c.color);
  const body = shadow(new THREE.Mesh(new THREE.CapsuleGeometry(0.32 * h, 0.75 * h, 6, 16), bodyMat));
  body.position.y = 0.7 * h;
  rig.add(body);
  const head = new THREE.Group();
  head.position.y = 1.42 * h;
  rig.add(head);
  head.add(shadow(new THREE.Mesh(new THREE.SphereGeometry(0.27 * h, 20, 16), mat(c.headColor))));
  for (const sx of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.045 * h, 8, 8), mat("#111"));
    eye.position.set(sx * 0.1 * h, 0.04 * h, -0.24 * h);
    head.add(eye);
  }
  const arms = [-1, 1].map((sx) => {
    const pivot = new THREE.Group();
    pivot.position.set(sx * 0.4 * h, 1.08 * h, 0);
    const arm = shadow(new THREE.Mesh(new THREE.CapsuleGeometry(0.08 * h, 0.45 * h, 4, 8), bodyMat));
    arm.position.y = -0.3 * h;
    pivot.add(arm);
    rig.add(pivot);
    return pivot;
  });
  const hand = new THREE.Object3D();
  hand.position.y = -0.62 * h;
  arms[1].add(hand);

  const ring = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.62, 32), new THREE.MeshBasicMaterial({ color: 0xffe27a, transparent: true, opacity: 0.9 }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.03;
  ring.visible = false;
  group.add(ring);

  const tagObj = label("nametag", `${c.name}<span class="ctl"></span>`);
  tagObj.position.y = 1.85 * h;
  group.add(tagObj);
  const bubbleObj = label("bubble hidden", "");
  bubbleObj.position.y = 2.3 * h;
  group.add(bubbleObj);

  group.traverse((o) => { o.userData.charId = c.id; });
  sceneRoot.add(group);
  return {
    info: c, h, group, rig, body, head, arms, hand, ring,
    tag: tagObj.element, ctl: tagObj.element.querySelector(".ctl"), bubble: bubbleObj.element,
    pos: new THREE.Vector2(), facing: 0, init: false, s: null, bubbleKey: "",
  };
}

function makeItem(it) {
  const size = it.shape === "sphere" ? [0.45, 0.45, 0.45] : it.shape === "cylinder" ? [0.25, 0.5, 0.25] : [0.45, 0.35, 0.35];
  const m = shadow(new THREE.Mesh(geometryFor(it.shape, size), mat(it.color, { emissive: it.color, emissiveIntensity: 0.15 })));
  m.userData = { kind: "item", name: it.name, id: it.id, lift: size[1] / 2 };
  sceneRoot.add(m);
  return { info: it, mesh: m };
}

// ------------------------------------------------------------------ animation

const timer = new THREE.Timer();
const angleLerp = (a, b, t) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;

function animateCharacter(v, s, t, dt) {
  if (!v.init) { v.pos.set(s.x, s.z); v.facing = s.facing; v.init = true; }
  const k = 1 - Math.exp(-dt * 10);
  v.pos.lerp(new THREE.Vector2(s.x, s.z), k);
  v.facing = angleLerp(v.facing, s.facing, k);
  v.group.position.set(v.pos.x, 0, v.pos.y);
  v.group.rotation.y = -v.facing;

  const { rig, head, arms, h } = v;
  // reset pose
  rig.position.set(0, 0, 0); rig.rotation.set(0, 0, 0);
  head.rotation.set(0, 0, 0);
  arms[0].rotation.set(0, 0, 0.08); arms[1].rotation.set(0, 0, -0.08);

  if (s.posture === "sitting") { rig.position.y = -0.38 * h; arms[0].rotation.x = arms[1].rotation.x = -0.9; }
  if (s.moving) {
    const f = s.pace === "run" ? 16 : 10;
    rig.position.y += Math.abs(Math.sin(t * f)) * 0.06 * h;
    arms[0].rotation.x = Math.sin(t * f) * 0.7;
    arms[1].rotation.x = -Math.sin(t * f) * 0.7;
    if (s.pace === "run") rig.rotation.x = -0.15;
  } else {
    rig.position.y += Math.sin(t * 2 + v.pos.x) * 0.008; // breathing
  }
  switch (s.emote) {
    case "wave": arms[1].rotation.z = -2.6 + Math.sin(t * 10) * 0.35; break;
    case "point": arms[1].rotation.x = -1.5; break;
    case "cheer": arms[0].rotation.z = 2.7; arms[1].rotation.z = -2.7; rig.position.y += Math.abs(Math.sin(t * 8)) * 0.12; break;
    case "shrug": arms[0].rotation.z = 0.9; arms[1].rotation.z = -0.9; head.rotation.z = 0.2; break;
    case "jump": rig.position.y += Math.abs(Math.sin(t * 6)) * 0.6; break;
    case "dance": rig.rotation.y = t * 5; rig.position.y += Math.abs(Math.sin(t * 8)) * 0.15; arms[0].rotation.z = 2 + Math.sin(t * 8) * 0.5; arms[1].rotation.z = -2 - Math.cos(t * 8) * 0.5; break;
    case "bow": rig.rotation.x = 0.6; break;
    case "nod": head.rotation.x = Math.sin(t * 9) * 0.3; break;
    case "shake_head": head.rotation.y = Math.sin(t * 10) * 0.45; break;
    case "laugh": rig.rotation.x = -0.12; rig.position.y += Math.abs(Math.sin(t * 18)) * 0.04; head.rotation.x = -0.25; break;
    case "cry": head.rotation.x = 0.5; arms[0].rotation.x = arms[1].rotation.x = -1.8; break;
    case "think": arms[1].rotation.x = -2.2; arms[1].rotation.z = 0.5; head.rotation.z = 0.15; break;
  }
  v.ring.visible = selectedId === s.id;
  v.tag.classList.toggle("selected", selectedId === s.id);

  // speech / action / thinking bubble
  let key = "", cls = "bubble hidden", html = "";
  if (s.speech) { key = "s" + s.speech.text; cls = `bubble ${s.speech.volume}`; html = escapeHtml(s.speech.text); }
  else if (s.action) { key = "a" + s.action; cls = "bubble action"; html = `*${escapeHtml(s.action)}*`; }
  else if (s.thinking && s.control !== "player") { key = "t"; cls = "bubble thinking"; html = `<span class="dots"><span>•</span><span>•</span><span>•</span></span>`; }
  if (key !== v.bubbleKey) { v.bubbleKey = key; v.bubble.className = cls; v.bubble.innerHTML = html; }
}

const handPos = new THREE.Vector3();
function animateItems(t) {
  for (const s of state.items) {
    const v = items.get(s.id);
    if (!v) continue;
    const holder = s.holder && chars.get(s.holder);
    if (holder) {
      holder.hand.getWorldPosition(handPos);
      v.mesh.position.copy(handPos);
    } else {
      v.mesh.position.set(s.x, v.mesh.userData.lift + 0.05 + Math.sin(t * 2 + s.x) * 0.04, s.z);
      v.mesh.rotation.y = t * 0.6;
    }
  }
}

function frame() {
  timer.update();
  const dt = Math.min(timer.getDelta(), 0.1);
  const t = timer.getElapsed();
  for (const s of state.characters) {
    const v = chars.get(s.id);
    if (v) animateCharacter(v, s, t, dt);
  }
  scene.updateMatrixWorld();
  animateItems(t);
  if (follow && selectedId && chars.get(selectedId)) {
    const p = chars.get(selectedId).group.position;
    const delta = p.clone().sub(controls.target).multiplyScalar(0.08);
    controls.target.add(delta);
    camera.position.add(delta);
  }
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
  requestAnimationFrame(frame);
}
let follow = false;
requestAnimationFrame(frame);

// ------------------------------------------------------------------ picking

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(sceneRoot.children, true).find((h) => h.object.isMesh);
  if (!hit) return;
  const o = hit.object;
  if (placing) {
    const p = hit.point;
    post("/api/spawn", { ...placing, x: round(p.x), z: round(p.z) });
    placing = null;
    $("#spawn-btn").textContent = "Place…";
    return;
  }
  if (o.userData.charId) {
    select(o.userData.charId);
    const name = chars.get(o.userData.charId).info.name;
    setTarget(name);
    if (e.shiftKey) playerDo("walk_to", { destination: name, pace: "walk" });
    return;
  }
  if (o.userData.kind === "item" || o.userData.kind === "landmark") {
    setTarget(o.userData.name);
    playerDo("walk_to", { destination: o.userData.name, pace: e.shiftKey ? "run" : "walk" });
    return;
  }
  if (o.userData.kind === "ground") {
    playerDo("walk_to", { destination: `${round(hit.point.x)}, ${round(hit.point.z)}`, pace: e.shiftKey ? "run" : "walk" });
  }
});
const round = (n) => Math.round(n * 10) / 10;

// ------------------------------------------------------------------ server link

async function post(url, body = {}) {
  const res = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) addLog({ t: state.time ?? 0, kind: "error", text: data.error ?? res.statusText });
  return data;
}

const es = new EventSource("/api/events");
es.addEventListener("scene", (e) => buildScene(JSON.parse(e.data)));
es.addEventListener("state", (e) => {
  Object.assign(state, JSON.parse(e.data));
  $("#clock").textContent = `${state.time.toFixed(1)}s`;
});
es.addEventListener("status", (e) => renderStatus(JSON.parse(e.data)));
es.addEventListener("backlog", (e) => { $("#log").innerHTML = ""; JSON.parse(e.data).forEach(addLog); });
es.addEventListener("log", (e) => addLog(JSON.parse(e.data)));
es.addEventListener("agent", (e) => { const a = JSON.parse(e.data); if (a.id === selectedId) renderAgent(a); });

// ------------------------------------------------------------------ status & controls

let status = null;
function renderStatus(s) {
  status = s;
  $("#play-btn").textContent = s.running ? "⏸ Pause" : "▶ Play";
  const pill = $("#status-pill");
  pill.textContent = s.running ? (s.inFlight ? `running · ${s.inFlight} thinking` : "running") : "paused";
  pill.classList.toggle("running", s.running);
  $("#brain-line").textContent = `Brain: ${s.brain}`;
  const t = s.totals;
  const tokens = t.input + t.cacheRead + t.cacheWrite;
  $("#totals").textContent = `${t.turns} turns · ${fmtK(tokens)} in (${tokens ? Math.round((100 * t.cacheRead) / tokens) : 0}% cached) · ${fmtK(t.output)} out` +
    (t.costUsd ? ` · ≈$${t.costUsd.toFixed(3)}` : "") + ` · budget ${s.maxTurns}`;
  $("#lockstep").checked = s.lockstep;
  $("#timescale").value = s.timeScale;
  $("#timescale-val").textContent = `${s.timeScale}×`;
  if (document.activeElement !== $("#max-turns")) $("#max-turns").value = s.maxTurns;
}
const fmtK = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

$("#play-btn").onclick = () => post(status?.running ? "/api/pause" : "/api/play");
$("#step-btn").onclick = () => post("/api/step");
$("#lockstep").onchange = (e) => post("/api/settings", { lockstep: e.target.checked });
$("#timescale").oninput = (e) => post("/api/settings", { timeScale: Number(e.target.value) });
$("#max-turns").onchange = (e) => post("/api/settings", { maxTurns: Number(e.target.value) });

document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.id === `tab-${name}`));
}

// ------------------------------------------------------------------ log

const logEl = $("#log");
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }
function addLog(e) {
  const li = document.createElement("li");
  li.className = e.kind;
  let text = escapeHtml(e.text);
  const name = e.actor && chars.get(e.actor)?.info.name;
  if (name && e.text.startsWith(name)) text = `<span class="who" data-id="${e.actor}">${escapeHtml(name)}</span>${text.slice(escapeHtml(name).length)}`;
  li.innerHTML = `<span class="t">${Number(e.t).toFixed(1)}</span><span class="x">${text}</span>`;
  const stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 30;
  logEl.appendChild(li);
  while (logEl.children.length > 600) logEl.firstChild.remove();
  if (stick) logEl.scrollTop = logEl.scrollHeight;
}
logEl.addEventListener("click", (e) => { const id = e.target.dataset?.id; if (id) { select(id); showTab("character"); } });
const filt = (box, cls) => { const f = () => logEl.classList.toggle(cls, !$(box).checked); $(box).onchange = f; f(); };
filt("#show-thoughts", "hide-thought"); filt("#show-moves", "hide-move"); filt("#show-errors", "hide-error");

// ------------------------------------------------------------------ character inspector

async function select(id) {
  selectedId = id;
  follow = false;
  const res = await fetch(`/api/agent/${encodeURIComponent(id)}`);
  if (res.ok) renderAgent(await res.json());
  showTab("character");
  if (isPlayer(id)) $("#player-select").value = id;
}

function renderAgent(a) {
  $("#char-empty").hidden = true;
  $("#char-panel").hidden = false;
  $("#char-name").textContent = a.name;
  $("#char-swatch").style.background = chars.get(a.id)?.info.color ?? "#888";
  $("#char-control").checked = a.control === "player";
  $("#char-persona").textContent = a.persona;
  $("#char-goals").innerHTML = a.goals.length ? `<b>Goals</b><ul>${a.goals.map((g) => `<li>${escapeHtml(g)}</li>`).join("")}</ul>` : "";
  $("#char-secrets").innerHTML = a.secrets.map((s) => `<li>${escapeHtml(s)}</li>`).join("") || "<li class='muted'>none</li>";
  $("#char-observation").textContent = a.observation;
  $("#char-prompt").textContent = a.systemPrompt;
  $("#direct-form").hidden = a.control === "player";
  $("#char-history").innerHTML = a.history.map((h) => `
    <li>
      <div class="t">t=${h.t.toFixed(1)}s</div>
      ${h.error ? `<div class="call bad">⚠ ${escapeHtml(h.error)}</div>` : ""}
      ${h.thinking ? `<details><summary>reasoning</summary><div class="thinking">${escapeHtml(h.thinking)}</div></details>` : ""}
      ${h.monologue ? `<div class="mono">💭 ${escapeHtml(h.monologue)}</div>` : ""}
      ${h.calls.map((c) => `<div class="call ${c.ok ? "" : "bad"}">${c.name}(${escapeHtml(JSON.stringify(c.input))})<span class="res">→ ${escapeHtml(c.message)}</span></div>`).join("")}
      ${!h.error && !h.calls.length ? `<div class="muted">(no action)</div>` : ""}
    </li>`).join("");
}

$("#char-control").onchange = (e) => post("/api/control", { id: selectedId, control: e.target.checked ? "player" : "llm" }).then(() => select(selectedId));
$("#direct-form").onsubmit = (e) => {
  e.preventDefault();
  const text = $("#direct-text").value.trim();
  if (!text || !selectedId) return;
  post("/api/direct", { id: selectedId, text });
  $("#direct-text").value = "";
};
$("#char-name").ondblclick = () => { follow = !follow; };

// ------------------------------------------------------------------ player

const isPlayer = (id) => world?.characters.find((c) => c.id === id)?.control === "player";
function playerId() { return $("#player-select").value; }

function refreshPlayerBar() {
  const players = world.characters.filter((c) => c.control === "player");
  $("#player-bar").hidden = players.length === 0;
  const sel = $("#player-select");
  const prev = sel.value;
  sel.innerHTML = players.map((c) => `<option value="${c.id}">${escapeHtml(c.name)}</option>`).join("");
  if (players.some((p) => p.id === prev)) sel.value = prev;
  const to = $("#say-to");
  const prevTo = to.value;
  to.innerHTML = `<option value="everyone">to everyone</option>` +
    world.characters.map((c) => `<option value="${escapeHtml(c.name)}">to ${escapeHtml(c.name)}</option>`).join("");
  to.value = [...to.options].some((o) => o.value === prevTo) ? prevTo : "everyone";
}

function setTarget(name) {
  targetName = name;
  const opt = [...$("#say-to").options].find((o) => o.value === name);
  if (opt) $("#say-to").value = name;
}

function playerDo(tool, input) {
  const id = playerId();
  if (!id || $("#player-bar").hidden) return;
  return post("/api/act", { id, tool, input });
}

$("#emote-select").innerHTML += EMOTES.map((e) => `<option value="${e}">${e.replace("_", " ")}</option>`).join("");
$("#emote-select").onchange = (e) => { if (e.target.value) playerDo("emote", { animation: e.target.value }); e.target.value = ""; };
$("#say-form").onsubmit = (e) => {
  e.preventDefault();
  const text = $("#say-text").value.trim();
  if (!text) return;
  playerDo("say", { text, to: $("#say-to").value, volume: $("#say-volume").value });
  $("#say-text").value = "";
};
$("#act-form").onsubmit = (e) => {
  e.preventDefault();
  const text = $("#act-text").value.trim();
  if (!text) return;
  playerDo("act", { description: text, target: "none" });
  $("#act-text").value = "";
};
$("#pickup-btn").onclick = () => targetName && playerDo("pick_up", { item: targetName });
$("#drop-btn").onclick = () => {
  const me = state.characters.find((c) => c.id === playerId());
  const item = me?.holding[0];
  if (item) playerDo("drop", { item });
};
$("#give-btn").onclick = () => {
  const me = state.characters.find((c) => c.id === playerId());
  const item = me?.holding[0];
  if (item && targetName) playerDo("give", { item, to: targetName });
};

// ------------------------------------------------------------------ director

$("#narrate-form").onsubmit = (e) => {
  e.preventDefault();
  const text = $("#narrate-text").value.trim();
  if (text) post("/api/narrate", { text });
  $("#narrate-text").value = "";
};
$("#spawn-form").onsubmit = (e) => {
  e.preventDefault();
  const name = $("#spawn-name").value.trim();
  if (!name) return;
  placing = { name, color: $("#spawn-color").value, description: $("#spawn-desc").value.trim() };
  $("#spawn-btn").textContent = "Click the ground…";
};
$("#reset-form").onsubmit = (e) => {
  e.preventDefault();
  selectedId = null;
  $("#char-panel").hidden = true;
  $("#char-empty").hidden = false;
  post("/api/reset", { scenario: $("#scenario-select").value, brain: $("#brain-select").value }).then(() => loadScenarios());
};

async function loadScenarios() {
  const r = await fetch("/api/scenarios").then((x) => x.json());
  $("#scenario-select").innerHTML = r.scenarios.map((s) => `<option value="${s.id}">${escapeHtml(s.title)}</option>`).join("");
  $("#scenario-select").value = r.current;
  $("#brain-select").value = r.brain;
}
loadScenarios();

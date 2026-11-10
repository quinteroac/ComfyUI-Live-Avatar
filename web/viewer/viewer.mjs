import * as THREE from "three";
import { LiveAvatar } from "../runtime/live-avatar.mjs";

const query = new URLSearchParams(location.search);
const shot = query.has("shot");
const renderer = new THREE.WebGLRenderer({ antialias: true, stencil: true, alpha: true, preserveDrawingBuffer: shot });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(22, 1, 0.01, 20);
const message = document.getElementById("msg");
const bar = document.getElementById("bar");
const panel = document.getElementById("params");
let avatar = null;
let mouth = null;
let tracking = null;
let focus = query.get("frame") ?? "body";
let follow = query.get("look") !== "0";
const pointer = { x: 0, y: 0, sx: 0, sy: 0 };
let dragging = false;
const light = { on: query.has("light"), x: 0.4, y: 0.5, strength: 0.8 };
if (light.on) [light.x, light.y, light.strength = 0.8] = query.get("light").split(",").map(Number);

function frame(width = innerWidth, height = innerHeight) {
  if (!avatar) return;
  const aspect = width / height;
  camera.aspect = aspect;
  const [w, h] = avatar.header.canvas;
  const head = avatar.boneByName.head.getWorldPosition(new THREE.Vector3());
  const view = focus === "body" ? 1.04 : 0.42;
  const y = focus === "body" ? 0.5 : head.y + 0.06;
  const fit = Math.max(view, (view * (w / h)) / aspect);
  camera.position.set(0, y, fit / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))));
  camera.lookAt(0, y, 0);
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function button(label, click, group = bar) {
  const b = document.createElement("button");
  b.textContent = label;
  b.onclick = () => click(b);
  group.append(b);
  return b;
}

function buildUi() {
  bar.replaceChildren();
  panel.replaceChildren();
  for (const name of Object.keys(avatar.clips)) {
    if (name === "idle") continue;
    button(name, (b) => {
      if (avatar.clips[name].loop) {
        const on = b.classList.toggle("on");
        on ? avatar.play(name) : avatar.stop(name);
      } else avatar.play(name);
    });
  }
  button("look", (b) => { follow = !follow; b.classList.toggle("on", follow); if (!follow) avatar.lookAt(0, 0); }).classList.toggle("on", follow);
  const mic = button("mic", toggleMic);
  const camera = query.get("tracking") === "0" ? null : button("camera", toggleCamera);
  button("frame", () => { focus = focus === "body" ? "head" : "body"; frame(); });
  button("light", (b) => { light.on = b.classList.toggle("on"); applyLight(); }).classList.toggle("on", light.on);
  button("poses", () => contactSheet().then(showSheet));
  button("params", () => panel.classList.toggle("open"));
  const say = Object.assign(document.createElement("input"), { placeholder: "say… (kana or text)", size: 18 });
  say.onkeydown = (e) => { if (e.key === "Enter" && say.value) avatar.speak(say.value); e.stopPropagation(); };
  bar.append(say);
  const faces = document.createElement("div");
  faces.id = "faces";
  bar.before(faces);
  for (const [i, name] of [null, ...Object.keys(avatar.expressions)].entries()) {
    const b = button(`${i + 1} ${name ?? "neutral"}`, () => setExpression(name), faces);
    b.dataset.expression = name ?? "";
  }
  setExpression(query.get("expression"));
  for (const name of Object.keys(avatar.params).sort()) {
    const label = document.createElement("label");
    const value = document.createElement("span");
    const input = Object.assign(document.createElement("input"), { type: "range", min: -1, max: 1, step: 0.01, value: 0 });
    value.textContent = "0";
    input.oninput = () => { avatar.setParam(name, +input.value); value.textContent = input.value; };
    label.append(name, value, input);
    panel.append(label);
  }
  if (query.get("mic") === "1") mic.click();
  if (query.get("camera") === "1") camera?.click();
}

function setExpression(name) {
  avatar.setExpression(name || null);
  for (const b of document.querySelectorAll("#faces button")) b.classList.toggle("on", b.dataset.expression === (name ?? ""));
}

function applyLight() {
  avatar.setLight({ x: light.x, y: light.y, strength: light.on ? light.strength : 0 });
}

async function toggleMic(b) {
  if (mouth) {
    mouth.stream.getTracks().forEach((t) => t.stop());
    mouth.context.close();
    mouth = null;
    avatar.setMouth(0);
    b.classList.remove("on");
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  context.createMediaStreamSource(stream).connect(analyser);
  mouth = { stream, context, analyser, data: new Float32Array(512), spectrum: new Float32Array(256), level: 0, centroid: 1200 };
  b.classList.add("on");
}

async function toggleCamera(b) {
  if (tracking) {
    tracking.stream.getTracks().forEach((t) => t.stop());
    tracking.tracker.close();
    tracking = null;
    avatar.autoBlink = true;
    b.classList.remove("on");
    return;
  }
  b.textContent = "camera…";
  const { FaceTracker } = await import("../runtime/face-tracker.mjs");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  const video = Object.assign(document.createElement("video"), { srcObject: stream, muted: true, playsInline: true });
  await video.play();
  tracking = { stream, video, tracker: await FaceTracker.create() };
  avatar.autoBlink = false;
  follow = false;
  b.textContent = "camera";
  b.classList.add("on");
}

function trackFace() {
  if (tracking.video.readyState < 2) return;
  const state = tracking.tracker.read(tracking.video, performance.now());
  if (!state) return;
  for (const name in state) if (name !== "mouthOpen") avatar.setParam(name, state[name]);
  if (!mouth) avatar.setMouth(state.mouthOpen, tracking.tracker.vowel);
}

function micLevel() {
  mouth.analyser.getFloatTimeDomainData(mouth.data);
  let sum = 0;
  for (const v of mouth.data) sum += v * v;
  const level = Math.min(Math.sqrt(sum / mouth.data.length) * 7, 1);
  mouth.level += (level - mouth.level) * 0.45;
  // The spectral centroid of the voice band gives a rough vowel: dark o, bright i, a in between.
  mouth.analyser.getFloatFrequencyData(mouth.spectrum);
  const bin = mouth.context.sampleRate / mouth.analyser.fftSize;
  let weight = 0, moment = 0;
  for (let i = Math.round(200 / bin); i < Math.round(4000 / bin); i++) {
    const power = 10 ** (mouth.spectrum[i] / 10);
    weight += power;
    moment += power * i * bin;
  }
  if (weight > 0) mouth.centroid += (moment / weight - mouth.centroid) * 0.2;
  avatar.setMouth(mouth.level, mouth.centroid < 900 ? "o" : mouth.centroid > 1900 ? "i" : "a");
}

async function show(buffer) {
  if (avatar) { scene.remove(avatar); avatar.dispose(); }
  avatar = await LiveAvatar.parse(buffer, renderer);
  scene.add(avatar);
  message.textContent = "";
  frame();
  applyLight();
  if (query.has("sheet")) {
    showSheet(await contactSheet());
    document.title = "ready";
  } else if (!shot) {
    buildUi();
    avatar.play("idle");
    if (query.has("clip")) avatar.play(query.get("clip"), { loop: true });
  }
}

if (query.has("bg")) {
  const background = query.get("bg");
  document.documentElement.style.background = document.body.style.background = background;
}
if (query.get("ui") === "0") bar.style.display = "none";

addEventListener("resize", () => frame());
addEventListener("pointermove", (e) => {
  pointer.x = (e.clientX / innerWidth) * 2 - 1;
  pointer.y = 1 - (e.clientY / innerHeight) * 2;
  dragging = e.buttons === 1 && e.target === renderer.domElement && !!avatar;
  if (!dragging) return;
  // Dragging over the avatar moves the light while it is on. Otherwise it moves the avatar, to see hair and chest follow.
  if (light.on) {
    light.x = pointer.x * 1.5;
    light.y = pointer.y * 1.5;
    applyLight();
  } else {
    const unit = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.position.z) / innerHeight;
    avatar.position.x += e.movementX * unit;
    avatar.position.y -= e.movementY * unit;
  }
});
addEventListener("pointerup", () => { dragging = false; });
addEventListener("keydown", (e) => {
  const b = document.querySelectorAll("#faces button")[+e.key - 1];
  if (b && !e.ctrlKey && !e.metaKey && !e.altKey) b.click();
});
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", async (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) await show(await file.arrayBuffer());
});

const clock = new THREE.Clock();
function tick() {
  requestAnimationFrame(tick);
  const dt = clock.getDelta();
  if (!avatar) return;
  if (follow) {
    pointer.sx += (pointer.x - pointer.sx) * 0.08;
    pointer.sy += (pointer.y - pointer.sy) * 0.08;
    avatar.lookAt(pointer.sx, pointer.sy);
  }
  if (tracking) trackFace();
  if (mouth) micLevel();
  if (!dragging) avatar.position.multiplyScalar(Math.exp(-dt * 5));
  avatar.update(dt);
  renderer.render(scene, camera);
}

// Fixed poses side by side, to check a build at a glance. Faces are framed on the head, the rest on the body.
function sheetPoses() {
  const has = (name) => avatar.params[name];
  const poses = [["rest", "body", {}], ["rest", "head", {}], ["blink", "head", { blinkL: 1, blinkR: 1 }], ["half blink", "head", { blinkL: 0.5, blinkR: 0.5 }]];
  if (has("eyeSmile")) poses.push(["smiling eyes", "head", { eyeSmile: 1 }]);
  for (const vowel of "aiueo") poses.push([vowel, "head", avatar.mouthValues(1, vowel)]);
  poses.push(["turn left", "head", { headYaw: 1 }], ["turn right", "head", { headYaw: -1 }], ["look up", "head", { headPitch: -1 }],
    ["look down", "head", { headPitch: 1 }], ["tilt", "head", { headRoll: 1 }], ["gaze", "head", { gazeX: 1, gazeY: 1 }]);
  for (const name of Object.keys(avatar.expressions)) poses.push([name, "head", { expression: name }]);
  poses.push(["body turn", "body", { bodyYaw: 1, headYaw: 0.5 }], ["body roll", "body", { bodyRoll: 1 }]);
  if (has("armLRaise")) poses.push(["arms up", "body", { armLRaise: 0.6, armRRaise: 0.6, armLBend: 0.3, armRBend: 0.3 }]);
  if (has("armLSwing")) poses.push(["arms swing", "body", { armLSwing: 0.6, armRSwing: -0.6 }]);
  if (has("legLSwing")) poses.push(["step", "body", { legLSwing: 0.5, legLBend: 0.5, legRSwing: -0.3 }]);
  return poses;
}

async function contactSheet() {
  const tile = [300, 380];
  const poses = sheetPoses();
  const columns = 6;
  const sheet = Object.assign(document.createElement("canvas"), { width: tile[0] * columns, height: tile[1] * Math.ceil(poses.length / columns) });
  const g = sheet.getContext("2d");
  g.fillStyle = "#fff";
  g.fillRect(0, 0, sheet.width, sheet.height);
  const saved = { focus, blink: avatar.autoBlink, actions: avatar.actions, values: avatar.values, ratio: renderer.getPixelRatio() };
  renderer.setPixelRatio(1);
  renderer.setClearColor(0xffffff, 1);
  avatar.autoBlink = false;
  avatar.actions = [];
  for (const [i, [label, view, values]] of poses.entries()) {
    const { expression, ...params } = values;
    avatar.values = params;
    avatar.face = {};
    avatar.setExpression(expression ?? null, 0.001);
    for (let k = 0; k < 90; k++) avatar.update(1 / 60);
    focus = view;
    frame(...tile);
    // Splat sorting runs in a worker and lands a few frames later.
    for (const end = performance.now() + (avatar.headSplat ? 400 : 0); performance.now() < end;) {
      renderer.render(scene, camera);
      await new Promise(requestAnimationFrame);
    }
    renderer.render(scene, camera);
    const x = (i % columns) * tile[0], y = Math.floor(i / columns) * tile[1];
    g.drawImage(renderer.domElement, x, y, tile[0], tile[1]);
    g.fillStyle = "#222";
    g.font = "15px system-ui, sans-serif";
    g.fillText(label, x + 8, y + 20);
  }
  Object.assign(avatar, { autoBlink: saved.blink, actions: saved.actions, values: saved.values });
  avatar.setExpression(null, 0.001);
  focus = saved.focus;
  renderer.setPixelRatio(saved.ratio);
  renderer.setClearColor(0x000000, 0);
  frame();
  return sheet;
}

function showSheet(sheet) {
  const overlay = document.createElement("div");
  overlay.id = "sheet";
  const link = Object.assign(document.createElement("a"), { href: sheet.toDataURL("image/png"), download: "live_avatar_poses.png", textContent: "Save PNG" });
  const close = button("close", () => overlay.remove(), overlay);
  overlay.append(link, sheet);
  close.after(link);
  document.body.append(overlay);
}

// ?shot=1&clip=wave&t=1.2&p=headYaw:0.8,mouthOpen:1 renders one fixed frame: used by the tests.
async function renderShot() {
  if (!query.has("bg")) document.documentElement.style.background = document.body.style.background = "#fff";
  avatar.autoBlink = false;
  frame();
  for (const pair of (query.get("p") ?? "").split(",").filter(Boolean)) {
    const [name, value] = pair.split(":");
    avatar.setParam(name, +value);
  }
  if (query.get("clip")) avatar.play(query.get("clip"), { fade: 0.001 });
  const t = +(query.get("t") ?? 0);
  for (let i = 0; i < Math.round(t * 60); i++) avatar.update(1 / 60);
  avatar.update(0);
  // Splat sorting runs in a worker and lands a few frames later.
  for (const end = performance.now() + (avatar.headSplat ? +(query.get("wait") ?? 5000) : 0); performance.now() < end;) {
    renderer.render(scene, camera);
    await new Promise(requestAnimationFrame);
  }
  renderer.render(scene, camera);
  document.title = "ready";
}

const file = query.get("file");
if (file) {
  message.textContent = "Loading…";
  fetch(file).then((r) => r.arrayBuffer()).then(show).then(() => (shot ? renderShot() : tick())).catch((e) => { message.textContent = e.message; });
} else {
  tick();
}

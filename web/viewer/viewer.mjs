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

function frame() {
  if (!avatar) return;
  const aspect = innerWidth / innerHeight;
  camera.aspect = aspect;
  const [w, h] = avatar.header.canvas;
  const head = avatar.boneByName.head.getWorldPosition(new THREE.Vector3());
  const view = focus === "body" ? 1.04 : 0.42;
  const y = focus === "body" ? 0.5 : head.y + 0.06;
  const fit = Math.max(view, (view * (w / h)) / aspect);
  camera.position.set(0, y, fit / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))));
  camera.lookAt(0, y, 0);
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight, false);
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
  button("params", () => panel.classList.toggle("open"));
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

async function toggleMic(b) {
  if (mouth) {
    mouth.stream.getTracks().forEach((t) => t.stop());
    mouth.context.close();
    mouth = null;
    avatar.setParam("mouthOpen", 0);
    b.classList.remove("on");
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  context.createMediaStreamSource(stream).connect(analyser);
  mouth = { stream, context, analyser, data: new Float32Array(512), level: 0 };
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
  if (state) for (const name in state) avatar.setParam(name, state[name]);
}

function micLevel() {
  mouth.analyser.getFloatTimeDomainData(mouth.data);
  let sum = 0;
  for (const v of mouth.data) sum += v * v;
  const level = Math.min(Math.sqrt(sum / mouth.data.length) * 7, 1);
  mouth.level += (level - mouth.level) * 0.45;
  avatar.setParam("mouthOpen", mouth.level);
}

async function show(buffer) {
  if (avatar) { scene.remove(avatar); avatar.dispose(); }
  avatar = await LiveAvatar.parse(buffer, renderer);
  scene.add(avatar);
  message.textContent = "";
  frame();
  if (!shot) {
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

addEventListener("resize", frame);
addEventListener("pointermove", (e) => {
  pointer.x = (e.clientX / innerWidth) * 2 - 1;
  pointer.y = 1 - (e.clientY / innerHeight) * 2;
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
  avatar.update(dt);
  renderer.render(scene, camera);
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

// Runtime for .lav avatars. The WebGLRenderer must be created with { stencil: true }: eyelids clip the iris with it.
import * as THREE from "three";
import { speechClip } from "./speech.mjs";

const decoder = new TextDecoder();
const FADE = 0.25;
const SPLAT_IN = THREE.MathUtils.degToRad(8);
const SPLAT_FULL = THREE.MathUtils.degToRad(18);
const LAYERS_OUT = THREE.MathUtils.degToRad(30);
// Turns add up along the spine. Flat layers shear when turned far, so the head stops early and the eyes carry the rest of the look.
const MAX_TURN = THREE.MathUtils.degToRad(40);
const MAX_TURN_FLAT = THREE.MathUtils.degToRad(22);
// Looking up opens a gap under the chin sooner than looking down. rotation.x is negative looking up.
const MAX_UP = THREE.MathUtils.degToRad(10);
const MAX_DOWN = THREE.MathUtils.degToRad(18);
const MAX_PITCH_SPLAT = THREE.MathUtils.degToRad(30);
// Drawn mouths standing in for each vowel, best first. Without any, the morph mouth opens by MOUTH_OPEN.
const DRAWN_MOUTH = { a: ["vowelA", "vowelO", "vowelI"], i: ["vowelI", "vowelA"], u: ["vowelO", "vowelA"], e: ["vowelI", "vowelA"], o: ["vowelO", "vowelA"] };
const MOUTH_OPEN = { a: 1, i: 0.45, u: 0.5, e: 0.7, o: 0.8 };

// Shading from the normal atlas. Normals follow the bones through the skin matrix; the light stays in world space.
const LIGHT_VERTEX = ["#include <skinnormal_vertex>", "#include <skinnormal_vertex>\n  vLightRot = mat3(modelMatrix) * mat3(skinMatrix);"];
const LIGHT_FRAGMENT = ["#include <map_fragment>", `#include <map_fragment>
  if (lavLight > 0.0) {
    vec3 n = normalize(vLightRot * (texture(lavNormals, vMapUv).xyz * 2.0 - 1.0));
    float lit = smoothstep(-0.3, 0.45, dot(n, lavLightDir));
    diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * mix(lavShadow, lavLightColor, lit), lavLight);
  }`];

export function parseHeader(buffer) {
  const view = new DataView(buffer);
  if (decoder.decode(new Uint8Array(buffer, 0, 4)) !== "LAV1") throw new Error("Not a .lav file");
  const length = view.getUint32(4, true);
  const header = JSON.parse(decoder.decode(new Uint8Array(buffer, 8, length)));
  return { header, base: 8 + length };
}

function smooth(t) {
  return t * t * (3 - 2 * t);
}

// Dense tracks (captured motion, speech) interpolate linearly: easing every key would stutter.
function sample(keys, time, linear) {
  if (time <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (time <= keys[i][0]) {
      const [t0, v0] = keys[i - 1];
      const [t1, v1] = keys[i];
      const t = (time - t0) / Math.max(t1 - t0, 1e-5);
      return v0 + (v1 - v0) * (linear ? t : smooth(t));
    }
  }
  return keys[keys.length - 1][1];
}

export class LiveAvatar extends THREE.Group {
  // renderer is only needed for avatars with a head splat.
  static async load(url, renderer) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}`);
    return LiveAvatar.parse(await response.arrayBuffer(), renderer);
  }

  static async parse(buffer, renderer) {
    const { header, base } = parseHeader(buffer);
    const [offset, length] = header.atlas.section;
    const blob = new Blob([new Uint8Array(buffer, base + offset, length)], { type: header.atlas.mime });
    const bitmap = await createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
    let normals = null;
    if (header.atlas.normals) {
      const [o, n] = header.atlas.normals;
      normals = await createImageBitmap(new Blob([new Uint8Array(buffer, base + o, n)], { type: header.atlas.mime }), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
    }
    const avatar = new LiveAvatar(header, buffer, base, bitmap, normals);
    if (header.headSplat) await avatar.attachHeadSplat(renderer, new Uint8Array(buffer, base + header.headSplat.section[0], header.headSplat.section[1]));
    return avatar;
  }

  // The splat sits under the face layers, so they can fade away over it without showing what is behind the head.
  async attachHeadSplat(renderer, bytes) {
    const { SparkRenderer, SplatMesh } = await import("../three/spark.module.min.mjs");
    const { position, scale, fade } = this.header.headSplat;
    this.spark = new SparkRenderer({ renderer });
    this.spark.renderOrder = this.meshes.face.renderOrder - 0.5;
    this.add(this.spark);
    this.headSplat = new SplatMesh({ fileBytes: bytes, fileType: "spz" });
    await this.headSplat.initialized;
    this.headSplat.position.fromArray(position);
    this.headSplat.scale.setScalar(scale);
    this.boneByName.head.add(this.headSplat);
    this.headFade = fade.filter((name) => this.meshes[name]).map((name) => this.meshes[name]);
  }

  // Total yaw ("y") or pitch ("x") of the head seen from the front.
  headTurn(axis = "y") {
    return ["spine", "chest", "neck", "head"].reduce((sum, name) => sum + this.boneByName[name].rotation[axis], 0);
  }

  // Past 70% of the limit the head eases toward it, so it never stops dead. Returns the share of the limit held back.
  limitHead(axis, limit) {
    const knee = limit * 0.7;
    const turn = this.headTurn(axis);
    const size = Math.abs(turn);
    if (size <= knee) return 0;
    const eased = knee + (limit - knee) * Math.tanh((size - knee) / (limit - knee));
    this.boneByName.head.rotation[axis] -= Math.sign(turn) * (size - eased);
    return (Math.sign(turn) * (size - eased)) / limit;
  }

  // What the head holds back, the eyes carry: gaze param plus the held share, kept within -1..1.
  carryGaze(param, held, values) {
    if (!held || !this.params[param]) return;
    const gaze = Math.min(Math.max(values[param] ?? 0, -1), 1);
    const extra = Math.min(Math.max(gaze + held * 2, -1), 1) - gaze;
    for (const { bone, channel, scale } of this.params[param].drivers) this.drive(this.boneByName[bone], channel, extra * scale);
  }

  // The splat rises under the layers from SPLAT_IN to SPLAT_FULL. The layers fade over it from halfway up until LAYERS_OUT.
  blendHead() {
    const head = this.boneByName.head.rotation;
    const neck = this.boneByName.neck.rotation;
    const turn = Math.hypot(this.headTurn(), (head.x + neck.x) * 0.7);
    const ramp = (a, b) => smooth(Math.min(Math.max((turn - a) / (b - a), 0), 1));
    const alwaysOn = this.header.headSplat.alwaysOn;
    this.headSplat.opacity = alwaysOn ? 1 : ramp(SPLAT_IN, SPLAT_FULL);
    this.headSplat.visible = this.headSplat.opacity > 0;
    const layers = 1 - ramp(alwaysOn ? SPLAT_IN : (SPLAT_IN + SPLAT_FULL) / 2, LAYERS_OUT);
    for (const mesh of this.headFade) mesh.material.opacity = layers;
  }

  constructor(header, buffer, base, bitmap, normals) {
    super();
    this.header = header;
    this.values = {};
    this.actions = [];
    this.expressions = header.expressions ?? {};
    this.face = {};
    this.expression = null;
    this.autoBlink = true;
    this.nextBlink = 2;
    this.blinkTime = -1;
    this.time = 0;
    this.clips = header.clips;
    this.params = header.params;

    const texture = new THREE.Texture(bitmap);
    texture.flipY = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.generateMipmaps = true;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.anisotropy = 4;
    texture.needsUpdate = true;
    this.texture = texture;
    this.normalTexture = normals ? new THREE.Texture(normals) : null;
    if (this.normalTexture) {
      this.normalTexture.flipY = false;
      this.normalTexture.needsUpdate = true;
    }
    this.light = {
      lavLight: { value: 0 }, lavLightDir: { value: new THREE.Vector3(0.4, 0.5, 0.8).normalize() },
      lavLightColor: { value: new THREE.Color(1.05, 1.03, 1.0) }, lavShadow: { value: new THREE.Color(0.62, 0.58, 0.74) },
      lavNormals: { value: this.normalTexture },
    };

    this.bones = header.bones.map((b) => {
      const bone = new THREE.Bone();
      bone.name = b.name;
      return bone;
    });
    // A bone with a roll hangs from a pivot holding it, so its own rotations and its children work in the leaning frame.
    const lean = [];
    const depth = new THREE.Vector3(0, 0, 1);
    header.bones.forEach((b, i) => {
      const parent = b.parent >= 0 ? header.bones[b.parent].position : [0, 0, 0];
      const offset = new THREE.Vector3(b.position[0] - parent[0], b.position[1] - parent[1], b.position[2] - parent[2]).applyAxisAngle(depth, -(lean[b.parent] ?? 0));
      let holder = b.parent >= 0 ? this.bones[b.parent] : this;
      if (b.roll) {
        const pivot = new THREE.Object3D();
        pivot.position.copy(offset);
        pivot.rotation.z = b.roll;
        holder.add(pivot);
        holder = pivot;
        offset.set(0, 0, 0);
      }
      this.bones[i].position.copy(offset);
      holder.add(this.bones[i]);
      lean[i] = (lean[b.parent] ?? 0) + (b.roll ?? 0);
    });
    this.boneByName = Object.fromEntries(this.bones.map((b) => [b.name, b]));
    this.rest = this.bones.map((b) => b.position.clone());
    this.updateMatrixWorld(true);
    this.skeleton = new THREE.Skeleton(this.bones);

    const f32 = ([o, n]) => new Float32Array(buffer.slice(base + o, base + o + n));
    const u16 = ([o, n]) => new Uint16Array(buffer.slice(base + o, base + o + n));
    const u32 = ([o, n]) => new Uint32Array(buffer.slice(base + o, base + o + n));
    this.meshes = {};
    const identity = new THREE.Matrix4();
    for (const part of header.parts) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(f32(part.position), 3));
      geometry.setAttribute("uv", new THREE.BufferAttribute(f32(part.uv), 2));
      geometry.setAttribute("skinIndex", new THREE.BufferAttribute(u16(part.skinIndex), 4));
      geometry.setAttribute("skinWeight", new THREE.BufferAttribute(f32(part.skinWeight), 4));
      geometry.setIndex(new THREE.BufferAttribute(u32(part.index), 1));
      const morphNames = Object.keys(part.morphs);
      if (morphNames.length) {
        geometry.morphTargetsRelative = true;
        geometry.morphAttributes.position = morphNames.map((name) => {
          const attribute = new THREE.BufferAttribute(f32(part.morphs[name]), 3);
          attribute.name = name;
          return attribute;
        });
      }
      const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide });
      if (this.normalTexture) {
        material.onBeforeCompile = (shader) => {
          Object.assign(shader.uniforms, this.light);
          shader.vertexShader = "varying mat3 vLightRot;\n" + shader.vertexShader.replace(...LIGHT_VERTEX);
          shader.fragmentShader = "uniform sampler2D lavNormals; uniform float lavLight; uniform vec3 lavLightDir, lavLightColor, lavShadow;\nvarying mat3 vLightRot;\n"
            + shader.fragmentShader.replace(...LIGHT_FRAGMENT);
        };
        material.customProgramCacheKey = () => "lav-light";
      }
      const side = part.name.endsWith("_l") ? 1 : 2;
      if (part.name.startsWith("eye_")) {
        material.alphaTest = 0.5;
        material.stencilWrite = true;
        material.stencilRef = side;
        material.stencilFunc = THREE.AlwaysStencilFunc;
        material.stencilZPass = THREE.ReplaceStencilOp;
      } else if (part.name.startsWith("iris_")) {
        material.stencilWrite = true;
        material.stencilRef = side;
        material.stencilFunc = THREE.EqualStencilFunc;
      }
      const mesh = new THREE.SkinnedMesh(geometry, material);
      mesh.name = part.name;
      mesh.renderOrder = part.order;
      mesh.frustumCulled = false;
      mesh.bind(this.skeleton, identity);
      if (morphNames.length) mesh.updateMorphTargets();
      this.add(mesh);
      this.meshes[part.name] = mesh;
    }

    this.springs = header.springs.filter((s) => this.boneByName[s.bone]).map((s) => {
      const node = this.boneByName[s.bone];
      // Where the tip of the bone goes when it turns: square to the way it points at rest, which is down unless a next bone says otherwise.
      const at = header.bones[this.bones.indexOf(node)].position;
      const next = header.bones.find((b) => header.bones[b.parent]?.name === s.bone)?.position;
      const across = next ? new THREE.Vector2(at[1] - next[1], next[0] - at[0]).normalize() : new THREE.Vector2(1, 0);
      return { ...s, node, across, angle: 0, speed: 0, last: new THREE.Vector3(), primed: false,
        offset: new THREE.Vector2(), velocity: new THREE.Vector2(), lastVelocity: new THREE.Vector3() };
    });
  }

  setParam(name, value) {
    this.values[name] = value;
  }

  // x and y in -1..1 of where the light comes from, y up. strength 0 turns shading off.
  setLight({ x = 0.4, y = 0.5, strength = 0.8, color, shadow } = {}) {
    this.light.lavLightDir.value.set(x, y, 1).normalize();
    this.light.lavLight.value = this.normalTexture ? strength : 0;
    if (color) this.light.lavLightColor.value.set(color);
    if (shadow) this.light.lavShadow.value.set(shadow);
  }

  // Holds a face from avatar.expressions over the params; null goes back to neutral.
  setExpression(name, fade = 0.3) {
    if (name && !this.expressions[name]) throw new Error(`Unknown expression ${name}`);
    this.expression = { from: { ...this.face }, to: name ? this.expressions[name] : {}, time: 0, fade };
  }

  // Param values for a mouth shape: level 0..1, vowel a, i, u, e or o. Drawn mouths replace the morph when the avatar has them.
  mouthValues(level, vowel = "a") {
    const drawn = DRAWN_MOUTH[vowel].find((name) => this.params[name]);
    const values = { mouthOpen: drawn ? 0 : level * MOUTH_OPEN[vowel] };
    for (const name of ["vowelA", "vowelI", "vowelO"]) if (this.params[name]) values[name] = name === drawn ? level : 0;
    return values;
  }

  setMouth(level, vowel = "a") {
    Object.assign(this.values, this.mouthValues(level, vowel));
  }

  // Lip sync from text: kana, or Latin letters as in Spanish or English. Returns the duration in seconds.
  speak(text, { rate = 8 } = {}) {
    const clip = speechClip(text, (level, vowel) => this.mouthValues(level, vowel), rate);
    this.actions = this.actions.filter((l) => l.name !== "speech");
    this.actions.push({ name: "speech", clip, time: 0, loop: false, weight: 1, fade: 0.05, stopping: false });
    return clip.duration;
  }

  // loop defaults to the clip's own setting. Clips add on top of each other and of setParam values.
  play(name, { loop, fade = FADE } = {}) {
    const clip = this.clips[name];
    if (!clip) throw new Error(`Unknown clip ${name}`);
    this.actions = this.actions.filter((l) => l.name !== name);
    this.actions.push({ name, clip, time: 0, loop: loop ?? clip.loop, weight: 0, fade, stopping: false });
  }

  stop(name, fade = FADE) {
    for (const layer of this.actions) if (layer.name === name) { layer.stopping = true; layer.fade = fade; }
  }

  // x and y in -1..1 of where to look, y up.
  lookAt(x, y) {
    this.values.headYaw = x * 0.5;
    this.values.headPitch = -y * 0.45;
    this.values.gazeX = x;
    this.values.gazeY = y;
  }

  update(dt) {
    this.time += dt;
    const values = { ...this.values };
    for (const layer of this.actions) {
      layer.time += dt;
      const rate = dt / Math.max(layer.fade, 1e-3);
      layer.weight = layer.stopping ? Math.max(layer.weight - rate, 0) : Math.min(layer.weight + rate, 1);
      const time = layer.loop ? layer.time % layer.clip.duration : layer.time;
      for (const track of layer.clip.tracks) {
        values[track.param] = (values[track.param] ?? 0) + sample(track.keys, time, track.linear) * layer.weight;
      }
    }
    this.actions = this.actions.filter((l) => (l.loop || l.time < l.clip.duration) && !(l.stopping && l.weight === 0));

    if (this.expression) {
      const { from, to, fade } = this.expression;
      this.expression.time += dt;
      const k = smooth(Math.min(this.expression.time / Math.max(fade, 1e-3), 1));
      this.face = {};
      for (const name of new Set([...Object.keys(from), ...Object.keys(to)])) this.face[name] = (from[name] ?? 0) + ((to[name] ?? 0) - (from[name] ?? 0)) * k;
      if (k === 1) this.expression = null;
    }
    for (const name in this.face) values[name] = (values[name] ?? 0) + this.face[name];

    if (this.autoBlink) {
      if (this.time > this.nextBlink) {
        this.blinkTime = 0;
        this.nextBlink = this.time + 2 + Math.random() * 4;
      }
      if (this.blinkTime >= 0) {
        this.blinkTime += dt;
        const t = this.blinkTime / 0.16;
        const closed = t < 1 ? t : Math.max(2 - t, 0);
        values.blinkL = Math.max(values.blinkL ?? 0, closed);
        values.blinkR = Math.max(values.blinkR ?? 0, closed);
        if (t >= 2) this.blinkTime = -1;
      }
    }

    this.bones.forEach((bone, i) => {
      bone.position.copy(this.rest[i]);
      bone.rotation.set(0, 0, 0);
      bone.scale.set(1, 1, 1);
    });
    const opacity = {};
    for (const [name, param] of Object.entries(this.params)) {
      const value = values[name] ?? 0;
      for (const [part, start, end] of param.fades ?? []) {
        opacity[part] = Math.max(opacity[part] ?? 0, smooth(Math.min(Math.max((value - start) / (end - start), 0), 1)));
      }
      if (param.drivers) {
        for (const { bone, channel, scale } of param.drivers) this.drive(this.boneByName[bone], channel, value * scale);
      }
      if (param.morphs) {
        for (const part of param.morphs) {
          const mesh = this.meshes[part];
          mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = Math.min(Math.max(value, -1), 1);
        }
      }
    }
    for (const [part, o] of Object.entries(opacity)) {
      this.meshes[part].material.opacity = o;
      this.meshes[part].visible = o > 0.003;
    }
    this.carryGaze("gazeX", this.limitHead("y", this.headSplat ? MAX_TURN : MAX_TURN_FLAT), values);
    const up = this.headTurn("x") < 0;
    // gazeY is positive looking up, while rotation.x is negative there.
    this.carryGaze("gazeY", -this.limitHead("x", this.headSplat ? MAX_PITCH_SPLAT : up ? MAX_UP : MAX_DOWN), values);
    if (this.headSplat) this.blendHead();
    this.updateSprings(dt);
  }

  drive(bone, channel, value) {
    switch (channel) {
      case "rx": bone.rotation.x += value; break;
      case "ry": bone.rotation.y += value; break;
      case "rz": bone.rotation.z += value; break;
      case "px": bone.position.x += value; break;
      case "py": bone.position.y += value; break;
      case "sx": bone.scale.x += value; break;
      case "sy": bone.scale.y += value; break;
    }
  }

  // Each spring is a damped angle pushed by the acceleration of its bone across the way it points. A strand lags behind its root.
  updateSprings(dt) {
    this.updateMatrixWorld(true);
    const step = Math.min(dt, 1 / 30);
    if (step <= 0) return;
    const position = new THREE.Vector3();
    const velocity = new THREE.Vector3();
    for (const spring of this.springs) {
      if (spring.mode === "bounce") {
        this.bounce(spring, step);
        continue;
      }
      position.setFromMatrixPosition(spring.node.parent.matrixWorld);
      if (!spring.primed) {
        spring.last.copy(position);
        spring.primed = true;
      }
      velocity.subVectors(position, spring.last).divideScalar(step);
      const push = ((velocity.x - spring.lastVelocity.x) * spring.across.x + (velocity.y - spring.lastVelocity.y) * spring.across.y) / step;
      spring.last.copy(position);
      spring.lastVelocity.copy(velocity);
      const wind = Math.sin(this.time * 1.3 + spring.gain * 3) * 0.3;
      const force = -spring.stiffness * spring.angle - spring.damping * spring.speed - push * spring.gain * 4 + wind * spring.gain;
      spring.speed += force * step;
      spring.angle = Math.min(Math.max(spring.angle + spring.speed * step, -0.8), 0.8);
      spring.node.rotation.z += spring.angle;
    }
    this.updateMatrixWorld(true);
  }

  // A soft body: the bone's offset is a damped spring pushed against its own acceleration, so it lags, overshoots and settles.
  bounce(spring, step) {
    const position = spring.node.getWorldPosition(new THREE.Vector3());
    if (!spring.primed) {
      spring.last.copy(position);
      spring.primed = true;
    }
    const velocity = position.clone().sub(spring.last).divideScalar(step);
    const push = velocity.clone().sub(spring.lastVelocity).divideScalar(step);
    spring.last.copy(position);
    spring.lastVelocity.copy(velocity);
    for (const axis of ["x", "y"]) {
      const force = -spring.stiffness * spring.offset[axis] - spring.damping * spring.velocity[axis] - push[axis] * spring.gain;
      spring.velocity[axis] += force * step;
      spring.offset[axis] = Math.min(Math.max(spring.offset[axis] + spring.velocity[axis] * step, -spring.reach), spring.reach);
    }
    spring.node.position.x += spring.offset.x;
    spring.node.position.y += spring.offset.y;
  }

  dispose() {
    for (const mesh of Object.values(this.meshes)) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.texture.dispose();
    this.normalTexture?.dispose();
    this.skeleton.dispose();
  }
}

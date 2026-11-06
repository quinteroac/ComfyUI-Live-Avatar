// Runtime for .lav avatars. The WebGLRenderer must be created with { stencil: true }: eyelids clip the iris with it.
import * as THREE from "three";

const decoder = new TextDecoder();
const FADE = 0.25;
const SPLAT_IN = THREE.MathUtils.degToRad(8);
const SPLAT_FULL = THREE.MathUtils.degToRad(18);
const LAYERS_OUT = THREE.MathUtils.degToRad(30);
// Turns add up along the spine. Past these limits the eye layers would leave the face, so the head gives way.
const MAX_TURN = THREE.MathUtils.degToRad(40);
const MAX_TURN_FLAT = THREE.MathUtils.degToRad(34);

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

function sample(keys, time) {
  if (time <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (time <= keys[i][0]) {
      const [t0, v0] = keys[i - 1];
      const [t1, v1] = keys[i];
      return v0 + (v1 - v0) * smooth((time - t0) / Math.max(t1 - t0, 1e-5));
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
    const avatar = new LiveAvatar(header, buffer, base, bitmap);
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

  // Total yaw of the head seen from the front.
  headTurn() {
    return ["spine", "chest", "neck", "head"].reduce((sum, name) => sum + this.boneByName[name].rotation.y, 0);
  }

  // Past 70% of the limit the head eases toward it, so it never stops dead.
  limitHeadTurn() {
    const limit = this.headSplat ? MAX_TURN : MAX_TURN_FLAT;
    const knee = limit * 0.7;
    const turn = this.headTurn();
    const size = Math.abs(turn);
    if (size <= knee) return;
    const eased = knee + (limit - knee) * Math.tanh((size - knee) / (limit - knee));
    this.boneByName.head.rotation.y -= Math.sign(turn) * (size - eased);
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

  constructor(header, buffer, base, bitmap) {
    super();
    this.header = header;
    this.values = {};
    this.actions = [];
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

    this.bones = header.bones.map((b) => {
      const bone = new THREE.Bone();
      bone.name = b.name;
      return bone;
    });
    header.bones.forEach((b, i) => {
      const parent = b.parent >= 0 ? header.bones[b.parent].position : [0, 0, 0];
      this.bones[i].position.set(b.position[0] - parent[0], b.position[1] - parent[1], b.position[2] - parent[2]);
      (b.parent >= 0 ? this.bones[b.parent] : this).add(this.bones[i]);
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

    this.springs = header.springs.filter((s) => this.boneByName[s.bone]).map((s) => ({
      ...s, node: this.boneByName[s.bone], angle: 0, speed: 0, last: new THREE.Vector3(), lastSpeed: 0, primed: false,
    }));
  }

  setParam(name, value) {
    this.values[name] = value;
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
    this.values.headYaw = x * 0.8;
    this.values.headPitch = -y * 0.6;
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
        values[track.param] = (values[track.param] ?? 0) + sample(track.keys, time) * layer.weight;
      }
    }
    this.actions = this.actions.filter((l) => (l.loop || l.time < l.clip.duration) && !(l.stopping && l.weight === 0));

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
    for (const [name, param] of Object.entries(this.params)) {
      const value = values[name] ?? 0;
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
    this.limitHeadTurn();
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

  // Each spring is a damped angle pushed by the sideways acceleration of its bone. A hanging strand lags behind its root.
  updateSprings(dt) {
    this.updateMatrixWorld(true);
    const step = Math.min(dt, 1 / 30);
    if (step <= 0) return;
    const position = new THREE.Vector3();
    for (const spring of this.springs) {
      position.setFromMatrixPosition(spring.node.parent.matrixWorld);
      if (!spring.primed) {
        spring.last.copy(position);
        spring.primed = true;
      }
      const speed = (position.x - spring.last.x) / step;
      const push = (speed - spring.lastSpeed) / step;
      spring.last.copy(position);
      spring.lastSpeed = speed;
      const wind = Math.sin(this.time * 1.3 + spring.gain * 3) * 0.3;
      const force = -spring.stiffness * spring.angle - spring.damping * spring.speed - push * spring.gain * 4 + wind * spring.gain;
      spring.speed += force * step;
      spring.angle = Math.min(Math.max(spring.angle + spring.speed * step, -0.8), 0.8);
      spring.node.rotation.z += spring.angle;
    }
    this.updateMatrixWorld(true);
  }

  dispose() {
    for (const mesh of Object.values(this.meshes)) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    this.texture.dispose();
    this.skeleton.dispose();
  }
}

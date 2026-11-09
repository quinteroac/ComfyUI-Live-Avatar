// Webcam face tracking with MediaPipe, fully local: wasm and model are served next to this file.
import * as THREE from "three";
import { FaceLandmarker } from "../mediapipe/vision_bundle.mjs";

const SMOOTH = 0.5;
const euler = new THREE.Euler();
const matrix = new THREE.Matrix4();

export class FaceTracker {
  static async create(base = new URL("../mediapipe/", import.meta.url).href) {
    const landmarker = await FaceLandmarker.createFromOptions(
      { wasmLoaderPath: base + "vision_wasm_internal.mjs", wasmBinaryPath: base + "vision_wasm_internal.wasm" },
      { baseOptions: { modelAssetPath: base + "face_landmarker.task", delegate: "GPU" }, runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true },
    );
    return new FaceTracker(landmarker);
  }

  constructor(landmarker) {
    this.landmarker = landmarker;
    this.state = {};
    this.vowel = "a";
  }

  // Maps one frame to avatar params, mirrored so the avatar behaves like a mirror. Returns null without a face.
  read(source, timeMs) {
    const result = this.landmarker.detectForVideo(source, timeMs);
    const shapes = result.faceBlendshapes?.[0]?.categories;
    const pose = result.facialTransformationMatrixes?.[0];
    if (!shapes || !pose) return null;
    const b = Object.fromEntries(shapes.map((c) => [c.categoryName, c.score]));
    euler.setFromRotationMatrix(matrix.fromArray(pose.data), "YXZ");
    const clamp = (v) => Math.min(Math.max(v, -1), 1);
    const target = {
      headYaw: clamp(-euler.y / 0.6),
      headPitch: clamp(euler.x / 0.45),
      headRoll: clamp(-euler.z / 0.5),
      bodyRoll: clamp(-euler.z / 0.9),
      blinkL: Math.max(clamp((b.eyeBlinkRight - 0.15) / 0.5), 0),
      blinkR: Math.max(clamp((b.eyeBlinkLeft - 0.15) / 0.5), 0),
      mouthOpen: clamp(b.jawOpen * 1.7),
      mouthSmile: clamp((b.mouthSmileLeft + b.mouthSmileRight) * 0.8),
      browL: clamp(b.browInnerUp + b.browOuterUpRight),
      browR: clamp(b.browInnerUp + b.browOuterUpLeft),
      gazeX: clamp((b.eyeLookInLeft + b.eyeLookOutRight - b.eyeLookOutLeft - b.eyeLookInRight) * 0.6),
      gazeY: clamp((b.eyeLookUpLeft + b.eyeLookUpRight - b.eyeLookDownLeft - b.eyeLookDownRight) * 0.6),
    };
    for (const name in target) this.state[name] = (this.state[name] ?? 0) + (target[name] - (this.state[name] ?? 0)) * SMOOTH;
    // Rounded lips read as o, stretched lips over a small jaw as i, the rest as a.
    const round = b.mouthFunnel + b.mouthPucker;
    const wide = (b.mouthStretchLeft + b.mouthStretchRight) / 2 + (b.mouthSmileLeft + b.mouthSmileRight) * 0.25;
    this.vowel = round > 0.35 ? "o" : wide > 0.3 && b.jawOpen < 0.35 ? "i" : "a";
    return this.state;
  }

  close() {
    this.landmarker.close();
  }
}

# ComfyUI-Live-Avatar

Turns one anime character image into a rigged 2.5D avatar in its own `.lav` format, played by a small three.js runtime with clips, spring physics, mic lip sync and webcam face tracking.

Needs [ComfyUI-WeVisOne-VL](https://huggingface.co/paralaif/ComfyUI-WeVisOne-VL). See-through is optional.

## Nodes

| Node | What it does |
|---|---|
| **Live Avatar Segment** | Splits the character with WeVisOne-VL: body parts on the full image, eyes, iris, lashes, brows and mouth on an enlarged head crop, plus joint points. |
| **Live Avatar Head Mask** | Matte of the head alone, for rebuilding it with Idol. |
| **Live Avatar Plate** | Prepares the body hidden behind arms and hair for any inpaint pass. |
| **Live Avatar Variant Mask** | Head crop and mask for inpainting one drawn variant: `eyes_closed`, `eyes_smile`, `mouth_a`, `mouth_i` or `mouth_o`. |
| **Live Avatar Variants** | Collects the inpainted variants for Build. Only each variant's masked area is used, with the crop's color shift removed. |
| **Live Avatar Build** | Builds layers, meshes, bones, morphs and clips. Optional: `SEETHROUGH_PARTS` for hidden pixels and depth, `depth` for relief, `plate` for the inpainted body, `head_splat` for wide head turns, `variants` for drawn eyes and mouths. |
| **Live Avatar Load** | Loads a saved `.lav` from `output/live_avatar`, to add clips or export it again without rebuilding. |
| **Live Avatar Motion** | Adds a clip retargeted from captured body motion: `MHR_POSE_DATA` from [InstantHMR](https://github.com/jtydhr88/ComfyUI-InstantHMR) or SAM 3D Body. |
| **Live Avatar Save** | Writes `output/live_avatar/*.lav` and shows a live viewer on the node. |
| **Live Avatar Export** | Writes a self-contained web package and `.zip` to `output/live_avatar_export`: page, runtime, avatar and a local server. Its `README.txt` covers OBS and your own app. |

## Workflows

In `example_workflows`:

- `live_avatar_wevisone.json`: WeVisOne, Depth Anything relief and an Anima inpaint of the hidden body. Describe the clothes in the prompt. Without `plate` hidden pixels are extended from visible ones.
- `live_avatar_splat_head.json`: the same plus an Idol head splat, about 150 s. Needs the Idol models listed in core's `3d_idol` template.
- `live_avatar_seethrough_splat_head.json`: See-through for the hidden body and depth, Idol for the head. About 9 minutes on a 16 GB GPU.
- `live_avatar_seethrough_wevisone.json`: WeVisOne picks the parts, See-through paints what is behind them. About 6 minutes on a 16 GB GPU.
- `live_avatar_variants.json`: `live_avatar_wevisone.json` plus Anima inpaints of closed and smiling eyes and the a, i and o mouths. Put the character's tags in the variant prompts but not the eye color in the closed-eye ones, or Anima paints the eyes open again.
- `live_avatar_motion.json`: a video through InstantHMR onto a saved avatar as a new clip. The performer's usual facing and lean become the avatar's front, so the camera angle does not matter.

Use a front-facing character with arms apart on a plain background. A larger face gives better eyes and mouth: at 1216 px tall a full-body face is only about 100 px.

Known limits: big arm raises leave a wedge at the shoulder of loose sleeves, `walk` is a front-view step in place, and without a head splat, turns past about 25 degrees show the flat layers.

## Using it in three.js

Copy `web/runtime` and, for webcam tracking, `web/mediapipe`. The renderer needs a stencil buffer.

```js
import * as THREE from "three";
import { LiveAvatar } from "./runtime/live-avatar.mjs";

const renderer = new THREE.WebGLRenderer({ stencil: true, alpha: true });
const avatar = await LiveAvatar.load("avatar.lav");
scene.add(avatar);            // 1 unit = canvas height, feet near y = 0
avatar.play("idle");

function frame(dt) {
  avatar.lookAt(mouseX, mouseY);
  avatar.setParam("mouthOpen", voiceLevel);
  avatar.update(dt);
  renderer.render(scene, camera);
}
```

- `avatar.play(name, { loop, fade })`, `avatar.stop(name)`: clips add to each other and to params.
- `avatar.setParam(name, value)`: values from -1 to 1. The list is in `avatar.params`.
- `avatar.lookAt(x, y)`: head and eyes, -1 to 1.
- `avatar.setExpression(name)`: holds a face from `avatar.expressions` (smile, surprise, angry, sad, wink, sleepy, shy); `null` returns to neutral.
- `avatar.setMouth(level, vowel)`: mouth from 0 to 1 for `a`, `i`, `u`, `e` or `o`. Drawn mouths replace the morph when the avatar has them.
- `avatar.speak(text, { rate })`: lip sync from kana or Latin text, such as Spanish or English. Returns the duration.
- `avatar.setLight({ x, y, strength, color, shadow })`: cel shading from the normal atlas, light direction in -1..1; strength 0 is off. The head splat is not shaded.
- `FaceTracker` in `runtime/face-tracker.mjs` turns a video frame into params, and `tracker.vowel` into a mouth shape.

Avatars with a head splat need the renderer at load, `LiveAvatar.load(url, renderer)`, and `web/three/spark.module.min.mjs`. Your import map must also map `three/addons/postprocessing/Pass.js`.

### Head splat

Idol rebuilds the head alone. Build aligns it to the face, fits it to the layers' outline, matches its brightness to the drawing and paints out its eyes and mouth. Those stay as layers on its surface, so they keep blinking and talking.

`splat_hair` on Live Avatar Build chooses how it joins the layers. On, the splat draws the hair at every angle and only the skin fades over it, from 8 to 30 degrees: no hair transition, slightly softer hair when facing front. Off, the whole head crossfades from the drawn layers, which keeps the drawn hair at the front but shows double strands around 18 to 24 degrees.

Clips: `idle`, `wave`, `nod`, `shake`, `happy`, `think`, `talk`, `walk`. Upper-body images have no leg params, so those tracks are dropped.

## The .lav format

Little-endian: `"LAV1"`, `u32` JSON length, JSON header, then binary sections addressed as `[offset, length]` from the end of the header.

```text
version, canvas [w, h]
atlas     section, normals (section, optional), size, mime (lossless WebP); both share the parts' uvs
bones     name, parent index, position
parts     draw order; position (f32 x3), uv (f32 x2), skinIndex (u16 x4), skinWeight (f32 x4), index (u32), morphs {name: f32 x3 deltas}
params    name -> drivers [{bone, channel, scale}], morphs [part names] and/or fades [[part, start, end]]
clips     name -> duration, loop, tracks [{param, keys [[time, value]], linear (optional)}]
expressions name -> {param: value}
springs   bone, stiffness, damping, gain
headSplat optional: section (.spz), scale, position on the head bone, fade (layers it replaces), alwaysOn
```

The viewer adds expression buttons (keys 1 to 8), a text box for `speak`, a `light` toggle (drag over the avatar to move the light) and `poses`, a contact sheet of fixed poses to check a build, saved as PNG.

Vertices carry real depth, so turning the head in 3D gives parallax without extra drawings. Layers keep a fixed draw order and ignore the depth buffer. Everything in the viewer runs locally.

three.js and Spark are MIT and MediaPipe is Apache-2.0; all are bundled in `web`.

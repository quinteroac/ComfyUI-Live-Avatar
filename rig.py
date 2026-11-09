"""Turns a part segmentation into a rigged 2.5D avatar: layers, grid meshes, bones, morphs, params and clips."""
import cv2
import numpy as np
from scipy import ndimage

from .clips import LOCK_BONES, build_clips, build_expressions, build_params, build_springs

SH_C0 = 0.28209479177387814
BACK_DEPTH = 0.6
SIDES = ("l", "r")
# First wins: every character pixel belongs to exactly one layer, so the rest pose reproduces the source image.
PRIORITY = ["iris_l", "iris_r", "lash_l", "lash_r", "eye_l", "eye_r", "brow_l", "brow_r", "mouth", "bangs",
            "animal_ear_l", "animal_ear_r", "hair", "face", "neck", "arm_l", "arm_r", "tail", "skirt", "leg_l", "leg_r"]
# Back to front.
DRAW_ORDER = ["hair_fill", "tail", "leg_r", "leg_l", "torso", "skirt", "neck", "arm_r", "arm_l", "lock_r", "lock_l", "hair_back", "animal_ear_r", "animal_ear_l", "face",
              "mouth_inner", "mouth_up", "mouth_low", "mouth_a", "mouth_i", "mouth_o", "eye_r", "eye_l", "iris_r", "iris_l", "lash_r", "lash_l",
              "closed_r", "closed_l", "smile_r", "smile_l", "brow_r", "brow_l", "hair_top", "bangs"]
HEAD_LAYERS = {"face", "mouth_inner", "mouth_up", "mouth_low", "mouth_a", "mouth_i", "mouth_o", "eye_r", "eye_l", "iris_r", "iris_l", "lash_r", "lash_l",
               "closed_r", "closed_l", "smile_r", "smile_l", "brow_r", "brow_l", "hair_top", "bangs", "hair_fill", "hair_back"}
# Drawn variants become these layers; eye variants split into the character's left and right.
VARIANT_LAYERS = {"eyes_closed": "closed", "eyes_smile": "smile", "mouth_a": "mouth_a", "mouth_i": "mouth_i", "mouth_o": "mouth_o"}
# See-through tags holding the hidden pixels of each layer, first tag on top.
SEETHROUGH_LAYERS = {"face": ["face"], "neck": ["neck"], "torso": ["topwear"], "skirt": ["bottomwear"], "leg_l": ["legwear"], "leg_r": ["legwear"],
                     "arm_l": ["handwear-l", "handwear"], "arm_r": ["handwear-r", "handwear"], "hair_fill": ["back hair"]}


def _from_square(square, width, height):
    """See-through works on a center-padded square. Brings an image of it back to the source canvas."""
    side = max(width, height)
    pad_x, pad_y = (side - width) // 2, (side - height) // 2
    return cv2.resize(square, (side, side), interpolation=cv2.INTER_LINEAR)[pad_y:pad_y + height, pad_x:pad_x + width]


def seethrough_depth(parts, width, height):
    """One depth map from See-through's per-layer depth, near = 1."""
    frame = parts["frame_size"][0]
    square = np.zeros((frame, frame), np.float32)
    for info in sorted(parts["tag2pinfo"].values(), key=lambda i: -i.get("depth_median", 1)):
        if info.get("img") is None or info.get("depth") is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in info["xyxy"])
        solid = info["img"][..., 3] >= 128
        square[y0:y1, x0:x1][solid] = 1 - info["depth"][solid].astype(np.float32) / 255
    return _from_square(square, width, height)


def seethrough_hidden(parts, width, height):
    """Layers holding the pixels hidden behind each avatar layer, on the source canvas."""
    frame = parts["frame_size"][0]
    hidden = {}
    for layer, tags in SEETHROUGH_LAYERS.items():
        square = np.zeros((frame, frame, 4), np.uint8)
        for tag in reversed(tags):
            info = parts["tag2pinfo"].get(tag)
            if info is None or info.get("img") is None:
                continue
            x0, y0, x1, y1 = (int(v) for v in info["xyxy"])
            tile = info["img"]
            target = square[y0:y1, x0:x1]
            target[tile[..., 3] >= 128] = tile[tile[..., 3] >= 128]
        if square[..., 3].any():
            hidden[layer] = _from_square(square, width, height)
    return hidden


def _kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def _dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius)) > 0 if radius > 0 else mask


def _erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), _kernel(radius)) > 0 if radius > 0 else mask


def _close(mask, radius):
    pad = radius + 2
    padded = np.pad(mask.astype(np.uint8), pad)
    return cv2.morphologyEx(padded, cv2.MORPH_CLOSE, _kernel(radius))[pad:-pad, pad:-pad] > 0


def _hull(mask):
    points = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if points is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(points), 1)
    return out > 0


def _largest(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _nearest_fill(rgb, source, region, blur=0.0):
    """Colors of region taken from the closest source pixel, optionally smoothed inside region."""
    if not source.any():
        return rgb
    index = ndimage.distance_transform_edt(~source, return_distances=False, return_indices=True)
    out = rgb.copy()
    out[region] = rgb[index[0][region], index[1][region]]
    if blur > 0:
        smooth = cv2.GaussianBlur(out, (0, 0), blur)
        hidden = region & ~source
        out[hidden] = smooth[hidden]
    return out


def partition(seg):
    """Assign every character pixel to one layer. Returns the label map, layer names and soft alpha."""
    masks = {name: mask for name, mask in seg["masks"].items()}
    for side in SIDES:
        hand = masks.pop("hand_" + side, None)
        if hand is not None:
            masks["arm_" + side] = np.maximum(masks.get("arm_" + side, 0), hand)

    alpha = seg["alpha"] if seg["alpha"] is not None else masks["person"]
    union = np.zeros(alpha.shape, bool)
    for name in PRIORITY:
        if name in masks:
            union |= masks[name] > 0.5
    alpha = np.maximum(alpha, union.astype(np.float32))
    alpha[~_dilate(_largest(alpha > 0.5), 3)] = 0

    names = ["torso"] + [name for name in PRIORITY if name in masks]
    label = np.zeros(alpha.shape, np.int32)
    for i in range(len(names) - 1, 0, -1):
        label[masks[names[i]] > 0.5] = i

    # Limbs and torso are one piece each; other islands and slivers between mattes are matte error.
    character = alpha > 0.02
    for i, name in enumerate(names):
        if name.startswith(("arm_", "leg_")):
            limb = (label == i) & character
            label[limb & ~_largest(limb)] = 0
    rest = character & (label == 0)
    solid = _largest(_dilate(_erode(rest, 2), 2) & rest)
    stray = rest & ~solid
    named = label > 0
    if stray.any() and named.any():
        index = ndimage.distance_transform_edt(~named, return_distances=False, return_indices=True)
        label[stray] = label[index[0][stray], index[1][stray]]
    label[~character] = -1
    return label, names, alpha


class Layers:
    """Full-canvas RGBA layers, each extended under whatever is drawn above it."""

    def __init__(self, seg, hidden=None, plate=None):
        self.rgb = np.ascontiguousarray((seg["image"] * 255).round().astype(np.uint8))
        self.label, names, self.alpha = partition(seg)
        self.height, self.width = self.label.shape
        self.hidden = hidden or {}
        self.plate = None if plate is None else cv2.resize(plate, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.visible = {name: self.label == i for i, name in enumerate(names)}
        solid = self.alpha >= 0.9
        if solid.any():  # Edge colors still contain the old background.
            self.rgb = _nearest_fill(self.rgb, solid, ~solid)
        self.layers = {}
        # A chest matte smaller than this share of the face is matte noise, not a bust that moves.
        face_area = self.visible["face"].sum()
        self.breasts = {}
        for side in SIDES:
            mask = seg["masks"].get("breast_" + side)
            if mask is not None and ((mask > 0.5) & (self.alpha > 0.5)).sum() > face_area * 0.15:
                self.breasts[side] = (mask > 0.5) & (self.alpha > 0.5)
        self._split_hair()
        self._split_face_parts()

    def _split_hair(self):
        hair = self.visible.pop("hair", None)
        face = self.visible["face"]
        x0, _, x1, y1 = _bbox(face)
        self.face_box = _bbox(face)
        self.chin_y = int(y1)
        # Short hair belongs to the head; only what hangs well below the chin moves as strands.
        self.hair_cut = min(int(y1 + (y1 - self.face_box[1]) * 0.45), self.label.shape[0] - 1)
        self.center_x = (x0 + x1) / 2
        if hair is None:
            return
        rows = np.arange(self.height)[:, None]
        cols = np.arange(self.width)[None, :]
        # The hair matte can swallow the neck, and no hair hangs in front of it right under the chin.
        throat = hair & (rows > self.chin_y) & (rows <= self.hair_cut) & (np.abs(cols - self.center_x) < (x1 - x0) * 0.28)
        if "neck" in self.visible:
            self.visible["neck"] |= throat
            hair = hair & ~throat
        self.visible["hair_top"] = hair & (rows <= self.hair_cut)
        below = hair & (rows > self.hair_cut)
        self.visible["lock_l"] = below & (cols >= self.center_x)
        self.visible["lock_r"] = below & (cols < self.center_x)
        for name in ("lock_l", "lock_r"):
            if self.visible[name].sum() < 40:
                self.visible["hair_top"] |= self.visible.pop(name)

    def _split_face_parts(self):
        self.skin = np.median(self.rgb[self.visible["face"]], axis=0)
        for side in SIDES:
            brow = self.visible.get("brow_" + side)
            if brow is None:
                continue
            # Under bangs the brow matte also catches blush and skin, and brows never fade out.
            dark = brow & (self.rgb.astype(np.float32).sum(2) < self.skin.sum() * 0.75)
            if dark.sum() >= brow.sum() * 0.3:
                dark = _largest(dark)
                self.visible["face"] |= brow & ~dark
                self.visible["brow_" + side] = dark
        # One moving iris next to a painted one looks worse than eyes that keep still.
        if "iris_l" not in self.visible or "iris_r" not in self.visible:
            for side in SIDES:
                iris = self.visible.pop("iris_" + side, None)
                if iris is not None and "eye_" + side in self.visible:
                    self.visible["eye_" + side] |= iris
        for side in SIDES:
            lash, eye, iris = ("lash_" + side, "eye_" + side, "iris_" + side)
            if lash in self.visible and eye in self.visible:
                pool = self.visible[eye] | self.visible[lash] | self.visible.get(iris, False)
                mid = (_bbox(pool)[1] + _bbox(pool)[3]) / 2
                lower = self.visible[lash] & (np.arange(self.height)[:, None] > mid)
                self.visible[lash] &= ~lower
                self.visible[eye] |= lower
        mouth = self.visible.pop("mouth", None)
        self.mouth_curve = None
        if mouth is not None and mouth.any():
            ys, xs = np.nonzero(mouth)
            self.mouth_curve = np.polyfit(xs, ys, 2)
            line = np.polyval(self.mouth_curve, np.arange(self.width))[None, :]
            rows = np.arange(self.height)[:, None]
            self.visible["mouth_up"] = mouth & (rows <= line)
            self.visible["mouth_low"] = mouth & (rows > line)
            self.mouth_box = _bbox(mouth)

    def above(self, name):
        """Pixels owned by layers drawn over name."""
        out = np.zeros(self.label.shape, bool)
        for other in DRAW_ORDER[DRAW_ORDER.index(name) + 1:]:
            if other in self.visible:
                out |= self.visible[other]
        return out

    def add(self, name, extension=None, erode_source=1, blur=1.0, source=None):
        visible = self.visible.get(name)
        if visible is None:
            visible = np.zeros(self.label.shape, bool)
        above = self.above(name)
        region = visible | (_dilate(visible, 4) & above)
        if extension is not None:
            region |= extension & above
        rgb = self.rgb
        fill = region & ~visible
        if fill.any():
            if source is None:
                source = _erode(visible, erode_source)
            rgb = _nearest_fill(rgb, source if source.any() else visible, region, blur)
        rgba = np.dstack([rgb, np.zeros(self.label.shape, np.uint8)])
        rgba[..., 3][region] = 255
        rgba[..., 3][visible] = (self.alpha[visible] * 255).round().astype(np.uint8)
        layer_hidden = self.hidden.get(name)
        if layer_hidden is not None:
            use = (layer_hidden[..., 3] > 128) & above & ~visible
            rgba[use] = layer_hidden[use]
            rgba[..., 3][use] = 255
        if name == "torso" and self.plate is not None:
            painted = self.plate_area(rgba)
            rgba[..., :3][painted] = self.plate[painted]
        self.layers[name] = rgba

    def plate_area(self, torso):
        """Hidden body for the inpaint pass. Around the neck the extended pixels stay: parallax shows them at every head pose."""
        neck = self.visible["face"] | self.visible.get("neck", False)
        unit = max(int((self.face_box[2] - self.face_box[0]) * 0.25), 3)
        return (torso[..., 3] > 0) & ~self.visible["torso"] & ~_dilate(neck, unit)

    def build(self):
        v = self.visible
        x0, y0, x1, y1 = self.face_box
        face_w, face_h = x1 - x0, y1 - y0
        unit = max(int(face_w * 0.5), 6)

        head_pool = v["face"].copy()
        for name in ("eye_l", "eye_r", "iris_l", "iris_r", "lash_l", "lash_r", "brow_l", "brow_r", "mouth_up", "mouth_low", "bangs"):
            if name in v:
                head_pool |= v[name]
        # Only clean skin feeds the fill. Under long bangs the face matte also holds hair shadow and blush.
        clean = _erode(v["face"], 3) & (np.abs(self.rgb.astype(np.float32) - self.skin).sum(2) < 70)
        self.add("face", _hull(head_pool), blur=max(face_w * 0.05, 1.5), source=clean)

        if "neck" in v and v["neck"].any():
            nx0, ny0, nx1, _ = _bbox(v["neck"])
            neck_up = np.zeros(self.label.shape, bool)
            jaw = int(face_w * 0.42)
            neck_up[max(ny0 - int(face_h * 0.6), 0):ny0 + unit // 2, int(self.center_x) - jaw:int(self.center_x) + jaw] = True
            self.add("neck", neck_up, blur=2.0)

        if "hair_top" in v:
            self.add("hair_top", _dilate(v["hair_top"], 3))
            # Hair plate behind the head, seen when the head turns or leans.
            skull = np.zeros(self.label.shape, np.uint8)
            cv2.ellipse(skull, (int(self.center_x), int(y0 + face_h * 0.35)), (int(face_w * 0.56), int(face_h * 0.72)), 0, 0, 360, 1, -1)
            skull = (skull > 0) & _hull(v["hair_top"] | v["face"])
            self.visible["hair_fill"] = np.zeros(self.label.shape, bool)
            hair_rgb = np.median(self.rgb[v["hair_top"]], axis=0) * 0.72
            fill = np.zeros((self.height, self.width, 4), np.uint8)
            fill[..., :3] = hair_rgb.astype(np.uint8)
            fill[..., 3][_erode(skull, 1)] = 255
            painted = self.hidden.get("hair_fill")
            if painted is not None:
                use = (painted[..., 3] > 128) & skull
                fill[use] = painted[use]
            self.layers["hair_fill"] = fill

        for side in SIDES:
            self._eye(side)
            if "brow_" + side in v:
                self.add("brow_" + side)
            if "lock_" + side in v:
                self.add("lock_" + side, blur=2.0)
            if "arm_" + side in v:
                arm = _close(v["arm_" + side], unit * 2)
                arm[:_bbox(v["arm_" + side])[1]] = False
                self.add("arm_" + side, arm, blur=2.0)
            if "leg_" + side in v:
                self.add("leg_" + side, _dilate(v["leg_" + side], unit // 2), blur=2.0)
            if "animal_ear_" + side in v:
                self.add("animal_ear_" + side, _dilate(v["animal_ear_" + side], unit), blur=2.0)
        for name in ("bangs", "mouth_up", "mouth_low", "tail", "skirt"):
            if name in v:
                self.add(name, blur=2.0)
        body = v["torso"].copy()
        for name in ("neck", "arm_l", "arm_r", "skirt"):
            if name in v:
                body |= v[name]
        plate = _close(body, unit * 3) & _dilate(v["torso"], unit * 3)
        # Below the waist the arms hang over empty space, not over the body.
        waist = _bbox(v["skirt"])[1] + unit if "skirt" in v and v["skirt"].any() else _bbox(v["torso"])[3]
        plate[waist:] = False
        plate[:_bbox(v["torso"])[1]] = False
        # A deep erode keeps thin trims out of the fill, so it comes from the solid panels.
        self.add("torso", plate, erode_source=max(unit // 8, 2), blur=3.0)
        return self.layers

    def _eye(self, side):
        eye, iris, lash = "eye_" + side, "iris_" + side, "lash_" + side
        v = self.visible
        if eye not in v:
            return
        pool = v[eye] | v.get(iris, False) | v.get(lash, False)
        # The eye only exists where the drawing shows it: painted under bangs it would appear when they fade.
        shape = _hull(pool) & ~(v.get("hair_top", False) | v.get("bangs", False))
        white = v[eye]
        # Small anime eyes may show no clean white at all, so the brightest texels only tint a near-white base.
        bright = self.rgb[white].astype(np.float32)
        sclera = np.array([238, 238, 244], np.float32)
        if len(bright):
            top = bright[bright.sum(1) >= np.percentile(bright.sum(1), 90)].mean(0)
            sclera = sclera * 0.75 + top * 0.25
        rgba = np.zeros((self.height, self.width, 4), np.uint8)
        rgba[..., :3][shape] = sclera.astype(np.uint8)
        rgba[..., :3][white] = self.rgb[white]
        rgba[..., 3][shape] = 255
        self.layers[eye] = rgba
        if iris in v and v[iris].any():
            ring = np.zeros(self.label.shape, np.uint8)
            points = cv2.findNonZero(v[iris].astype(np.uint8))
            (cx, cy), radius = cv2.minEnclosingCircle(points)
            cv2.circle(ring, (round(cx), round(cy)), round(radius), 1, -1)
            rgb = _nearest_fill(self.rgb, v[iris], ring > 0)
            # The ring only adds what is hidden: over the drawn eye it would repaint it.
            rgba = np.dstack([rgb, ((ring > 0) & ~v[eye]).astype(np.uint8) * 255])
            self.layers[iris] = rgba
        if lash in v:
            self.add(lash, erode_source=0, blur=0)


def grid_mesh(alpha, step):
    """Quads over every cell that touches visible alpha. Returns vertices [N,2] in pixels and triangles [M,3]."""
    height, width = alpha.shape
    x0, y0, x1, y1 = _bbox(alpha > 0)
    columns = np.arange(x0, x1 + step, step)
    rows = np.arange(y0, y1 + step, step)
    occupied = {}
    triangles = []
    for r in range(len(rows) - 1):
        for c in range(len(columns) - 1):
            if not alpha[rows[r]:rows[r + 1], columns[c]:columns[c + 1]].any():
                continue
            corners = []
            for rr, cc in ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)):
                corners.append(occupied.setdefault((rr, cc), len(occupied)))
            triangles += [(corners[0], corners[3], corners[1]), (corners[1], corners[3], corners[2])]
    vertices = np.zeros((len(occupied), 2), np.float32)
    for (r, c), i in occupied.items():
        vertices[i] = columns[c], rows[r]
    return vertices, np.array(triangles, np.uint32)


def _segment_distance(points, a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-6), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def skin(vertices, segments, tau):
    """Smooth weights from distance to bone segments. segments: [(bone_index, a, b)]. Returns indices and weights [N,4]."""
    distance = np.stack([_segment_distance(vertices, a, b) for _, a, b in segments], 1)
    weight = np.exp(-((distance - distance.min(1, keepdims=True)) / tau) ** 2)
    order = np.argsort(-weight, 1)[:, :4]
    top = np.take_along_axis(weight, order, 1)
    bones = np.array([s[0] for s in segments], np.uint16)[order]
    if top.shape[1] < 4:
        pad = 4 - top.shape[1]
        top = np.pad(top, ((0, 0), (0, pad)))
        bones = np.pad(bones, ((0, 0), (0, pad)))
    return bones, (top / top.sum(1, keepdims=True)).astype(np.float32)


class Skeleton:
    def __init__(self):
        self.bones = []
        self.index = {}

    def add(self, name, parent, point, z=0.0):
        self.index[name] = len(self.bones)
        self.bones.append({"name": name, "parent": self.index[parent] if parent else -1, "pos": [float(point[0]), float(point[1]), float(z)]})
        return self.index[name]

    def xy(self, name):
        return self.bones[self.index[name]]["pos"][:2]


def _limb_joints(mask, joints, names, side):
    """Shoulder/elbow/wrist style chain from WeVisOne points. A point outside the limb falls back to the limb axis."""
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], 1).astype(np.float32)
    center = points.mean(0)
    axis = np.linalg.svd(points - center, full_matrices=False)[2][0]
    if axis[1] < 0:
        axis = -axis
    t = (points - center) @ axis
    top, bottom = center + axis * t.min(), center + axis * t.max()
    near = _dilate(mask, max(int((mask.sum() ** 0.5) * 0.08), 4))
    out = []
    for i, name in enumerate(names):
        point = joints.get(name + "_" + side)
        fallback = top + (bottom - top) * (i * 0.46)
        if point is None or not near[min(int(point[1]), mask.shape[0] - 1), min(int(point[0]), mask.shape[1] - 1)]:
            point = fallback
        out.append(np.asarray(point, np.float32))
    return out, bottom


def build_skeleton(layers, joints):
    v = layers.visible
    x0, y0, x1, y1 = layers.face_box
    chin = np.array([layers.center_x, layers.chin_y], np.float32)
    body = layers.alpha > 0.5
    bottom = _bbox(body)[3]
    skeleton = Skeleton()
    limbs = {}
    for side in SIDES:
        if "arm_" + side in v and v["arm_" + side].any():
            limbs["arm_" + side] = _limb_joints(v["arm_" + side], joints, ("shoulder", "elbow", "wrist"), side)
        if "leg_" + side in v and v["leg_" + side].any():
            limbs["leg_" + side] = _limb_joints(v["leg_" + side], joints, ("hip", "knee", "ankle"), side)

    shoulders = [limbs[n][0][0] for n in ("arm_l", "arm_r") if n in limbs]
    hips = [limbs[n][0][0] for n in ("leg_l", "leg_r") if n in limbs]
    face_h = y1 - y0
    shoulder_mid = np.mean(shoulders, 0) if shoulders else chin + (0, face_h * 0.6)
    hip_mid = np.mean(hips, 0) if hips else np.array([chin[0], min(chin[1] + face_h * 4, bottom)], np.float32)

    skeleton.add("root", None, (chin[0], bottom))
    skeleton.add("hips", "root", hip_mid)
    skeleton.add("spine", "hips", hip_mid + (shoulder_mid - hip_mid) * 0.45)
    skeleton.add("chest", "spine", hip_mid + (shoulder_mid - hip_mid) * 0.8)
    neck = chin + (shoulder_mid - chin) * 0.75
    skeleton.add("neck", "chest", neck)
    skeleton.add("head", "neck", chin + (0, -face_h * 0.1))
    for side in SIDES:
        if "arm_" + side in limbs:
            (shoulder, elbow, wrist), end = limbs["arm_" + side]
            skeleton.add("arm_" + side, "chest", shoulder)
            skeleton.add("forearm_" + side, "arm_" + side, elbow)
            skeleton.add("hand_" + side, "forearm_" + side, wrist)
            skeleton.add("hand_end_" + side, "hand_" + side, end)
        if "leg_" + side in limbs:
            (hip, knee, ankle), end = limbs["leg_" + side]
            skeleton.add("thigh_" + side, "hips", hip)
            skeleton.add("shin_" + side, "thigh_" + side, knee)
            skeleton.add("foot_" + side, "shin_" + side, ankle)
            skeleton.add("foot_end_" + side, "foot_" + side, end)
        for name in ("eye_" + side, "iris_" + side):
            if name in layers.layers:
                box = _bbox(layers.layers[name][..., 3] > 0)
                skeleton.add(name, "head", ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        ear = "animal_ear_" + side
        if ear in layers.layers:
            box = _bbox(layers.layers[ear][..., 3] > 0)
            skeleton.add(ear, "head", ((box[0] + box[2]) / 2, box[3]))
        lock = "lock_" + side
        if lock in layers.layers:
            _lock_bones(skeleton, lock, layers.visible[lock], layers.hair_cut)
    if "bangs" in layers.layers:
        box = _bbox(layers.layers["bangs"][..., 3] > 0)
        skeleton.add("bangs", "head", ((box[0] + box[2]) / 2, box[1]))
    if "tail" in layers.layers:
        _lock_bones(skeleton, "tail", layers.visible["tail"], _bbox(layers.visible["tail"])[1], parent="hips")
    if "skirt" in layers.layers:
        box = _bbox(layers.visible["skirt"])
        skeleton.add("skirt", "hips", ((box[0] + box[2]) / 2, box[1]))
    for side, mask in layers.breasts.items():
        ys, xs = np.nonzero(mask)
        skeleton.add("breast_" + side, "chest", (xs.mean(), ys.mean()))
    return skeleton


def _lock_bones(skeleton, name, mask, top, parent="head"):
    """A chain following the middle of a hanging strand, one bone per equal height slice."""
    ys, xs = np.nonzero(mask)
    bottom = ys.max() + 1
    cuts = np.linspace(top, bottom, LOCK_BONES + 1)
    previous = parent
    for i in range(LOCK_BONES):
        rows = (ys >= cuts[i]) & (ys < cuts[i + 1])
        x = xs[rows].mean() if rows.any() else xs.mean()
        bone = name + "_" + str(i)
        skeleton.add(bone, previous, (x, cuts[i]))
        previous = bone
    skeleton.add(name + "_end", previous, (xs[ys >= cuts[-2]].mean(), bottom))


def _chain_segments(skeleton, names):
    return [(skeleton.index[a], skeleton.xy(a), skeleton.xy(b)) for a, b in zip(names, names[1:])]


def _layer_segments(name, skeleton):
    """Bones that may move a layer, with the blend distance as a fraction of the head size."""
    index = skeleton.index
    if name == "torso":
        return _chain_segments(skeleton, ["hips", "spine", "chest", "neck"]), 1.0
    if name == "neck":
        return _chain_segments(skeleton, ["chest", "neck", "head"]), 0.35
    if name.startswith("arm_"):
        side = name[-1]
        chain = ["arm_" + side, "forearm_" + side, "hand_" + side, "hand_end_" + side]
        return _chain_segments(skeleton, chain), 0.25
    if name.startswith("leg_"):
        side = name[-1]
        chain = ["thigh_" + side, "shin_" + side, "foot_" + side, "foot_end_" + side]
        return _chain_segments(skeleton, chain), 0.3
    if name.startswith("lock_") or name == "tail":
        chain = [name + "_" + str(i) for i in range(LOCK_BONES)] + [name + "_end"]
        segments = _chain_segments(skeleton, chain)
        if name.startswith("lock_"):
            segments.insert(0, (index["head"], [skeleton.xy(chain[0])[0], skeleton.xy(chain[0])[1] - 40], skeleton.xy(chain[0])))
        return segments, 0.2
    if name.startswith("iris_") or name.startswith("animal_ear_") or name in ("bangs", "skirt"):
        return [(index[name], skeleton.xy(name), skeleton.xy(name))], 1.0
    return [(index["head"], skeleton.xy("head"), skeleton.xy("head"))], 1.0


def head_depth(xy, ellipse, height):
    """Front of the head in pixels: a wide bell with gentle slopes everywhere. A round dome is steep at the eyes and squeezes the far one when the head turns."""
    cx, cy, rx, ry = ellipse
    u, v = (xy[:, 0] - cx) / rx, (xy[:, 1] - cy) / ry
    return (np.clip(1 - u * u, 0, 1) * np.clip(1 - v * v, 0, 1)) ** 2 * height


# Head depth at the nose in face widths, and the skull's half width in widths of the face layer.
HEAD_DEPTH = 0.4
HEAD_REACH = 0.9
LAYER_DEPTH = {"face": 0.0, "eye_l": 0.0, "eye_r": 0.0, "iris_l": 0.01, "iris_r": 0.01, "lash_l": 0.04, "lash_r": 0.04, "brow_l": 0.05, "brow_r": 0.05,
               "mouth_inner": -0.01, "mouth_up": 0.02, "mouth_low": 0.02, "mouth_a": 0.025, "mouth_i": 0.025, "mouth_o": 0.025,
               "closed_l": 0.045, "closed_r": 0.045, "smile_l": 0.045, "smile_r": 0.045, "hair_top": 0.06, "bangs": 0.14, "hair_fill": -0.3, "lock_l": 0.06, "lock_r": 0.06, "hair_back": 0.0}


class HeadSplat:
    """A splat of the head alone, aligned to the layers. The layers borrow its surface as depth, so both coincide when they swap."""

    def __init__(self, layers, positions, scales, rotations, opacities, sh):
        # Splat frame: the head faces +x, -y is up, +z is image right. Avatar frame: x right, y up, z to the viewer.
        solid = opacities.reshape(-1) > 0.02
        self.points = np.stack([positions[:, 2], -positions[:, 1], positions[:, 0]], 1)[solid]
        self.scales, self.opacities, self.sh = scales[solid], opacities.reshape(-1)[solid], sh[solid].copy()
        # (x,y,z) -> (z,-y,x) is a half turn about the (1,0,1) axis.
        half = np.array([0, 1, 0, 1], np.float32) / 2 ** 0.5
        self.rotations = _quat_mul(half, rotations[solid])

        x0, y0, x1, y1 = _bbox(_head_pixels(layers))
        dense = self.points[self.opacities > 0.3]
        low, high = np.percentile(dense[:, :2], [0.3, 99.7], axis=0)
        self.scale = ((x1 - x0) / (high[0] - low[0]) + (y1 - y0) / (high[1] - low[1])) / 2
        self.center = (low + high) / 2
        self.middle = float(np.median(dense[:, 2]))
        self.anchor = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.pixels = self.to_pixels(self.points)
        self._surface(layers)
        self._register(layers)
        self._fit_silhouette(layers)
        self._clear_neck(layers)
        self._surface(layers)
        self._match_colors(layers)

    def to_pixels(self, points):
        """Avatar-frame splat points to canvas x, y and depth, all in pixels."""
        return np.stack([self.anchor[0] + (points[:, 0] - self.center[0]) * self.scale,
                         self.anchor[1] - (points[:, 1] - self.center[1]) * self.scale,
                         (points[:, 2] - self.middle) * self.scale], 1)

    def _surface(self, layers):
        """Front surface of the splat as a depth map, extended past the head so strands continue it."""
        face_w = layers.face_box[2] - layers.face_box[0]
        cell = max(face_w // 30, 3)
        rows, columns = layers.height // cell + 1, layers.width // cell + 1
        z = np.full((rows, columns), -np.inf, np.float32)
        solid = self.opacities > 0.5
        x = np.clip(self.pixels[solid, 0] / cell, 0, columns - 1).astype(int)
        y = np.clip(self.pixels[solid, 1] / cell, 0, rows - 1).astype(int)
        np.maximum.at(z, (y, x), self.pixels[solid, 2])
        known = np.isfinite(z)
        index = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        z = ndimage.median_filter(z[index[0], index[1]], 5)
        self.depth = cv2.GaussianBlur(cv2.resize(z, (columns * cell, rows * cell), interpolation=cv2.INTER_LINEAR), (0, 0), face_w * 0.05)[:layers.height, :layers.width]
        self.soft = cv2.GaussianBlur(self.depth, (0, 0), face_w * 0.25)

    def front(self, layers):
        """Gaussians on the visible surface, with their canvas pixel."""
        x = np.clip(self.pixels[:, 0].astype(int), 0, layers.width - 1)
        y = np.clip(self.pixels[:, 1].astype(int), 0, layers.height - 1)
        near = self.pixels[:, 2] > self.depth[y, x] - (layers.face_box[2] - layers.face_box[0]) * 0.08
        return near, x, y

    def _register(self, layers):
        """Refines scale and offset on the face, where eyes and mouth give a sharp signal. Bounding boxes alone miss by a lot."""
        near, x, y = self.front(layers)
        x0, y0, x1, y1 = layers.face_box
        points = self.points[near & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)]
        points = points[::max(len(points) // 40000, 1)]
        colors = np.clip(self.sh[near & (x >= x0) & (x < x1) & (y >= y0) & (y < y1), 0, :][::max(len(points) // 40000, 1)] * SH_C0 + 0.5, 0, 1)[:len(points)]
        target = cv2.GaussianBlur(layers.rgb, (0, 0), 1.0).astype(np.float32) / 255
        face_w = x1 - x0

        def cost(scale, anchor):
            px = np.clip((anchor[0] + (points[:, 0] - self.center[0]) * scale).astype(int), 0, layers.width - 1)
            py = np.clip((anchor[1] - (points[:, 1] - self.center[1]) * scale).astype(int), 0, layers.height - 1)
            inside = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
            return np.abs(target[py[inside], px[inside]] - colors[inside]).mean() if inside.mean() > 0.5 else 9.0

        scale, anchor = self.scale, self.anchor
        spans = [(np.linspace(0.8, 1.2, 9), np.linspace(-0.3, 0.3, 13)), (np.linspace(0.97, 1.03, 5), np.linspace(-0.04, 0.04, 9))]
        for scales, offsets in spans:
            trials = [(cost(scale * s, (anchor[0] + dx * face_w, anchor[1] + dy * face_w)), s, dx, dy) for s in scales for dx in offsets for dy in offsets]
            _, s, dx, dy = min(trials)
            scale, anchor = scale * s, (anchor[0] + dx * face_w, anchor[1] + dy * face_w)
        self.scale, self.anchor = scale, anchor
        self.pixels = self.to_pixels(self.points)

    def _fit_silhouette(self, layers):
        """Pulls the splat inside the layers' outline and flattens its back. Idol's extra volume would pop out when it takes over."""
        outline = _head_pixels(layers)
        for name in ("lock_l", "lock_r"):
            if name in layers.visible:
                outline |= layers.visible[name]
        ys, xs = np.nonzero(_head_pixels(layers))
        center = xs.mean(), ys.mean()
        bins = 96

        def profile(x, y, percentile):
            angle = np.arctan2(y - center[1], x - center[0])
            index = ((angle + np.pi) / (2 * np.pi) * bins).astype(int) % bins
            radius = np.hypot(x - center[0], y - center[1])
            out = np.array([np.percentile(radius[index == i], percentile) if (index == i).any() else 0 for i in range(bins)], np.float32)
            return out, index, radius

        ys, xs = np.nonzero(outline)
        target = profile(xs, ys, 100)[0]
        dense = self.opacities > 0.3
        current = profile(self.pixels[dense, 0], self.pixels[dense, 1], 99.5)[0]
        ratio = np.where(current > 0, np.minimum(target / np.maximum(current, 1), 1), 1)
        ratio = ndimage.gaussian_filter1d(ratio, 2, mode="wrap")
        reach = ndimage.gaussian_filter1d(current, 2, mode="wrap")

        _, index, radius = profile(self.pixels[:, 0], self.pixels[:, 1], 100)
        # Full pull at the edge, none at the center, so the face stays where it was registered.
        edge = np.clip(radius / np.maximum(reach[index], 1), 0, 1.5) ** 2
        shrink = 1 - (1 - ratio[index]) * edge
        shrink = np.maximum(shrink, ratio[index] * reach[index] / np.maximum(radius, 1))
        shrink = np.minimum(shrink, 1)
        moved_x = center[0] + (self.pixels[:, 0] - center[0]) * shrink
        moved_y = center[1] + (self.pixels[:, 1] - center[1]) * shrink
        self.points[:, 0] += (moved_x - self.pixels[:, 0]) / self.scale
        self.points[:, 1] -= (moved_y - self.pixels[:, 1]) / self.scale
        behind = self.points[:, 2] < self.middle
        self.points[behind, 2] = self.middle + (self.points[behind, 2] - self.middle) * BACK_DEPTH
        self.pixels = self.to_pixels(self.points)

    def _clear_neck(self, layers):
        """Idol closes the head with a stub of neck, which would sit over the neck layer."""
        neck = layers.visible.get("neck")
        if neck is None:
            return
        x = np.clip(self.pixels[:, 0].astype(int), 0, layers.width - 1)
        y = np.clip(self.pixels[:, 1].astype(int), 0, layers.height - 1)
        keep = ~neck[y, x]
        self.points, self.pixels, self.scales, self.rotations = self.points[keep], self.pixels[keep], self.scales[keep], self.rotations[keep]
        self.opacities, self.sh = self.opacities[keep], self.sh[keep]

    def _match_colors(self, layers):
        """Matches brightness to the drawing, then paints out the eyes and mouth the layers draw. Per-channel matching would shift hues."""
        near, x, y = self.front(layers)
        head = _head_pixels(layers)
        fit = near & head[y, x]
        rgb = self.sh[:, 0, :] * SH_C0 + 0.5
        drawing = layers.rgb[head].astype(np.float32) / 255
        steps = np.linspace(0, 100, 65)
        light = rgb.mean(1)
        rgb += (np.interp(light, np.percentile(light[fit], steps), np.percentile(drawing.mean(1), steps)) - light)[:, None]
        features = np.zeros(head.shape, bool)
        for name in ("eye_l", "eye_r", "iris_l", "iris_r", "lash_l", "lash_r", "mouth_up", "mouth_low"):
            if name in layers.visible:
                features |= layers.visible[name]
        unit = max((layers.face_box[2] - layers.face_box[0]) // 16, 1)
        hair = layers.visible.get("hair_top", False) | layers.visible.get("bangs", False)
        blank = near & (_dilate(features, unit) & ~hair)[y, x]
        skin = layers.layers["face"][..., :3].astype(np.float32) / 255
        rgb[blank] = skin[y[blank], x[blank]]
        self.sh[:, 0, :] = (rgb - 0.5) / SH_C0


def _quat_mul(a, b):
    """Hamilton product of one wxyz quaternion with an array of them."""
    w, x, y, z = a
    bw, bx, by, bz = b.T
    return np.stack([w * bw - x * bx - y * by - z * bz, w * bx + x * bw + y * bz - z * by,
                     w * by - x * bz + y * bw + z * bx, w * bz + x * by - y * bx + z * bw], 1).astype(np.float32)


def _head_pixels(layers):
    head = np.zeros(layers.label.shape, bool)
    for name, visible in layers.visible.items():
        if name in HEAD_LAYERS or name.startswith("animal_ear_") or name.startswith("mouth_"):
            head |= visible
    return _largest(head)


def _relief(depth, layers):
    """Depth detail in pixels for each body layer, taken from that form alone: monocular depth jumps between parts, and those jumps would tear the meshes.
    The head and its hair get none. They share one surface, and detail of their own pulls skin and hair apart when the head turns."""
    face_w = layers.face_box[2] - layers.face_box[0]
    low, high = np.percentile(depth[layers.alpha > 0.5], [2, 98])
    near = np.clip((depth - low) / max(high - low, 1e-5), 0, 1)
    relief = {}
    for name, visible in layers.visible.items():
        if name in HEAD_LAYERS or name.startswith("lock_") or not visible.any():
            continue
        solid = visible.astype(np.float32)

        def blur(sigma):
            return cv2.GaussianBlur(near * solid, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(solid, (0, 0), sigma), 1e-3)

        # Detail finer than the coarse grid would alias into saw teeth along the layer's edge.
        relief[name] = np.clip(blur(face_w * 0.1) - blur(face_w * 0.4), -0.06, 0.06) * face_w
    return relief


def build_parts(layers, skeleton, relief=None, splat=None):
    x0, y0, x1, y1 = layers.face_box
    face_w, face_h = x1 - x0, y1 - y0
    # The face matte stops at the hair, so the skull is sized from the face layer, which reaches under it.
    hx0, _, hx1, _ = _bbox(layers.layers["face"][..., 3] > 0)
    ellipse = ((hx0 + hx1) / 2, y0 + face_h * 0.5, (hx1 - hx0) * HEAD_REACH, face_h * 0.8)
    unit = face_w * 0.56
    coarse = max(layers.height // 90, 6)
    fine = max(face_w // 22, 2)
    parts = {}
    for name in DRAW_ORDER:
        rgba = layers.layers.get(name)
        if rgba is None or not (rgba[..., 3] > 0).any():
            continue
        step = fine if name in HEAD_LAYERS and name not in ("hair_top", "hair_fill", "hair_back") else coarse
        vertices, triangles = grid_mesh(rgba[..., 3] > 0, int(step))
        segments, blend = _layer_segments(name, skeleton)
        bones, weights = skin(vertices, segments, max(blend * face_h, 1.0))
        if name == "torso":
            _soft_chest(layers, skeleton, vertices, bones, weights)
        # Depth over the part's whole tile: vertices sample it, and the encoder turns it into the normal map.
        bx0, by0 = vertices.min(0).astype(int)
        bx1, by1 = vertices.max(0).astype(int) + 1
        px = np.clip(np.arange(bx0, bx1), 0, layers.width - 1)
        py = np.clip(np.arange(by0, by1), 0, layers.height - 1)
        if name in LAYER_DEPTH and splat is not None:
            # Hair follows a soft version: the surface steps at the hairline and would fold its mesh.
            surface = splat.soft if name in ("hair_top", "bangs", "hair_back") else splat.depth
            field = surface[np.ix_(py, px)] + LAYER_DEPTH[name] * unit * (name != "hair_fill")
            if name.startswith("lock_"):
                # Strands hang at the sides of the head; the surface under the chin would throw them across the face.
                field = np.minimum(field, unit * 0.3)
        elif name in LAYER_DEPTH:
            grid = np.stack(np.meshgrid(np.arange(bx0, bx1), np.arange(by0, by1)), -1).reshape(-1, 2).astype(np.float32)
            field = head_depth(grid, ellipse, face_w * HEAD_DEPTH).reshape(len(py), len(px)) + LAYER_DEPTH[name] * unit
        else:
            mask = cv2.GaussianBlur((rgba[..., 3] > 0).astype(np.float32), (0, 0), 2)
            dt = cv2.GaussianBlur(cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5), (0, 0), 3)
            field = np.minimum(dt[np.ix_(py, px)], face_w * 0.6) * 0.6
            if name.startswith("arm_"):
                field += face_w * 0.1
        if relief is not None and name in relief:
            field = field + relief[name][np.ix_(py, px)]
        field = field.astype(np.float32)
        z = field[vertices[:, 1].astype(int) - by0, vertices[:, 0].astype(int) - bx0]
        parts[name] = {"rgba": rgba, "vertices": vertices, "triangles": triangles, "z": z, "depth": field,
                       "bones": bones, "weights": weights, "morphs": {}}
    _add_mouth(layers, skeleton, parts, ellipse, unit, splat)
    _add_face_morphs(layers, parts)
    return {name: parts[name] for name in DRAW_ORDER if name in parts}


def _soft_chest(layers, skeleton, vertices, bones, weights):
    """Torso vertices over the breasts partly follow their soft-body bone, fading out toward the matte's edge."""
    face_w = layers.face_box[2] - layers.face_box[0]
    px = np.clip(vertices[:, 0].astype(int), 0, layers.width - 1)
    py = np.clip(vertices[:, 1].astype(int), 0, layers.height - 1)
    for side, mask in layers.breasts.items():
        soft = cv2.GaussianBlur(_erode(mask, max(int(face_w * 0.04), 1)).astype(np.float32), (0, 0), face_w * 0.08)[py, px]
        moving = soft > 0.01
        weights[moving] *= (1 - soft[moving])[:, None]
        # The weakest influence makes room for the breast bone.
        bones[moving, 3] = skeleton.index["breast_" + side]
        weights[moving, 3] = soft[moving]
    weights /= weights.sum(1, keepdims=True)


def _add_mouth(layers, skeleton, parts, ellipse, unit, splat):
    if layers.mouth_curve is None:
        return
    x0, y0, x1, y1 = layers.mouth_box
    width = x1 - x0
    center = (x0 + x1) / 2
    columns, rows = 13, 7
    u = np.linspace(-1, 1, columns)
    grid_x = center + u * width * 0.46
    line = np.polyval(layers.mouth_curve, grid_x).astype(np.float32)
    vertices = np.zeros((rows * columns, 2), np.float32)
    uv = np.zeros((rows * columns, 2), np.float32)
    open_delta = np.zeros((rows * columns, 2), np.float32)
    span = width * 0.5
    envelope = 1 - u ** 2
    triangles = []
    for r in range(rows):
        s = r / (rows - 1)
        for c in range(columns):
            i = r * columns + c
            vertices[i] = grid_x[c], line[c]
            uv[i] = c / (columns - 1), s
            open_delta[i, 1] = (s * 1.25 - 0.25) * span * envelope[c]
            if r < rows - 1 and c < columns - 1:
                triangles += [(i, i + columns, i + 1), (i + 1, i + columns, i + columns + 1)]
    texture = np.zeros((rows * 8, columns * 8, 4), np.uint8)
    gradient = np.linspace(0, 1, texture.shape[0])[:, None]
    texture[..., 0] = (58 + 150 * np.clip((gradient - 0.55) * 2.2, 0, 1)).astype(np.uint8)
    texture[..., 1] = (18 + 62 * np.clip((gradient - 0.55) * 2.2, 0, 1)).astype(np.uint8)
    texture[..., 2] = (28 + 66 * np.clip((gradient - 0.55) * 2.2, 0, 1)).astype(np.uint8)
    texture[..., 3] = 255
    bones = np.zeros((len(vertices), 4), np.uint16)
    bones[:, 0] = skeleton.index["head"]
    weights = np.zeros((len(vertices), 4), np.float32)
    weights[:, 0] = 1
    parts["mouth_inner"] = {"texture": texture, "uv": uv, "vertices": vertices, "triangles": np.array(triangles, np.uint32),
                            "z": _mouth_depth(vertices, layers, ellipse, unit, splat),
                            "bones": bones, "weights": weights, "morphs": {"mouthOpen": open_delta}}
    for name, share in (("mouth_up", -0.25), ("mouth_low", 1.0)):
        if name in parts:
            xy = parts[name]["vertices"]
            t = np.clip((xy[:, 0] - center) / (width * 0.5), -1, 1)
            delta = np.zeros_like(xy)
            delta[:, 1] = share * span * (1 - t ** 2)
            parts[name]["morphs"]["mouthOpen"] = delta
    smile_lift = width * 0.16
    for name in ("mouth_inner", "mouth_up", "mouth_low"):
        if name in parts:
            xy = parts[name]["vertices"]
            t = np.clip((xy[:, 0] - center) / (width * 0.5), -1, 1)
            delta = np.zeros_like(xy)
            delta[:, 1] = -smile_lift * t ** 2
            parts[name]["morphs"]["mouthSmile"] = delta


def _mouth_depth(vertices, layers, ellipse, unit, splat):
    offset = LAYER_DEPTH["mouth_inner"] * unit
    if splat is None:
        face_w = layers.face_box[2] - layers.face_box[0]
        return (head_depth(vertices, ellipse, face_w * HEAD_DEPTH) + offset).astype(np.float32)
    return splat.depth[vertices[:, 1].astype(int), vertices[:, 0].astype(int)] + offset


def _add_face_morphs(layers, parts):
    for side in SIDES:
        eye, lash, brow = "eye_" + side, "lash_" + side, "brow_" + side
        param = "blink" + side.upper()
        if eye in parts:
            alpha = parts[eye]["rgba"][..., 3] > 0
            x0, y0, x1, y1 = _bbox(alpha)
            ys, xs = np.nonzero(alpha)
            lower_x, lower_y = [], []
            for x in range(x0, x1):
                column = ys[xs == x]
                if len(column):
                    lower_x.append(x)
                    lower_y.append(column.max())
            curve = np.polyfit(lower_x, lower_y, 2)
            xy = parts[eye]["vertices"]
            delta = np.zeros_like(xy)
            delta[:, 1] = (np.polyval(curve, xy[:, 0]) - xy[:, 1]) * 0.97
            parts[eye]["morphs"][param] = delta
            if lash in parts:
                lash_xy = parts[lash]["vertices"]
                alpha = parts[lash]["rgba"][..., 3] > 0
                top_line = np.polyval(np.polyfit(*np.nonzero(alpha)[::-1], 2), lash_xy[:, 0])
                delta = np.zeros_like(lash_xy)
                delta[:, 1] = np.polyval(curve, lash_xy[:, 0]) - top_line
                parts[lash]["morphs"][param] = delta
        if brow in parts:
            xy = parts[brow]["vertices"]
            delta = np.zeros_like(xy)
            delta[:, 1] = -(layers.face_box[3] - layers.face_box[1]) * 0.07
            parts[brow]["morphs"]["brow" + side.upper()] = delta


def plate_request(seg):
    """The character without arms and strands, plus the mask of the body they hid, for an inpaint pass."""
    layers = Layers(seg)
    layers.build()
    torso = layers.layers["torso"]
    hidden = layers.plate_area(torso)
    image = np.array(seg["image"])
    for name in ("arm_l", "arm_r", "lock_l", "lock_r"):
        if name in layers.visible:
            image[layers.visible[name]] = 0.5
    image[hidden] = torso[..., :3][hidden] / 255
    return image, _dilate(hidden, 2)


def head_mask(seg):
    """Matte of the head alone: skin, face parts, ears and hair down to the chin."""
    layers = Layers(seg)
    return np.where(_dilate(_head_pixels(layers), 1), layers.alpha, 0).astype(np.float32)


def _hand_hair_to_splat(layers):
    """Drops the hair layers and trims the face to its visible skin, since the splat now draws the hair around it."""
    hair = layers.visible.get("hair_top", False) | layers.visible.get("bangs", False)
    layers.layers["face"][..., 3][hair] = 0
    # Below the chin the splat has gaps; the drawn hair stays behind it so the fill under it never shows.
    if "hair_top" in layers.layers:
        back = layers.layers["hair_top"].copy()
        back[:layers.chin_y, :, 3] = 0
        layers.layers["hair_back"] = back
    for name in ("hair_top", "bangs", "hair_fill"):
        layers.layers.pop(name, None)


def _add_variants(layers, variants):
    columns = np.arange(layers.width)[None, :]
    for variant, rgba in variants.items():
        name = VARIANT_LAYERS[variant]
        if not variant.startswith("eyes_"):
            layers.layers[name] = rgba
            continue
        for side, keep in (("l", columns >= layers.center_x), ("r", columns < layers.center_x)):
            half = rgba.copy()
            half[..., 3] = np.where(keep, half[..., 3], 0)
            if half[..., 3].any():
                layers.layers[name + "_" + side] = half


def _chest_reach(layers):
    """How far each breast may move, in world units: a share of its height."""
    return {"breast_" + side: float(_bbox(mask)[3] - _bbox(mask)[1]) * 0.1 / layers.height for side, mask in layers.breasts.items()}


def build_avatar(seg, hidden=None, depth=None, plate=None, head_splat=None, splat_hair=False, variants=None, chest_physics=1.0):
    """head_splat: positions, scales, rotations (wxyz), opacities and sh arrays of a splat of the head alone.
    variants: full-canvas RGBA patches keyed by VARIANT_LAYERS. chest_physics scales the breast bounce; 0 leaves the chest rigid."""
    layers = Layers(seg, hidden, plate)
    if chest_physics <= 0:
        layers.breasts = {}
    layers.build()
    _add_variants(layers, variants or {})
    splat = None if head_splat is None else HeadSplat(layers, *head_splat)
    if splat is not None and splat_hair:
        _hand_hair_to_splat(layers)
    skeleton = build_skeleton(layers, seg["joints"])
    parts = build_parts(layers, skeleton, None if depth is None else _relief(depth, layers), splat)
    face_unit = (layers.face_box[2] - layers.face_box[0]) / layers.height
    params = build_params(skeleton.index, parts, face_unit, splat is not None)
    return {"width": layers.width, "height": layers.height, "bones": skeleton.bones, "parts": parts, "params": params,
            "clips": build_clips(params), "expressions": build_expressions(params), "springs": build_springs(skeleton.index, _chest_reach(layers), chest_physics), "face_box": [int(v) for v in layers.face_box], "head_splat": splat, "splat_hair": splat_hair}

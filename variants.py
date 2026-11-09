"""Drawn eye and mouth variants: an inpaint request on the head crop, and its import back onto the canvas.
Only the masked area is imported, so the rest of the drawing stays pixel exact."""
import cv2
import numpy as np

from .rig import SIDES, Layers, _bbox, _dilate, _erode, _hull

VARIANTS = ["eyes_closed", "eyes_smile", "mouth_a", "mouth_i", "mouth_o"]
CROP = 1024


def head_crop(seg):
    """The square around the face the variants are painted on: (x0, y0, x1, y1) on the canvas and the scale to CROP."""
    height, width = seg["image"].shape[:2]
    x0, y0, x1, y1 = _bbox(seg["masks"]["face"] > 0.5)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    radius = int(max(x1 - x0, y1 - y0) * 0.85)
    box = max(cx - radius, 0), max(cy - radius, 0), min(cx + radius, width), min(cy + radius, height)
    return box, CROP / max(box[2] - box[0], box[3] - box[1])


def variant_mask(layers, variant):
    """Canvas area a variant may repaint."""
    v = layers.visible
    x0, _, x1, _ = layers.face_box
    face_w = x1 - x0
    hair = v.get("hair_top", False) | v.get("bangs", False)
    if variant.startswith("eyes_"):
        area = np.zeros(layers.label.shape, bool)
        for side in SIDES:
            for name in ("eye_", "iris_", "lash_"):
                area |= v.get(name + side, False)
        brows = v.get("brow_l", False) | v.get("brow_r", False)
        return _dilate(area, max(int(face_w * 0.05), 2)) & ~_dilate(brows, 1) & ~hair
    if layers.mouth_curve is None:
        raise ValueError("Live Avatar: no mouth found for " + variant + ".")
    mx0, my0, mx1, my1 = layers.mouth_box
    w = max(mx1 - mx0, face_w * 0.12)
    area = np.zeros(layers.label.shape, bool)
    # An open mouth reaches well below the closed one.
    area[max(int(my0 - w * 0.3), 0):int(my1 + w * 0.8), max(int(mx0 - w * 0.35), 0):int(mx1 + w * 0.35)] = True
    # The jaw outline stays the drawing's own.
    inside = _erode(_hull(v["face"] | v.get("mouth_up", False) | v.get("mouth_low", False)), max(int(face_w * 0.04), 2))
    return area & inside & ~hair


def variant_request(seg, variant):
    """Head crop at CROP size and the mask to inpaint, a little wider than what is imported so the edge blends."""
    layers = Layers(seg)
    (x0, y0, x1, y1), scale = head_crop(seg)
    face_w = layers.face_box[2] - layers.face_box[0]
    mask = _dilate(variant_mask(layers, variant), max(int(face_w * 0.02), 2)).astype(np.float32)
    size = round((x1 - x0) * scale), round((y1 - y0) * scale)
    image = cv2.resize(seg["image"][y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC).clip(0, 1)
    return image, cv2.resize(mask[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)


def import_variant(seg, variant, painted):
    """painted: the inpainted head crop, float RGB. Returns the full-canvas RGBA patch."""
    layers = Layers(seg)
    (x0, y0, x1, y1), _ = head_crop(seg)
    face_w = layers.face_box[2] - layers.face_box[0]
    area = variant_mask(layers, variant)
    source = layers.rgb.astype(np.float32)
    canvas = source.copy()
    canvas[y0:y1, x0:x1] = cv2.resize(painted, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA) * 255
    # VAE round trips shift the whole crop's color a little; the ring just outside the area should match the drawing.
    ring = _dilate(area, max(int(face_w * 0.03), 2)) & ~area & (layers.alpha > 0.9)
    if ring.any():
        canvas[area] -= (canvas[ring] - source[ring]).mean(0)
    alpha = cv2.GaussianBlur(_erode(area, 1).astype(np.float32), (0, 0), max(face_w * 0.006, 0.8)) * area
    rgba = np.dstack([canvas.clip(0, 255).round().astype(np.uint8), (alpha * 255).round().astype(np.uint8)])
    return rgba

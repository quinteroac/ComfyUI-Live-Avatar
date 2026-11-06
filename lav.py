"""The .lav container: "LAV1", u32 JSON length, JSON, then 4-byte aligned binary sections (mesh arrays and the atlas)."""
import io
import json
import struct

import numpy as np
import torch
from comfy_extras.nodes_gaussian_splat import _gaussian_spz_bytes
from PIL import Image
from scipy import ndimage

MAGIC = b"LAV1"
VERSION = 1
PAD = 2


def _tile(rgba, vertices):
    """The part's texels over the whole mesh. Grid cells overhang the image, and uvs must stay inside the tile."""
    x0, y0 = np.floor(vertices.min(0)).astype(int)
    x1, y1 = np.ceil(vertices.max(0)).astype(int) + 1
    tile = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, rgba.shape[1]), min(y1, rgba.shape[0])
    tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgba[sy0:sy1, sx0:sx1]
    return x0, y0, tile


def _pack(sizes):
    """Shelf packer. sizes: {name: (w, h)}. Returns positions and the atlas size."""
    total = sum((w + PAD) * (h + PAD) for w, h in sizes.values())
    side = 512
    while side * side < total * 1.25:
        side *= 2
    while True:
        x = y = shelf = 0
        placed = {}
        for name, (w, h) in sorted(sizes.items(), key=lambda item: -item[1][1]):
            if x + w + PAD > side:
                x, y, shelf = 0, y + shelf + PAD, 0
            placed[name] = (x, y)
            x += w + PAD
            shelf = max(shelf, h)
        if y + shelf <= side:
            return placed, side
        side *= 2


def _bleed(rgba):
    """Spread color into transparent texels so bilinear filtering never picks up black."""
    empty = rgba[..., 3] == 0
    if not empty.any() or empty.all():
        return rgba
    index = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    out = rgba.copy()
    out[..., :3] = rgba[index[0], index[1], :3]
    return out


def _head_splat(avatar, section, scale):
    """Splat bytes in their own unit frame, plus the transform that seats them on the head bone."""
    splat = avatar["head_splat"]
    data = _gaussian_spz_bytes(*(torch.from_numpy(a) for a in (splat.points, splat.scales, splat.rotations, splat.opacities[:, None], splat.sh)))
    origin = splat.to_pixels(np.zeros((1, 3), np.float32))[0]
    head = next(bone for bone in avatar["bones"] if bone["name"] == "head")["pos"]
    return {"section": section(data), "scale": float(splat.scale * scale),
            "position": [float((origin[0] - head[0]) * scale), float((head[1] - origin[1]) * scale), float((origin[2] - head[2]) * scale)],
            "fade": ["face", "hair_top", "bangs", "hair_fill", "brow_l", "brow_r"], "alwaysOn": avatar["splat_hair"]}


def encode(avatar):
    width, height = avatar["width"], avatar["height"]
    crops, sizes = {}, {}
    for name, part in avatar["parts"].items():
        if "texture" in part:
            crops[name] = (0, 0, part["texture"])
        else:
            x0, y0, tile = _tile(part["rgba"], part["vertices"])
            crops[name] = (x0, y0, _bleed(tile))
        sizes[name] = crops[name][2].shape[1], crops[name][2].shape[0]
    placed, side = _pack(sizes)
    atlas = np.zeros((side, side, 4), np.uint8)
    for name, (x, y) in placed.items():
        tile = crops[name][2]
        atlas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile

    blob = io.BytesIO()

    def section(data):
        offset = blob.tell()
        blob.write(data)
        blob.write(b"\0" * (-blob.tell() % 4))
        return [offset, len(data)]

    scale = 1 / height

    def world(xy, z):
        """Pixels to world units: y up, one unit is the canvas height, x centered."""
        return np.stack([(xy[:, 0] - width / 2) * scale, (height - xy[:, 1]) * scale, z * scale], 1).astype(np.float32)

    parts = []
    for order, (name, part) in enumerate(avatar["parts"].items()):
        x, y = placed[name]
        tile_w, tile_h = sizes[name]
        vertices = part["vertices"]
        if "uv" in part:
            u = (x + 0.5 + part["uv"][:, 0] * (tile_w - 1)) / side
            v = (y + 0.5 + part["uv"][:, 1] * (tile_h - 1)) / side
        else:
            u = (x + vertices[:, 0] - crops[name][0]) / side
            v = (y + vertices[:, 1] - crops[name][1]) / side
        morphs = {}
        for morph, delta in part["morphs"].items():
            data = np.zeros((len(vertices), 3), np.float32)
            data[:, 0] = delta[:, 0] * scale
            data[:, 1] = -delta[:, 1] * scale
            morphs[morph] = section(data.tobytes())
        parts.append({"name": name, "order": order, "vertexCount": len(vertices), "indexCount": int(part["triangles"].size),
                      "position": section(world(vertices, part["z"]).tobytes()),
                      "uv": section(np.stack([u, v], 1).astype(np.float32).tobytes()),
                      "skinIndex": section(part["bones"].astype(np.uint16).tobytes()),
                      "skinWeight": section(part["weights"].astype(np.float32).tobytes()),
                      "index": section(part["triangles"].astype(np.uint32).tobytes()),
                      "morphs": morphs})

    buffer = io.BytesIO()
    Image.fromarray(atlas, "RGBA").save(buffer, "WEBP", lossless=True, exact=True)
    atlas_section = section(buffer.getvalue())

    bones = []
    for bone in avatar["bones"]:
        x, y, z = bone["pos"]
        bones.append({"name": bone["name"], "parent": bone["parent"], "position": [(x - width / 2) * scale, (height - y) * scale, z * scale]})
    header = {"version": VERSION, "canvas": [width, height], "atlas": {"section": atlas_section, "size": side, "mime": "image/webp"},
              "bones": bones, "parts": parts, "params": avatar["params"], "clips": avatar["clips"], "springs": avatar["springs"]}
    if avatar["head_splat"] is not None:
        header["headSplat"] = _head_splat(avatar, section, scale)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-(len(text) + 8) % 4)
    return MAGIC + struct.pack("<I", len(text)) + text + blob.getvalue()

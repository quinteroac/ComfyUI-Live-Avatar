"""Body-part segmentation with WeVisOne-VL: full-frame parts, then face parts on an enlarged head crop."""
import json
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BODY_QUERIES = {"person": "person", "hair": "hair", "face": "face", "neck": "neck", "arms": "arms", "hands": "hands",
                "legs": "legs", "skirt": "skirt", "tail": "tail", "animal_ears": "animal ears", "breasts": "breasts"}
HEAD_QUERIES = {"face": "face skin", "bangs": "bangs", "eye": "eye", "iris": "iris", "lash": "eyelashes", "brow": "eyebrow",
                "mouth": "mouth"}
JOINTS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
HEAD_CROP = 1024
# A head leaning less than this is built as drawn: straightening it costs a resampling of the drawing.
MIN_ROLL = math.radians(5)


def _run(perceive, clip, image, task, query):
    out = perceive.execute(clip, image, task, query, 512, True).args
    return out[3].float().cpu(), json.loads(out[2])


def _split_by_side(masks, center_x):
    """Instances to (image-right, image-left). Image-right is the character's left side."""
    masks = [m for m in masks if (m > 0.5).any()]
    centers = [torch.nonzero(m > 0.5)[:, 1].float().mean() for m in masks]
    # A leaning body can put both limbs on one side of the face, so a pair is told apart by its own middle.
    if len(centers) > 1:
        center_x = (min(centers) + max(centers)) / 2
    left, right = None, None
    for m, x in zip(masks, centers):
        if x >= center_x:
            left = m if left is None else torch.maximum(left, m)
        else:
            right = m if right is None else torch.maximum(right, m)
    return left, right


def _split_at_center(masks, center_x):
    """Face parts never cross the middle of the face, but WeVisOne may return both eyes as one instance."""
    union = masks.amax(0)
    side_l, side_r = union.clone(), union.clone()
    side_l[:, :center_x] = 0
    side_r[:, center_x:] = 0
    return (side if (side > 0.5).any() else None for side in (side_l, side_r))


def _bbox(mask):
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    return xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1


def _head_roll(eyes, below):
    """How far the head leans clockwise, in radians: both eyes spread along one line, and the mouth lies under it."""
    ys, xs = torch.nonzero(eyes > 0.5, as_tuple=True)
    points = torch.stack([xs, ys], 1).float()
    center = points.mean(0)
    across = torch.linalg.svd(points - center, full_matrices=False)[2][0]
    up = torch.stack([across[1], -across[0]])
    ys, xs = torch.nonzero(below > 0.5, as_tuple=True)
    if up @ (center - torch.stack([xs, ys]).float().mean(1)) < 0:
        up = -up
    return math.atan2(up[0], -up[1])


def _upright(pose):
    """Affine turning the drawing until the head stands straight, and the size of the canvas holding all of it."""
    width, height = pose["size"]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), math.degrees(pose["roll"]), 1)
    size = [math.ceil(abs(matrix[0, 0]) * width + abs(matrix[0, 1]) * height), math.ceil(abs(matrix[0, 1]) * width + abs(matrix[0, 0]) * height)]
    matrix[:, 2] += (size[0] - width) / 2, (size[1] - height) / 2
    return matrix, size


def upright(image, pose, flags=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT):
    """A numpy image of the drawing, brought onto the canvas where the head stands straight."""
    if pose["roll"] == 0:
        return image
    matrix, size = _upright(pose)
    return cv2.warpAffine(image, matrix, size, flags=flags, borderMode=border)


def segment(perceive, clip, image, alpha=None):
    """image: [1,H,W,3]. Returns numpy masks keyed by part, plus joints in pixels.
    Everything comes back on a canvas where the head stands straight; pose tells how to put it back on the drawing."""
    height, width = image.shape[1:3]
    found = {name: _run(perceive, clip, image, "matting", query)[0] for name, query in BODY_QUERIES.items()}
    if len(found["face"]) == 0:
        raise ValueError("Live Avatar: no face found in the image.")

    x0, y0, x1, y1 = _bbox(found["face"].amax(0))
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    radius = int(max(x1 - x0, y1 - y0) * 0.85)
    cx0, cy0, cx1, cy1 = max(center_x - radius, 0), max(center_y - radius, 0), min(center_x + radius, width), min(center_y + radius, height)
    scale = HEAD_CROP / max(cx1 - cx0, cy1 - cy0)
    crop_size = (round((cy1 - cy0) * scale), round((cx1 - cx0) * scale))
    crop = F.interpolate(image[:, cy0:cy1, cx0:cx1].movedim(-1, 1), size=crop_size, mode="bicubic", antialias=True).clamp(0, 1).movedim(1, -1)

    def paste(masks):
        small = F.interpolate(masks[:, None], size=(cy1 - cy0, cx1 - cx0), mode="area")[:, 0]
        full = torch.zeros((masks.shape[0], height, width))
        full[:, cy0:cy1, cx0:cx1] = small
        return full

    head = {name: paste(_run(perceive, clip, crop, "matting", query)[0]) for name, query in HEAD_QUERIES.items()}

    joints = {}
    for joint in JOINTS:
        for suffix, word in (("l", "left"), ("r", "right")):
            points = _run(perceive, clip, image, "point", word + " " + joint)[1]
            if points:
                joints[joint + "_" + suffix] = (points[0]["x"], points[0]["y"])

    pose = {"roll": 0.0, "size": [width, height]}
    eyes = torch.cat([head[name] for name in ("eye", "iris", "lash")])
    if len(eyes):
        roll = _head_roll(eyes.amax(0), head["mouth"].amax(0) if len(head["mouth"]) else found["face"].amax(0))
        if abs(roll) > MIN_ROLL:
            pose["roll"] = roll
            matrix, (width, height) = _upright(pose)

            def turn(masks):
                return torch.stack([torch.from_numpy(upright(m.cpu().numpy(), pose)) for m in masks]) if len(masks) else torch.zeros((0, height, width))

            image = torch.from_numpy(upright(image[0].cpu().numpy(), pose, cv2.INTER_LANCZOS4, cv2.BORDER_REPLICATE)).clamp(0, 1)[None]
            found = {name: turn(masks) for name, masks in found.items()}
            head = {name: turn(masks) for name, masks in head.items()}
            if alpha is not None:
                alpha = turn(alpha)
            joints = {name: tuple(matrix @ (x, y, 1)) for name, (x, y) in joints.items()}
            x0, _, x1, _ = _bbox(found["face"].amax(0))
            center_x = (x0 + x1) // 2

    masks = {}
    for name in ("hair", "neck", "skirt", "tail"):
        if len(found[name]):
            masks[name] = found[name].amax(0)
    masks["person"] = found["person"].amax(0) if len(found["person"]) else torch.zeros((height, width))
    masks["face"] = head["face"].amax(0) if len(head["face"]) else found["face"].amax(0)
    if len(head["bangs"]):
        masks["bangs"] = head["bangs"].amax(0)
    if len(head["mouth"]):
        masks["mouth"] = head["mouth"].amax(0)
    paired = [(_split_by_side(found[name], center_x), part) for name, part in (("arms", "arm"), ("hands", "hand"), ("legs", "leg"), ("animal_ears", "animal_ear"))]
    paired += [(_split_at_center(head[name], center_x), name) for name in ("eye", "iris", "lash", "brow") if len(head[name])]
    # Breasts are no layer of their own: they only weight the torso toward a soft-body bone.
    if len(found["breasts"]):
        bx0, _, bx1, _ = _bbox(found["breasts"].amax(0))
        paired.append((_split_at_center(found["breasts"], (bx0 + bx1) // 2), "breast"))
    for sides, part in paired:
        for suffix, side in zip(("l", "r"), sides):
            if side is not None:
                masks[part + "_" + suffix] = side

    return {"image": image[0].cpu().numpy(), "alpha": None if alpha is None else alpha[0].cpu().numpy(),
            "masks": {name: mask.numpy() for name, mask in masks.items()}, "joints": joints, "pose": pose}

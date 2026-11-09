"""Body-part segmentation with WeVisOne-VL: full-frame parts, then face parts on an enlarged head crop."""
import json

import torch
import torch.nn.functional as F

BODY_QUERIES = {"person": "person", "hair": "hair", "face": "face", "neck": "neck", "arms": "arms", "hands": "hands",
                "legs": "legs", "skirt": "skirt", "tail": "tail", "animal_ears": "animal ears", "breasts": "breasts"}
HEAD_QUERIES = {"face": "face skin", "bangs": "bangs", "eye": "eye", "iris": "iris", "lash": "eyelashes", "brow": "eyebrow",
                "mouth": "mouth"}
JOINTS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
HEAD_CROP = 1024


def _run(perceive, clip, image, task, query):
    out = perceive.execute(clip, image, task, query, 512, True).args
    return out[3].float().cpu(), json.loads(out[2])


def _split_by_side(masks, center_x):
    """Instances to (image-right, image-left). Image-right is the character's left side."""
    left, right = None, None
    for m in masks:
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if len(xs) == 0:
            continue
        if xs.float().mean() >= center_x:
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


def segment(perceive, clip, image, alpha=None):
    """image: [1,H,W,3]. Returns numpy masks keyed by part, plus joints in pixels."""
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
        paired.append((_split_at_center(found["breasts"], center_x), "breast"))
    for sides, part in paired:
        for suffix, side in zip(("l", "r"), sides):
            if side is not None:
                masks[part + "_" + suffix] = side

    joints = {}
    for joint in JOINTS:
        for suffix, word in (("l", "left"), ("r", "right")):
            points = _run(perceive, clip, image, "point", word + " " + joint)[1]
            if points:
                joints[joint + "_" + suffix] = (points[0]["x"], points[0]["y"])

    return {"image": image[0].cpu().numpy(), "alpha": None if alpha is None else alpha[0].cpu().numpy(),
            "masks": {name: mask.numpy() for name, mask in masks.items()}, "joints": joints}

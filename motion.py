"""Clips from captured body motion: MHR pose data (InstantHMR, SAM 3D Body) retargeted onto an avatar's params.
Angles are solved in the frame of each bone's parent, so they land on the same Euler channels the runtime drives."""
import math

import numpy as np
from comfy_extras.sam3d_body.export.glb_shared import OPENPOSE18_TO_MHR70
from scipy.spatial.transform import Rotation

NOSE, NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE, R_EYE, L_EYE, R_EAR, L_EAR = range(18)
LIMBS = {"l": (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE), "r": (R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE)}
LIMIT = 1.5
SIDEWAYS = 0.2


def _keypoints(pose_data, person):
    """COCO-18 keypoints and root translation per frame, x right, y up, z to the camera. Frames without the person hold the last one."""
    points, roots = [], []
    for people in pose_data["frames"]:
        if len(people) > person:
            p = people[person]
            points.append(np.asarray(p["pred_keypoints_3d"], np.float64)[OPENPOSE18_TO_MHR70] * (1, -1, -1))
            roots.append(np.asarray(p["pred_cam_t"], np.float64) * (1, -1, -1))
        elif points:
            points.append(points[-1])
            roots.append(roots[-1])
    if not points:
        raise ValueError(f"Live Avatar: person {person} is in no frame of the pose data.")
    points, roots = np.stack(points), np.stack(roots)
    # The avatar is drawn from the front: the performer's usual facing becomes the camera, whatever angle the video was shot from.
    across = points[:, L_SHOULDER] - points[:, R_SHOULDER]
    facing = np.angle(np.exp(1j * np.arctan2(-across[:, 2], across[:, 0])).mean())
    turn = Rotation.from_euler("y", -facing).as_matrix().T
    return points @ turn, roots @ turn


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _basis(x, forward=None, up=None):
    """Rotation whose columns are x, y, z, from an x axis and a forward (z) or up (y) hint."""
    x = _unit(x)
    if forward is not None:
        z = _unit(forward - forward @ x * x)
        y = np.cross(z, x)
    else:
        y = _unit(up - up @ x * x)
        z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _drivers(params, name, bones, channel):
    return sum(d["scale"] for d in params.get(name, {}).get("drivers", []) if d["bone"] in bones and d["channel"] == channel)


def _bone_rotation(params, values, bone):
    angles = [0.0, 0.0, 0.0]
    for name, value in values.items():
        for d in params.get(name, {}).get("drivers", []):
            if d["bone"] == bone and d["channel"] in ("rx", "ry", "rz"):
                angles["xyz".index(d["channel"][1])] += value * d["scale"]
    return Rotation.from_euler("XYZ", angles).as_matrix()


def _arm(u, rest):
    """Raise (about z) then swing (about x) taking the rest direction to u. Of the two solutions, the one turning least:
    the other is the same pose with both angles past the param range."""
    r = math.hypot(u[1], u[2])
    swing = math.atan2(u[2], u[1])
    rest_angle = math.atan2(rest[1], rest[0])
    options = [(_wrap(math.atan2(r, u[0]) - rest_angle), swing), (_wrap(math.atan2(-r, u[0]) - rest_angle), _wrap(swing - math.pi))]
    raise_angle, swing = min(options, key=lambda o: abs(o[0]) + abs(o[1]))
    # An arm held straight out to the side has no defined swing, yet the swing sets the elbow's bend plane: fade it out there.
    return raise_angle, swing * min(r / SIDEWAYS, 1)


def _bend(v, rest, axis):
    """Signed angle from rest to v around one axis of the parent frame, both projected onto its plane."""
    a, b = {"x": (2, 1), "z": (1, 0)}[axis]
    return _wrap(math.atan2(v[a], v[b]) - math.atan2(rest[a], rest[b]))


def _usual(rotations, channels):
    """Median rotation of a clip on the given XYZ channels."""
    return Rotation.from_euler("XYZ", np.median([Rotation.from_matrix(r).as_euler("XYZ") for r in rotations], 0) * channels).as_matrix()


def _body_length(neck, hips, legs):
    return np.linalg.norm(neck - hips) + (np.mean([np.linalg.norm(h - k) + np.linalg.norm(k - a) for h, k, a in legs]) if legs else 0)


def motion_clip(bones, params, pose_data, fps, loop, person=0):
    """bones: rest positions in world units by name. Returns a clip with one linear track per param it can drive."""
    points, roots = _keypoints(pose_data, person)
    has = lambda *names: all(n in bones for n in names)
    rest = lambda a, b: _unit(bones[b] - bones[a])
    keys = {}

    hips_mid = (points[:, L_HIP] + points[:, R_HIP]) / 2
    performer = np.median([_body_length(p[NECK], h, [(p[LIMBS[s][3]], p[LIMBS[s][4]], p[LIMBS[s][5]]) for s in LIMBS]) for p, h in zip(points, hips_mid)])
    legs = [(bones["thigh_" + s], bones["shin_" + s], bones["foot_" + s]) for s in LIMBS if has("thigh_" + s, "shin_" + s, "foot_" + s)]
    character = _body_length(bones["neck"], bones["hips"], legs)
    if not legs:
        performer = np.median(np.linalg.norm(points[:, NECK] - hips_mid, axis=1))
    roots = (roots - np.median(roots, 0)) * character / performer
    head_bones = ("head", "neck")
    torsos = [_basis(p[L_SHOULDER] - p[R_SHOULDER], up=p[NECK] - h) for p, h in zip(points, hips_mid)]
    heads = []
    for p, torso in zip(points, torsos):
        ears = p[L_EAR] - p[R_EAR]
        across = ears if np.linalg.norm(ears) > 1e-3 else p[L_EYE] - p[R_EYE]
        heads.append(torso.T @ _basis(across, forward=p[NOSE] - (p[L_EAR] + p[R_EAR]) / 2))
    # Keypoints give no level reference: MHR's neck sits forward of the hips, and the nose below the ears.
    # The performer's usual lean and head tilt are the avatar's rest.
    lean = _usual(torsos, (1, 0, 1))
    tilt = _usual(heads, (1, 0, 1))

    for i, p in enumerate(points):
        t = i / fps
        torso = torsos[i] @ lean.T
        pitch, yaw, roll = Rotation.from_matrix(torso).as_euler("XYZ")
        values = {}
        for name, angle, channel in (("bodyPitch", pitch, "rx"), ("bodyYaw", yaw, "ry"), ("bodyRoll", roll, "rz")):
            scale = _drivers(params, name, ("spine", "chest"), channel)
            if scale:
                values[name] = angle / scale
        chest = _bone_rotation(params, values, "spine") @ _bone_rotation(params, values, "chest")

        head_pitch, head_yaw, head_roll = Rotation.from_matrix(heads[i] @ tilt.T).as_euler("XYZ")
        for name, angle, channel in (("headPitch", head_pitch, "rx"), ("headYaw", head_yaw, "ry"), ("headRoll", head_roll, "rz")):
            scale = _drivers(params, name, head_bones, channel)
            if scale:
                values[name] = angle / scale

        for side, (shoulder, elbow, wrist, hip, knee, ankle) in LIMBS.items():
            S = side.upper()
            if has("arm_" + side, "forearm_" + side, "hand_" + side):
                upper = chest.T @ _unit(p[elbow] - p[shoulder])
                raise_angle, swing = _arm(upper, rest("arm_" + side, "forearm_" + side))
                arm = Rotation.from_euler("XYZ", [swing, 0, raise_angle]).as_matrix()
                lower = arm.T @ chest.T @ _unit(p[wrist] - p[elbow])
                bend = _bend(lower, rest("forearm_" + side, "hand_" + side), "z")
                for name, angle, bone, channel in (("arm" + S + "Raise", raise_angle, "arm_", "rz"), ("arm" + S + "Swing", swing, "arm_", "rx"), ("arm" + S + "Bend", bend, "forearm_", "rz")):
                    scale = _drivers(params, name, (bone + side,), channel)
                    if scale:
                        values[name] = angle / scale
            if has("thigh_" + side, "shin_" + side, "foot_" + side):
                pelvis = _basis(p[L_HIP] - p[R_HIP], up=p[NECK] - hips_mid[i])
                thigh = pelvis.T @ _unit(p[knee] - p[hip])
                swing = _bend(thigh, rest("thigh_" + side, "shin_" + side), "x")
                shin = Rotation.from_euler("X", swing).as_matrix().T @ pelvis.T @ _unit(p[ankle] - p[knee])
                bend = _bend(shin, rest("shin_" + side, "foot_" + side), "x")
                for name, angle, bone in (("leg" + S + "Swing", swing, "thigh_"), ("leg" + S + "Bend", bend, "shin_")):
                    scale = _drivers(params, name, (bone + side,), "rx")
                    if scale:
                        values[name] = angle / scale

        for name, offset, channel in (("sway", roots[i, 0], "px"), ("bounce", roots[i, 1], "py")):
            scale = _drivers(params, name, ("root",), channel)
            if scale:
                values[name] = offset / scale
        for name, value in values.items():
            keys.setdefault(name, []).append([round(t, 4), round(float(np.clip(value, -LIMIT, LIMIT)), 4)])

    return {"duration": round(len(points) / fps, 4), "loop": loop, "tracks": [{"param": name, "keys": k, "linear": True} for name, k in keys.items()]}

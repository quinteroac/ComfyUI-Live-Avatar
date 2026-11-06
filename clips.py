"""Parameters, clips and springs stored in the avatar. Runtime code stays generic: clips only key params."""
import math

LOCK_BONES = 4


def build_params(bones, parts, face_unit, wide_turn):
    params = {}

    def bone_param(name, drivers):
        drivers = [{"bone": bone, "channel": channel, "scale": scale} for bone, channel, scale in drivers if bone in bones]
        if drivers:
            params[name] = {"drivers": drivers}

    # Strands hang from the head: their first bone undoes most of its rotation.
    # With a head splat the turn may go past what flat layers can show.
    head_yaw, neck_yaw = (0.6, 0.2) if wide_turn else (0.45, 0.15)
    bone_param("headYaw", [("head", "ry", head_yaw), ("neck", "ry", neck_yaw), ("lock_l_0", "ry", -head_yaw - neck_yaw + 0.1), ("lock_r_0", "ry", -head_yaw - neck_yaw + 0.1)])
    bone_param("headPitch", [("head", "rx", 0.4), ("neck", "rx", 0.15), ("lock_l_0", "rx", -0.45), ("lock_r_0", "rx", -0.45)])
    bone_param("headRoll", [("head", "rz", 0.28), ("neck", "rz", 0.1), ("lock_l_0", "rz", -0.34), ("lock_r_0", "rz", -0.34)])
    bone_param("bodyYaw", [("spine", "ry", 0.22), ("chest", "ry", 0.22)])
    bone_param("bodyRoll", [("spine", "rz", 0.12), ("chest", "rz", 0.1), ("neck", "rz", -0.1)])
    bone_param("bodyPitch", [("spine", "rx", 0.15), ("chest", "rx", 0.15)])
    bone_param("breath", [("chest", "sy", 0.012), ("chest", "sx", 0.008), ("arm_l", "rz", 0.015), ("arm_r", "rz", -0.015)])
    bone_param("bounce", [("root", "py", 0.05)])
    bone_param("sway", [("root", "px", 0.04)])
    for side, sign in (("L", 1), ("R", -1)):
        s = side.lower()
        bone_param("arm" + side + "Raise", [("arm_" + s, "rz", 2.6 * sign)])
        bone_param("arm" + side + "Swing", [("arm_" + s, "rx", -1.0)])
        bone_param("arm" + side + "Bend", [("forearm_" + s, "rz", 2.2 * sign)])
        bone_param("hand" + side, [("hand_" + s, "rz", 0.7 * sign)])
        bone_param("leg" + side + "Swing", [("thigh_" + s, "rx", -0.8)])
        bone_param("leg" + side + "Bend", [("shin_" + s, "rx", 1.2)])
        bone_param("ear" + side, [("animal_ear_" + s, "rz", 0.35 * sign)])
    bone_param("gazeX", [("iris_l", "px", face_unit * 0.07), ("iris_r", "px", face_unit * 0.07)])
    bone_param("gazeY", [("iris_l", "py", face_unit * 0.035), ("iris_r", "py", face_unit * 0.035)])
    bone_param("tail", [("tail_0", "rz", 0.5), ("tail_1", "rz", 0.3)])

    morphs = {}
    for name, part in parts.items():
        for morph in part["morphs"]:
            morphs.setdefault(morph, []).append(name)
    for morph, names in morphs.items():
        params[morph] = {"morphs": names}
    return params


def _track(param, *keys):
    return {"param": param, "keys": [list(k) for k in keys]}


def _wave(duration, period, amplitude, phase=0.0, offset=0.0, samples=None):
    count = samples or max(int(duration / period * 8), 8)
    return [(duration * i / count, offset + amplitude * math.sin(math.tau * (duration * i / count / period) + phase)) for i in range(count + 1)]


def build_clips(params):
    """Keys are [time, value]. Loop clips return to their first value; one-shot clips return to zero."""
    def clip(duration, loop, *tracks):
        return {"duration": duration, "loop": loop, "tracks": [t for t in tracks if t["param"] in params]}

    clips = {}
    clips["idle"] = clip(
        6.0, True,
        _track("breath", *_wave(6.0, 3.0, 0.5, offset=0.5)),
        _track("bodyRoll", *_wave(6.0, 6.0, 0.1)),
        _track("sway", *_wave(6.0, 6.0, 0.04, phase=1.0)),
        _track("headRoll", *_wave(6.0, 6.0, 0.09, phase=0.6)),
        _track("headYaw", *_wave(6.0, 6.0, 0.08, phase=2.0)),
        _track("headPitch", *_wave(6.0, 3.0, 0.05, phase=1.0)),
    )
    wave_keys = [(0, 0), (0.45, 0.55), (2.35, 0.55), (2.8, 0)]
    bend_keys = [(0, 0), (0.45, 0.6)] + [(0.45 + i * 0.238, 0.6 + (0.16 if i % 2 == 0 else -0.04)) for i in range(1, 8)] + [(2.35, 0.6), (2.8, 0)]
    clips["wave"] = clip(2.8, False, _track("armLRaise", *wave_keys), _track("armLBend", *bend_keys),
                         _track("headRoll", (0, 0), (0.5, -0.35), (2.3, -0.35), (2.8, 0)), _track("mouthSmile", (0, 0), (0.4, 1), (2.4, 1), (2.8, 0)),
                         _track("bodyRoll", (0, 0), (0.5, -0.3), (2.3, -0.3), (2.8, 0)))
    clips["nod"] = clip(1.3, False, _track("headPitch", (0, 0), (0.2, 0.75), (0.45, -0.1), (0.7, 0.6), (1.0, -0.05), (1.3, 0)))
    clips["shake"] = clip(1.4, False, _track("headYaw", (0, 0), (0.2, 0.7), (0.5, -0.7), (0.8, 0.6), (1.1, -0.4), (1.4, 0)))
    clips["happy"] = clip(1.6, False, _track("bounce", (0, 0), (0.15, -0.25), (0.35, 0.85), (0.55, -0.2), (0.75, 0.6), (0.95, -0.1), (1.2, 0)),
                          _track("mouthSmile", (0, 0), (0.2, 1), (1.3, 1), (1.6, 0)), _track("mouthOpen", (0, 0), (0.2, 0.5), (1.2, 0.5), (1.5, 0)),
                          _track("armLRaise", (0, 0), (0.35, 0.28), (1.2, 0.22), (1.6, 0)), _track("armRRaise", (0, 0), (0.35, 0.28), (1.2, 0.22), (1.6, 0)),
                          _track("browL", (0, 0), (0.3, 1), (1.3, 1), (1.6, 0)), _track("browR", (0, 0), (0.3, 1), (1.3, 1), (1.6, 0)))
    clips["think"] = clip(3.0, False, _track("headRoll", (0, 0), (0.5, 0.55), (2.5, 0.5), (3.0, 0)), _track("headYaw", (0, 0), (0.5, 0.3), (2.5, 0.3), (3.0, 0)),
                          _track("gazeX", (0, 0), (0.45, 0.8), (2.5, 0.8), (3.0, 0)), _track("gazeY", (0, 0), (0.45, 0.7), (2.5, 0.7), (3.0, 0)),
                          _track("browL", (0, 0), (0.5, 0.8), (2.5, 0.8), (3.0, 0)))
    clips["talk"] = clip(1.8, True, _track("mouthOpen", (0, 0), (0.1, 0.6), (0.22, 0.15), (0.36, 0.8), (0.5, 0.2), (0.62, 0.55), (0.8, 0.05), (0.98, 0.7), (1.12, 0.25), (1.3, 0.6), (1.5, 0.1), (1.8, 0)),
                         _track("headPitch", *_wave(1.8, 0.9, 0.08)), _track("headRoll", *_wave(1.8, 1.8, 0.06)))
    half = 0.45
    swing = [(0, 0.45), (half, -0.45), (half * 2, 0.45)]
    opposite = [(0, -0.45), (half, 0.45), (half * 2, -0.45)]
    clips["walk"] = clip(
        half * 2, True,
        _track("legLSwing", *swing), _track("legRSwing", *opposite),
        _track("legLBend", (0, 0.1), (half * 0.5, 0.45), (half, 0.05), (half * 2, 0.1)),
        _track("legRBend", (0, 0.05), (half, 0.1), (half * 1.5, 0.45), (half * 2, 0.05)),
        _track("armLSwing", *opposite), _track("armRSwing", *swing),
        _track("bounce", *_wave(half * 2, half, 0.14, offset=0.05)),
        _track("bodyRoll", *_wave(half * 2, half * 2, 0.25)),
        _track("bodyYaw", *_wave(half * 2, half * 2, 0.3, phase=1.57)),
    )
    return clips


def build_springs(bones):
    springs = []
    for chain, stiffness, gain in (("lock_l", 26.0, 1.0), ("lock_r", 26.0, 1.0), ("tail", 18.0, 1.3)):
        for i in range(LOCK_BONES):
            name = chain + "_" + str(i)
            if name in bones:
                springs.append({"bone": name, "stiffness": stiffness * (1 - 0.13 * i), "damping": 4.5, "gain": gain * (0.5 + 0.28 * i)})
    for name, stiffness, gain in (("bangs", 60.0, 0.45), ("skirt", 45.0, 0.5), ("animal_ear_l", 80.0, 0.3), ("animal_ear_r", 80.0, 0.3)):
        if name in bones:
            springs.append({"bone": name, "stiffness": stiffness, "damping": 7.0, "gain": gain})
    return springs

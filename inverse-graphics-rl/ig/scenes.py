"""Scene validation and random target generation (the 'infinite dataset')."""
import numpy as np

PALETTE = {
    "red": [0.85, 0.25, 0.2], "blue": [0.2, 0.4, 0.85], "green": [0.25, 0.7, 0.3],
    "yellow": [0.9, 0.8, 0.2], "white": [0.9, 0.9, 0.9], "brown": [0.55, 0.35, 0.2],
}
MAX_PRIMS = 24


def validate(scene):
    """Raise ValueError on anything the renderer can't handle; return a cleaned copy."""
    if not isinstance(scene, list) or not 1 <= len(scene) <= MAX_PRIMS:
        raise ValueError(f"scene must be a list of 1..{MAX_PRIMS} primitives")
    out = []
    for p in scene:
        kind = p.get("type")
        q = {"type": str(kind), "center": _vec(p["center"], 3),
             "color": [float(np.clip(v, 0, 1)) for v in _vec(p.get("color", [0.8] * 3), 3)]}
        if kind == "sphere":
            q["radius"] = _pos(p["radius"])
        elif kind == "box":
            q["size"] = [_pos(v) for v in _vec(p["size"], 3)]
            q["rot_y"] = float(p.get("rot_y", 0.0))
        elif kind == "cylinder":
            q["radius"], q["height"] = _pos(p["radius"]), _pos(p["height"])
            q["axis"] = p.get("axis", "y")
            if q["axis"] not in ("x", "y", "z"):
                raise ValueError("cylinder axis must be x, y or z")
        else:
            raise ValueError(f"unknown primitive type {kind!r}")
        out.append(q)
    return out


def _vec(v, n):
    v = [float(x) for x in v]
    if len(v) != n:
        raise ValueError(f"expected {n} numbers, got {v}")
    return v


def _pos(x):
    x = float(x)
    if not 1e-3 <= x <= 5:
        raise ValueError(f"size {x} out of range")
    return x


def random_primitive(rng, on_top_of=None):
    kind = str(rng.choice(["sphere", "box", "cylinder"]))
    color = PALETTE[str(rng.choice(list(PALETTE)))]
    s = rng.uniform(0.12, 0.35)
    if kind == "sphere":
        dims, half_h = {"radius": s}, s
    elif kind == "box":
        size = [rng.uniform(0.15, 0.7), rng.uniform(0.15, 0.7), rng.uniform(0.15, 0.7)]
        dims, half_h = {"size": size, "rot_y": float(rng.uniform(0, 90))}, size[1] / 2
    else:
        h = rng.uniform(0.2, 0.8)
        dims, half_h = {"radius": s * 0.8, "height": h, "axis": "y"}, h / 2
    if on_top_of is None:
        x, z, base = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 0.0
    else:
        x, z = on_top_of["center"][0], on_top_of["center"][2]
        base = on_top_of["_top"]
    prim = {"type": kind, "center": [x, base + half_h, z], "color": color, **dims}
    prim["_top"] = base + 2 * half_h
    return prim


def random_scene(rng, level=1):
    """level 1: one primitive. level 2: 2-3 primitives. level 3: 3-5 with stacking."""
    n = {1: 1, 2: int(rng.integers(2, 4)), 3: int(rng.integers(3, 6))}[level]
    prims = []
    for _ in range(n):
        stack = level >= 3 and prims and rng.random() < 0.4
        prims.append(random_primitive(rng, on_top_of=prims[-1] if stack else None))
    for p in prims:
        p.pop("_top")
    return validate(prims)

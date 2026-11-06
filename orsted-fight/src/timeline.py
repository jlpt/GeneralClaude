"""Keyframe tracks. Keys are partial pose dicts that inherit everything they don't set from the previous key.

Vectors and numbers interpolate with the *incoming* key's easing; strings / bools / None snap (they switch
when the key is reached). Easing "step" holds the previous key until the frame of the next one.
"""
import math
import numpy as np

EASE = {
    "linear": lambda t: t,
    "in": lambda t: t * t,
    "in3": lambda t: t * t * t,
    "out": lambda t: 1 - (1 - t) ** 2,
    "out3": lambda t: 1 - (1 - t) ** 3,
    "inout": lambda t: t * t * (3 - 2 * t),
    "step": lambda t: 0.0,
}


def _is_vec(v):
    return isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2 and not isinstance(v[0], str)


def _lerp(a, b, t):
    if _is_vec(a) and _is_vec(b):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    if isinstance(a, bool) or isinstance(b, bool):
        return a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    return a


# fields that never interpolate (signs, z-order, flags, names)
DISCRETE = {"face", "z", "hide", "flash", "expr", "wstate", "wz", "zn", "zf", "two", "thrust", "smear", "warm",
            "shape_n", "shape_f", "bn", "bf", "kn", "kf", "hk", "aura_col"}


class Track:
    def __init__(self, name, **base):
        self.name = name
        self.base = base
        self.keys = []          # (frame, ease, fields)
        self._resolved = None

    def key(self, f, ease="inout", **fields):
        self.keys.append((f, ease, fields))
        self._resolved = None
        return self

    def hold(self, f):
        """Pin the current pose until frame f (an unchanged key)."""
        return self.key(f, "linear")

    def last_frame(self):
        return max(k[0] for k in self.keys) if self.keys else 0

    def _resolve(self):
        keys = sorted(self.keys, key=lambda k: k[0])
        cur = dict(self.base)
        out = []
        for f, ease, fields in keys:
            cur = dict(cur)
            for k, v in fields.items():
                if _is_vec(v):
                    v = (float(v[0]), float(v[1]))
                cur[k] = v
            out.append((f, ease, cur))
        self._resolved = out

    def at(self, f):
        if self._resolved is None:
            self._resolve()
        ks = self._resolved
        if not ks:
            return dict(self.base)
        if f <= ks[0][0]:
            return dict(ks[0][2])
        for i in range(1, len(ks)):
            f1, ease, p1 = ks[i]
            f0, _, p0 = ks[i - 1]
            if f < f1:
                t = (f - f0) / max(f1 - f0, 1e-9)
                t = EASE[ease](t)
                out = {}
                for k in set(p0) | set(p1):
                    a, b = p0.get(k, p1.get(k)), p1.get(k, p0.get(k))
                    out[k] = a if k in DISCRETE else _lerp(a, b, t)
                # hit-flash lasts `flash` frames (True = 1) from its key, not until the next key
                fl = p0.get("flash")
                if fl:
                    n = 1 if fl is True else int(fl)
                    out["flash"] = (f - f0) < n
                return out
        return dict(ks[-1][2])


def finalize(pose, f):
    """Procedural life on top of the keyed pose: breathing bob, hair phase, wind gusts."""
    p = dict(pose)
    b = p.get("breath", 0)
    if b:
        bob = 1.0 if ((f + p.get("breath_off", 0)) // 8) % 2 else 0.0
        for k in ("N",):
            if k in p:
                p[k] = (p[k][0], p[k][1] + bob * b)
        if p.get("breath_hands", True):
            for k in ("hn", "hf"):
                if k in p:
                    p[k] = (p[k][0], p[k][1] + bob * b)
    rate = p.get("phase_rate", 0.32)
    p["phase"] = p.get("phase", 0.0) + f * rate
    g = p.get("gust", 1.0)
    if g:
        wx, wy = p.get("wind", (0, 0))
        p["wind"] = (wx + math.sin(f * 0.37) * 0.12 * g, wy + math.cos(f * 0.29) * 0.08 * g)
        cx, cy = p.get("coat", (0, 0))
        p["coat"] = (cx + math.sin(f * 0.61 + 1.0) * 0.25 * g, cy + math.cos(f * 0.47) * 0.15 * g)
    return p

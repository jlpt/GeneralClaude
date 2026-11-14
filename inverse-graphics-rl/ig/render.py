"""Tiny numpy ray caster for a scene DSL of spheres, boxes and cylinders.

Coordinates: y is up, the ground is y = 0, objects live roughly in [-1, 1] on x/z.
A scene is a list of dicts, for example:
    {"type": "sphere",   "center": [x, y, z], "radius": r, "color": [r, g, b]}
    {"type": "box",      "center": [x, y, z], "size": [sx, sy, sz], "rot_y": deg, "color": [...]}
    {"type": "cylinder", "center": [x, y, z], "radius": r, "height": h, "axis": "y", "color": [...]}
Colors are 0..1 floats.
"""
from dataclasses import dataclass

import numpy as np

EPS = 1e-6
LIGHT = np.array([0.5, 1.0, 0.35]) / np.linalg.norm([0.5, 1.0, 0.35])
LOOK_AT = np.array([0.0, 0.4, 0.0])


@dataclass(frozen=True)
class Camera:
    azimuth: float          # degrees, 0 = looking from +z toward the origin
    elevation: float        # degrees above the horizon
    distance: float = 2.6
    fov: float = 40.0       # vertical field of view, degrees

    def position(self):
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        return LOOK_AT + self.distance * np.array(
            [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])

    def describe(self):
        p = self.position()
        return (f"azimuth={self.azimuth:.0f}deg elevation={self.elevation:.0f}deg "
                f"distance={self.distance} position=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) "
                f"looking at {tuple(LOOK_AT.tolist())}, vertical fov={self.fov}deg")


def camera_rays(cam: Camera, res: int):
    origin = cam.position()
    fwd = LOOK_AT - origin
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0]) if abs(fwd[1]) < 0.99 else np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    half = np.tan(np.radians(cam.fov) / 2)
    coords = (np.arange(res) + 0.5) / res * 2 - 1
    xs, ys = np.meshgrid(coords * half, -coords * half)
    dirs = fwd + xs[..., None] * right + ys[..., None] * up
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return origin, dirs.reshape(-1, 3)


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _safe(d):
    return np.where(np.abs(d) < EPS, np.where(d < 0, -EPS, EPS), d)


def _hit_sphere(o, d, prim):
    c = np.asarray(prim["center"], float)
    r = float(prim["radius"])
    oc = o - c
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    sq = np.sqrt(np.maximum(disc, 0))
    t0, t1 = -b - sq, -b + sq
    t = np.where(t0 > EPS, t0, t1)
    t = np.where((disc >= 0) & (t > EPS), t, np.inf)
    n = (o + t[:, None] * d - c) / r
    return t, n


def _hit_box(o, d, prim):
    c = np.asarray(prim["center"], float)
    h = np.asarray(prim["size"], float) / 2
    R = _rot_y(float(prim.get("rot_y", 0.0)))
    ol = (o - c) @ R          # R^T (o - c): world -> local
    dl = _safe(d @ R)
    t1, t2 = (-h - ol) / dl, (h - ol) / dl
    tnear, tfar = np.minimum(t1, t2), np.maximum(t1, t2)
    tmin, tmax = tnear.max(axis=1), tfar.min(axis=1)
    hit = (tmax >= np.maximum(tmin, EPS))
    t = np.where(tmin > EPS, tmin, tmax)
    t = np.where(hit, t, np.inf)
    axis = tnear.argmax(axis=1)
    nl = np.zeros_like(dl)
    nl[np.arange(len(dl)), axis] = -np.sign(dl[np.arange(len(dl)), axis])
    return t, nl @ R.T


_AXES = {"x": (0, 1, 2), "y": (1, 0, 2), "z": (2, 0, 1)}


def _hit_cylinder(o, d, prim):
    c = np.asarray(prim["center"], float)
    r, hh = float(prim["radius"]), float(prim["height"]) / 2
    a, b, cc = _AXES[prim.get("axis", "y")]
    p = o - c
    # side
    A = d[:, b] ** 2 + d[:, cc] ** 2
    B = 2 * (p[b] * d[:, b] + p[cc] * d[:, cc])
    C = p[b] ** 2 + p[cc] ** 2 - r * r
    disc = B * B - 4 * A * C
    sq = np.sqrt(np.maximum(disc, 0))
    A_s = np.maximum(A, EPS)
    ts = []
    for t_side in ((-B - sq) / (2 * A_s), (-B + sq) / (2 * A_s)):
        ok = (disc >= 0) & (A > EPS) & (t_side > EPS) & (np.abs(p[a] + t_side * d[:, a]) <= hh)
        ts.append(np.where(ok, t_side, np.inf))
    # caps
    da = _safe(d[:, a])
    for sgn in (-1, 1):
        t_cap = (sgn * hh - p[a]) / da
        qb, qc = p[b] + t_cap * d[:, b], p[cc] + t_cap * d[:, cc]
        ok = (t_cap > EPS) & (qb ** 2 + qc ** 2 <= r * r)
        ts.append(np.where(ok, t_cap, np.inf))
    ts = np.stack(ts, axis=1)
    which = ts.argmin(axis=1)
    t = ts[np.arange(len(ts)), which]
    q = p + np.where(np.isfinite(t), t, 0)[:, None] * d
    n = np.zeros_like(d)
    side = which < 2
    n[side, b] = q[side, b] / r
    n[side, cc] = q[side, cc] / r
    n[which == 2, a] = -1
    n[which == 3, a] = 1
    return t, n


HITTERS = {"sphere": _hit_sphere, "box": _hit_box, "cylinder": _hit_cylinder}


def render(scene, cam: Camera, res: int = 64):
    """Return (rgb HxWx3 float, mask HxW bool, depth HxW float with inf for background)."""
    origin, dirs = camera_rays(cam, res)
    n_rays = len(dirs)
    best_t = np.full(n_rays, np.inf)
    normal = np.zeros((n_rays, 3))
    color = np.zeros((n_rays, 3))
    for prim in scene:
        t, n = HITTERS[prim["type"]](origin, dirs, prim)
        closer = t < best_t
        best_t[closer] = t[closer]
        normal[closer] = n[closer]
        color[closer] = np.asarray(prim.get("color", [0.8, 0.8, 0.8]), float)
    mask = np.isfinite(best_t)
    # shade double-sided so normals pointing away from the camera still light up
    facing = np.sign(-(normal * dirs).sum(axis=1, keepdims=True))
    lambert = np.clip(((normal * facing) @ LIGHT), 0, 1)
    rgb = color * (0.3 + 0.7 * lambert)[:, None]
    rgb[~mask] = 0
    return rgb.reshape(res, res, 3), mask.reshape(res, res), best_t.reshape(res, res)


def to_png(rgb, path, scale=1):
    from PIL import Image
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)

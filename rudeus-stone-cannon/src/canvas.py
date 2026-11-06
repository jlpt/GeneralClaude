"""Canvas, palette and drawing primitives.

The scene is drawn into a float RGB buffer that is a little larger than the
final frame (so screen shake can crop a shifted window without exposing edges).
Every pixel remembers which layer last drew it, and whether it should be
ordered-dithered when it is snapped to the palette. At the end each layer is
quantised against its own sub-palette, which keeps the art to a fixed set of
hand-picked colours.
"""
import math
import numpy as np

W, H = 320, 180          # final frame
M = 10                   # margin for screen shake
CW, CH = W + 2 * M, H + 2 * M

# screen-space coordinates of every canvas pixel
YS, XS = np.mgrid[0:CH, 0:CW].astype(np.float32)
XS -= M
YS -= M


def hexc(s):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# ---------------------------------------------------------------- palette
P = {
    # night sky, zenith -> horizon
    "sky0": "0a0818", "sky1": "110f2a", "sky2": "1a1840", "sky3": "252257", "sky4": "33306d",
    "sky5": "463f80", "sky6": "5c4d8e", "sky7": "7a5a96", "sky8": "9c6a98", "sky9": "c27e94",
    "star": "fff6e0", "star_d": "aaa6d0",
    "moon_h": "fffbea", "moon_l": "f0e8cc", "moon_b": "d2c8ac", "moon_d": "a69e8e", "moon_s": "7e7890",
    # mountains and plain
    "mtn0": "2a2650", "mtn1": "3c3768", "mtn2": "1b1938", "mtn3": "24224a",
    "mtn4": "48345a", "mtn5": "6c465e", "mtn6": "965a58",
    "gr0": "0b1418", "gr1": "122222", "gr2": "1a3229", "gr3": "244630", "gr4": "325e38",
    "gr5": "487a40", "gr6": "6c9e4a",
    "dt0": "1c140e", "dt1": "33251a", "dt2": "50392a", "dt3": "735440",
    # crag
    "rk0": "160f12", "rk1": "2a1e22", "rk2": "413034", "rk3": "5b4544", "rk4": "7a5e54",
    "rk5": "9c7c68", "rk6": "c2a084",
    # earth magic (warm)
    "mg0": "3e160c", "mg1": "6e260e", "mg2": "a8401a", "mg3": "dc6a22", "mg4": "ff9a36",
    "mg5": "ffc655", "mg6": "ffe996", "mg7": "fffae2", "white": "ffffff",
    # stone bullet
    "st0": "1c181c", "st1": "383236", "st2": "564e52", "st3": "7a7072", "st4": "a49a98", "st5": "d0c8c2",
    # air / shock
    "air0": "7c90c4", "air1": "b4c6ee", "air2": "e4eeff",
    # Rudeus
    "ol": "1a1014",
    "sk0": "5a2e28", "sk1": "b4705a", "sk2": "e6a07e", "sk3": "fcc8a2", "sk4": "ffe4ca",
    "hr0": "1e110c", "hr1": "3a2216", "hr2": "583620", "hr3": "7a502c", "hr4": "a0703c",
    "ey0": "12241e", "ey1": "2e8a60", "ey2": "8ee0b0", "mo": "8a3a34",
    "rb0": "171922", "rb1": "282c3a", "rb2": "3e4454", "rb3": "586070", "rb4": "7c8494", "rb5": "a4acb8",
    "tr0": "6a5034", "tr1": "a08458", "tr2": "d2b67e",
    "pa0": "241a14", "pa1": "3e2e22", "pa2": "5a4430",
    "bt0": "1a100c", "bt1": "38241a", "bt2": "5a3c28",
    "be0": "3a2414", "be1": "6a4426", "gold": "e8c462",
}
C = {k: np.array(hexc(v), np.float32) for k, v in P.items()}
NAMES = list(P.keys())

LAYERS = ["Sky", "Land", "Crag", "Back FX", "Back arm", "Rudeus", "Front arm", "Magic", "FX"]
L = {n: i for i, n in enumerate(LAYERS)}


def _group(prefixes):
    return [n for n in NAMES if any(n.startswith(p) for p in prefixes)]


WARM = _group(["mg", "white"])
SUBPAL = {
    "Sky": _group(["sky", "star", "moon"]) + WARM + _group(["air"]),
    "Land": _group(["mtn", "gr", "dt", "sky"]) + WARM,
    "Crag": _group(["rk", "dt", "mtn"]) + WARM,
    "Back FX": NAMES,
    "Back arm": _group(["ol", "sk", "rb", "tr"]) + WARM,
    "Rudeus": _group(["ol", "sk", "hr", "ey", "mo", "rb", "tr", "pa", "bt", "be", "gold"]) + WARM,
    "Front arm": _group(["ol", "sk", "rb", "tr"]) + WARM,
    "Magic": NAMES,
    "FX": NAMES,
}
SUBPAL_RGB = {k: np.stack([C[n] for n in v]) for k, v in SUBPAL.items()}

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0 - 0.47
BAYER = np.tile(BAYER4, (CH // 4 + 1, CW // 4 + 1))[:CH, :CW]


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self):
        self.rgb = np.zeros((CH, CW, 3), np.float32)
        self.owner = np.full((CH, CW), -1, np.int16)
        self.dith = np.zeros((CH, CW), np.float32)   # dither spread per pixel (0 = exact)
        self.protect = None                             # bool mask of pixels draws must not touch

    # -- masks (full canvas, bool)
    def put(self, mask, color, layer, dither=0.0):
        if self.protect is not None:
            mask = mask & ~self.protect
        col = C[color] if isinstance(color, str) else np.asarray(color, np.float32)
        self.rgb[mask] = col
        self.owner[mask] = L[layer]
        self.dith[mask] = dither

    def put_rgb(self, mask, rgb, layer, dither=0.0):
        """rgb: (CH, CW, 3) array; copies where mask."""
        if self.protect is not None:
            mask = mask & ~self.protect
        self.rgb[mask] = rgb[mask]
        self.owner[mask] = L[layer]
        self.dith[mask] = dither

    # -- integer pixel lists in screen space
    def put_px(self, xs, ys, color, layer, dither=0.0):
        xs = np.asarray(np.round(xs), int) + M
        ys = np.asarray(np.round(ys), int) + M
        ok = (xs >= 0) & (xs < CW) & (ys >= 0) & (ys < CH)
        xs, ys = xs[ok], ys[ok]
        if self.protect is not None:
            keep = ~self.protect[ys, xs]
            xs, ys = xs[keep], ys[keep]
        col = C[color] if isinstance(color, str) else np.asarray(color, np.float32)
        self.rgb[ys, xs] = col
        self.owner[ys, xs] = L[layer]
        self.dith[ys, xs] = dither

    def light(self, intensity, color, strength=1.0, dither=36.0, mul=None):
        """Additive light: rgb += color * intensity. Lit pixels get dithered."""
        col = C[color] if isinstance(color, str) else np.asarray(color, np.float32)
        I = np.clip(intensity * strength, 0, None)
        m = I > 0.015
        if not m.any():
            return
        if mul is not None:
            self.rgb[m] *= 1.0 + mul * I[m, None]
        self.rgb[m] += col * I[m, None]
        self.dith[m] = np.maximum(self.dith[m], dither)

    def darken(self, factor, mask=None):
        if mask is None:
            self.rgb *= factor
        else:
            self.rgb[mask] *= factor

    # -- output
    def finish(self, shake=(0, 0)):
        sx, sy = int(round(shake[0])), int(round(shake[1]))
        win = (slice(M + sy, M + sy + H), slice(M + sx, M + sx + W))
        rgb = self.rgb[win]
        owner = self.owner[win]
        dith = self.dith[win]
        bay = BAYER[win]
        out = np.zeros((H, W, 3), np.uint8)
        q = rgb + (bay * dith)[..., None]
        for name, i in L.items():
            m = owner == i
            if not m.any():
                continue
            pal = SUBPAL_RGB[name]
            px = q[m]
            d = (px[:, None, :] - pal[None, :, :])
            d = (d * np.array([0.30, 0.59, 0.11], np.float32) ** 0.5) ** 2
            idx = d.sum(2).argmin(1)
            out[m] = pal[idx].astype(np.uint8)
        return out, owner.copy()


# ---------------------------------------------------------------- geometry
def v2(a):
    return np.array(a, np.float32)


def rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return v2((v[0] * c - v[1] * s, v[0] * s + v[1] * c))


def unit(deg):
    a = math.radians(deg)
    return v2((math.cos(a), math.sin(a)))


def disc(c, r):
    return (XS - c[0]) ** 2 + (YS - c[1]) ** 2 <= r * r


def ellipse(c, rx, ry):
    return ((XS - c[0]) / rx) ** 2 + ((YS - c[1]) / ry) ** 2 <= 1.0


def dist(c):
    return np.hypot(XS - c[0], YS - c[1])


def seg_dist(p0, p1):
    p0, p1 = v2(p0), v2(p1)
    d = p1 - p0
    ll = max(float(d @ d), 1e-9)
    t = np.clip(((XS - p0[0]) * d[0] + (YS - p0[1]) * d[1]) / ll, 0, 1)
    return np.hypot(XS - (p0[0] + t * d[0]), YS - (p0[1] + t * d[1])), t


def capsule(p0, p1, r0, r1=None):
    r1 = r0 if r1 is None else r1
    dd, t = seg_dist(p0, p1)
    return dd <= r0 + (r1 - r0) * t


def polygon(pts):
    pts = [v2(p) for p in pts]
    inside = np.zeros((CH, CW), bool)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        cond = (a[1] > YS) != (b[1] > YS)
        xint = (b[0] - a[0]) * (YS - a[1]) / ((b[1] - a[1]) + 1e-12) + a[0]
        inside ^= cond & (XS < xint)
    return inside


def nb(mask, dx, dy):
    """out[y, x] = mask[y + dy, x + dx] (False outside)."""
    out = np.zeros_like(mask)
    hs, ws = mask.shape
    out[max(-dy, 0):hs + min(-dy, 0), max(-dx, 0):ws + min(-dx, 0)] = \
        mask[max(dy, 0):hs + min(dy, 0), max(dx, 0):ws + min(dx, 0)]
    return out


def dilate4(mask):
    return mask | nb(mask, 1, 0) | nb(mask, -1, 0) | nb(mask, 0, 1) | nb(mask, 0, -1)


def outline_of(mask):
    return dilate4(mask) & ~mask


def line_pts(p0, p1):
    """Bresenham line as integer (xs, ys) in screen space."""
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    xs, ys = [], []
    for _ in range(2000):
        xs.append(x0)
        ys.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return np.array(xs), np.array(ys)


def curve_pts(fn, n):
    """Sample a parametric curve fn(t)->(x, y) for t in [0,1], dedupe to pixel path."""
    t = np.linspace(0, 1, n)
    x, y = fn(t)
    xs, ys = np.round(x).astype(int), np.round(y).astype(int)
    keep = np.ones(len(xs), bool)
    keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
    return xs[keep], ys[keep]


def smooth01(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def ease_out(t, p=3):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** p


def ease_in(t, p=2):
    t = min(max(t, 0.0), 1.0)
    return t ** p


def lerp(a, b, t):
    return a + (b - a) * t


def phase(f, a, b):
    """0..1 progress of frame f through [a, b)."""
    return min(max((f - a) / max(b - a, 1e-6), 0.0), 1.0)

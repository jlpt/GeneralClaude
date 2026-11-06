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
    # sunset sky, zenith -> horizon
    "s0": "1a1030", "s1": "2a1848", "s2": "48225e", "s3": "6e2c66", "s4": "9a3a62", "s5": "c4505a",
    "s6": "e2704e", "s7": "f49a4c", "s8": "ffc45a", "s9": "ffe28a",
    "sun0": "fff4c8", "sun1": "ffd88a",
    "cl0": "5a2a5a", "cl1": "8e3e5e", "cl2": "c86a5e", "cl3": "f2a070",
    "mt0": "3a1e4a", "mt1": "52285a", "mt2": "2a1638", "mt3": "3c1e46",
    # ruins + floor
    "rs0": "1c1420", "rs1": "2e2230", "rs2": "463444", "rs3": "614a58", "rs4": "80646c", "rs5": "a48888",
    "rs6": "caa8a0",
    "fl0": "241a24", "fl1": "362834", "fl2": "4e3a46", "fl3": "6a525a", "fl4": "8a6c6c",
    "du0": "4a3a40", "du1": "7a6266", "du2": "a68a86", "du3": "d4bcb0",
    # eclipse
    "ec0": "06040c", "ec1": "140a24", "ec2": "28103e", "ec3": "3e1a5c",
    # shared
    "ol": "140c14", "white": "ffffff", "sp0": "ffe08a", "sp1": "fffbe4", "red": "ff3a3a", "gold": "ffd24a",
    "sk0": "5a2e28", "sk1": "b4705a", "sk2": "e6a07e", "sk3": "fcc8a2", "sk4": "ffe4ca",
    # Kael
    "kh0": "0e1a1a", "kh1": "1c3432", "kh2": "2c4e4a", "kh3": "44706a", "kb0": "1e6a3e", "kb1": "2e9a5a",
    "kt0": "1e2416", "kt1": "34402a", "kt2": "4e5c3c", "kt3": "6e7c54", "kp0": "16141a", "kp1": "2a2630",
    "kw0": "a89a86", "kw1": "e0d4c0", "kbo0": "3a2618", "kbo1": "5a3c24", "kamb": "e8a030",
    "gs0": "141418", "gs1": "2a2c34", "gs2": "444854", "gs3": "686e7c", "gs4": "9aa2b0",
    "gn0": "0a2a18", "gn1": "126a3a", "gn2": "1ea85a", "gn3": "3dff9a", "gn4": "b8ffd8",
    # Ronan
    "rh0": "4a4a58", "rh1": "7a7a8a", "rh2": "aeaebc", "rh3": "e2e2ea",
    "rc0": "10122e", "rc1": "1c2050", "rc2": "2c3478", "rc3": "4250a8", "rc4": "6a7ccc",
    "ru0": "a8a49c", "ru1": "d8d4cc",
    "bl0": "2a2c3a", "bl1": "8890a8", "bl2": "d4dcec",
    "in0": "06082a", "in1": "141a5e", "in2": "2a3aa8", "in3": "5a7cff", "in4": "b8d4ff",
    # Morvane
    "mh0": "0a080c", "mh1": "1e1a24", "mh2": "34303e",
    "mc0": "2a0610", "mc1": "4e0c1c", "mc2": "7a1428", "mc3": "a8203a", "mc4": "d24050",
    "ms0": "8a6a74", "ms1": "c8b0b4", "ms2": "f0e0e0", "mpink": "ff7aa8",
    "cr0": "3a0418", "cr1": "7a0c34", "cr2": "c81e5a", "cr3": "ff4c86", "cr4": "ffa8c8", "cr5": "fff0f6",
    # the Ashen Warden
    "wc0": "16141a", "wc1": "26232c", "wc2": "3a3642", "wc3": "524c5a", "wc4": "6e6878",
    "wm0": "6a6260", "wm1": "b8b0a8", "wm2": "e8e2d8", "wm3": "fffaf0",
    "wg0": "1e1c24", "wg1": "3a3846", "wg2": "5e5a6e",
    "vd0": "12041e", "vd1": "2e0c4e", "vd2": "5a1e9a", "vd3": "8a4cff", "vd4": "c8a4ff", "vd5": "f2e6ff",
}
C = {k: np.array(hexc(v), np.float32) for k, v in P.items()}
NAMES = list(P.keys())

LAYERS = ["Sky", "Arena", "Back FX", "Morvane", "Ronan", "Kael", "Warden", "FX", "Text"]
L = {n: i for i, n in enumerate(LAYERS)}


def _group(prefixes):
    return [n for n in NAMES if any(n.startswith(p) for p in prefixes)]


ENERGY = _group(["gn", "in", "cr", "vd", "sp", "white"])
SUBPAL = {
    "Sky": _group(["s", "cl", "mt", "ec", "vd", "white"]),
    "Arena": _group(["rs", "fl", "du", "mt", "ec", "ol"]) + ENERGY,
    "Back FX": NAMES,
    "Kael": _group(["ol", "sk", "kh", "kb", "kt", "kp", "kw", "kbo", "kamb", "gs", "red"]) + ENERGY,
    "Ronan": _group(["ol", "sk", "rh", "rc", "ru", "bl", "gold", "red"]) + ENERGY,
    "Morvane": _group(["ol", "mh", "mc", "ms", "mpink", "red"]) + ENERGY,
    "Warden": _group(["ol", "wc", "wm", "wg", "ec"]) + ENERGY,
    "FX": NAMES,
    "Text": NAMES,
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

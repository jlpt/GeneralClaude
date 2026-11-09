"""Canvas, palette and raster helpers shared by the mage rig, the backdrop and the effects.

Everything is drawn as boolean pixel masks on a canvas that is PAD pixels larger than the
sprite on every side, so the whole picture can be shifted for screen shake and cropped.
Coordinates are sprite-space: (0, 0) is the top-left pixel of the final frame.
"""
import math
import numpy as np

W, H = 176, 112
PAD = 8
HORIZON = 82          # first ground row
GROUND = 97           # row the mage's feet stand on
YS, XS = np.mgrid[-PAD:H + PAD, -PAD:W + PAD].astype(float)
XI, YI = XS.astype(int), YS.astype(int)
SHAPE = XS.shape

_B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
BAYER = _B4[YI % 4, XI % 4]
CHECK = (XI + YI) % 2 == 0


def dith(level):
    """Ordered-dither mask: True on roughly `level` (0..1, scalar or array) of the pixels."""
    return BAYER < level


def hexc(s, a=255):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)


# ---------------------------------------------------------------- palette
C = {
    # hat / cloak (indigo)
    "ind_o": hexc("120b2e"), "ind_d": hexc("201754"), "ind_b": hexc("31257c"),
    "ind_l": hexc("4737a6"), "ind_h": hexc("6d5ad2"),
    # gold trim
    "gold_o": hexc("5a3210"), "gold_d": hexc("9a6220"), "gold_b": hexc("dc9c30"),
    "gold_l": hexc("ffd45c"), "gold_h": hexc("fff4b8"),
    # cloak lining / tunic (teal)
    "teal_o": hexc("0a2c36"), "teal_d": hexc("10505a"), "teal_b": hexc("1d7f84"), "teal_l": hexc("42bcae"),
    # skin
    "skin_o": hexc("3e1e1c"), "skin_s": hexc("7c4632"), "skin_b": hexc("ab6c46"), "skin_l": hexc("d39564"),
    # hair (silver)
    "hair_o": hexc("3a4062"), "hair_s": hexc("8a94ba"), "hair_b": hexc("c6cee6"), "hair_l": hexc("f4f7ff"),
    # face details
    "eye_d": hexc("1c1232"), "eye_i": hexc("38c8d8"), "mouth": hexc("6a3028"),
    # trousers / boots
    "leg_d": hexc("181226"), "leg_b": hexc("2a2040"), "boot_d": hexc("3a2218"), "boot_b": hexc("6a4028"),
    "boot_l": hexc("96643c"),
    # staff
    "wood_o": hexc("1e100e"), "wood_d": hexc("4c2c20"), "wood_b": hexc("7e4e32"), "wood_l": hexc("ae7648"),
    # fx: white / cyan
    "fx_w": hexc("ffffff"), "cy_1": hexc("d4fbff"), "cy_2": hexc("78f0ff"), "cy_3": hexc("2cbcf0"),
    "cy_4": hexc("1e74d8"), "cy_5": hexc("2a3ea8"),
    # fx: gold / ember
    "au_1": hexc("fff4b8"), "au_2": hexc("ffd45c"), "au_3": hexc("ff9a3c"), "au_4": hexc("e0502c"),
    # fx: violet
    "vi_1": hexc("f2ccff"), "vi_2": hexc("c488ff"), "vi_3": hexc("844ae4"), "vi_4": hexc("4a2aa2"),
    # smoke / dust
    "sm_1": hexc("a8a2d0"), "sm_2": hexc("6e68a4"), "sm_3": hexc("423e74"),
    # misc
    "shadow": hexc("0e0c22"), "sil": hexc("0a0820"),
}

LAYERS = ["Background", "FX (back)", "Braid + lining", "Back arm", "Body", "Head + hat", "Staff",
          "Front arm", "FX (front)"]
CHAR_LAYERS = LAYERS[2:8]


class Scene:
    """One frame: a stack of real RGBA layers (every pixel is fully opaque or empty)."""

    def __init__(self):
        self.img = {n: np.zeros(SHAPE + (4,), np.uint8) for n in LAYERS}

    def put(self, mask, color, layer):
        self.img[layer][mask] = C[color] if isinstance(color, str) else color

    def erase(self, mask, layer):
        self.img[layer][mask] = 0

    def alpha(self, names):
        m = np.zeros(SHAPE, bool)
        for n in names:
            m |= self.img[n][:, :, 3] > 0
        return m

    def _crop(self, img, shake):
        sx, sy = shake
        return img[PAD + sy:PAD + sy + H, PAD + sx:PAD + sx + W].copy()

    def layers(self, shake=(0, 0)):
        return [self._crop(self.img[n], shake) for n in LAYERS]

    def flat(self, shake=(0, 0)):
        out = np.zeros(SHAPE + (4,), np.uint8)
        for n in LAYERS:
            m = self.img[n][:, :, 3] > 0
            out[m] = self.img[n][m]
        return self._crop(out, shake)


# ---------------------------------------------------------------- mask helpers
def nb(mask, dx, dy):
    """out[y, x] = mask[y + dy, x + dx] (False outside)."""
    out = np.zeros_like(mask)
    hs, ws = mask.shape
    ys_src = slice(max(dy, 0), hs + min(dy, 0))
    ys_dst = slice(max(-dy, 0), hs + min(-dy, 0))
    xs_src = slice(max(dx, 0), ws + min(dx, 0))
    xs_dst = slice(max(-dx, 0), ws + min(-dx, 0))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def dilate4(mask):
    return mask | nb(mask, 1, 0) | nb(mask, -1, 0) | nb(mask, 0, 1) | nb(mask, 0, -1)


def dilate8(mask):
    return dilate4(mask) | nb(mask, 1, 1) | nb(mask, -1, 1) | nb(mask, 1, -1) | nb(mask, -1, -1)


def erode4(mask):
    return mask & nb(mask, 1, 0) & nb(mask, -1, 0) & nb(mask, 0, 1) & nb(mask, 0, -1)


def v2(a):
    return np.array(a, float)


def seg_dist(p0, p1):
    p0, p1 = v2(p0), v2(p1)
    d = p1 - p0
    ll = max(d @ d, 1e-9)
    t = np.clip(((XS - p0[0]) * d[0] + (YS - p0[1]) * d[1]) / ll, 0, 1)
    return np.hypot(XS - (p0[0] + t * d[0]), YS - (p0[1] + t * d[1])), t


def capsule(p0, p1, r0, r1=None):
    r1 = r0 if r1 is None else r1
    dist, t = seg_dist(p0, p1)
    return dist <= r0 + (r1 - r0) * t


def disc(c, r):
    return np.hypot(XS - c[0], YS - c[1]) <= r


def ellipse(c, rx, ry):
    if rx <= 0 or ry <= 0:
        return np.zeros(SHAPE, bool)
    return ((XS - c[0]) / rx) ** 2 + ((YS - c[1]) / ry) ** 2 <= 1.0


def ering(c, rx, ry, th=1.0):
    """Elliptical ring, `th` pixels thick, outer radii rx / ry."""
    return ellipse(c, rx, ry) & ~ellipse(c, rx - th, ry - th)


def polygon(pts):
    pts = [v2(p) for p in pts]
    inside = np.zeros(SHAPE, bool)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if a[1] == b[1]:
            continue
        cond = (a[1] > YS) != (b[1] > YS)
        xint = (b[0] - a[0]) * (YS - a[1]) / (b[1] - a[1]) + a[0]
        inside ^= cond & (XS < xint)
    return inside


def pixels(points):
    """Mask with the given (x, y) pixels set (points off the padded canvas are dropped)."""
    m = np.zeros(SHAPE, bool)
    for x, y in points:
        xi, yi = int(round(x)) + PAD, int(round(y)) + PAD
        if 0 <= xi < SHAPE[1] and 0 <= yi < SHAPE[0]:
            m[yi, xi] = True
    return m


def line_px(p0, p1):
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    pts = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return pixels(pts)


def polyline(pts):
    m = np.zeros(SHAPE, bool)
    for a, b in zip(pts[:-1], pts[1:]):
        m |= line_px(a, b)
    return m


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def unit(deg):
    """Unit vector for a screen-space angle: 0 = right, 90 = up."""
    r = math.radians(deg)
    return v2((math.cos(r), -math.sin(r)))

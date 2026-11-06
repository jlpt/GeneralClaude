"""Pixel canvas + raster primitives in coordinate space.

Generalised from eris-sword-slash/src/rig.py: instead of one global 112x80 grid, every Canvas carries its own
coordinate grids (pixel (i, j) sits at (x0 + j, y0 + i)), and primitives are evaluated only inside their
bounding box, so a character can be drawn in local "root on the ground" coordinates and composited anywhere.
"""
import math
import numpy as np


def hexc(s, a=255):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)


def v2(a):
    return np.array(a, float)


def rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return v2((v[0] * c - v[1] * s, v[0] * s + v[1] * c))


def unit(deg):
    a = math.radians(deg)
    return v2((math.cos(a), math.sin(a)))


def norm(v):
    v = v2(v)
    return v / max(float(np.hypot(*v)), 1e-9)


def ang_of(v):
    return math.degrees(math.atan2(v[1], v[0]))


def ik2(a, t, l1, l2, bend):
    a, t = v2(a), v2(t)
    d = t - a
    dist = float(np.hypot(*d))
    dirn = d / max(dist, 1e-6)
    dist = min(max(dist, abs(l1 - l2) + 0.05), l1 + l2 - 0.05)
    cosang = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
    ang = math.degrees(math.acos(max(-1, min(1, cosang))))
    elbow = a + rot(dirn, bend * ang) * l1
    end = a + dirn * dist
    return elbow, end


# ---------------------------------------------------------------- mask ops (shape agnostic)
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
    m = dilate4(mask)
    return m | nb(mask, 1, 1) | nb(mask, -1, 1) | nb(mask, 1, -1) | nb(mask, -1, -1)


def ring4(mask):
    return dilate4(mask) & ~mask


class Canvas:
    def __init__(self, w, h, x0=0, y0=0):
        self.w, self.h, self.x0, self.y0 = w, h, x0, y0
        ys, xs = np.mgrid[0:h, 0:w]
        self.XS = (xs + x0).astype(np.float32)
        self.YS = (ys + y0).astype(np.float32)
        self.clear()

    def clear(self):
        self.rgba = np.zeros((self.h, self.w, 4), np.uint8)
        self.owner = np.full((self.h, self.w), -1, np.int16)

    def empty(self):
        return np.zeros((self.h, self.w), bool)

    def put(self, mask, color, owner=0):
        self.rgba[mask] = color
        self.owner[mask] = owner

    # ---- bounding boxes
    def _box(self, xmin, xmax, ymin, ymax):
        j0 = max(int(math.floor(xmin)) - self.x0, 0)
        j1 = min(int(math.ceil(xmax)) - self.x0 + 1, self.w)
        i0 = max(int(math.floor(ymin)) - self.y0, 0)
        i1 = min(int(math.ceil(ymax)) - self.y0 + 1, self.h)
        if j0 >= j1 or i0 >= i1:
            return None
        return slice(i0, i1), slice(j0, j1)

    def _fill(self, box, fn):
        m = self.empty()
        if box is None:
            return m
        sy, sx = box
        m[sy, sx] = fn(self.XS[sy, sx], self.YS[sy, sx])
        return m

    # ---- primitives
    def seg_t(self, p0, p1, X=None, Y=None):
        p0, p1 = v2(p0), v2(p1)
        X = self.XS if X is None else X
        Y = self.YS if Y is None else Y
        d = p1 - p0
        ll = max(d @ d, 1e-9)
        t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / ll, 0, 1)
        return np.hypot(X - (p0[0] + t * d[0]), Y - (p0[1] + t * d[1])), t

    def capsule(self, p0, p1, r0, r1=None):
        r1 = r0 if r1 is None else r1
        p0, p1 = v2(p0), v2(p1)
        rm = max(r0, r1) + 1
        box = self._box(min(p0[0], p1[0]) - rm, max(p0[0], p1[0]) + rm,
                        min(p0[1], p1[1]) - rm, max(p0[1], p1[1]) + rm)

        def fn(X, Y):
            dist, t = self.seg_t(p0, p1, X, Y)
            return dist <= r0 + (r1 - r0) * t
        return self._fill(box, fn)

    def disc(self, c, r):
        box = self._box(c[0] - r - 1, c[0] + r + 1, c[1] - r - 1, c[1] + r + 1)
        return self._fill(box, lambda X, Y: np.hypot(X - c[0], Y - c[1]) <= r)

    def ellipse(self, c, rx, ry):
        box = self._box(c[0] - rx - 1, c[0] + rx + 1, c[1] - ry - 1, c[1] + ry + 1)
        return self._fill(box, lambda X, Y: ((X - c[0]) / rx) ** 2 + ((Y - c[1]) / ry) ** 2 <= 1.0)

    def ring(self, c, r_out, r_in, ry_scale=1.0):
        rx = r_out
        ry = r_out * ry_scale
        box = self._box(c[0] - rx - 1, c[0] + rx + 1, c[1] - ry - 1, c[1] + ry + 1)

        def fn(X, Y):
            d = np.hypot(X - c[0], (Y - c[1]) / ry_scale)
            return (d <= r_out) & (d > r_in)
        return self._fill(box, fn)

    def polygon(self, pts):
        pts = [v2(p) for p in pts]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = self._box(min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1)

        def fn(X, Y):
            inside = np.zeros(X.shape, bool)
            n = len(pts)
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                cond = (a[1] > Y) != (b[1] > Y)
                xint = (b[0] - a[0]) * (Y - a[1]) / ((b[1] - a[1]) + 1e-12) + a[0]
                inside ^= cond & (X < xint)
            return inside
        return self._fill(box, fn)

    def line(self, p0, p1):
        """1px Bresenham line as a mask."""
        m = self.empty()
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        for _ in range(4000):
            j, i = x0 - self.x0, y0 - self.y0
            if 0 <= j < self.w and 0 <= i < self.h:
                m[i, j] = True
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return m

    def point(self, p):
        m = self.empty()
        j, i = int(round(p[0])) - self.x0, int(round(p[1])) - self.y0
        if 0 <= j < self.w and 0 <= i < self.h:
            m[i, j] = True
        return m

    def checker(self, phase=0):
        return ((self.XS + self.YS + phase) % 2) == 0

    def bayer(self, level):
        """Ordered-dither mask covering roughly `level` (0..1) of the pixels."""
        B = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], float) / 16.0
        bx = (self.XS.astype(int) % 4)
        by = (self.YS.astype(int) % 4)
        return B[by, bx] < level


# ---------------------------------------------------------------- shading
def paint_shaded(cv, mask, ramp, owner=0, outline=True, rim=True, shadow=1, outline_col=None, light_rim=None,
                 inner_col=None):
    """ramp = (outline, dark, base, light). Light comes from the upper left.
    inner_col: colour for the part's outline where it overlaps the same owner (softer clump lines)."""
    o, d, b, l = ramp
    if outline:
        ring = ring4(mask)
        if inner_col is not None:
            over_same = ring & (cv.owner == owner) & np.any(cv.rgba != np.array(o, np.uint8), axis=2)
        cv.put(ring, outline_col or o, owner)
        if inner_col is not None:
            cv.put(over_same, inner_col, owner)
    cv.put(mask, b, owner)
    if shadow:
        sh = mask & ~nb(mask, 1, 1)
        if shadow > 1:
            sh |= mask & ~nb(mask, 2, 1)
        cv.put(sh, d, owner)
    if rim:
        lt = mask & ~(nb(mask, -1, 0) & nb(mask, 0, -1)) & nb(mask, 1, 1)
        cv.put(lt, light_rim or l, owner)


# ---------------------------------------------------------------- sprites
def rot_offset(dx, dy, k):
    """Rotate a screen-space offset by k quarter turns counter-clockwise (y points down)."""
    for _ in range(k % 4):
        dx, dy = dy, -dx
    return dx, dy


def blit_sprite(cv, rows, key_colors, anchor_xy, at, k=0, owner=0, flip=False):
    """Draw an ASCII sprite so that its `anchor_xy` pixel lands on `at`, rotated k quarter turns CCW.
    key_colors maps a character to an RGBA tuple (missing / None = transparent)."""
    ax, ay = anchor_xy
    bx, by = int(round(at[0])), int(round(at[1]))
    wspr = len(rows[0])
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            col = key_colors.get(ch)
            if col is None:
                continue
            sx = (wspr - 1 - x) if flip else x
            dx, dy = rot_offset(sx - ax, y - ay, k)
            j, i = bx + dx - cv.x0, by + dy - cv.y0
            if 0 <= j < cv.w and 0 <= i < cv.h:
                cv.rgba[i, j] = col
                cv.owner[i, j] = owner


def sprite_point(anchor_xy, at, p, k=0):
    """Where sprite pixel p lands (same transform as blit_sprite)."""
    dx, dy = rot_offset(p[0] - anchor_xy[0], p[1] - anchor_xy[1], k)
    return v2((round(at[0]) + dx, round(at[1]) + dy))

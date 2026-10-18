"""Procedural pixel-art rig for Eris (3/4 view, facing right).

Every frame is drawn from a pose description: body parts are rasterised as
pixel masks, shaded with hand-picked colour ramps, outlined, and assigned to
named layers so the result stays editable in Aseprite.
"""
import math
import numpy as np

W, H = 112, 80
GROUND = 76  # last row the feet touch

YS, XS = np.mgrid[0:H, 0:W].astype(float)


def hexc(s, a=255):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), a)


# ---------------------------------------------------------------- palette
C = {
    # hair
    "hair_o": hexc("3b0d16"), "hair_d": hexc("6b1824"), "hair_s": hexc("952a2b"),
    "hair_b": hexc("c23c31"), "hair_l": hexc("e2623f"), "hair_h": hexc("f89a68"),
    # skin
    "skin_o": hexc("7e3a36"), "skin_s": hexc("e0a084"), "skin_b": hexc("f8d0b4"), "skin_l": hexc("fff0e0"),
    # eyes / mouth
    "eye_d": hexc("3a0812"), "eye_r": hexc("d8343a"), "mouth": hexc("a84a4c"),
    # jacket
    "jak_o": hexc("100a10"), "jak_d": hexc("1e161c"), "jak_b": hexc("2e242a"),
    "jak_l": hexc("483b42"), "jak_h": hexc("66565e"), "stripe": hexc("d6ccc8"), "stripe_s": hexc("9a8c8c"),
    # fur collar
    "fur_d": hexc("6e5850"), "fur_b": hexc("a98a76"), "fur_l": hexc("d9c2a8"), "fur_h": hexc("f0e2cc"),
    # crop top
    "top_s": hexc("b8a69c"), "top_b": hexc("e8dace"), "top_l": hexc("faf2ea"), "bow": hexc("2a1c1e"),
    # shorts / belt
    "sho_d": hexc("1a1418"), "sho_b": hexc("2c2428"), "sho_l": hexc("40363a"),
    "belt": hexc("6c4632"), "belt_l": hexc("90643f"), "buckle": hexc("c8ccd4"),
    # legs
    "leg_o": hexc("1a110e"), "leg_d": hexc("362820"), "leg_b": hexc("4f3c33"), "leg_l": hexc("6c5446"),
    "sock": hexc("1e1a1e"), "sock_l": hexc("383038"),
    "shoe_b": hexc("1c181c"), "shoe_l": hexc("3c3440"),
    # sword
    "bl_o": hexc("262a36"), "bl_d": hexc("6a7688"), "bl_b": hexc("aab6c6"), "bl_l": hexc("e2eaf2"), "bl_h": hexc("ffffff"),
    "guard": hexc("c4c8d2"), "guard_d": hexc("7c8090"), "grip": hexc("3a2a28"), "grip_l": hexc("6c4c3c"),
    "scab": hexc("2e2420"), "scab_l": hexc("4a3c34"), "scab_tip": hexc("a8acb6"),
    # fx
    "fx_w": hexc("ffffff"), "fx_1": hexc("e6f6ff"), "fx_2": hexc("a8dcff"), "fx_3": hexc("ffb070"), "fx_4": hexc("ff6a3c"),
    "dust_l": hexc("d8cfc4"), "dust_d": hexc("a0958a"),
}

LAYERS = ["Hair (back)", "Back arm", "Body", "Head", "Sword", "Front arm", "FX"]
L = {n: i for i, n in enumerate(LAYERS)}


# ---------------------------------------------------------------- canvas
class Scene:
    def __init__(self):
        self.rgba = np.zeros((H, W, 4), np.uint8)
        self.owner = np.full((H, W), -1, np.int16)
        self.solid = np.zeros((H, W), bool)   # pixels that get an outer outline

    def put(self, mask, color, layer, solid=True):
        self.rgba[mask] = color
        self.owner[mask] = L[layer] if isinstance(layer, str) else layer
        if solid:
            self.solid |= mask
        else:
            self.solid &= ~mask

    def layers(self):
        out = []
        for i in range(len(LAYERS)):
            img = np.zeros((H, W, 4), np.uint8)
            m = self.owner == i
            img[m] = self.rgba[m]
            out.append(img)
        return out

    def flat(self):
        return self.rgba.copy()


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


# ---------------------------------------------------------------- geometry
def v2(a):
    return np.array(a, float)


def seg_dist(p0, p1):
    p0, p1 = v2(p0), v2(p1)
    d = p1 - p0
    ll = max(d @ d, 1e-9)
    t = np.clip(((XS - p0[0]) * d[0] + (YS - p0[1]) * d[1]) / ll, 0, 1)
    px = p0[0] + t * d[0]
    py = p0[1] + t * d[1]
    return np.hypot(XS - px, YS - py), t


def capsule(p0, p1, r0, r1=None):
    r1 = r0 if r1 is None else r1
    dist, t = seg_dist(p0, p1)
    return dist <= r0 + (r1 - r0) * t


def disc(c, r):
    return np.hypot(XS - c[0], YS - c[1]) <= r


def ellipse(c, rx, ry):
    return ((XS - c[0]) / rx) ** 2 + ((YS - c[1]) / ry) ** 2 <= 1.0


def polygon(pts):
    pts = [v2(p) for p in pts]
    inside = np.zeros((H, W), bool)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        cond = (a[1] > YS) != (b[1] > YS)
        xint = (b[0] - a[0]) * (YS - a[1]) / ((b[1] - a[1]) + 1e-12) + a[0]
        inside ^= cond & (XS < xint)
    return inside


def line_px(p0, p1):
    """1px line (Bresenham) as mask."""
    m = np.zeros((H, W), bool)
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            m[y0, x0] = True
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


def rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return v2((v[0] * c - v[1] * s, v[0] * s + v[1] * c))


def unit(deg):
    a = math.radians(deg)
    return v2((math.cos(a), math.sin(a)))


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


# ---------------------------------------------------------------- shading
def paint_shaded(sc, mask, ramp, layer, outline=True, rim=True, shadow=1, outline_col=None, light_rim=None,
                 inner_col=None):
    """ramp = (outline, dark, base, light). Light comes from the upper left.
    inner_col: colour for the part's outline where it overlaps the same layer (softer clump lines)."""
    o, d, b, l = ramp
    if outline:
        ring = dilate4(mask) & ~mask
        over_same = ring & (sc.owner == L[layer]) & np.any(sc.rgba != np.array(o, np.uint8), axis=2)
        sc.put(ring, outline_col or o, layer)
        if inner_col is not None:
            sc.put(over_same, inner_col, layer)
    sc.put(mask, b, layer)
    if shadow:
        sh = mask & ~nb(mask, 1, 1)
        if shadow > 1:
            sh |= mask & ~nb(mask, 2, 1)
        sc.put(sh, d, layer)
    if rim:
        lt = mask & ~(nb(mask, -1, 0) & nb(mask, 0, -1)) & nb(mask, 1, 1)
        sc.put(lt, light_rim or l, layer)


# ---------------------------------------------------------------- head sprites
HEAD_KEY = {
    ".": None, "o": "hair_o", "d": "hair_d", "r": "hair_s", "R": "hair_b", "H": "hair_l", "h": "hair_h",
    "k": "skin_o", "s": "skin_s", "S": "skin_b", "L": "skin_l",
    "e": "eye_d", "E": "eye_r", "m": "mouth",
}

from faces import HEADS  # noqa: E402
HEAD_ANCHOR = (8, 19)  # sprite pixel that sits on the neck base


def draw_head(sc, neck, variant="smirk", offset=(0, 0)):
    rows = HEADS[variant]
    ax, ay = HEAD_ANCHOR
    ox = int(round(neck[0] + offset[0])) - ax
    oy = int(round(neck[1] + offset[1])) - ay
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            key = HEAD_KEY.get(ch)
            if key is None:
                continue
            px, py = ox + x, oy + y
            if 0 <= px < W and 0 <= py < H:
                m = np.zeros((H, W), bool)
                m[py, px] = True
                sc.put(m, C[key], "Head")
    return v2((ox, oy))


# ---------------------------------------------------------------- body parts
def interp(v, vs, xs):
    return np.interp(v, vs, xs)


def torso_frame(P, N):
    P, N = v2(P), v2(N)
    up = (N - P) / np.hypot(*(N - P))
    right = v2((-up[1], up[0]))
    return P, N, up, right, float(np.hypot(*(N - P)))


def draw_neck(sc, N, head_off):
    top = v2(N) + v2(head_off) + v2((0.5, -3))
    m = capsule(v2(N) + v2((0.0, 1)), top, 1.1)
    sc.put(dilate4(m) & ~m, C["skin_o"], "Body")
    sc.put(m, C["skin_b"], "Body")
    sc.put(m & (XS > top[0] - 0.2), C["skin_s"], "Body")


def draw_jacket_tail(sc, P, N, wind=(0.0, 0.0)):
    P, N, up, right, Lt = torso_frame(P, N)
    wx, wy = wind

    def at(u, v):
        return P + right * u + up * (v * Lt)

    pts = [at(-3.0, 0.45), at(-5.2, 0.25), at(-6.5 + wx * 2, -0.25 + wy * 0.2),
           at(-7.5 + wx * 3, -0.55 + wy * 0.4), at(-4.0 + wx * 1.5, -0.42 + wy * 0.2), at(-2.0, -0.1)]
    m = polygon(pts)
    paint_shaded(sc, m, (C["jak_o"], C["jak_d"], C["jak_b"], C["jak_l"]), "Body", shadow=1)


def draw_torso(sc, P, N, twist=0.0, breath=0):
    P, N, up, right, Lt = torso_frame(P, N)
    dx, dy = XS - P[0], YS - P[1]
    v = (dx * up[0] + dy * up[1]) / Lt
    u = dx * right[0] + dy * right[1]
    vs = [-0.32, -0.12, 0.12, 0.32, 0.52, 0.70, 0.86, 1.0, 1.08]
    left = interp(v, vs, [-4.6, -5.0, -4.6, -4.0, -4.4, -5.2, -6.2, -6.0, -3.5])
    right_e = interp(v, vs, [4.0, 4.6, 4.2, 3.6, 4.6, 5.6, 5.0, 4.6, 2.5])
    body = (v >= -0.32) & (v <= 1.08) & (u >= left) & (u <= right_e)
    body |= (v >= 0.80) & (v <= 1.12) & (u >= left - 0.8) & (u <= right_e + 0.3)

    m = 1.2 + twist          # front midline (sternum) in u
    jn = m - 2.6             # inner edge of the near jacket panel
    jf = m + 3.2             # inner edge of the far jacket panel
    shorts = body & (v < 0.10)
    belt = body & (v >= 0.10) & (v < 0.18)
    skin = body & (v >= 0.18) & (v < 0.42)
    top = body & (v >= 0.42) & (v < 0.86)
    collar = (v >= 0.80) & (v <= 1.12) & (u >= left - 0.8) & (u <= right_e + 0.3)
    jacket = body & (v >= 0.06) & ((u < jn) | (u > jf)) & ~collar
    # the near panel hangs a bit lower than the shorts at the side
    sc.put(dilate4(body) & ~body, C["jak_o"], "Body")
    sc.put(shorts, C["sho_b"], "Body")
    sc.put(shorts & (u > m + 1.5), C["sho_d"], "Body")
    sc.put(shorts & (np.abs(u - (m - 0.4)) < 0.5) & (v < 0.0), C["sho_d"], "Body")   # crotch seam
    sc.put(belt, C["belt"], "Body")
    sc.put(belt & (v >= 0.14), C["belt_l"], "Body")
    sc.put(belt & (np.abs(u - (m - 0.5)) < 1.0), C["buckle"], "Body")
    sc.put(skin, C["skin_b"], "Body")
    sc.put(skin & (u > m + 1.8), C["skin_s"], "Body")
    sc.put(skin & (v >= 0.36), C["skin_s"], "Body")                          # shadow under top
    sc.put(skin & (np.abs(u - (m - 0.3)) < 0.5) & (np.abs(v - 0.26) < 0.04), C["skin_s"], "Body")  # navel
    sc.put(top, C["top_b"], "Body")
    sc.put(top & (u < m - 0.5) & (v > 0.56), C["top_l"], "Body")
    sc.put(top & (u > m + 1.5), C["top_s"], "Body")
    sc.put(top & (v < 0.49), C["top_s"], "Body")                             # under-bust shadow
    sc.put(top & (np.abs(u - (m + 0.2)) < 0.5) & (v > 0.5), C["top_s"], "Body")  # cleavage/zip line
    sc.put(jacket, C["jak_b"], "Body")
    sc.put(jacket & (u < left + 1.2), C["jak_l"], "Body")
    sc.put(jacket & (u > jn - 1.0) & (u < jn), C["jak_d"], "Body")
    sc.put(jacket & (u > jf), C["jak_d"], "Body")
    # fur collar with a scalloped lower edge
    scallop = (np.floor(u) % 2 == 0)
    fur = collar | ((v >= 0.74) & (v < 0.80) & scallop & (u > left) & (u < right_e))
    sc.put(fur, C["fur_l"], "Body")
    sc.put(fur & ~nb(fur, 0, 1), C["fur_b"], "Body")
    sc.put(fur & (u > m + 2.5), C["fur_b"], "Body")
    sc.put(fur & ~nb(fur, 0, -1) & (u < m), C["fur_h"], "Body")
    sc.put(fur & (u > m + 3.5) & ~nb(fur, 0, 1), C["fur_d"], "Body")
    # bow at the throat
    bow = body & (np.abs(u - (m + 0.3)) < 1.6) & (np.abs(v - 0.84) < 0.05)
    sc.put(bow, C["bow"], "Body")


def draw_leg(sc, hip, ankle, bend, layer="Body", dark=False, inner=1):
    knee, ank = ik2(hip, ankle, 13.5, 13.0, bend)
    thigh = capsule(hip, knee, 2.4, 1.7)
    shin = capsule(knee, ank, 1.75, 1.15)
    leg = thigh | shin
    toe_dir = 1.0
    shoe = capsule(ank + v2((-0.4, 1.0)), ank + v2((3.6 * toe_dir, 1.6)), 1.1, 1.0)
    whole = leg | shoe
    sc.put(dilate4(whole) & ~whole, C["leg_o"], layer)
    ramp_d, ramp_b, ramp_l = (C["leg_d"], C["leg_d"], C["leg_b"]) if dark else (C["leg_d"], C["leg_b"], C["leg_l"])
    sc.put(leg, ramp_b, layer)
    sc.put(leg & ~nb(leg, 1, 0), ramp_d, layer)
    sc.put(leg & ~nb(leg, -1, 0) & nb(leg, 1, 0), ramp_l, layer)
    # black thigh-high showing above the chaps
    _, t = seg_dist(hip, knee)
    tdir = (knee - hip) / max(np.hypot(*(knee - hip)), 1e-6)
    side = ((XS - hip[0]) * -tdir[1] + (YS - hip[1]) * tdir[0]) * inner
    sock = thigh & (((t < 0.62) & (side < -0.2)) | (t < 0.22))
    sc.put(sock, C["sock"], layer)
    sc.put(sock & ~nb(sock, -1, 0), C["sock_l"], layer)
    _, t2 = seg_dist(knee, ank)
    sc.put(shin & (t2 > 0.86), C["sock"], layer)      # ankle wrap
    sc.put(shoe & ~leg, C["shoe_b"], layer)
    sc.put(shoe & ~leg & ~nb(shoe, 0, -1), C["shoe_l"], layer)
    return knee, ank


def solve_arm(shoulder, hand_c, bend, l1=8.5, l2=8.0):
    """IK that places the centre of the hand (not the wrist) on hand_c."""
    hand_c = v2(hand_c)
    fdir = (hand_c - v2(shoulder)) / max(np.hypot(*(hand_c - v2(shoulder))), 1e-6)
    for _ in range(4):
        elbow, wrist = ik2(shoulder, hand_c - fdir * 1.8, l1, l2, bend)
        fdir = (wrist - elbow) / max(np.hypot(*(wrist - elbow)), 1e-6)
    return elbow, wrist, wrist + fdir * 1.8


def draw_arm(sc, shoulder, hand_c, bend, layer, hand=True, dark=False):
    elbow, wrist, hand_c = solve_arm(shoulder, hand_c, bend)
    sleeve = capsule(shoulder, elbow, 2.1, 1.8) | capsule(elbow, wrist, 1.8, 2.1)
    hm = disc(hand_c, 1.5) & ~sleeve
    base = C["jak_d"] if dark else C["jak_l"]
    if hand:
        sc.put(dilate4(hm) & ~hm, C["skin_o"], layer)
    sc.put(dilate4(sleeve) & ~sleeve, C["jak_o"], layer)
    sc.put(sleeve, base, layer)
    if not dark:
        sc.put(sleeve & ~nb(sleeve, 1, 1), C["jak_b"], layer)
        lt = sleeve & ~(nb(sleeve, -1, 0) & nb(sleeve, 0, -1)) & nb(sleeve, 1, 1)
        sc.put(lt, C["jak_h"], layer)
        _, t = seg_dist(elbow, wrist)
        fore = capsule(elbow, wrist, 1.8, 2.1)
        sc.put(fore & (np.abs(t - 0.8) < 0.07), C["stripe"], layer)
    if hand:
        sc.put(hm, C["skin_b"], layer)
        sc.put(hm & ~nb(hm, 1, 1), C["skin_s"], layer)
    return elbow, wrist, hand_c


def draw_blade(sc, guard_c, dirv, length, layer):
    perp = v2((-dirv[1], dirv[0]))
    b0 = guard_c + dirv * 1.0
    tip = guard_c + dirv * length
    blade = capsule(b0, tip, 1.05, 0.55)
    sc.put(dilate4(blade) & ~blade, C["bl_o"], layer)
    side = (XS - b0[0]) * perp[0] + (YS - b0[1]) * perp[1]
    # choose the side facing the upper-left light as the bright edge
    lit_sign = -1 if (perp[0] + perp[1]) > 0 else 1
    sc.put(blade, C["bl_b"], layer)
    sc.put(blade & (side * lit_sign > 0), C["bl_l"], layer)
    _, t = seg_dist(b0, tip)
    sc.put(blade & (t > 0.9), C["bl_h"], layer)
    return tip


def draw_hilt(sc, guard_c, dirv, layer, grip_len=4.0):
    perp = v2((-dirv[1], dirv[0]))
    pommel = guard_c - dirv * (grip_len + 1.0)
    grip = capsule(guard_c - dirv * 0.5, pommel, 0.75)
    guard = capsule(guard_c - perp * 2.4, guard_c + perp * 2.4, 0.7)
    pm = disc(pommel, 1.0)
    allm = grip | guard | pm
    sc.put(dilate4(allm) & ~allm, C["bl_o"], layer)
    sc.put(grip, C["grip"], layer)
    _, t = seg_dist(guard_c, pommel)
    sc.put(grip & (np.floor(t * grip_len * 1.4) % 2 == 0), C["grip_l"], layer)
    sc.put(guard | pm, C["guard"], layer)
    sc.put((guard | pm) & ~nb(guard | pm, 1, 1), C["guard_d"], layer)


SWORD_LEN = 20


def sword_geom(grip_pt, ang):
    dirv = unit(ang)
    guard_c = v2(grip_pt) + dirv * 2.2
    return guard_c, guard_c + dirv * SWORD_LEN


def draw_sword(sc, grip_pt, ang, layer="Sword"):
    dirv = unit(ang)
    guard_c, tip = sword_geom(grip_pt, ang)
    draw_hilt(sc, guard_c, dirv, layer)
    draw_blade(sc, guard_c, dirv, SWORD_LEN, layer)
    return guard_c, tip


def draw_scabbard(sc, mouth, ang, sheathed, layer="Body"):
    dirv = unit(ang)
    mouth = v2(mouth)
    end = mouth + dirv * 21
    m = capsule(mouth, end, 1.2, 0.9)
    sc.put(dilate4(m) & ~m, C["jak_o"], layer)
    sc.put(m, C["scab"], layer)
    sc.put(m & ~(nb(m, -1, 0) & nb(m, 0, -1)), C["scab_l"], layer)
    _, t = seg_dist(mouth, end)
    sc.put(m & ((t > 0.94) | (t < 0.06)), C["scab_tip"], layer)
    if sheathed:
        draw_hilt(sc, mouth - dirv * 0.5, -dirv, layer)


# ---------------------------------------------------------------- hair
BACK_STRANDS = [
    # root (sprite px), rest angle, length, root radius, tone (0 dark .. 2 light), wave phase, curl
    ((13.0, 9.0), 64, 26, 2.6, 0, 0.9, 1.4),
    ((12.0, 7.0), 76, 30, 3.2, 0, 1.7, 0.8),
    ((13.0, 11.0), 70, 26, 2.8, 0, 2.6, 0.6),
    ((8.0, 4.0), 90, 30, 3.0, 0, 3.1, 0.0),
    ((3.0, 5.0), 118, 31, 3.4, 0, 4.0, -1.2),
    ((5.0, 4.0), 104, 33, 3.8, 1, 0.0, -0.4),
    ((2.5, 7.0), 112, 34, 4.2, 1, 1.3, -0.6),
    ((2.0, 11.0), 100, 33, 3.6, 2, 2.1, -0.5),
    ((6.0, 7.0), 93, 31, 3.0, 2, 0.7, -0.2),
]
FRONT_LOCK = ((4.0, 12.0), 92, 13, 1.5, 2, 0.4, 0.3)


def strand_mask(origin, spec, wind, phase):
    (rx, ry), a0, length, r0, _tone, ph, curl = spec
    pos = v2(origin) + v2((rx, ry))
    wind = v2(wind)
    n = 28
    seg = length / n
    m = np.zeros((H, W), bool)
    upper = np.zeros((H, W), bool)
    for i in range(n + 1):
        t = i / n
        r = r0 * (1.0 - 0.8 * t) * (0.85 + 0.35 * math.sin(math.pi * min(1.0, t * 1.3)))
        dm = disc(pos, max(r, 0.55))
        m |= dm
        if 0.12 < t < 0.42:
            upper |= dm
        rest = unit(a0 + curl * 10 * t)
        d = rest + wind * (t ** 1.1)
        d = d / max(np.hypot(*d), 1e-6)
        perp = v2((-d[1], d[0]))
        wave = math.sin(t * 8.0 + phase + ph) * 2.8 * min(1.0, t * 1.6) * (8.0 / n)
        pos = pos + d * seg + perp * wave
    return m, upper


def draw_hair_back(sc, head_origin, wind, phase):
    tones = [
        (C["hair_o"], C["hair_d"], C["hair_d"], C["hair_s"]),
        (C["hair_o"], C["hair_d"], C["hair_s"], C["hair_b"]),
        (C["hair_o"], C["hair_s"], C["hair_b"], C["hair_l"]),
    ]
    for spec in BACK_STRANDS:
        m, upper = strand_mask(head_origin, spec, wind, phase)
        paint_shaded(sc, m, tones[spec[4]], "Hair (back)", inner_col=C["hair_d"] if spec[4] else C["hair_o"])
        if spec[4] == 2:
            shine = upper & m & nb(m, -2, 0) & ~nb(m, -3, 0)
            sc.put(shine, C["hair_h"], "Hair (back)")


def draw_front_lock(sc, head_origin, wind, phase):
    m, _ = strand_mask(head_origin, FRONT_LOCK, v2(wind) * 0.6, phase)
    paint_shaded(sc, m, (C["hair_o"], C["hair_s"], C["hair_b"], C["hair_l"]), "Head")

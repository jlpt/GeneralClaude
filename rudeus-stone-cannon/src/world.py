"""Backdrop: night sky, moon, mountains, the plain, grass and the target crag."""
import math
import numpy as np
from canvas import (C, XS, YS, W, H, CW, CH, M, v2, disc, ellipse, polygon, nb, outline_of, dilate4,
                    line_pts, smooth01)

RNG = np.random.default_rng(7)
HORIZON = 120
GROUND_Y = 159          # where Rudeus' boots touch

# ---------------------------------------------------------------- static data
STARS = []
for _ in range(110):
    x = RNG.uniform(-8, W + 8)
    y = (RNG.uniform(0, 1) ** 1.6) * (HORIZON - 16) - 6
    STARS.append((x, y, RNG.uniform(0, 6.28), RNG.uniform(0.15, 0.5), RNG.uniform(0, 1)))
BIG_STARS = [(30, 18), (128, 10), (232, 22), (300, 8), (58, 52), (165, 60), (16, 80)]

MOON = v2((182, 30))
MOON_R = 15
CRATERS = [((-5, -4), 3.2), ((4, 3), 2.4), ((-2, 6), 1.8), ((6, -6), 1.6), ((-8, 3), 1.3)]


def ridge(x, seed, base, amp, scales):
    r = np.random.default_rng(seed)
    y = np.full_like(x, base, dtype=np.float32)
    for s in scales:
        ph = r.uniform(0, 6.28)
        y += amp * s[1] * np.sin(x / s[0] + ph)
    return y


XLINE = np.arange(-M, W + M, dtype=np.float32)
FAR_RIDGE = ridge(XLINE, 3, 100, 1.0, [(37, 8), (17, 4), (7, 1.4), (3.1, 0.6)])
NEAR_RIDGE = ridge(XLINE, 11, 112, 1.0, [(53, 6), (21, 3), (9, 1.2), (4, 0.5)])

CLOUDS = [  # y, x0, length, thickness, speed
    (44, -40, 120, 3, 0.18), (64, 150, 90, 2, 0.12), (24, 230, 70, 2, 0.22), (82, 20, 140, 2, 0.08),
]

# crag silhouette (screen px)
CRAG_PTS = [(232, 166), (236, 152), (239, 136), (242, 122), (240, 110), (244, 98), (242, 86),
            (247, 72), (251, 60), (256, 50), (262, 45), (268, 46), (272, 41), (278, 43), (283, 50),
            (286, 61), (291, 74), (290, 88), (295, 102), (299, 118), (305, 134), (313, 150),
            (322, 160), (334, 166)]
HOLE = v2((252, 98))     # where the bullet strikes the left face

# grass blades: (x, base_y, height, tone)
GRASS_FG = [(x + RNG.uniform(-0.6, 0.6), 181 - RNG.uniform(0, 5), RNG.uniform(4, 10), RNG.integers(0, 2))
            for x in np.arange(-12, W + 12, 1.7)]
GRASS_MID = []
for _ in range(240):
    x = RNG.uniform(-6, W + 6)
    y = RNG.uniform(HORIZON + 6, 172)
    near = (y - HORIZON) / (180 - HORIZON)
    GRASS_MID.append((x, y, 1.5 + 3.5 * near, near))
PEBBLES = [(RNG.uniform(0, W), RNG.uniform(140, 176), RNG.uniform(0.8, 1.8)) for _ in range(26)]


def _crag_cells():
    """Voronoi blocks that give the crag its stacked-stone look."""
    mask = polygon(CRAG_PTS)
    r = np.random.default_rng(5)
    pts = np.stack([r.uniform(232, 322, 70), r.uniform(38, 170, 70)], 1)
    d = None
    idx = None
    for i, (px, py) in enumerate(pts):
        dd = (XS - px) ** 2 + ((YS - py) * 2.6) ** 2
        if d is None:
            d, idx = dd, np.zeros(dd.shape, np.int32)
        else:
            closer = dd < d
            d = np.where(closer, dd, d)
            idx = np.where(closer, i, idx)
    tone = np.zeros(len(pts))
    for i, (px, py) in enumerate(pts):
        tone[i] = 2.5 - (px - 256) / 20.0 + r.uniform(-0.5, 0.5) - max(0, (py - 135) / 25)
    return mask, idx, tone


CRAG_MASK, CRAG_CELL, CRAG_TONE = _crag_cells()
RAMP_RK = ["rk1", "rk2", "rk3", "rk4", "rk5"]


# ---------------------------------------------------------------- drawing
def draw_sky(sc, f, dark=0.0):
    t = np.clip((YS + 10) / (HORIZON + 10), 0, 1)
    stops = [(0.0, "sky0"), (0.22, "sky1"), (0.42, "sky2"), (0.58, "sky3"), (0.70, "sky4"),
             (0.80, "sky5"), (0.88, "sky6"), (0.94, "sky7"), (0.98, "sky8"), (1.0, "sky9")]
    xs = [s[0] for s in stops]
    rgb = np.stack([np.interp(t, xs, [C[s[1]][k] for s in stops]) for k in range(3)], -1)
    sc.rgb[:] = rgb * (1 - 0.45 * dark)
    sc.owner[:] = 0
    sc.dith[:] = 26.0
    # moon halo
    d = np.hypot(XS - MOON[0], YS - MOON[1])
    halo = np.clip(1 - (d - MOON_R) / 36, 0, 1) ** 2 * (1 - 0.6 * dark)
    sc.light(halo, (60, 52, 90), 1.0, dither=30)
    # drifting cloud streaks
    for (cy, x0, ln, th, sp) in CLOUDS:
        x = ((x0 + f * sp + 60) % (W + 160)) - 60
        band = (np.abs(YS - cy) < th) & (XS > x) & (XS < x + ln)
        edge = (np.abs(XS - x - ln / 2) / (ln / 2)) > 0.7
        sc.light(band.astype(np.float32) * np.where(edge, 0.5, 1.0) * (1 - 0.5 * dark), (34, 28, 60), 1.0)
        lower = (np.abs(YS - cy - th) < 1) & (XS > x + 6) & (XS < x + ln - 10)
        sc.light(lower.astype(np.float32) * (1 - 0.5 * dark), (44, 30, 52), 1.0)
    # stars
    for (x, y, ph, sp, b) in STARS:
        tw = 0.5 + 0.5 * math.sin(f * sp + ph)
        if tw * (1 - dark * 0.7) < 0.25:
            continue
        col = "star" if (b > 0.6 and tw > 0.6) else "star_d"
        sc.put_px([x], [y], col, "Sky")
    for i, (x, y) in enumerate(BIG_STARS):
        tw = 0.5 + 0.5 * math.sin(f * 0.37 + i * 1.7)
        if dark > 0.6 and i % 2:
            continue
        sc.put_px([x], [y], "star", "Sky")
        if tw > 0.45:
            sc.put_px([x - 1, x + 1, x, x], [y, y, y - 1, y + 1], "star_d", "Sky")
        if tw > 0.85:
            sc.put_px([x - 2, x + 2, x, x], [y, y, y - 2, y + 2], "star_d", "Sky")
    # moon
    mm = disc(MOON, MOON_R)
    sc.put(mm, "moon_l", "Sky")
    sc.put(mm & ~disc(MOON + v2((-1.5, -1.5)), MOON_R - 0.6), "moon_b", "Sky")
    sc.put(mm & ~disc(MOON + v2((-3.5, -3)), MOON_R + 0.3), "moon_d", "Sky")
    sc.put(mm & disc(MOON + v2((-4, -4)), 6.0) & ~disc(MOON + v2((-3, -3)), 4.5), "moon_h", "Sky")
    for (cx, cy), r in CRATERS:
        c = MOON + v2((cx, cy))
        cm = disc(c, r) & mm
        sc.put(cm, "moon_b", "Sky")
        sc.put(cm & ~disc(c + v2((0.8, 0.8)), r), "moon_d", "Sky")
        sc.put(cm & ~disc(c + v2((-0.8, -0.8)), r) & disc(c, r + 0.2), "moon_h", "Sky")
    if dark > 0:
        sc.rgb[mm] *= 1 - 0.35 * dark


def draw_land(sc, f, dark=0.0):
    k = 1 - 0.4 * dark
    xi = np.clip((XS + M).astype(int), 0, len(XLINE) - 1)
    far = YS >= FAR_RIDGE[xi]
    sc.put(far, C["mtn0"] * k, "Land")
    rim = far & ~nb(far, 0, -1)
    sc.put(rim, C["mtn1"] * k, "Land")
    # moonlit snow-ish streaks on the far range
    sc.put(far & (YS < FAR_RIDGE[xi] + 4) & ((XS * 0.7 + YS) % 9 < 1.2), C["mtn1"] * k, "Land")
    near = YS >= NEAR_RIDGE[xi]
    sc.put(near, C["mtn2"] * k, "Land")
    sc.put(near & ~nb(near, 0, -1), C["mtn3"] * k, "Land")
    # plain
    plain = YS >= HORIZON
    t = np.clip((YS - HORIZON) / (180 - HORIZON), 0, 1)
    stops = [(0.0, "gr1"), (0.25, "gr2"), (0.6, "gr3"), (1.0, "gr2")]
    rgb = np.stack([np.interp(t, [s[0] for s in stops], [C[s[1]][c] for s in stops]) for c in range(3)], -1)
    sc.put_rgb(plain, rgb * k, "Land", dither=22)
    sc.put(plain & ~nb(plain, 0, -1), C["gr2"] * k, "Land")
    # dirt patch Rudeus stands on
    patch = ellipse((80, GROUND_Y + 1), 34, 7) & plain
    sc.put(patch, C["dt1"] * k, "Land", dither=0)
    sc.put(patch & ~ellipse((80, GROUND_Y + 2.2), 31, 5.5), C["dt0"] * k, "Land")
    sc.put(patch & ellipse((72, GROUND_Y), 18, 3), C["dt2"] * k, "Land")
    for (x, y, r) in PEBBLES:
        pm = ellipse((x, y), r * 1.3, r * 0.8)
        sc.put(pm, C["rk3"] * k, "Land")
        sc.put(pm & ~nb(pm, 0, -1), C["rk5"] * k, "Land")
    # mid-ground grass
    for (x, y, h, near_) in GRASS_MID:
        sway = math.sin(f * 0.21 + x * 0.09) * 0.8 * near_
        xs, ys = line_pts((x, y), (x + sway + 0.6, y - h))
        col = "gr4" if near_ > 0.5 else "gr3"
        sc.put_px(xs, ys, C[col] * k, "Land")


def draw_crag(sc, f, hit=None, dark=0.0):
    """hit: None or dict(t=frames since impact) to carve the hole, cracks and the
    fresh rock face."""
    k = 1 - 0.35 * dark
    m = CRAG_MASK
    lvl = np.clip(np.round(CRAG_TONE[CRAG_CELL]), 0, 4).astype(int)
    up_diff = nb(CRAG_CELL, 0, -1) != CRAG_CELL
    dn_diff = nb(CRAG_CELL, 0, 1) != CRAG_CELL
    lf_diff = nb(CRAG_CELL, -1, 0) != CRAG_CELL
    lvl_top = np.clip(lvl + 1, 0, 4)
    for i, name in enumerate(RAMP_RK):
        sc.put(m & (lvl == i), C[name] * k, "Crag")
    for i, name in enumerate(RAMP_RK):
        sc.put(m & up_diff & (lvl_top == i), C[name] * k, "Crag")
    for i, name in enumerate(RAMP_RK):
        sc.put(m & dn_diff & (np.clip(lvl - 1, 0, 4) == i), C[name] * k, "Crag")
    sc.put(m & dn_diff & (lvl <= 1), C["rk0"] * k, "Crag")
    sc.put(m & lf_diff & (lvl <= 2) & ((XS + YS) % 3 == 0), C["rk1"] * k, "Crag")
    # silhouette: dark outline, moonlit left rim
    sc.put(outline_of(m), C["rk0"] * k, "Crag")
    lit = m & ~nb(m, -1, 0) & (YS < 150)
    sc.put(lit, C["rk6"] * k, "Crag")
    sc.put(m & ~nb(m, -1, -1) & ~lit & (YS < 120), C["rk5"] * k, "Crag")
    # grass and rubble at the foot
    for x in range(232, 330, 2):
        h = 2 + (x * 7 % 5)
        sc.put_px([x, x], [163 - h // 2, 163 - h // 2 - 1], C["gr3"] * k, "Crag")
    if hit is None:
        return
    t = hit["t"]
    hc = HOLE
    # fresh fractured face around the hole
    bite = polygon([hc + v2((math.cos(a) * r, math.sin(a) * r * 1.1)) for a, r in zip(
        np.linspace(0, 2 * math.pi, 12, endpoint=False), [11, 8, 10, 12, 9, 13, 10, 8, 11, 9, 12, 10])]) & m
    sc.put(bite, C["rk4"] * k, "Crag")
    sc.put(bite & ~nb(bite, 1, 1), C["rk5"] * k, "Crag")
    sc.put(outline_of(bite) & m, C["rk1"] * k, "Crag")
    # radiating cracks
    r = np.random.default_rng(21)
    grow = min(1.0, (t + 1) / 4)
    for j in range(9):
        a = j / 9 * 2 * math.pi + r.uniform(-0.3, 0.3)
        p = hc.copy()
        ln = r.uniform(14, 34) * grow
        for s in range(5):
            a += r.uniform(-0.5, 0.5)
            q = p + v2((math.cos(a), math.sin(a) * 1.2)) * ln / 5
            xs, ys = line_pts(p, q)
            inside = m[np.clip(ys + M, 0, CH - 1), np.clip(xs + M, 0, CW - 1)]
            sc.put_px(xs[inside], ys[inside], C["rk0"] * k, "Crag")
            inside2 = m[np.clip(ys + 1 + M, 0, CH - 1), np.clip(xs + 1 + M, 0, CW - 1)]
            sc.put_px(xs[inside2] + 1, ys[inside2] + 1, C["rk3"] * k, "Crag")
            p = q
    # the hole itself, with a molten rim that cools
    hole = ellipse(hc, 4.6, 5.4)
    sc.put(hole, C["rk0"], "Crag")
    sc.put(hole & ellipse(hc + v2((1.4, 0.4)), 3.0, 3.6), C["st0"], "Crag")
    rim = outline_of(hole) | (outline_of(dilate4(hole)) & ((XS + YS) % 2 == 0))
    heat = max(0.0, 1 - t / 30)
    if heat > 0:
        cols = ["mg7", "mg6", "mg5", "mg4", "mg3", "mg2", "mg1"]
        ci = min(len(cols) - 1, int((1 - heat) * len(cols)))
        sc.put(rim & m, C[cols[ci]], "Crag")
        sc.put(outline_of(hole) & m & (YS > hc[1]), C[cols[min(len(cols) - 1, ci + 1)]], "Crag")
        sc.light(np.clip(1 - np.hypot(XS - hc[0], YS - hc[1]) / 16, 0, 1) ** 2 * heat, C["mg3"], 0.9)
    else:
        sc.put(rim & m, C["rk2"] * k, "Crag")


def draw_grass_fg(sc, f, blast=0.0, blast_x=80, dark=0.0):
    k = 1 - 0.4 * dark
    for (x, y, h, tone) in GRASS_FG:
        sway = math.sin(f * 0.25 + x * 0.13) * 1.2
        push = 0.0
        if blast:
            near = max(0.0, 1 - abs(x - blast_x) / 140)
            push = -blast * near * h * 0.9
        tip = (x + sway + push, y - h * (1 - 0.35 * min(1.0, abs(push) / max(h, 1))))
        xs, ys = line_pts((x, y), tip)
        sc.put_px(xs, ys, C["gr0" if tone else "gr1"] * k, "Land")
        sc.put_px(xs[-1:], ys[-1:], C["gr2"] * k, "Land")

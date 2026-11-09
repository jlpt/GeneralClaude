"""Effect primitives: glows, sparkles, magic circles, the beam, shockwaves, meteors, smoke."""
import math
import numpy as np
from canvas import (C, XS, YS, BAYER, CHECK, HORIZON, SHAPE, dith, v2, nb, disc, ellipse, ering, pixels,
                    line_px, polyline, seg_dist)

FRONT, BACK = "FX (front)", "FX (back)"
CY = ("fx_w", "cy_1", "cy_2", "cy_3", "cy_4", "cy_5")
AU = ("fx_w", "au_1", "au_2", "au_3", "au_4", "vi_4")


def glow(sc, c, r, layer=FRONT, halo=True, ramp=CY):
    d = np.hypot(XS - c[0], YS - c[1])
    if halo:
        sc.put((d <= r + 5.5) & dith(0.2), ramp[5], layer)
        sc.put((d <= r + 3.4) & dith(0.5), ramp[4], layer)
    sc.put(d <= r + 1.3, ramp[4], layer)
    sc.put(d <= r + 0.5, ramp[3], layer)
    sc.put(d <= r - 0.4, ramp[2], layer)
    sc.put(d <= r * 0.62, ramp[1], layer)
    sc.put(d <= r * 0.34, ramp[0], layer)


def sparkle(sc, c, n, layer=FRONT, core="fx_w", arm="cy_2", diag=True):
    """Four-point twinkle with arms `n` pixels long."""
    x, y = int(round(c[0])), int(round(c[1]))
    arms, inner = [], [(x, y)]
    for i in range(1, n + 1):
        pts = [(x + i, y), (x - i, y), (x, y + i), (x, y - i)]
        (inner if i <= n // 2 else arms).extend(pts)
    sc.put(pixels(arms), arm, layer)
    if diag and n >= 3:
        sc.put(pixels([(x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1), (x - 1, y - 1)]), arm, layer)
    sc.put(pixels(inner), core, layer)


def magic_circle(sc, c, rx, ry, rot, layer, progress=1.0, fade=1.0, ring="cy_2", hot="cy_1", star="au_2",
                 points=6, detail=True):
    """A summoning circle lying on a plane seen at a shallow angle (so: an ellipse).
    Outer + inner ring, a star polygon between them, counter-rotating glyph ticks and a hot arc."""
    if rx < 2 or ry < 1 or fade <= 0:
        return
    th = np.degrees(np.arctan2((YS - c[1]) / ry, (XS - c[0]) / rx))
    keep = np.mod(th - rot, 360) <= progress * 360
    if fade < 1:
        keep &= dith(fade)
    outer = ering(c, rx, ry)
    sc.put(outer & keep, ring, layer)
    if detail:
        k_in = 0.74
        sc.put(ering(c, rx * k_in, ry * k_in) & keep, ring, layer)
        pts = [(c[0] + rx * k_in * math.cos(math.radians(rot + i * 360 / points)),
                c[1] + ry * k_in * math.sin(math.radians(rot + i * 360 / points))) for i in range(points)]
        m = np.zeros(SHAPE, bool)
        step = 2 if points > 4 else 1
        for i in range(points):
            m |= line_px(pts[i], pts[(i + step) % points])
        sc.put(m & keep, star, layer)
        ticks = [(c[0] + rx * 0.87 * math.cos(math.radians(-rot * 1.5 + i * 30)),
                  c[1] + ry * 0.87 * math.sin(math.radians(-rot * 1.5 + i * 30))) for i in range(12)]
        sc.put(pixels(ticks) & keep, hot, layer)
        sc.put(ering(c, rx * 0.3, ry * 0.3) & keep, hot, layer)
        sc.put(pixels(pts) & keep, "fx_w", layer)
    sc.put(outer & keep & (np.mod(th - rot * 3, 360) < 75), hot, layer)


def light_wall(sc, c, rx, ry, h, front, level=0.55):
    """Soft curtain of light rising from the far or near arc of a ground circle."""
    if h <= 0:
        return
    dx = (XS - c[0]) / rx
    arc = ry * np.sqrt(np.clip(1 - dx ** 2, 0, 1))
    up = (c[1] + arc if front else c[1] - arc) - YS
    m = (np.abs(dx) <= 1) & (up >= 0) & (up <= h)
    lv = (1 - up / h) * level
    layer = FRONT if front else BACK
    sc.put(m & (BAYER < lv), "cy_4", layer)
    sc.put(m & (BAYER < lv * 0.45), "cy_3", layer)


def beam(sc, x0, y0, y1, hw, t, layer=BACK):
    """Pillar of light: white core, cyan sheath, dithered halo, with a wobbling edge."""
    if hw <= 0:
        return
    wob = np.sin(YS * 0.55 + t * 2.3) * 1.1 + np.sin(YS * 0.21 - t * 1.7) * 0.8
    w = hw + wob * min(1.0, hw / 7.0)
    d = np.abs(XS - x0)
    rows = (YS >= y0) & (YS <= y1)
    if hw >= 3:
        sc.put(rows & (d <= w + 6) & dith(0.25), "cy_5", layer)
        sc.put(rows & (d <= w + 3.5) & dith(0.55), "cy_4", layer)
    sc.put(rows & (d <= w + 1.4), "cy_3", layer)
    sc.put(rows & (d <= w), "cy_2", layer)
    sc.put(rows & (d <= w * 0.74), "cy_1", layer)
    sc.put(rows & (d <= w * 0.48), "fx_w", layer)
    if hw >= 6:   # streaks of light racing down inside the core
        for k in range(4):
            xk = x0 + (k - 1.5) * hw * 0.42
            seg = np.mod(YS + t * 11 + k * 17, 29) < 8
            sc.put(rows & (np.abs(XS - xk) < 0.6) & seg, "cy_1" if k % 2 else "cy_2", layer)


def shock(sc, c, rx, ry, th, col, fade=1.0):
    """Ground shockwave ring; the near half goes in front of the character."""
    m = ering(c, rx, ry, th) & (YS >= HORIZON)
    if fade < 1:
        m &= dith(fade)
    sc.put(m & (YS < c[1]), col, BACK)
    sc.put(m & (YS >= c[1]), col, FRONT)


def puff(sc, c, r, layer=FRONT, cols=("sm_1", "sm_2", "sm_3"), fade=1.0, seed=0):
    """Billow of dust or smoke: three overlapping lobes, lit from the upper left."""
    if r <= 0.6 or fade <= 0:
        return
    c = v2(c)
    a = seed * 2.4
    lobes = [(c, r * 0.8), (c + v2((math.cos(a), -0.45 * abs(math.sin(a)) - 0.2)) * r * 0.75, r * 0.6),
             (c + v2((-math.cos(a) * 0.9, 0.25)) * r * 0.8, r * 0.52)]
    full, hi = np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)
    for q, rr in lobes:
        full |= disc(q, rr)
        hi |= disc(q + v2((-0.3, -0.35)) * rr, rr * 0.6)
    keep = dith(fade) if fade < 1 else np.ones(SHAPE, bool)
    sc.put(full & keep, cols[1], layer)
    sc.put(full & ~hi & ~nb(full, 0, 1) & keep, cols[2], layer)
    sc.put(full & ~hi & ~nb(full, 1, 1) & CHECK & keep, cols[2], layer)
    sc.put(hi & full & keep, cols[0], layer)


def meteor(sc, head, tail, layer=FRONT, ramp=CY):
    dist, t = seg_dist(head, tail)
    m = dist <= (1.8 - 1.5 * t)
    sc.put(m & (t > 0.7) & dith(0.4), ramp[4], layer)
    sc.put(m & (t <= 0.7) & (t > 0.42) & dith(0.75), ramp[3], layer)
    sc.put(m & (t <= 0.42), ramp[2], layer)
    sc.put(m & (t <= 0.22) & (dist <= 1.0), ramp[1], layer)
    sc.put(disc(head, 1.7), "fx_w", layer)
    sparkle(sc, head, 3, layer, arm=ramp[1])


def smear(sc, pivot, r_in, r_out, a_from, a_to, cols, layer=FRONT, fade=1.0):
    """Crescent swept by the staff head between two screen angles (0 = right, 90 = up)."""
    px, py = pivot
    r = np.hypot(XS - px, YS - py)
    a = np.degrees(np.arctan2(-(YS - py), XS - px))
    span = a_to - a_from
    t = np.mod((a - a_from) * np.sign(span), 360) / abs(span)   # 0 trailing end .. 1 leading edge
    thick = (r_out - r_in) * np.clip(t, 0, 1) ** 1.6 * fade
    band = (t <= 1.0) & (r <= r_out + 0.5) & (r >= r_out - thick - 0.5)
    sc.put(band & (t < 0.45) & CHECK, cols[2], layer)
    sc.put(band & (t >= 0.45), cols[1], layer)
    sc.put(band & (t > 0.78) & (r > r_out - thick * 0.6), cols[0], layer)


def bolt(sc, p0, p1, rng, jag, col, layer=FRONT, n=6):
    """Jagged lightning between two points."""
    p0, p1 = v2(p0), v2(p1)
    d = p1 - p0
    nrm = v2((-d[1], d[0])) / max(np.hypot(*d), 1e-6)
    pts = [p0]
    for i in range(1, n):
        pts.append(p0 + d * (i / n) + nrm * rng.uniform(-jag, jag))
    pts.append(p1)
    sc.put(polyline(pts), col, layer)

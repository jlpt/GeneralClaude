"""The spinning stone bullet: rifled drill body, glowing rune band, spin blur, spiral air
vortex, tangential sparks and the drilling spray when it bores into rock."""
import math
import numpy as np
from canvas import C, XS, YS, CW, CH, M, v2, unit, outline_of, nb, line_pts, disc

BACK, SHOULDER, TIP, RAD = -8.5, 0.5, 11.5, 4.4


def _frame(ang):
    d = unit(ang)
    return d, v2((-d[1], d[0]))


def _px(sc, x, y, color, layer):
    xs, ys = np.round(np.atleast_1d(x)).astype(int), np.round(np.atleast_1d(y)).astype(int)
    sc.put_px(xs, ys, color, layer)


def profile(u, s):
    back, sh, tip, R = BACK * s, SHOULDER * s, TIP * s, RAD * s
    x = np.clip((u - sh) / (tip - sh), 0, 1)
    rad = np.where(u < sh, R, R * np.clip(1 - x ** 1.6, 0, 1))
    rad = np.where(u < back + 1.2 * s, rad - (back + 1.2 * s - u) * 1.1, rad)
    return rad


def drill_bullet(sc, c, ang, s, turn, heat=0.0, blur=0.0, layer="Magic", underglow=1.0, clip=None, f=0,
                 band=1.0):
    """turn: rotation about the axis in turns (drives the flutes and rune band).
    blur 0..1: how smeared the flutes are. clip: (point, dir) plane; pixels beyond it are hidden."""
    if s < 0.08:
        return None
    d, p = _frame(ang)
    dx, dy = XS - c[0], YS - c[1]
    u = dx * d[0] + dy * d[1]
    w = dx * p[0] + dy * p[1]
    rad = profile(u, s)
    m = (u >= BACK * s) & (u <= TIP * s) & (np.abs(w) <= rad + 0.3)
    if clip is not None:
        cp, cd = clip
        m &= ((XS - cp[0]) * cd[0] + (YS - cp[1]) * cd[1]) < 0
    if not m.any():
        return m
    wn = np.clip(w / np.maximum(rad, 0.6), -1, 1)
    th = np.arcsin(wn)                                   # angle around the axis on the visible half
    lightv = np.cos(th + 1.0)                            # key light from above
    lvl = 1 + np.round(4 * np.clip((lightv + 0.3) / 1.3, 0, 1)).astype(int)
    # three helical flutes that crawl backwards as it turns
    pitch = 7.5 * s
    g = np.mod(3 * th / (2 * math.pi) + u / pitch + 3 * turn, 1.0)
    flute = g < 0.27
    lip = (g >= 0.27) & (g < 0.38)
    if blur > 0.55:
        flute &= ((XS + YS + f) % 2 == 0)
        lvl = np.where(flute, lvl - 1, lvl)
    else:
        lvl = np.where(flute, lvl - 2, lvl)
        lvl = np.where(lip & (lightv > 0.1), lvl + 1, lvl)
    lvl = np.clip(lvl, 1, 5)
    sc.put(outline_of(m), "st0", layer)
    for i in range(1, 6):
        sc.put(m & (lvl == i), "st%d" % i, layer)
    # flat back face
    sc.put(m & (u < BACK * s + 1.1 * s), "st2", layer)
    sc.put(m & (u < BACK * s + 1.1 * s) & (wn < -0.3), "st3", layer)
    # glowing rune band near the back; the dashes run round with the spin
    if band > 0:
        bu = BACK * s + 2.3 * s
        bandm = m & (np.abs(u - bu) <= max(0.6, 0.55 * s))
        dash = np.mod(th / (2 * math.pi) * 8 + 8 * turn, 1.0) < 0.5
        if blur > 0.7:
            sc.put(bandm, "mg4", layer)
            sc.put(bandm & (wn < -0.2), "mg6", layer)
        else:
            sc.put(bandm, "mg2", layer)
            sc.put(bandm & dash, "mg5" if band > 0.6 else "mg4", layer)
            sc.put(bandm & dash & (wn < -0.3), "mg7", layer)
    # motion arcs across the body once it is really moving
    if blur > 0.2:
        span = TIP * s * 0.75 - BACK * s
        for k in range(3):
            uk = TIP * s * 0.75 - np.mod(f * 0.43 + k / 3.0, 1.0) * span
            arc = m & (np.abs(u - uk - 0.9 * s * (1 - wn * wn)) < 0.55) & (u > bu + 0.8 * s if band > 0 else True)
            sc.put(arc & (wn < 0.1), "white" if blur > 0.6 else "st5", layer)
            sc.put(arc & (wn >= 0.1), "st4", layer)
    # heated tip
    if heat > 0:
        tz = m & (u > TIP * s - 7.0 * s * heat)
        sc.put(tz & (lvl <= 3), "mg3", layer)
        sc.put(tz & (lvl > 3), "mg4", layer)
        sc.put(m & (u > TIP * s - 4.0 * s * heat), "mg5", layer)
        sc.put(m & (u > TIP * s - 1.8 * s * heat) | (m & (u > TIP * s - 4 * s * heat) & (wn < -0.35)), "mg7", layer)
    if underglow > 0.3:
        below = ((XS - c[0]) * p[0] + (YS - c[1]) * p[1]) > 0
        ol = outline_of(m) & below
        if clip is not None:
            ol &= ((XS - clip[0][0]) * clip[1][0] + (YS - clip[0][1]) * clip[1][1]) < 0
        sc.put(ol, "mg3" if underglow < 0.8 else "mg4", layer)
    return m


def vortex(sc, c, ang, s, turn, power, front, tail=26, layer="Magic", strands=3, twist=0.19, flare=0.07,
           start=None, clip=None):
    """Spiral air streams wrapping the bullet and trailing behind it as a widening cone."""
    if power <= 0:
        return
    d, p = _frame(ang)
    tip = TIP * s if start is None else start
    us = np.arange(tip, BACK * s - tail, -0.45)
    dist = tip - us
    r = RAD * s * (1.25 + flare * dist / max(s, 0.4))
    for j in range(strands):
        phi = 2 * math.pi * (j / strands - turn) + dist * twist / max(s, 0.4) ** 0.5
        depth = np.sin(phi)
        sel = (depth > 0) == front
        # dashed so the streams visibly screw along
        sel &= np.mod(phi / (2 * math.pi) * 2 + j * 0.3, 1.0) < 0.5
        frac = dist / (tip - (BACK * s - tail))
        sel &= frac < power + 0.05
        sel &= ~((frac > 0.6) & ((np.arange(len(us)) % 2) == 1))
        x = c[0] + d[0] * us + p[0] * r * np.cos(phi)
        y = c[1] + d[1] * us + p[1] * r * np.cos(phi)
        if clip is not None:
            cp, cd = clip
            sel &= ((x - cp[0]) * cd[0] + (y - cp[1]) * cd[1]) < 0
        for lo, hi, col_f, col_b in ((0, 0.25, "white", "mg4"), (0.25, 0.6, "air2", "mg3"), (0.6, 1.1, "air1", "mg2")):
            k = sel & (frac >= lo) & (frac < hi)
            if k.any():
                _px(sc, x[k], y[k], col_f if front else col_b, layer)


def spin_rings(sc, c, ang, s, f, power, layer="Magic"):
    """Cross-section rings (front half) peeling back off the body: reads as rotation + wind."""
    d, p = _frame(ang)
    for k in range(3):
        ph = np.mod(f * 0.37 + k / 3.0, 1.0)
        uu = TIP * s * 0.4 - ph * (TIP * s * 0.4 - BACK * s + 6 * s)
        rr = RAD * s * (1.3 + 0.5 * ph)
        t = np.linspace(-0.95, 0.95, int(rr * 3) + 6)
        x = c[0] + d[0] * (uu + 0.9 * s * (1 - t * t)) + p[0] * t * rr
        y = c[1] + d[1] * (uu + 0.9 * s * (1 - t * t)) + p[1] * t * rr
        keep = np.mod(t * 3 + f * 0.5 + k, 1.0) < (0.75 if power > 0.6 else 0.5)
        if ph > 0.6:
            keep &= (np.arange(len(t)) % 2) == 0
        _px(sc, x[keep], y[keep], "air2" if ph < 0.5 else "air1", layer)


def fling(sc, c, ang, s, f, power, n=8, layer="FX", reach=1.0):
    """Grit and sparks thrown off tangentially by the spin."""
    if power <= 0:
        return
    d, p = _frame(ang)
    rng = np.random.default_rng(f * 13 + 5)
    for k in range(int(n * power)):
        u0 = rng.uniform(BACK * s + 2 * s, TIP * s * 0.6)
        phi = rng.uniform(0, 2 * math.pi)
        r = float(profile(np.array([u0]), s)[0])
        start = c + d * u0 + p * r * math.cos(phi)
        vel = -p * math.sin(phi) * 1.0 + p * math.cos(phi) * 0.35 - d * 0.55
        ln = rng.uniform(3, 8) * reach * (0.6 + 0.4 * power)
        a = rng.uniform(0.2, 1.0)
        head = start + vel * ln * (0.4 + a)
        tail = head - vel * ln * 0.45
        hot = rng.uniform() < 0.6
        xs, ys = line_pts(tail, head)
        sc.put_px(xs, ys, ("mg4" if hot else "rk4"), layer)
        sc.put_px(xs[-1:], ys[-1:], "mg7" if hot else "rk5", layer)


def drill_spray(sc, hole, ang, R, f, power, layer="FX", reach=1.0):
    """Catherine wheel: sparks and chips sprayed from the contact ring as it bores in."""
    d, p = _frame(ang)
    K = 7
    for age in range(7):
        for k in range(K):
            phi = 2 * math.pi * k / K + (f - age) * 1.9 + age * 0.17
            start = hole + p * R * math.cos(phi) - d * 1.0
            vt = -p * math.sin(phi) * 3.2 * reach
            vo = p * math.cos(phi) * 1.4 * reach
            vb = -d * (1.6 + 0.4 * k % 3) * reach
            vel = vt + vo + vb
            pos = start + vel * (age + 0.6) + v2((0, 0.18 * age * age))
            prev = start + vel * max(age - 0.4, 0.0) + v2((0, 0.18 * max(age - 1, 0) ** 2))
            if age > 5 * power + 1:
                continue
            col = ["white", "mg7", "mg6", "mg5", "mg4", "mg3", "mg2"][age]
            xs, ys = line_pts(prev, pos)
            sc.put_px(xs, ys, col, layer)
            if (k + age) % 3 == 0 and age > 1:
                # a rock chip riding the spray
                q = pos + v2((0.5, 0.5))
                sc.put_px([int(round(q[0])), int(round(q[0])) + 1], [int(round(q[1]))] * 2, "rk4", layer)
                sc.put_px([int(round(q[0]))], [int(round(q[1])) + 1], "rk1", layer)


SPIN_RATE_FORM = 0.03

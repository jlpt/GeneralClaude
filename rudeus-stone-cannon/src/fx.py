"""Spell effects: magic circles, gathering motes, rising rocks, the stone bullet,
launch streaks and the impact explosion."""
import math
import numpy as np
from canvas import (C, L, XS, YS, W, H, M, CW, CH, v2, rot, unit, disc, ellipse, polygon, nb, outline_of,
                    dilate4, line_pts, smooth01, lerp)
from world import HOLE, GROUND_Y

CORE = v2((81, 81))                 # where the stone forms
OVER = v2((81, 98))                 # overhead circle (between the raised hands)
FEET = v2((80, GROUND_Y + 1))       # ground circle
AIM = HOLE - CORE
AIM_LEN = float(np.hypot(*AIM))
AIM_DIR = AIM / AIM_LEN
AIM_ANG = math.degrees(math.atan2(AIM_DIR[1], AIM_DIR[0]))
PERP = v2((-AIM_DIR[1], AIM_DIR[0]))   # points "down" relative to the shot

R = np.random.default_rng(1234)


def _px(sc, pts, color, layer, dither=0.0):
    if len(pts[0]):
        sc.put_px(pts[0], pts[1], color, layer, dither)


def _uniq(x, y):
    xs, ys = np.round(x).astype(int), np.round(y).astype(int)
    if len(xs) == 0:
        return xs, ys
    k = np.unique(xs * 4096 + ys)
    return k // 4096, k % 4096


# ---------------------------------------------------------------- magic circles
def _ring(c, ax, ay, r, a0, a1, dens=1.6):
    per = 2 * math.pi * r * max(np.hypot(*ax), np.hypot(*ay)) * abs(a1 - a0) / (2 * math.pi)
    n = max(int(per * dens) + 2, 3)
    th = np.linspace(a0, a1, n)
    X, Y = r * np.cos(th), r * np.sin(th)
    return c[0] + X * ax[0] + Y * ay[0], c[1] + X * ax[1] + Y * ay[1]


def _seg(c, ax, ay, p, q, dens=1.6):
    n = int(np.hypot(*(np.subtract(q, p) * [np.hypot(*ax), np.hypot(*ay)])) * dens) + 2
    t = np.linspace(0, 1, n)
    X, Y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
    return c[0] + X * ax[0] + Y * ay[0], c[1] + X * ax[1] + Y * ay[1]


def magic_circle(sc, c, ax, ay, spin, sweep=1.0, power=1.0, layer="Magic", glow=True, inner_spin=None):
    """Circle drawn in the plane spanned by ax/ay (screen vectors for unit X/Y).
    sweep 0..1 draws it in; power 0..1 picks the brightness."""
    c, ax, ay = v2(c), v2(ax), v2(ay)
    inner_spin = -spin * 1.6 if inner_spin is None else inner_spin
    hi = "mg7" if power > 0.85 else ("mg6" if power > 0.5 else "mg4")
    mid = "mg5" if power > 0.5 else "mg3"
    lo = "mg4" if power > 0.5 else "mg2"
    a_end = spin + sweep * 2 * math.pi
    if glow:
        # soft light disc in the circle's plane
        inv = np.linalg.inv(np.array([[ax[0], ay[0]], [ax[1], ay[1]]], np.float32))
        lx = inv[0, 0] * (XS - c[0]) + inv[0, 1] * (YS - c[1])
        ly = inv[1, 0] * (XS - c[0]) + inv[1, 1] * (YS - c[1])
        er = np.sqrt(lx * lx + ly * ly)
        g = np.exp(-((er - 1.0) * 6) ** 2) * 0.55 + np.clip(1 - er, 0, 1) * 0.22 + np.exp(-((er - 0.5) * 9) ** 2) * 0.2
        sc.light(g * sweep, C["mg3"], 0.55 * power + 0.15, dither=40)
    if sweep <= 0:
        return
    _px(sc, _uniq(*_ring(c, ax, ay, 1.0, spin, a_end)), hi, layer)
    _px(sc, _uniq(*_ring(c, ax, ay, 0.86, -spin, -spin + sweep * 2 * math.pi)), mid, layer)
    # ticks + dots in the rune band
    nt = 24
    for i in range(int(nt * sweep)):
        a = spin * 0.5 + i * 2 * math.pi / nt
        if i % 3 == 0:
            p, q = (0.88 * math.cos(a), 0.88 * math.sin(a)), (0.99 * math.cos(a), 0.99 * math.sin(a))
            _px(sc, _uniq(*_seg(c, ax, ay, p, q)), lo, layer)
        else:
            X, Y = 0.93 * math.cos(a), 0.93 * math.sin(a)
            col = hi if (i % 3 == 1) else mid
            _px(sc, _uniq(np.array([c[0] + X * ax[0] + Y * ay[0]]), np.array([c[1] + X * ax[1] + Y * ay[1]])), col, layer)
    # hexagram
    if sweep > 0.45:
        k = min(1.0, (sweep - 0.45) / 0.45)
        for tri in range(2):
            for j in range(3):
                a0 = inner_spin + tri * math.pi / 3 + j * 2 * math.pi / 3
                a1 = a0 + 2 * math.pi / 3
                p = (0.84 * math.cos(a0), 0.84 * math.sin(a0))
                q = (0.84 * math.cos(a1), 0.84 * math.sin(a1))
                q = (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)
                _px(sc, _uniq(*_seg(c, ax, ay, p, q)), mid, layer)
    if sweep > 0.25:
        _px(sc, _uniq(*_ring(c, ax, ay, 0.46, inner_spin, inner_spin + 2 * math.pi * min(1, sweep * 1.3))), hi, layer)
        x, y = _ring(c, ax, ay, 0.36, -inner_spin, -inner_spin + 2 * math.pi)
        th = np.linspace(0, 1, len(x))
        keep = (np.floor(th * 18) % 2) == 0
        _px(sc, _uniq(x[keep], y[keep]), lo, layer)
    if sweep > 0.7:
        _px(sc, _uniq(*_ring(c, ax, ay, 0.16, 0, 2 * math.pi)), mid, layer)


def ground_circle(sc, f, sweep, power, spin):
    magic_circle(sc, FEET, (34, 0), (0, 8.5), spin, sweep, power, layer="Back FX")


def over_circle(sc, f, sweep, power, spin):
    magic_circle(sc, OVER, (17, 0), (0, 4.4), -spin * 1.3, sweep, power, layer="Magic")


BARRELS = [(19, 12.0), (31, 9.5), (42, 7.0)]


def barrel_ring(sc, i, sweep, power, spin, burst=None):
    d, r = BARRELS[i]
    c = CORE + AIM_DIR * d
    ax = PERP * -r        # vertical-ish diameter
    ay = AIM_DIR * r * 0.32
    if burst is None:
        magic_circle(sc, c, ax, ay, spin + i, sweep, power, layer="Magic", glow=True)
        return
    # shattering: ring of shards flying outward in the ring plane
    t = burst
    n = 22
    for j in range(n):
        a = j / n * 2 * math.pi + i
        rr = 1.0 + t * 0.35
        p = c + ax * math.cos(a) * rr + ay * math.sin(a) * rr
        q = c + ax * math.cos(a) * (rr + 0.18) + ay * math.sin(a) * (rr + 0.18)
        if t < 4 and (j + t) % 3 != 0:
            col = "white" if t < 1 else ("mg6" if t < 2 else "mg4")
            _px(sc, line_pts(p, q), col, "FX")


# ---------------------------------------------------------------- gathering motes
MOTES = [dict(a=R.uniform(0, 2 * math.pi), r0=R.uniform(36, 95), life=R.integers(9, 17),
              off=R.integers(0, 17), spin=R.uniform(0.6, 1.8) * R.choice([-1, 1]))
         for _ in range(64)]


def motes(sc, f, density, speed=1.0, target=CORE):
    if density <= 0:
        return
    for i, m in enumerate(MOTES):
        if i / len(MOTES) > density:
            break
        life = m["life"] / speed
        t = ((f + m["off"]) % life) / life
        pos, prev = [], []
        for tt in (t, max(t - 0.05, 0)):
            r = m["r0"] * (1 - tt) ** 1.4
            a = m["a"] + m["spin"] * tt * 2.2 + (f // 40)
            pos.append(target + v2((math.cos(a) * r, math.sin(a) * r * 0.85)))
        p, q = pos
        dv = q - p
        ln = float(np.hypot(*dv))
        if ln > 4:
            q = p + dv / ln * 4
        col = "mg3" if t < 0.4 else ("mg5" if t < 0.75 else "mg7")
        _px(sc, line_pts(q, p), col if t > 0.2 else "mg2", "Magic")
        _px(sc, ([int(round(p[0]))], [int(round(p[1]))]), "mg6" if t < 0.75 else "white", "Magic")


# ---------------------------------------------------------------- rising rocks
ROCKS = []
for i in range(20):
    side = -1 if i % 2 else 1
    x0 = 80 + side * R.uniform(8, 44)
    y0 = GROUND_Y + R.uniform(-3, 6)
    ROCKS.append(dict(x0=x0, y0=y0, t0=22 + i * 1.2 + R.uniform(-1, 1), dur=R.uniform(14, 18),
                      size=R.uniform(1.6, 3.4), ph=R.uniform(0, 6.28), orbit=R.uniform(10, 24),
                      spin=R.uniform(-0.5, 0.5),
                      shape=[R.uniform(0.7, 1.15) for _ in range(6)]))


def rock_shape(c, size, ang, shape):
    pts = []
    for j, s in enumerate(shape):
        a = ang + j * math.pi / 3
        pts.append(c + v2((math.cos(a) * size * s, math.sin(a) * size * s * 0.85)))
    return polygon(pts)


def draw_rock(sc, c, size, ang, shape, layer, glow_below=0.0, hot=0.0):
    if size < 1.1:
        _px(sc, ([int(round(c[0]))], [int(round(c[1]))]), "rk4" if hot < 0.5 else "mg4", layer)
        return
    m = rock_shape(c, size, ang, shape)
    if not m.any():
        m = disc(c, 0.8)
    sc.put(outline_of(m), "rk0", layer)
    sc.put(m, "rk3", layer)
    sc.put(m & ~nb(m, -1, -1), "rk5", layer)
    sc.put(m & ~nb(m, 1, 1), "rk2", layer)
    if glow_below > 0.3:
        sc.put(outline_of(m) & (YS > c[1] + size * 0.4), "mg3" if glow_below < 0.7 else "mg4", layer)
    if hot > 0:
        col = "mg6" if hot > 0.7 else ("mg4" if hot > 0.4 else "mg2")
        sc.put(m & (YS > c[1]), col, layer)


def rocks(sc, f, rumble=0.0, resting=True, flying=True):
    for i, r in enumerate(ROCKS):
        s = (f - r["t0"]) / r["dur"]
        base = v2((r["x0"], r["y0"]))
        if s < 0:
            if not resting:
                continue
            if rumble > 0.2 and f > 16:
                jig = v2((0, -1 if (f + i) % 3 == 0 else 0))
                draw_rock(sc, base + jig, r["size"] * 0.8, r["ph"], r["shape"], "Back FX")
            else:
                draw_rock(sc, base, r["size"] * 0.8, r["ph"], r["shape"], "Back FX")
            continue
        if s >= 1 or not flying:
            continue
        e = s ** 1.5
        lin = base + (CORE - base) * e
        a = r["ph"] + s * 9.0
        orb = v2((math.cos(a), math.sin(a) * 0.45)) * r["orbit"] * (1 - s) * math.sin(math.pi * min(1, s * 1.4))
        p = lin + orb
        size = r["size"] * (1 - 0.65 * s ** 2.5)
        draw_rock(sc, p, size, r["ph"] + r["spin"] * f, r["shape"], "Magic", glow_below=1 - s)
        if s < 0.12:   # dust kicked up at lift-off
            for k in range(4):
                dx = (k - 1.5) * 2.2
                _px(sc, ([int(base[0] + dx)], [int(base[1] + 1 - s * 18 - abs(dx) * 0.3)]), "dt3", "Magic")
        else:          # a short trail of grit
            for k in range(1, 3):
                q = base + (CORE - base) * max(0, (s - 0.05 * k)) ** 1.5
                q = q + v2((math.cos(a - 0.6 * k), math.sin(a - 0.6 * k) * 0.45)) * r["orbit"] * (1 - s) * \
                    math.sin(math.pi * min(1, max(0, s - 0.05 * k) * 1.4))
                _px(sc, ([int(round(q[0]))], [int(round(q[1]))]), "mg3" if k == 1 else "rk3", "Magic")


# ---------------------------------------------------------------- ground cracks
CRACKS = []
for i in range(9):
    a = math.pi * (i / 9) * 2 + R.uniform(-0.2, 0.2)
    pts = [FEET + v2((math.cos(a) * 5, math.sin(a) * 1.4))]
    for k in range(6):
        a += R.uniform(-0.45, 0.45)
        step = R.uniform(3.5, 6)
        pts.append(pts[-1] + v2((math.cos(a) * step, math.sin(a) * step * 0.28)))
    CRACKS.append(pts)


def ground_cracks(sc, grow, heat):
    if grow <= 0:
        return
    for pts in CRACKS:
        n = int(len(pts) * grow)
        for k in range(min(n, len(pts) - 1)):
            xs, ys = line_pts(pts[k], pts[k + 1])
            sc.put_px(xs, ys + 1, "mg2" if heat > 0.3 else "dt0", "Back FX")
            sc.put_px(xs, ys, "mg5" if heat > 0.6 else ("mg4" if heat > 0.3 else "dt0"), "Back FX")


def light_pillars(sc, f, power):
    """Motes of light rising from the ground circle."""
    for i in range(18):
        a = i * 2.39996 + f * 0.04
        base = FEET + v2((math.cos(a) * 34 * 0.95, math.sin(a) * 8.5 * 0.95))
        life = 12 + (i * 5) % 7
        t = ((f + i * 3) % life) / life
        y = base[1] - t * (26 + (i % 4) * 6)
        if power * (1 - t) < 0.15:
            continue
        col = "mg6" if t < 0.3 else ("mg4" if t < 0.65 else "mg2")
        layer = "Back FX" if math.sin(a) < 0 else "Magic"
        _px(sc, ([int(base[0])] * 2, [int(y), int(y) + 1]), col, layer)


# ---------------------------------------------------------------- the stone itself
def gravel_ball(sc, c, rad, spin, f):
    if rad < 0.6:
        return
    dx, dy = XS - c[0], YS - c[1]
    r = np.hypot(dx, dy)
    m = r <= rad
    th = np.arctan2(dy, dx)
    cell = (np.floor((th / (2 * math.pi) + spin) * (6 + rad * 1.2)) * 7 + np.floor(r * 0.9) * 13).astype(int)
    tone = (cell * 2654435761 % 97) / 97.0
    sc.put(outline_of(m), "st0", "Magic")
    sc.put(m, "rk3", "Magic")
    sc.put(m & (tone > 0.55), "rk4", "Magic")
    sc.put(m & (tone < 0.22), "rk2", "Magic")
    sc.put(m & ~nb(m, -1, -1), "rk5", "Magic")
    sc.put(m & ~nb(m, 1, 1), "rk1", "Magic")
    sc.put(outline_of(m) & (YS > c[1] + rad * 0.3), "mg4", "Magic")
    # loose grit orbiting the ball
    for k in range(10):
        a = k * 0.63 + f * (0.5 + 0.07 * k) * (1 if k % 2 else -1)
        rr = rad + 2 + (k % 4) * 1.6
        p = c + v2((math.cos(a) * rr, math.sin(a) * rr * 0.7))
        _px(sc, ([int(round(p[0]))], [int(round(p[1]))]), "rk4" if k % 3 else "mg5", "Magic")


def bullet_mask(c, ang, s, stretch=0.0):
    d = unit(ang)
    p = v2((-d[1], d[0]))
    dx, dy = XS - c[0], YS - c[1]
    u = dx * d[0] + dy * d[1]
    w = dx * p[0] + dy * p[1]
    back, shoulder, tip = -8.5 * s - stretch, 1.5 * s, 10.5 * s
    Rr = 4.4 * s
    rad = np.where(u < shoulder, Rr, Rr * np.clip(1 - np.clip((u - shoulder) / (tip - shoulder), 0, 1) ** 1.7, 0, 1))
    rad = np.where(u < back + 1.2 * s, rad - (back + 1.2 * s - u) * 1.2, rad)
    m = (u >= back) & (u <= tip) & (np.abs(w) <= rad + 0.25)
    return m, u, w / np.maximum(rad, 0.6)


def draw_bullet(sc, c, ang, s, spin, heat=0.0, layer="Magic", underglow=1.0):
    if s < 0.08:
        return
    m, u, wn = bullet_mask(c, ang, s)
    if not m.any():
        return
    lvl = np.select([wn < -0.62, wn < -0.18, wn < 0.32, wn < 0.72], [5, 4, 3, 2], 1)
    groove = (np.mod(u / max(s, 0.3) * 0.30 + wn * 0.85 + spin, 1.0) < 0.24) & (u < 1.5 * s)
    lvl = np.where(groove, lvl - 1, lvl)
    lvl = np.clip(lvl, 1, 5)
    sc.put(outline_of(m), "st0", layer)
    for i in range(1, 6):
        sc.put(m & (lvl == i), "st%d" % i, layer)
    # flat back face
    sc.put(m & (u < -8.5 * s + 1.3), "st2", layer)
    if heat > 0:
        tipz = m & (u > 10.5 * s - 6.5 * s * heat)
        sc.put(tipz, "mg4", layer)
        sc.put(tipz & (u > 10.5 * s - 3.5 * s * heat), "mg6", layer)
        sc.put(tipz & (u > 10.5 * s - 1.5 * s * heat) | (tipz & (wn < -0.3)), "mg7", layer)
    if underglow > 0.3:
        d = unit(ang)
        p = v2((-d[1], d[0]))
        below = ((XS - c[0]) * p[0] + (YS - c[1]) * p[1]) > 0
        sc.put(outline_of(m) & below, "mg3" if underglow < 0.8 else "mg4", layer)


def spin_arcs(sc, c, ang, s, f):
    """White speed ticks wrapping around the spinning body."""
    d = unit(ang)
    p = v2((-d[1], d[0]))
    for k in range(3):
        uu = (-6 + k * 3.5 + (f * 1.7) % 3.5) * s
        a0 = (f * 0.9 + k) % 2
        for t in np.linspace(-0.9, 0.9, 7):
            if (t + a0) % 0.6 < 0.3:
                q = c + d * (uu + 1.6 * s * (1 - t * t)) + p * (t * 6.4 * s)
                _px(sc, ([int(round(q[0]))], [int(round(q[1]))]), "air2", "Magic")


def orbit_sparks(sc, c, f, power, front):
    for ring in range(2):
        tilt = 25 if ring else -30
        for j in range(5):
            th = f * (0.55 if ring else -0.45) + j * 2 * math.pi / 5
            if (math.sin(th) > 0) != front:
                continue
            for tr in range(3):
                t2 = th - tr * 0.18 * (1 if ring else -1)
                q = c + rot(v2((math.cos(t2) * 15, math.sin(t2) * 4.5)), tilt)
                col = ["white", "mg6", "mg4"][tr] if power > 0.5 else ["mg6", "mg4", "mg2"][tr]
                _px(sc, ([int(round(q[0]))], [int(round(q[1]))]), col, "Magic")


def implosion_ring(sc, c, t, rmax=46):
    """t 0..1: ring collapsing onto c."""
    if not 0 <= t < 1:
        return
    r = rmax * (1 - t) ** 1.3 + 2
    x, y = _ring(c, (1, 0), (0, 0.82), r, 0, 2 * math.pi, dens=1.2)
    xs, ys = _uniq(x, y)
    if t < 0.4:
        keep = (xs + ys) % 2 == 0
        xs, ys = xs[keep], ys[keep]
    _px(sc, (xs, ys), "mg4" if t < 0.6 else "mg6", "Magic")


def crackle(sc, c, f, n, rad):
    rr = np.random.default_rng(f * 31 + 7)
    for _ in range(n):
        a = rr.uniform(0, 2 * math.pi)
        p = c + v2((math.cos(a), math.sin(a))) * rad
        for k in range(rr.integers(3, 6)):
            a += rr.uniform(-0.9, 0.9)
            q = p + v2((math.cos(a), math.sin(a))) * rr.uniform(2, 4)
            xs, ys = line_pts(p, q)
            sc.put_px(xs, ys, "mg6" if k < 2 else "mg4", "Magic")
            p = q


def aura(sc, f, power):
    """Flickering flame-like outline around Rudeus (pixels owned by his layers)."""
    if power <= 0:
        return
    body = (sc.owner >= L["Back arm"]) & (sc.owner <= L["Front arm"])
    r1 = dilate4(body) & ~body
    rr = np.random.default_rng(f)
    col_noise = rr.uniform(0, 1, CW)
    # short wisps rising off the top edges only, in a few columns
    top = body & ~nb(body, 0, -1)
    flame = np.zeros_like(body)
    hmax = int(2 + 4 * power)
    for k in range(1, hmax + 1):
        flame |= nb(top, 0, k) & (col_noise[None, :] > 0.55 + 0.4 * k / hmax)
    flame &= ~body & ~r1
    free = sc.owner < L["Domain"]
    sc.put(flame & free, "mg4" if power > 0.7 else "mg3", "Back FX")
    sc.put(r1 & free & ((XS + YS + f) % 2 == 0), "mg3" if power < 0.7 else "mg4", "Back FX")
    if power > 0.7:
        sc.put(r1 & free & top_edge(body) & ((XS + f) % 3 == 0), "mg6", "Back FX")


def top_edge(body):
    return nb(body & ~nb(body, 0, -1), 0, 1)


# ---------------------------------------------------------------- launch
def path_point(s):
    return CORE + AIM * s


def muzzle_flash(sc, t):
    c = CORE + AIM_DIR * 6
    if t == 0:
        sc.put(disc(c, 9), "white", "FX")
        sc.put(outline_of(disc(c, 9)), "mg6", "FX")
        for k in range(10):
            a = k * 36 + 18
            ln = 26 if k % 2 == 0 else 14
            _px(sc, line_pts(c, c + unit(a) * ln), "mg7", "FX")
        sc.light(np.clip(1 - np.hypot(XS - c[0], YS - c[1]) / 90, 0, 1) ** 1.5, C["mg5"], 1.0)
    elif t < 4:
        r = 6 + t * 7
        x, y = _ring(c, PERP * r, AIM_DIR * r * 0.35, 1.0, 0, 2 * math.pi)
        xs, ys = _uniq(x, y)
        if t >= 2:
            keep = (xs + ys) % 2 == 0
            xs, ys = xs[keep], ys[keep]
        _px(sc, (xs, ys), "air2" if t < 2 else "air1", "FX")
        sc.put(disc(c, 5 - t), "mg6", "FX")


def bullet_streak(sc, s_head, s_tail, fade=0.0):
    """Tapered energy beam along the shot from s_tail to s_head (path params)."""
    p0, p1 = path_point(s_tail), path_point(s_head)
    dd, t = _seg_dist(p0, p1)
    w = 0.6 + 3.2 * t ** 0.8
    w = w * (1 - fade)
    if w.max() < 0.3:
        return
    outer = dd <= w + 1.2
    mid = dd <= w * 0.65 + 0.3
    core = dd <= w * 0.25
    sc.put(outer & ((XS + YS) % 2 == 0), "mg3" if fade < 0.5 else "mg2", "FX")
    sc.put(mid, "mg5" if fade < 0.5 else "mg4", "FX")
    sc.put(core, "mg7" if fade < 0.4 else "mg6", "FX")


def _seg_dist(p0, p1):
    d = p1 - p0
    ll = max(float(d @ d), 1e-9)
    t = np.clip(((XS - p0[0]) * d[0] + (YS - p0[1]) * d[1]) / ll, 0, 1)
    return np.hypot(XS - (p0[0] + t * d[0]), YS - (p0[1] + t * d[1])), t


def helix_trail(sc, s_head, f, age=0):
    n = int(s_head * AIM_LEN)
    for k in range(0, n, 1):
        ph = k * 0.42 + f * 0.0
        for strand in range(2):
            off = math.sin(ph + strand * math.pi) * (3.5 + k * 0.012)
            front = math.cos(ph + strand * math.pi) > 0
            p = CORE + AIM_DIR * k + PERP * off
            if age > 0 and (k + age * 7) % (age + 2) != 0:
                continue
            col = ("air2" if front else "mg4") if age < 2 else ("air1" if front else "mg2")
            _px(sc, ([int(round(p[0]))], [int(round(p[1]))]), col, "FX")


def vapor_rings(sc, s_head, f_since):
    for i, d in enumerate((26, 58, 92, 126)):
        if d > s_head * AIM_LEN:
            continue
        age = f_since - i * 0.8
        if age < 0 or age > 6:
            continue
        r = 5 + age * 2.6
        c = CORE + AIM_DIR * d
        x, y = _ring(c, PERP * r, AIM_DIR * r * 0.3, 1.0, 0, 2 * math.pi)
        xs, ys = _uniq(x, y)
        if age > 2.5:
            keep = (xs * 3 + ys) % 3 == 0
            xs, ys = xs[keep], ys[keep]
        _px(sc, (xs, ys), "air2" if age < 2 else "air1", "FX")


SPEED_LINES = [(R.uniform(-30, 180), R.uniform(20, 90), R.uniform(-60, 320)) for _ in range(26)]


def speed_lines(sc, t):
    for i, (y, ln, x0) in enumerate(SPEED_LINES):
        x = x0 + t * 70
        p0 = v2((x, y + (x - CORE[0]) * AIM_DIR[1] / AIM_DIR[0] * 0.3))
        p1 = p0 - AIM_DIR * ln
        _px(sc, line_pts(p0, p1), "air2" if i % 3 else "white", "FX")


# ---------------------------------------------------------------- impact
DEBRIS = []
for i in range(56):
    a = math.radians(R.uniform(110, 250))
    sp = R.uniform(2.2, 8.5)
    DEBRIS.append(dict(v=v2((math.cos(a) * sp, math.sin(a) * sp * 0.9 - 1.6)), size=R.uniform(0.8, 3.6),
                       ph=R.uniform(0, 6), spin=R.uniform(-0.6, 0.6), hot=R.uniform(0, 1) < 0.25,
                       shape=[R.uniform(0.7, 1.15) for _ in range(6)], floor=R.uniform(160, 176),
                       start=R.uniform(0, 1.5)))
EXIT = []
for i in range(22):
    a = math.radians(R.uniform(-60, 30))
    sp = R.uniform(2.5, 7)
    EXIT.append(dict(v=v2((math.cos(a) * sp, math.sin(a) * sp - 1.0)), size=R.uniform(0.8, 2.6),
                     ph=R.uniform(0, 6), spin=R.uniform(-0.6, 0.6), shape=[R.uniform(0.7, 1.15) for _ in range(6)]))
EXIT_P = v2((292, 104))
PUFFS = [dict(off=v2((R.uniform(-10, 6), R.uniform(-9, 9))), r0=R.uniform(3, 6), grow=R.uniform(6, 13),
              v=v2((R.uniform(-1.3, 0.2), R.uniform(-0.6, 0.15))), t0=R.uniform(0, 4), life=R.uniform(16, 30))
         for _ in range(26)]
GRAV = 0.42


def debris(sc, t):
    for d in DEBRIS:
        tt = t - d["start"]
        if tt < 0:
            continue
        p = HOLE + d["v"] * tt + v2((0, 0.5 * GRAV * tt * tt))
        if p[1] > d["floor"]:
            p = v2((p[0], d["floor"]))
            # settle: only the bigger chunks stay as rubble
            if d["size"] < 1.6:
                continue
            draw_rock(sc, p, d["size"] * 0.8, d["ph"], d["shape"], "FX")
            continue
        hot = 0.0
        if d["hot"]:
            hot = max(0.0, 1 - tt / 14)
            prev = HOLE + d["v"] * (tt - 1.2) + v2((0, 0.5 * GRAV * (tt - 1.2) ** 2))
            _px(sc, line_pts(prev, p), "mg4" if hot > 0.5 else "mg2", "FX")
        draw_rock(sc, p, d["size"], d["ph"] + d["spin"] * tt, d["shape"], "FX", hot=hot)
    for d in EXIT:
        if t < 0.5:
            continue
        p = EXIT_P + d["v"] * t + v2((0, 0.5 * GRAV * t * t))
        if p[1] < 175:
            draw_rock(sc, p, d["size"], d["ph"] + d["spin"] * t, d["shape"], "Back FX")


def _cloud(sc, puffs, layer, warm=0.0):
    """puffs: list of (centre, radius, age 0..1). Drawn as one merged mass, lit from the upper left,
    thinning out with a dither as it ages."""
    U = np.zeros((CH, CW), bool)
    core = np.zeros((CH, CW), bool)
    for c, r, k in puffs:
        m = disc(c, r)
        if k > 0.5:
            lv = (k - 0.5) / 0.5
            m &= (((XS * 2 + YS * 3) % 7) / 7.0 >= lv) | (((XS + YS) % 2 == 0) & (lv < 0.5))
        U |= m
        if k < 0.25:
            core |= disc(c + v2((1, 1.5)), r * 0.4)
    if not U.any():
        return
    sc.put(U, "rk2", layer)
    for c, r, k in sorted(puffs, key=lambda p: p[0][1]):
        sc.put(disc(c + v2((-0.15 * r, -0.2 * r)), 0.8 * r) & U, "rk3", layer)
        sc.put(disc(c + v2((-0.38 * r, -0.45 * r)), 0.42 * r) & U, "rk4", layer)
    sc.put(U & ~nb(U, 0, 3), "rk2", layer)                   # underside in shadow
    sc.put(U & ~nb(U, 1, 1), "rk1", layer)
    sc.put(U & ~nb(U, -1, -1), "rk5", layer)
    if warm > 0:
        hot = core & U
        sc.put(hot & ((XS + YS) % 2 == 0), "mg1", layer)
        sc.put(hot & erode(hot) & ((XS + YS) % 2 == 0), "mg2", layer)
        sc.put(hot & erode(erode(hot)), "mg3" if warm > 0.5 else "mg2", layer)
        if warm > 0.6:
            sc.put(hot & erode(erode(erode(hot))), "mg5", layer)


def erode(m):
    return m & nb(m, 1, 0) & nb(m, -1, 0) & nb(m, 0, 1) & nb(m, 0, -1)


def dust_cloud(sc, t, layer="FX"):
    items = []
    for d in PUFFS:
        a = t - d["t0"]
        if a < 0 or a > d["life"]:
            continue
        k = a / d["life"]
        r = d["r0"] + d["grow"] * (1 - math.exp(-a / 5))
        c = HOLE + d["off"] * (1 + a * 0.12) + d["v"] * a + v2((0, -a * 0.12))
        items.append((c, r, k))
    _cloud(sc, items, layer, warm=max(0.0, 1 - t / 8))


def impact_flash(sc, t):
    c = HOLE
    if t == 0:
        sc.put(disc(c, 16), "white", "FX")
        sc.put(outline_of(disc(c, 16)) | outline_of(disc(c, 17)), "mg6", "FX")
        for k in range(12):
            a = k * 30 + 9
            ln = 44 if k % 3 == 0 else 24
            _px(sc, line_pts(c, c + unit(a) * ln), "mg7", "FX")
        sc.light(np.clip(1 - np.hypot(XS - c[0], YS - c[1]) / 220, 0, 1) ** 1.2, C["mg6"], 0.9)
    elif t < 3:
        sc.put(disc(c, 10 - t * 3), "mg7", "FX")
        sc.put(outline_of(disc(c, 10 - t * 3)), "mg5", "FX")
        sc.light(np.clip(1 - np.hypot(XS - c[0], YS - c[1]) / 140, 0, 1) ** 1.5, C["mg4"], 0.9 - t * 0.25)


def shockwave(sc, t):
    if t < 1 or t > 9:
        return
    for k, lag in enumerate((0, 2.5)):
        tt = t - lag
        if tt < 1:
            continue
        r = 6 + tt * 10.5
        x, y = _ring(HOLE, (r, 0), (0, r * 0.9), 1.0, 0, 2 * math.pi, dens=1.4)
        xs, ys = _uniq(x, y)
        if tt > 4:
            keep = (xs + 2 * ys) % 3 != 0
            xs, ys = xs[keep], ys[keep]
        if tt > 6.5:
            keep = (xs + ys) % 2 == 0
            xs, ys = xs[keep], ys[keep]
        col = ("white" if tt < 2.5 else "air2") if k == 0 else "air1"
        _px(sc, (xs, ys), col, "FX")
        if tt < 4 and k == 0:
            x2, y2 = _ring(HOLE, (r - 1.5, 0), (0, (r - 1.5) * 0.9), 1.0, 0, 2 * math.pi, dens=1.4)
            _px(sc, _uniq(x2, y2), "air1", "FX")


def sparks(sc, t):
    if t > 6:
        return
    rr = np.random.default_rng(99)
    for k in range(18):
        a = rr.uniform(100, 260) if k % 4 else rr.uniform(-80, 80)
        sp = rr.uniform(5, 11)
        d = unit(a)
        p = HOLE + d * (sp * t + 4)
        q = p - d * (3 + sp * 0.4)
        _px(sc, line_pts(q, p), "mg6" if t < 3 else "mg4", "FX")


def embers(sc, f, t):
    """Slow glowing bits drifting up from the hole after the blast."""
    for k in range(14):
        life = 18 + k % 5 * 3
        a = ((t + k * 4) % life) / life
        if t + k * 4 < life * 0 + k * 4:
            continue
        x = HOLE[0] - 6 + (k * 5) % 14 + math.sin(f * 0.3 + k) * 2 - a * 10
        y = HOLE[1] + 4 - a * 34 - (k % 3) * 2
        col = "mg5" if a < 0.4 else ("mg3" if a < 0.75 else "mg1")
        _px(sc, ([int(round(x))], [int(round(y))]), col, "FX")


def ground_dust(sc, t, x0=80, y0=GROUND_Y, direction=-1):
    """Dust shoved outward from Rudeus' feet by the recoil (mostly backwards)."""
    if t < 0 or t > 14:
        return
    k = t / 14
    puffs = []
    for j in range(7):
        side = direction if j < 5 else -direction
        sp = (2.2 + 0.4 * j) if j < 5 else 1.2
        r = 1.6 + t * 0.35 + (j % 2)
        x = x0 + side * (7 + t * sp)
        puffs.append((v2((x, y0 - 1 - (j % 3) * 0.8 - t * 0.2)), r, k))
    _cloud(sc, puffs, "FX")


# ---------------------------------------------------------------- Rudeus' earth wall
WALL_X0, WALL_X1, WALL_H = 96, 116, 48
WALL_TOP = [(0, 0), (4, -3), (8, 1), (12, -2), (16, 2), (20, -1)]


def earth_wall(sc, rise, cuts, t, layer="FX"):
    """A slab of stone shoved up from the ground. cuts: list of (t_cut, point, angle); once cut, the
    pieces slide apart along each cut and the loose ones drop."""
    if rise <= 0:
        return
    gy = GROUND_Y + 2
    top = gy - WALL_H * smooth01(rise)
    tx = np.interp(XS - WALL_X0, [p[0] for p in WALL_TOP], [p[1] for p in WALL_TOP])
    m = (XS >= WALL_X0) & (XS <= WALL_X1) & (YS >= top + tx) & (YS <= gy)
    sc_t = np.zeros_like(m)
    rgb = np.zeros((CH, CW, 3), np.float32)
    rgb[:] = C["rk3"]
    strata = np.mod(YS - top + (XS * 0.3).astype(int), 7) == 0
    rgb[strata] = C["rk2"]
    rgb[m & ~nb(m, -1, 0)] = C["rk5"]
    rgb[m & ~nb(m, 0, -1)] = C["rk5"]
    rgb[m & ~nb(m, 1, 0)] = C["rk1"]
    rgb[m & ((XS - WALL_X0) < 4) & ~strata] = C["rk4"]
    outline = outline_of(m)
    # which side of every active cut each pixel is on -> piece id
    pid = np.zeros((CH, CW), np.int64)
    offx = np.zeros((CH, CW), np.float32)
    offy = np.zeros((CH, CW), np.float32)
    active = [(i, c) for i, c in enumerate(cuts) if t >= c[0]]
    for i, (tc, pt, a) in active:
        n = v2((-math.sin(math.radians(a)), math.cos(math.radians(a))))
        side = ((XS - pt[0]) * n[0] + (YS - pt[1]) * n[1]) > 0
        pid += side.astype(np.int64) << i
        sep = min(2.5, (t - tc) * 0.9)
        offx += np.where(side, 1, -1) * n[0] * sep * 0.5
        offy += np.where(side, 1, -1) * n[1] * sep * 0.5
    # loose pieces fall once enough cuts are in
    if len(active) >= 3:
        ids = np.unique(pid[m])
        for pidv in ids:
            pm = m & (pid == pidv)
            if (pm & (YS >= gy - 1)).any():
                continue
            t0 = max(c[0] for c in cuts[:len(active)]) + 1 + (int(pidv) * 7) % 4
            if t > t0:
                drop = 0.35 * (t - t0) ** 2
                offy[pm] += drop
                offx[pm] += ((int(pidv) * 13) % 5 - 2) * 0.3 * (t - t0)
    ys, xs = np.nonzero(m)
    dy = np.round(offy[ys, xs]).astype(int)
    dx = np.round(offx[ys, xs]).astype(int)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < CH) & (nx >= 0) & (nx < CW) & (ny <= gy + M + 4)
    moved = np.zeros_like(m)
    moved[ny[ok], nx[ok]] = True
    out_rgb = np.zeros_like(rgb)
    out_rgb[ny[ok], nx[ok]] = rgb[ys[ok], xs[ok]]
    # dark crack where pieces have parted
    sc.put(outline_of(moved), "rk0", layer)
    sc.put_rgb(moved, out_rgb, layer)
    for i, (tc, pt, a) in active:
        if t - tc <= 1:
            d = unit(a)
            xs2, ys2 = line_pts(v2(pt) - d * 30, v2(pt) + d * 30)
            col = "white" if t == tc else "dm6"
            sc.put_px(xs2, ys2, col, layer)
    if rise < 1:
        # dirt thrown up at the base while it rises
        for k in range(6):
            x = WALL_X0 - 3 + k * 5
            sc.put_px([x, x + 1], [gy - 2 - (k % 3) - rise * 4] * 2, "dt3", layer)

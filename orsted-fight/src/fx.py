"""World-space effects. Every public function has the Event signature fn(world, t, ctx, **kw),
where t is the frame number relative to the event's start."""
import math
import numpy as np
from px import v2, unit, rot, norm, nb, ring4, dilate4, dilate8, ang_of
from palette import FX


def _P(ctx, spec):
    """Resolve a point spec: a tuple, or ("Actor", "attachment", (dx, dy))."""
    if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
        name, key, off = spec
        return ctx["att"][name][key] + v2(off)
    if callable(spec):
        return v2(spec(ctx))
    return v2(spec)


# ---------------------------------------------------------------- small primitives
def star(wd, c, size, col="w", inner="c2", layer="FX"):
    cv = wd.cv
    x, y = c
    m = cv.empty()
    for i in range(-size, size + 1):
        m |= cv.point((x + i, y)) | cv.point((x, y + i))
    wd.put(m, FX[col], layer)
    if size >= 2:
        d = cv.empty()
        for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            d |= cv.point((x + dx, y + dy))
        wd.put(d, FX[inner], layer)


def puff(wd, c, r, phase=0, layer="FX", light="dust_l", mid="dust_m"):
    cv = wd.cv
    full = cv.disc(c, r)
    core = cv.disc(v2(c) + v2((-0.6, -0.8)), r * 0.72)
    wd.put(full & cv.checker(phase) & ~core, FX[mid], layer)
    wd.put(core, FX[light], layer)
    wd.put(core & ~nb(core, 1, 1), FX[mid], layer)


def streak(wd, p0, p1, col="c1", layer="FX"):
    wd.put(wd.cv.line(p0, p1), FX[col], layer)


# ---------------------------------------------------------------- events
def glint(wd, t, ctx, at, size=2, col="w"):
    s = [1, 2, size + 1, size, 1][min(t, 4)]
    star(wd, _P(ctx, at), s, col)


def eye_glow(wd, t, ctx, actor="Orsted", dur=8):
    """Golden flare off Orsted's eyes."""
    for key in ("eyes", "eye_n"):
        c = ctx["att"][actor][key]
        s = [1, 2, 3, 3, 2, 2, 1, 1, 1][min(t, 8)]
        cv = wd.cv
        m = cv.empty()
        for i in range(-s * 2, s * 2 + 1):
            m |= cv.point((c[0] + i, c[1]))
        for i in range(-s, s + 1):
            m |= cv.point((c[0], c[1] + i))
        wd.put(m, FX["g2"], "FX")
        wd.put(cv.point(c), FX["w"], "FX")


def impact(wd, t, ctx, at, size=1.0, warm=False, seed=0):
    """Hit spark: a flash disc, radiating spikes, then flying embers."""
    c = _P(ctx, at)
    cv = wd.cv
    rng = np.random.default_rng(seed)
    hot = ("o1", "o2", "o3") if warm else ("w", "c1", "c2")
    if t == 0:
        m = cv.disc(c, 4 * size)
        wd.put(m, FX[hot[0]], "FX")
        wd.put(ring4(m), FX[hot[1]], "FX")
    n = 9
    for i in range(n):
        a = i * 360 / n + rng.uniform(-14, 14)
        ln = (7 + rng.uniform(0, 7)) * size
        r0 = (2 + t * 4.5) * size
        r1 = r0 + ln * max(0.0, 1 - t * 0.28)
        if r1 - r0 < 1:
            continue
        p0, p1 = c + unit(a) * r0, c + unit(a) * r1
        streak(wd, p0, p1, hot[1] if i % 2 else hot[0])
    if 1 <= t <= 6:
        for i in range(7):
            a = rng.uniform(0, 360)
            sp = rng.uniform(2.5, 5.5) * size
            p = c + unit(a) * sp * t + v2((0, 0.35 * t * t))
            wd.put(cv.point(p), FX[hot[min(2, t // 2)]], "FX")
    if t <= 1:
        star(wd, c, int(3 + 3 * size) - t * 2, hot[0], hot[2])


def dust(wd, t, ctx, at, n=3, spread=8, rise=0.6, drift=(0.0, 0.0), r=3.0, seed=0, life=10, layer="FX"):
    c = _P(ctx, at)
    rng = np.random.default_rng(seed)
    for i in range(n):
        off = v2((rng.uniform(-spread, spread), rng.uniform(-2, 1)))
        vel = v2((rng.uniform(-0.6, 0.6), -rng.uniform(0.2, rise))) + v2(drift)
        k = t / life
        if k >= 1:
            continue
        rr = r * (0.6 + 0.8 * k) * rng.uniform(0.8, 1.2) * (1 - k * 0.55)
        p = c + off + vel * t
        if k > 0.6:
            wd.put(wd.cv.disc(p, rr) & wd.cv.checker(i + t), FX["dust_m"], layer)
        else:
            puff(wd, p, rr, i + t, layer)


def skid(wd, t, ctx, actor, life=12, side=-1):
    """Dust kicked up behind a sliding character (spawned along their path)."""
    r = ctx["root"][actor]
    fc = ctx["pose"][actor].get("face", 1)
    for i in range(3):
        p = r + v2((side * fc * (4 + i * 6), -1 - i * 0.5))
        rr = 3.5 - i * 0.9 + (t % 2) * 0.5
        puff(wd, p, max(rr, 1.0), t + i)


def speed_lines(wd, t, ctx, actor, n=6, length=26, spread=22, seed=0, col="c1", height=(-50, -6)):
    """Horizontal streaks trailing a dashing character."""
    r = ctx["root"][actor]
    fc = ctx["pose"][actor].get("face", 1)
    rng = np.random.default_rng(seed + t * 13)
    for i in range(n):
        y = r[1] + rng.uniform(*height)
        x0 = r[0] - fc * rng.uniform(8, spread)
        ln = rng.uniform(0.5, 1.0) * length
        streak(wd, (x0, y), (x0 - fc * ln, y), col if i % 3 else "w")


def magic_circle(wd, t, ctx, at, r=9, ry=1.0, col=("c1", "c3", "c2"), spin=12, grow=4, layer="FX", seed=0):
    """Rotating rune ring. ry < 1 squashes it into a ground circle; rx scale < 1 gives a side-on circle."""
    c = _P(ctx, at)
    cv = wd.cv
    k = min(1.0, (t + 1) / grow)
    rad = r * k
    rx_s, ry_s = (1.0, ry) if ry <= 1 else (1.0 / ry, 1.0)
    X, Y = cv.XS, cv.YS
    d = np.hypot((X - c[0]) / max(rx_s, 1e-3), (Y - c[1]) / max(ry_s, 1e-3))
    outer = (d <= rad) & (d > rad - 1.2)
    inner = (d <= rad * 0.66) & (d > rad * 0.66 - 1.0)
    wd.put(outer, FX[col[0]], layer)
    wd.put(inner & cv.checker(t), FX[col[1]], layer)
    # runes: dots orbiting between the rings
    for i in range(8):
        a = math.radians(i * 45 + t * spin)
        p = c + v2((math.cos(a) * rad * 0.83 * rx_s, math.sin(a) * rad * 0.83 * ry_s))
        wd.put(cv.point(p), FX[col[2] if i % 2 else col[0]], layer)
    # a hexagram of lines inside
    for i in range(3):
        a0 = math.radians(i * 120 - t * spin * 0.7)
        a1 = a0 + math.radians(120)
        p0 = c + v2((math.cos(a0) * rad * 0.62 * rx_s, math.sin(a0) * rad * 0.62 * ry_s))
        p1 = c + v2((math.cos(a1) * rad * 0.62 * rx_s, math.sin(a1) * rad * 0.62 * ry_s))
        wd.put(cv.line(p0, p1) & cv.checker(), FX[col[1]], layer)


def rock(wd, c, r, spin, layer="FX"):
    """A faceted stone bullet, spinning."""
    cv = wd.cv
    pts = []
    for i in range(6):
        a = spin + i * 60 + (11 if i % 2 else -7)
        rr = r * (1.0 if i % 2 else 0.86)
        pts.append(v2(c) + unit(a) * rr)
    m = cv.polygon(pts)
    wd.put(ring4(m), FX["rock_o"], layer)
    wd.put(m, FX["rock_b"], layer)
    wd.put(m & ~nb(m, 1, 1), FX["rock_d"], layer)
    wd.put(m & ~nb(m, -1, -1), FX["rock_l"], layer)
    hl = cv.disc(v2(c) + v2((-r * 0.35, -r * 0.35)), max(r * 0.28, 0.6)) & m
    wd.put(hl, FX["rock_h"], layer)
    facet = cv.line(v2(c) + unit(spin + 30) * r * 0.8, v2(c) + unit(spin + 200) * r * 0.5) & m
    wd.put(facet, FX["rock_d"], layer)


def stone_charge(wd, t, ctx, at, r_max=5.0, grow=10):
    """Stone Cannon forming at the staff: grit spirals in, the rock grows and spins."""
    c = _P(ctx, at)
    cv = wd.cv
    k = min(1.0, t / grow)
    for i in range(10):
        a = i * 36 + t * 31
        rr = 16 * (1 - ((t * 0.12 + i * 0.1) % 1.0))
        p = c + unit(a) * rr
        wd.put(cv.point(p), FX["rock_l" if i % 2 else "dust_l"], "FX")
    if k > 0.1:
        rock(wd, c, 1.5 + (r_max - 1.5) * k, t * 37)
    if t % 4 < 2:
        star(wd, c + v2((-r_max - 2, -r_max - 1)), 1, "c1")


def stone_shot(wd, t, ctx, p0, p1, frames=9, r=5.0, stop=1.0, arc=0.0):
    """Stone Cannon flying from p0 to p1 over `frames` frames along an arc (px of lift at mid-flight), with a
    speed trail. It vanishes once it has covered `stop` of the path (Ran Ma takes it apart there)."""
    a, b = _P(ctx, p0), _P(ctx, p1)

    def at(s):
        return a + (b - a) * s + v2((0, -arc * math.sin(math.pi * s)))
    s = t / frames
    if s > stop:
        return
    c = at(s)
    d = norm(at(s + 0.02) - at(max(s - 0.02, 0)))
    perp = v2((-d[1], d[0]))
    for i, off in enumerate((0, -2.5, 2.5, -1.2, 1.2)):
        ln = (18 if off == 0 else 12) + (t % 2) * 4
        s0 = c - d * (r + 1) + perp * off
        streak(wd, s0, s0 - d * ln, ("w", "c1", "c2", "c1", "c2")[i])
    for i in range(3):
        q = at(max(s - 0.05 * (i + 1), 0))
        puff(wd, q + perp * ((i % 2) * 2 - 1), 2.2 - i * 0.5, t + i)
    rock(wd, c, r, t * 61)


def stone_point(p0, p1, s, arc):
    p0, p1 = v2(p0), v2(p1)
    return p0 + (p1 - p0) * s + v2((0, -arc * math.sin(math.pi * s)))


def ranma(wd, t, ctx, at, rings=4, period=3, r_max=26, squash=1.25):
    """Ran Ma — disturbance waves pulsing out of an open palm: dithered gold/violet ripples."""
    c = _P(ctx, at)
    cv = wd.cv
    for i in range(rings):
        age = t - i * period
        if age < 0:
            continue
        rr = 3 + age * 3.2
        if rr > r_max:
            continue
        k = rr / r_max
        col = ("g1", "g2", "p2", "p3")[min(3, int(k * 4))]
        m = cv.ring(c, rr, rr - (2.0 if k < 0.5 else 1.2), ry_scale=squash)
        if k > 0.55:
            m &= cv.checker(t)
        wd.put(m, FX[col], "FX")
    if t % 2 == 0:
        star(wd, c, 2, "g1", "p2")
    # warped air: short wavy arcs
    for i in range(5):
        a = i * 72 + t * 23
        p = c + unit(a) * (6 + (t * 2 + i * 3) % 14)
        wd.put(cv.line(p, p + unit(a + 90) * 2.5), FX["p1"], "FX")


def crumble(wd, t, ctx, at, n=14, seed=3, power=1.0, dirx=1.0):
    """The stone falls apart: chunks fly out and drop, grit sparkles, dust blooms."""
    c = _P(ctx, at)
    rng = np.random.default_rng(seed)
    cv = wd.cv
    for i in range(n):
        a = rng.uniform(0, 360)
        sp = rng.uniform(1.0, 3.2) * power
        vel = unit(a) * sp + v2((dirx * rng.uniform(0.5, 1.6), -rng.uniform(0.0, 1.2)))
        p = c + vel * t + v2((0, 0.22 * t * t))
        size = rng.uniform(0.8, 2.4) * max(0.0, 1 - t / 22)
        if size < 0.6:
            continue
        if size > 1.2:
            rock(wd, p, size, a + t * 40)
        else:
            wd.put(cv.point(p), FX["rock_l"], "FX")
    if t < 14:
        k = t / 14
        for i in range(4):
            off = unit(i * 90 + 45) * (2 + t * 1.1)
            puff(wd, c + off, 3.2 * (1 - k * 0.5), t + i)
    if t < 8:
        for i in range(6):
            a = i * 60 + t * 20
            wd.put(cv.point(c + unit(a) * (3 + t * 2.2)), FX["p1" if i % 2 else "g1"], "FX")


def quagmire(wd, t, ctx, at, w=26, grow=6, sink_layer="back", bubbles=True):
    """Mud pool spreading on the ground (call twice: back layer for the pool, front layer for the lip)."""
    c = _P(ctx, at)
    cv = wd.cv
    k = min(1.0, (t + 1) / grow)
    rx, ry = w * k, 4.5 * k
    if rx < 1:
        return
    pool = cv.ellipse(c, rx, ry)
    if sink_layer == "back":
        wd.put(ring4(pool), FX["mud_o"], "FX (back)")
        wd.put(pool, FX["mud_b"], "FX (back)")
        wd.put(pool & ~nb(pool, 0, -1), FX["mud_l"], "FX (back)")
        wd.put(pool & cv.ellipse(c + v2((2, 1)), rx * 0.6, ry * 0.5), FX["mud_d"], "FX (back)")
        # swirling streaks
        for i in range(3):
            a = t * 25 + i * 120
            p = c + v2((math.cos(math.radians(a)) * rx * 0.6, math.sin(math.radians(a)) * ry * 0.5))
            wd.put(cv.line(p, p + v2((4, 0))) & pool, FX["mud_h"], "FX (back)")
    else:
        # front lip of the pool hides sunk feet
        lip = pool & (cv.YS > c[1] - 0.5)
        wd.put(lip, FX["mud_b"], "FX")
        wd.put(lip & ~nb(lip, 0, 1), FX["mud_o"], "FX")
        wd.put(lip & ~nb(lip, 0, -1), FX["mud_l"], "FX")
        if bubbles:
            rng = np.random.default_rng(t // 3)
            for i in range(3):
                p = c + v2((rng.uniform(-rx * 0.8, rx * 0.8), rng.uniform(-1, 2)))
                b = cv.disc(p, 1.0 + (t + i) % 3 * 0.4)
                wd.put(ring4(b), FX["mud_o"], "FX")
                wd.put(b, FX["mud_h"], "FX")


def shockwave(wd, t, ctx, at, r_max=90, speed=11, seed=4, debris=True, cols=("w", "g1", "g2", "p2")):
    """Mana burst: a bright expanding ring plus a ground ring, with rocks thrown up."""
    c = _P(ctx, at)
    cv = wd.cv
    rr = 6 + t * speed
    if rr < r_max:
        k = rr / r_max
        th = max(1.2, 5 * (1 - k))
        m = cv.ring(c, rr, rr - th, ry_scale=0.85)
        if k > 0.6:
            m &= cv.checker(t)
        wd.put(m, FX[cols[min(3, int(k * 4))]], "FX")
        inner = cv.ring(c, rr * 0.7, rr * 0.7 - 1.0, ry_scale=0.85) & cv.checker(t + 1)
        wd.put(inner, FX[cols[min(3, int(k * 4) + 1)]], "FX")
        from world import GROUND
        g = cv.ring(v2((c[0], GROUND)), rr * 1.1, rr * 1.1 - 2, ry_scale=0.12)
        wd.put(g, FX["dust_l"], "FX")
    if t <= 1:
        m = cv.disc(c, 10 - t * 3)
        wd.put(m, FX["w"], "FX")
        wd.put(ring4(m), FX["g1"], "FX")
    if debris:
        from world import GROUND
        rng = np.random.default_rng(seed)
        for i in range(16):
            x0 = c[0] + rng.uniform(-20, 20)
            vx = (x0 - c[0]) * 0.25 + rng.uniform(-1, 1)
            vy = -rng.uniform(3.0, 6.5)
            y = GROUND - 1 + vy * t + 0.45 * t * t
            if y > GROUND:
                continue
            p = v2((x0 + vx * t, y))
            sz = rng.uniform(0.9, 2.2)
            if sz > 1.3:
                rock(wd, p, sz, i * 50 + t * 30)
            else:
                wd.put(cv.point(p), FX["rock_l"], "FX")
        if t < 12:
            for i in range(6):
                x = c[0] + (i - 2.5) * 12 * (1 + t * 0.15)
                puff(wd, (x, GROUND - 2 - t * 0.4), 4.5 - t * 0.3, t + i)


def palm_burst(wd, t, ctx, at, dirx=-1):
    """Directional blast from Orsted's palm strike: a cone of speed streaks and a bright core."""
    c = _P(ctx, at)
    cv = wd.cv
    rng = np.random.default_rng(t * 7 + 1)
    for i in range(14):
        a = 180 * (dirx < 0) + rng.uniform(-28, 28)
        r0 = 4 + t * 6
        ln = rng.uniform(14, 40) * max(0.0, 1 - t * 0.18)
        if ln < 2:
            continue
        p0 = c + unit(a) * r0
        streak(wd, p0, p0 + unit(a) * ln, ("w", "c1", "g1", "g2")[i % 4])
    if t < 3:
        m = cv.disc(c, 8 - t * 2)
        wd.put(m, FX["w"], "FX")
        wd.put(ring4(m), FX["g2"], "FX")
        wd.put(cv.ring(c, 14 + t * 8, 12 + t * 8, ry_scale=1.0) & cv.checker(), FX["g1"], "FX")


def crater(wd, t, ctx, at, w=18, seed=8):
    """Debris and dust where somebody slammed into the cliff / ground."""
    c = _P(ctx, at)
    rng = np.random.default_rng(seed)
    cv = wd.cv
    for i in range(10):
        a = rng.uniform(-170, -10)
        sp = rng.uniform(1.5, 4.0)
        p = c + unit(a) * sp * t + v2((0, 0.35 * t * t))
        sz = rng.uniform(0.8, 2.0) * max(0.0, 1 - t / 18)
        if sz > 1.2:
            rock(wd, p, sz, t * 40 + i * 30)
        elif sz > 0.4:
            wd.put(cv.point(p), FX["rock_l"], "FX")
    if t < 16:
        for i in range(5):
            puff(wd, c + v2(((i - 2) * w / 4, -t * 0.5 - (i % 2) * 3)), 4.5 + t * 0.15 - (t > 10) * 1.5, t + i)


def motes(wd, t, ctx, n=26, seed=1, layer="FX"):
    """Ambient wind: specks and petals drifting right-to-left across the pass."""
    from world import W, H
    rng = np.random.default_rng(seed)
    cv = wd.cv
    for i in range(n):
        x0, y0 = rng.uniform(0, W + 60), rng.uniform(90, 250)
        sp = rng.uniform(1.2, 3.0)
        x = (x0 - sp * t) % (W + 60) - 30
        y = y0 + math.sin(t * 0.15 + i) * 4
        col = "dust_l" if i % 3 else "p1"
        wd.put(cv.point((x, y)), FX[col], layer)
        if i % 4 == 0:
            wd.put(cv.point((x + 1, y)), FX["dust_m"], layer)


def ghost(wd, t, ctx, actor, lags=(1, 2, 3), lo=0, hi=10 ** 6, col="p3", col2="p2", layer="FX (back)"):
    """Afterimages: the actor's silhouette at earlier frames, dithered, in violet."""
    scene = ctx["scene"]
    a = next(x for x in scene.actors if x.layer == actor)
    frames = [ctx["f"] - l for l in lags if lo <= ctx["f"] - l <= hi]
    for j, f in enumerate(frames):
        p = a.pose(f)
        a.cv.clear()
        a.m.render(a.cv, p, None)
        sil = a.cv.rgba[:, :, 3] > 0
        if not sil.any():
            continue
        r = a.root(p)
        fc = p.get("face", 1)
        tmp = np.zeros_like(a.cv.rgba)
        edge = sil & ~(nb(sil, 1, 0) & nb(sil, -1, 0) & nb(sil, 0, 1) & nb(sil, 0, -1))
        tmp[sil & a.cv.checker(j)] = FX[col]
        tmp[edge] = FX[col2]
        a.cv.rgba[:] = tmp
        a.cv.owner[:] = 0
        wd.blit(a.cv, r, fc, layer)

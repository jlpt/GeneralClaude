"""Timeline: what happens on every frame of the Stone Cannon cast."""
import math
import numpy as np
from canvas import (Scene, C, L, LAYERS, XS, YS, W, H, M, CW, CH, v2, unit, phase, ease_out, ease_in, lerp,
                    smooth01, line_pts, hexc, BAYER)
import world
import fx
from rudeus import draw_rudeus_full

N = 124
# key frames
RAISE = (12, 20)
CH1 = (20, 50)
CH2 = (50, 68)
FLASH = (68, 70)
FLY = {70: 0.24, 71: 0.54, 72: 0.84}
HIT = 73
RECOVER = (100, 112)
FADE_OUT = 116


def duration(f):
    if 64 <= f < 68:
        return 80
    if f in (68, 69):
        return 50
    if f in FLY:
        return 40
    if f == HIT:
        return 110
    if HIT < f <= HIT + 3:
        return 50
    return 60


DURATIONS = [duration(f) for f in range(N)]

HIP = v2((80, 140))
SH_NEAR = HIP + v2((-3.2, -13.0))
SH_FAR = HIP + v2((3.4, -13.6))
RAISED = (v2((69, 104)), v2((93, 104)))
THRUST = (HIP + v2((20, -22)), HIP + v2((23, -25)))


def _polar(sh, ang, d):
    return sh + unit(ang) * d


def arm_targets(f):
    """(near hand, far hand), (near bend, far bend)"""
    if f < RAISE[0]:
        breathe = math.sin(f * 0.5) * 0.4
        return (_polar(SH_NEAR, 100, 16 + breathe), _polar(SH_FAR, 80, 16 + breathe)), (1, 1)
    if f < RAISE[1] + 1:
        t = smooth01(phase(f, RAISE[0], RAISE[1]))
        an = lerp(100, 251.3, t)
        af = lerp(80, -66.8, t)
        d = lerp(16, 24.3, t) + math.sin(t * math.pi) * 3
        return (_polar(SH_NEAR, an, d), _polar(SH_FAR, af, d)), (1 if an < 175 else -1, 1)
    if f < 70:
        jit = 0.0
        if f >= CH2[0]:
            jit = 0.7 if (f % 2) else -0.7
        n, fa = RAISED
        return (n + v2((jit, 0)), fa + v2((-jit, 0))), (-1, 1)
    if f < RECOVER[0]:
        return THRUST, (1, -1)
    t = smooth01(phase(f, RECOVER[0], RECOVER[1]))
    idle = (_polar(SH_NEAR, 100, 16), _polar(SH_FAR, 80, 16))
    n = lerp(THRUST[0], idle[0], t) + v2((0, math.sin(t * math.pi) * 3))
    fa = lerp(THRUST[1], idle[1], t) + v2((0, math.sin(t * math.pi) * 3))
    return (n, fa), (1, -1 if t < 0.5 else 1)


def pose(f):
    hands, bends = arm_targets(f)
    p = dict(hands=hands, bends=bends, hip=HIP.copy(), spread=f >= RAISE[0] + 4 and f < RECOVER[1])
    p["face"] = "calm" if (f < 14 or f >= 94) else ("focus" if f < 70 else "shout")
    if f < CH1[0]:
        p["flutter"] = 0.4 * math.sin(f * 0.3)
        p["sway"] = 0.5 * math.sin(f * 0.25)
        p["sleeve"] = phase(f, RAISE[0] + 2, RAISE[1])
    elif f < 70:
        ch = phase(f, CH1[0], CH2[1])
        p["flutter"] = (0.6 + 0.8 * ch) * math.sin(f * 0.9)
        p["flare"] = 1.4 * ch
        p["sway"] = math.sin(f * 0.7) * ch
        p["sleeve"] = 1.0
        p["hair_lift"] = 1 if f < CH2[0] else 2
    else:
        k = math.exp(-(f - 70) / 9)
        p["flutter"] = 3.6 * k + 0.3 * math.sin(f * 0.5)
        p["flare"] = 1.4 * k
        p["sleeve"] = 0.3 * (1 - phase(f, RECOVER[0], RECOVER[1]))
        lean = -6 * (1 - smooth01(phase(f, 84, 104)))
        p["lean"] = lean
        p["hip"] = HIP + v2((-2 * min(1, (f - 69) / 2), 0))
        p["sway"] = math.sin(f * 0.35) * k * 2
    if f >= 70 and f < RECOVER[1]:
        # thrust hands follow the hip slide
        p["hands"] = tuple(h + v2((-2, 0)) for h in p["hands"])
    return p


def shake(f):
    if f < CH1[0] or f in (68, 69):
        a = 0.0
    elif f < CH2[0]:
        a = 0.9 * phase(f, CH1[0], CH2[0])
    elif f < 68:
        a = 0.9 + 1.2 * phase(f, CH2[0], 68)
    elif f < HIT:
        a = 3.0
    else:
        a = 7.0 * math.exp(-(f - HIT) / 5.0)
    if a < 0.35:
        return (0, 0)
    r = np.random.default_rng(f * 17 + 3)
    v = r.uniform(-1, 1, 2) * a
    return (int(round(v[0])), int(round(v[1])))


def darkness(f):
    if f < CH1[0]:
        return 0.0
    if f < 68:
        return 0.15 + 0.5 * phase(f, CH1[0], 66)
    if f < HIT + 2:
        return 0.65
    return 0.65 * (1 - smooth01(phase(f, HIT + 2, 100)))


def render(f):
    sc = Scene()
    dark = darkness(f)
    hit = dict(t=f - HIT) if f >= HIT else None
    world.draw_sky(sc, f, dark)
    world.draw_land(sc, f, dark)
    world.draw_crag(sc, f, hit=hit, dark=dark)

    # pebbles resting / lifting are partly behind him
    if f < 70:
        fx.rocks(sc, f, rumble=phase(f, CH1[0], 30), flying=False)

    p = pose(f)
    draw_rudeus_full(sc, p)
    body = (sc.owner >= L["Back arm"]) & (sc.owner <= L["Front arm"])
    sc.rgb[body] *= 1 - 0.3 * dark

    # ---- light pass (affects world + Rudeus)
    ch = phase(f, 22, 66)
    if CH1[0] <= f < 70:
        d = np.hypot(XS - fx.CORE[0], YS - fx.CORE[1])
        sc.light(np.clip(1 - d / 115, 0, 1) / (1 + (d / 24) ** 2), C["mg4"], 0.18 + 0.75 * ch, dither=34, mul=0.5)
        dg = np.hypot((XS - fx.FEET[0]) / 3.5, (YS - fx.FEET[1]))
        sc.light(np.clip(1 - dg / 26, 0, 1) ** 2, C["mg3"], 0.35 * phase(f, 14, 26) + 0.3 * ch, dither=34)
    if f in FLY:
        c = fx.path_point(FLY[f])
        d = np.hypot(XS - c[0], YS - c[1])
        sc.light(np.clip(1 - d / 120, 0, 1) / (1 + (d / 26) ** 2), C["mg5"], 0.9, dither=34, mul=0.3)
    if HIT <= f < HIT + 30:
        t = f - HIT
        d = np.hypot(XS - world.HOLE[0], YS - world.HOLE[1])
        sc.light(np.clip(1 - d / 150, 0, 1) / (1 + (d / 34) ** 2), C["mg4"], 0.9 * math.exp(-t / 6), dither=34, mul=0.4)

    blast = math.exp(-(f - 70) / 4) if f >= 70 else 0.0
    world.draw_grass_fg(sc, f, blast=blast, blast_x=p["hip"][0], dark=dark)

    # ---- behind Rudeus
    sc.protect = body
    if RAISE[0] + 1 <= f < 78:
        sweep = phase(f, RAISE[0] + 1, 22)
        power = 0.45 + 0.55 * ch if f < 70 else max(0.0, 1 - (f - 70) / 7)
        if f < 74 or f % 2 == 0:
            fx.ground_circle(sc, f, sweep, power, f * 0.05)
    crack_grow = phase(f, 26, 60)
    crack_heat = (0.35 + 0.65 * phase(f, 40, 66)) if f < 70 else max(0.0, 1 - (f - 70) / 18)
    fx.ground_cracks(sc, crack_grow, crack_heat)
    if 22 <= f < 70:
        fx.light_pillars(sc, f, 0.4 + 0.6 * ch)
    aura_p = 0.6 * phase(f, 28, 40) + 0.4 * phase(f, CH2[0], 66) if f < 70 else max(0.0, 1 - (f - 69) / 5) * 0.8
    if f >= 28 and f not in (68, 69):
        fx.aura(sc, f, aura_p)
    sc.protect = None

    # ---- the spell
    if 20 <= f < 70:
        fx.over_circle(sc, f, phase(f, 20, 27), 0.5 + 0.5 * ch, f * 0.05)
    elif 70 <= f < 73:
        k = 1 + (f - 69) * 0.35
        fx.magic_circle(sc, fx.OVER, (17 * k, 0), (0, 4.4 * k), -f * 0.07, 1.0, 0.4, layer="Magic", glow=False)
    if 20 <= f < 68:
        fx.motes(sc, f, phase(f, 20, 32) * (1.0 if f < 62 else 1 - phase(f, 62, 68)), 1.0 + 0.8 * phase(f, 50, 66))
    if 20 <= f < 70:
        fx.rocks(sc, f, resting=False)
    # gravel ball -> bullet
    jitter = v2((0, 0))
    if 60 <= f < 68:
        jitter = v2(((f % 2) * 2 - 1, 0)) * 0.6
    core = fx.CORE + jitter
    if 24 <= f < 57:
        rad = 1 + 6 * ease_out(phase(f, 24, 50), 2)
        if f >= 50:
            rad *= 1 - phase(f, 50, 57)
        fx.gravel_ball(sc, core, rad, f * 0.09, f)
    if 50 <= f < 70:
        s = ease_out(phase(f, 50, 57), 2) * (1 + 0.35 * smooth01(phase(f, 57, 66)))
        spin = f * (0.25 + 0.2 * phase(f, 50, 68))
        heat = phase(f, 58, 68)
        if f >= 52:
            fx.orbit_sparks(sc, core, f, ch, front=False)
        fx.draw_bullet(sc, core, fx.AIM_ANG, s, spin, heat=heat)
        if f >= 56:
            fx.spin_arcs(sc, core, fx.AIM_ANG, s, f)
        if f >= 52:
            fx.orbit_sparks(sc, core, f, ch, front=True)
        if f >= 56:
            fx.crackle(sc, core, f, 1 + (f - 56) // 4, 7)
    for start in (30, 38, 44, 50, 54, 58, 61, 64, 66):
        if start <= f < start + 6:
            fx.implosion_ring(sc, fx.CORE, (f - start) / 6)
    for i in range(3):
        a = 58 + 2 * i
        if a <= f < 70:
            fx.barrel_ring(sc, i, phase(f, a, a + 3), 0.6 + 0.4 * phase(f, a + 2, 68), f * 0.12)
        elif 70 <= f < 74:
            fx.barrel_ring(sc, i, 1, 1, f * 0.12, burst=f - 70)

    # ---- launch
    if 70 <= f < HIT + 5:
        t = f - 70
        fx.muzzle_flash(sc, t)
        if f < HIT:
            s = FLY[f]
            fx.helix_trail(sc, s, f, age=0)
            fx.bullet_streak(sc, s, max(0.0, s - 0.5))
            fx.vapor_rings(sc, s, t)
            fx.speed_lines(sc, t)
            fx.draw_bullet(sc, fx.path_point(s), fx.AIM_ANG, 1.4, f * 0.7, heat=1.0, layer="FX", underglow=0)
        else:
            age = f - HIT
            fx.helix_trail(sc, 1.0, f, age=age + 1)
            fx.bullet_streak(sc, 1.0, 0.45 + 0.13 * age, fade=0.25 + 0.2 * age)
            fx.vapor_rings(sc, 1.0, t)
            if age < 2:
                fx.speed_lines(sc, t)
    if f >= 70:
        fx.ground_dust(sc, f - 70, x0=p["hip"][0])
    # ---- impact
    if f >= HIT:
        t = f - HIT
        fx.debris(sc, t)
        fx.dust_cloud(sc, t)
        fx.sparks(sc, t)
        fx.shockwave(sc, t)
        fx.impact_flash(sc, t)
        if t > 3:
            fx.embers(sc, f, t)

    img, owner = sc.finish(shake(f))
    if f in (68, 69):
        img = impact_frame(img, owner, inverted=(f == 68))
    fade = 0.0
    if f < 4:
        fade = 1 - f / 4
    elif f >= FADE_OUT:
        fade = (f - FADE_OUT + 1) / (N - FADE_OUT)
    if fade > 0:
        bay = BAYER[M:M + H, M:M + W] + 0.47
        m = bay < fade
        img[m] = np.array(hexc("0a0818"), np.uint8)
        owner[m] = L["FX"]
    return img, owner


IMPACT_CREAM = np.array(hexc("fffae2"), np.uint8)
IMPACT_RED = np.array(hexc("a8401a"), np.uint8)
IMPACT_BLACK = np.array(hexc("0a0818"), np.uint8)


def impact_frame(img, owner, inverted):
    """Anime-style impact frame: the scene collapses to stark silhouettes."""
    out = np.empty_like(img)
    lum = img.astype(np.float32) @ np.array([0.3, 0.59, 0.11], np.float32) / 255.0
    fig = ((owner >= L["Back FX"]) & (owner <= L["Magic"])) | (owner == L["Crag"]) | \
        ((owner == L["Sky"]) & (lum > 0.72))
    ys, xs = np.mgrid[0:H, 0:W]
    ang = np.degrees(np.arctan2(ys - fx.CORE[1], xs - fx.CORE[0]))
    rad = np.hypot(ys - fx.CORE[1], xs - fx.CORE[0])
    rays = (np.mod(ang + rad * 0.15, 9.0) < 1.6) & (rad > 36)
    if inverted:
        out[:] = IMPACT_CREAM
        out[rays] = IMPACT_RED
        out[fig] = IMPACT_BLACK
    else:
        out[:] = IMPACT_BLACK
        out[rays] = IMPACT_RED
        out[fig] = IMPACT_CREAM
        edge = fig & ~(np.roll(fig, 1, 0) & np.roll(fig, -1, 0) & np.roll(fig, 1, 1) & np.roll(fig, -1, 1))
        out[edge] = IMPACT_RED
    return out

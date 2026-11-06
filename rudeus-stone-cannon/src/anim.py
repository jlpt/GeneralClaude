"""Timeline: Rudeus' Stone Cannon, and what turns out to be waiting inside the rock."""
import math
import numpy as np
from canvas import (Scene, C, L, XS, YS, W, H, M, CW, CH, v2, unit, phase, ease_out, lerp, smooth01, line_pts,
                    hexc, BAYER, disc, outline_of, dilate4)
import world
import fx
import spin
import shots
import sukuna as SK
from rudeus import draw_rudeus_full

SEGMENTS = [
    ("idle", 12), ("raise", 8), ("gather", 30), ("form", 8), ("spinup", 26), ("cutin", 14), ("lock", 10),
    ("iframe", 2), ("fire", 2), ("track", 14), ("arrive", 1), ("drill", 10), ("blast", 16), ("collapse", 16),
    ("reveal", 22), ("face", 40), ("sign", 14), ("expand", 12), ("shrine", 18), ("slash", 32), ("finale", 10),
]
S, LEN = {}, {}
_acc = 0
for _n, _l in SEGMENTS:
    S[_n], LEN[_n] = _acc, _l
    _acc += _l
N = _acc


def seg(f):
    for n, l in SEGMENTS:
        if S[n] <= f < S[n] + l:
            return n, f - S[n], l
    return SEGMENTS[-1][0], f - S[SEGMENTS[-1][0]], SEGMENTS[-1][1]


def at(name, t=0):
    return S[name] + t


def duration(f):
    n, t, l = seg(f)
    table = {"iframe": 50, "fire": 40, "track": 50, "arrive": 40, "drill": 50, "reveal": 70, "face": 70,
             "expand": 50, "slash": 50}
    if n == "spinup" and t >= l - 8:
        return 50
    if n == "lock" and t >= l - 2:
        return 110
    if n == "blast":
        return 130 if t == 0 else (50 if t < 4 else 60)
    if n == "finale":
        return 80 if t == 0 else (60 if t < 3 else 90)
    return table.get(n, 60)


DURATIONS = [duration(f) for f in range(N)]

# ---------------------------------------------------------------- spin of the bullet
RATE = np.zeros(N)
for _f in range(N):
    _n, _t, _l = seg(_f)
    if _n == "form":
        RATE[_f] = 0.03 * _t / _l
    elif _n == "spinup":
        RATE[_f] = 0.03 + 0.14 * (_t / _l) ** 1.6
    elif _n == "cutin":
        RATE[_f] = 0.17 + 0.03 * _t / _l
    elif _f >= S["lock"] and _f < S["blast"] + 2:
        RATE[_f] = 0.2
TURN = np.cumsum(np.minimum(RATE, 0.085))     # what the flutes visibly do (kept below aliasing)


def blur_of(f):
    return float(np.clip((RATE[f] - 0.07) / 0.08, 0, 1))


# ---------------------------------------------------------------- Rudeus
HIP = v2((80, 140))
SH_NEAR = HIP + v2((-3.2, -13.0))
SH_FAR = HIP + v2((3.4, -13.6))
RAISED = (v2((69, 104)), v2((93, 104)))
THRUST = (HIP + v2((20, -22)), HIP + v2((23, -25)))
GUARD = (HIP + v2((12, -10)), HIP + v2((15, -12)))
PUSH = (HIP + v2((19, -13)), HIP + v2((21, -15)))
IDLE_H = (SH_NEAR + unit(100) * 16, SH_FAR + unit(80) * 16)


def arm_targets(f):
    n, t, l = seg(f)
    if n == "idle":
        b = math.sin(f * 0.5) * 0.4
        return (SH_NEAR + unit(100) * (16 + b), SH_FAR + unit(80) * (16 + b)), (1, 1)
    if n == "raise":
        k = smooth01(t / (l - 1))
        an, af = lerp(100, 251.3, k), lerp(80, -66.8, k)
        d = lerp(16, 24.3, k) + math.sin(k * math.pi) * 3
        return (SH_NEAR + unit(an) * d, SH_FAR + unit(af) * d), (1 if an < 175 else -1, 1)
    if f < S["fire"]:
        jit = 0.0
        if f >= S["spinup"]:
            jit = (0.5 + 0.6 * phase(f, S["spinup"], S["iframe"])) * (1 if f % 2 else -1)
        return (RAISED[0] + v2((jit, 0)), RAISED[1] + v2((-jit, 0))), (-1, 1)
    if f < S["reveal"] + 6:
        return THRUST, (1, -1)
    if f < S["shrine"] + 6:
        k = smooth01(phase(f, S["reveal"] + 6, S["reveal"] + 12))
        return (lerp(THRUST[0], GUARD[0], k), lerp(THRUST[1], GUARD[1], k)), (1, -1)
    k = smooth01(phase(f, S["shrine"] + 6, S["shrine"] + 8))
    return (lerp(GUARD[0], PUSH[0], k), lerp(GUARD[1], PUSH[1], k)), (1, -1)


def pose(f):
    n, t, l = seg(f)
    hands, bends = arm_targets(f)
    p = dict(hands=hands, bends=bends, hip=HIP.copy(), spread=f >= S["raise"] + 4)
    if f < S["raise"] + 2:
        p["face"] = "calm"
    elif f < S["fire"]:
        p["face"] = "focus"
    elif f < S["reveal"] + 6:
        p["face"] = "shout"
    else:
        p["face"] = "shock"
    if f < S["gather"]:
        p["flutter"] = 0.4 * math.sin(f * 0.3)
        p["sway"] = 0.5 * math.sin(f * 0.25)
        p["sleeve"] = phase(f, S["raise"] + 2, S["gather"])
    elif f < S["fire"]:
        ch = phase(f, S["gather"], S["iframe"])
        p["flutter"] = (0.6 + 1.4 * ch) * math.sin(f * 0.9)
        p["flare"] = 1.6 * ch
        p["sway"] = math.sin(f * 0.7) * ch * 1.4
        p["sleeve"] = 1.0
        p["hair_lift"] = 1 if f < S["form"] else (2 if f < S["spinup"] + 10 else 3)
    else:
        k = math.exp(-(f - S["fire"]) / 9)
        p["flutter"] = 3.6 * k + 0.3 * math.sin(f * 0.5)
        p["flare"] = 1.4 * k
        p["sleeve"] = 0.3
        p["lean"] = -6 * (1 - smooth01(phase(f, S["drill"], S["collapse"])))
        p["hip"] = HIP + v2((-2 * min(1, (f - S["fire"] + 1) / 2), 0))
        p["sway"] = math.sin(f * 0.35) * k * 2
        p["hands"] = tuple(h + v2((-2, 0)) for h in p["hands"])
        if f >= S["sign"]:
            # the domain's pressure shoves at him
            dom = phase(f, S["sign"], S["shrine"])
            p["flutter"] += 1.5 * dom * (1 + 0.5 * math.sin(f * 1.3))
            p["flare"] = 1.2 * dom
    return p


# ---------------------------------------------------------------- camera
def shake(f):
    n, t, l = seg(f)
    a = 0.0
    if n == "gather":
        a = 0.6 * t / l
    elif n in ("form", "spinup"):
        a = 0.7 + 1.0 * phase(f, S["form"], S["cutin"])
    elif n == "lock":
        a = 1.7 + 0.8 * t / l
    elif n == "fire":
        a = 3.0
    elif n == "drill":
        a = 1.5 + 0.25 * t
    elif n == "blast":
        a = 7.0 * math.exp(-t / 5.0)
    elif n == "collapse":
        a = 2.4 * math.exp(-t / 8.0) + 1.0
    elif n == "reveal":
        a = 0.6 * math.exp(-t / 4)
    elif n == "sign":
        a = 0.5 + 1.0 * t / l
    elif n == "expand":
        a = 2.5
    elif n == "shrine":
        a = 2.0 if t < 12 else 1.0
    elif n == "slash":
        a = 1.5 + 2.0 * t / l
    elif n == "finale":
        a = 6.0 if t < 3 else 0.0
    if a < 0.35:
        return (0, 0)
    r = np.random.default_rng(f * 17 + 3)
    v = r.uniform(-1, 1, 2) * a
    return (int(round(v[0])), int(round(v[1])))


def darkness(f):
    if f < S["gather"]:
        return 0.0
    if f < S["fire"]:
        return 0.15 + 0.5 * phase(f, S["gather"], S["lock"] + 8)
    if f < S["blast"] + 2:
        return 0.65
    if f < S["reveal"]:
        return 0.65 - 0.25 * smooth01(phase(f, S["blast"] + 2, S["reveal"]))
    return 0.4 + 0.45 * phase(f, S["reveal"], S["expand"])


# ---------------------------------------------------------------- impact frame
IMPACT_CREAM = np.array(hexc("fffae2"), np.uint8)
IMPACT_RED = np.array(hexc("a8401a"), np.uint8)
IMPACT_BLACK = np.array(hexc("0a0818"), np.uint8)


def impact_frame(img, owner, inverted, center, red=IMPACT_RED):
    out = np.empty_like(img)
    lum = img.astype(np.float32) @ np.array([0.3, 0.59, 0.11], np.float32) / 255.0
    fig = ((owner >= L["Back FX"]) & (owner <= L["Magic"])) | (owner == L["Crag"]) | \
        ((owner <= L["Domain"]) & (lum > 0.72))
    ys, xs = np.mgrid[0:H, 0:W]
    ang = np.degrees(np.arctan2(ys - center[1], xs - center[0]))
    rad = np.hypot(ys - center[1], xs - center[0])
    rays = (np.mod(ang + rad * 0.15, 9.0) < 1.6) & (rad > 36)
    if inverted:
        out[:] = IMPACT_CREAM
        out[rays] = red
        out[fig] = IMPACT_BLACK
    else:
        out[:] = IMPACT_BLACK
        out[rays] = red
        out[fig] = IMPACT_CREAM
        edge = fig & ~(np.roll(fig, 1, 0) & np.roll(fig, -1, 0) & np.roll(fig, 1, 1) & np.roll(fig, -1, 1))
        out[edge] = red
    return out


# ---------------------------------------------------------------- pieces of the wide shot
SUK_X, SUK_FEET = 266, 151
BASE_PUFFS = [dict(off=v2((np.random.default_rng(i).uniform(-34, 40), np.random.default_rng(i + 50).uniform(-8, 4))),
                   r0=np.random.default_rng(i + 99).uniform(5, 9), t0=(i * 3) % 9,
                   v=v2((np.random.default_rng(i + 7).uniform(-0.5, 0.5), -np.random.default_rng(i + 8).uniform(0.4, 1.0))))
              for i in range(18)]
WALL_CUTS = [(4, (105, 128), 20), (7, (106, 140), -35), (9, (104, 120), 70), (12, (107, 146), 5),
             (14, (103, 133), -60), (17, (108, 124), 40), (20, (105, 150), -15), (23, (106, 136), 85)]


def base_dust(sc, t, thin=0.0):
    """Billowing dust where the crag came down; thin 0..1 lets it clear."""
    puffs = []
    for d in BASE_PUFFS:
        a = t - d["t0"]
        if a < 0:
            continue
        r = d["r0"] + 10 * (1 - math.exp(-a / 7))
        c = v2((270, 156)) + d["off"] + d["v"] * a * 0.8
        k = min(0.99, 0.15 + thin * 0.85 + max(0, a - 30) / 60)
        puffs.append((c, r, k))
    fx._cloud(sc, puffs, "FX")


def title(sc, text, t, y=18):
    if t < 0:
        return
    x = 160 - SK.text_width(text, 2) // 2
    if t == 0:
        SK.draw_text(sc, text, x, y, 2, fill="white", edge="dm5")
    else:
        SK.draw_text(sc, text, x, y, 2, fill="re2", edge="ob0", shade="re1")


def domain_scene(f, shrine_rise, dark):
    dom = Scene()
    SK.domain_world(dom, f, shrine_rise=shrine_rise)
    return dom


def draw_characters(sc, f, p, dark, suk_pose="stand", suk_sil=0.0, domain=False, eye_glow=0.0):
    draw_rudeus_full(sc, p)
    body = (sc.owner >= L["Back arm"]) & (sc.owner <= L["Front arm"])
    sc.rgb[body] *= 1 - 0.3 * dark
    if domain:
        sc.rgb[body] = sc.rgb[body] * 0.8 + C["dm4"] * 0.2
    return body


# ---------------------------------------------------------------- the wide shot
def render_wide(f):
    n, t, l = seg(f)
    sc = Scene()
    dark = darkness(f)
    in_domain = f >= S["shrine"]
    p = pose(f)

    # ---------------- backdrop
    crag_hit, crag_dx, collapse = None, 0, None
    if n == "drill":
        k = t / l
        crag_hit = dict(crack=0.25 + 0.6 * k, heat=1.0, hole=0.3 + 0.7 * k)
        crag_dx = (1 if f % 2 else -1) if t > 2 else 0
    elif f >= S["blast"]:
        tb = f - S["blast"]
        crag_hit = dict(crack=1.0, spread=min(1.0, tb / 10), heat=max(0.0, 1 - tb / 22), hole=1.0)
        if f >= S["blast"] + 8:
            collapse = f - (S["blast"] + 8)
        elif tb > 2:
            crag_dx = 1 if f % 2 else -1
    if in_domain:
        dom = domain_scene(f, smooth01(phase(f, S["shrine"], S["shrine"] + 12)), dark)
        sc.rgb[:], sc.owner[:], sc.dith[:] = dom.rgb, dom.owner, dom.dith
    else:
        world.draw_sky(sc, f, dark)
        world.draw_land(sc, f, dark)
        world.draw_crag(sc, f, hit=crag_hit, dark=dark, dx=crag_dx, collapse=collapse)
        if collapse is not None:
            world.draw_rubble(sc, min(1.0, collapse / 14), dark)
    if f < S["fire"]:
        fx.rocks(sc, f, rumble=phase(f, S["gather"], S["gather"] + 10), flying=False)

    # ---------------- Sukuna (behind the dust while it settles)
    suk_on = f >= S["collapse"] + 10
    suk_mask = None
    suk_pose = "sign" if f >= S["sign"] + 3 else "stand"
    if suk_on:
        sil = 1.0 - smooth01(phase(f, S["reveal"] + 8, S["reveal"] + 20))
        suk_mask = SK.draw_sukuna(sc, SUK_X, SUK_FEET, suk_pose, sil=sil, eye_glow=1.0 if f < S["face"] else 0.0)
        suk_mask = sc.owner == L["Sukuna"]

    body = draw_characters(sc, f, p, dark, domain=in_domain)

    # ---------------- light
    ch = phase(f, S["gather"] + 2, S["lock"] + 8)
    if S["gather"] <= f < S["fire"]:
        d = np.hypot(XS - fx.CORE[0], YS - fx.CORE[1])
        sc.light(np.clip(1 - d / 115, 0, 1) / (1 + (d / 24) ** 2), C["mg4"], 0.18 + 0.8 * ch, dither=34, mul=0.5)
        dg = np.hypot((XS - fx.FEET[0]) / 3.5, (YS - fx.FEET[1]))
        sc.light(np.clip(1 - dg / 26, 0, 1) ** 2, C["mg3"], 0.35 + 0.3 * ch, dither=34)
    if n == "fire":
        c = fx.path_point([0.10, 0.26][t])
        d = np.hypot(XS - c[0], YS - c[1])
        sc.light(np.clip(1 - d / 120, 0, 1) / (1 + (d / 26) ** 2), C["mg5"], 0.9, dither=34, mul=0.3)
    if n in ("arrive", "drill") or (n == "blast"):
        tb = f - S["drill"]
        d = np.hypot(XS - world.HOLE[0], YS - world.HOLE[1])
        st = 0.6 + 0.04 * tb if n != "blast" else 1.0 * math.exp(-t / 6)
        sc.light(np.clip(1 - d / 150, 0, 1) / (1 + (d / 34) ** 2), C["mg4"], st, dither=34, mul=0.4)
    if f >= S["sign"] and not in_domain:
        d = np.hypot(XS - SUK_X, YS - (SUK_FEET - 30))
        sc.light(np.clip(1 - d / 110, 0, 1) / (1 + (d / 30) ** 2) * ~(sc.owner == L["Sukuna"]), C["dm5"],
                 0.3 + 0.4 * phase(f, S["sign"], S["expand"]), dither=30)
    if in_domain:
        d = np.hypot(XS - SUK_X, YS - 100)
        sc.light(np.clip(1 - d / 300, 0, 1) * 0.5, C["dm4"], 0.5, dither=30)

    if not in_domain:
        blast = math.exp(-(f - S["fire"]) / 4) if f >= S["fire"] else 0.0
        world.draw_grass_fg(sc, f, blast=blast, blast_x=p["hip"][0], dark=dark)

    # ---------------- behind Rudeus
    sc.protect = body
    if S["raise"] + 1 <= f < S["fire"] + 6:
        sweep = phase(f, S["raise"] + 1, S["gather"] + 2)
        power = 0.45 + 0.55 * ch if f < S["fire"] else max(0.0, 1 - (f - S["fire"]) / 6)
        if f < S["fire"] + 3 or f % 2 == 0:
            fx.ground_circle(sc, f, sweep, power, f * (0.05 + 0.25 * RATE[f]))
    if not in_domain:
        crack_grow = phase(f, S["gather"] + 6, S["spinup"])
        crack_heat = (0.35 + 0.65 * phase(f, S["gather"] + 20, S["lock"])) if f < S["fire"] else \
            max(0.0, 1 - (f - S["fire"]) / 18)
        fx.ground_cracks(sc, crack_grow, crack_heat)
    if S["gather"] + 2 <= f < S["fire"]:
        fx.light_pillars(sc, f, 0.4 + 0.6 * ch)
    if S["gather"] + 8 <= f < S["fire"] + 5 and n != "iframe":
        ap = 0.6 * phase(f, S["gather"] + 8, S["gather"] + 20) + 0.4 * phase(f, S["form"], S["lock"])
        if f >= S["fire"]:
            ap = max(0.0, 1 - (f - S["fire"] + 1) / 5) * 0.8
        fx.aura(sc, f, ap)
    sc.protect = None
    if suk_on and f >= S["sign"] and suk_mask is not None:
        SK.red_aura(sc, suk_mask, f, 0.4 + 0.6 * phase(f, S["sign"], S["expand"]))

    # ---------------- the spell (charge)
    if S["gather"] <= f < S["fire"]:
        fx.over_circle(sc, f, phase(f, S["gather"], S["gather"] + 7), 0.5 + 0.5 * ch, f * (0.05 + 0.3 * RATE[f]))
    elif n == "fire":
        k = 1 + (t + 1) * 0.35
        fx.magic_circle(sc, fx.OVER, (17 * k, 0), (0, 4.4 * k), -f * 0.07, 1.0, 0.4, layer="Magic", glow=False)
    if S["gather"] <= f < S["lock"]:
        fade = 1.0 - phase(f, S["lock"] - 4, S["lock"])
        fx.motes(sc, f, phase(f, S["gather"], S["gather"] + 12) * fade, 1.0 + 1.2 * phase(f, S["form"], S["lock"]))
    if S["gather"] <= f < S["fire"]:
        fx.rocks(sc, f, resting=False)
    jitter = v2((0, 0))
    if f >= S["spinup"] + 10 and f < S["fire"]:
        jitter = v2(((f % 2) * 2 - 1, 0)) * 0.6
    core = fx.CORE + jitter
    if S["gather"] + 4 <= f < S["form"] + 7:
        rad = 1 + 6 * ease_out(phase(f, S["gather"] + 4, S["form"]), 2)
        if f >= S["form"]:
            rad *= 1 - phase(f, S["form"], S["form"] + 7)
        fx.gravel_ball(sc, core, rad, f * 0.09, f)
    if S["form"] <= f < S["fire"]:
        s = ease_out(phase(f, S["form"], S["form"] + 7), 2) * (1 + 0.55 * smooth01(phase(f, S["form"] + 7, S["lock"])))
        heat = phase(f, S["spinup"] + 6, S["lock"] + 6)
        bl = blur_of(f)
        vp = phase(f, S["spinup"], S["spinup"] + 14)
        if f >= S["form"] + 2:
            fx.orbit_sparks(sc, core, f, ch, front=False)
        spin.vortex(sc, core, fx.AIM_ANG, s, TURN[f], vp, front=False, tail=10 + 24 * vp)
        spin.drill_bullet(sc, core, fx.AIM_ANG, s, TURN[f], heat=heat, blur=bl, f=f)
        if f >= S["spinup"]:
            spin.spin_rings(sc, core, fx.AIM_ANG, s, f, vp)
        spin.vortex(sc, core, fx.AIM_ANG, s, TURN[f], vp, front=True, tail=10 + 24 * vp)
        if f >= S["form"] + 2:
            fx.orbit_sparks(sc, core, f, ch, front=True)
        if f >= S["spinup"] + 4:
            spin.fling(sc, core, fx.AIM_ANG, s, f, phase(f, S["spinup"] + 4, S["lock"]), n=10, layer="Magic")
            fx.crackle(sc, core, f, 1 + (f - S["spinup"]) // 8, 8)
    for start in (S["gather"] + 10, S["gather"] + 18, S["gather"] + 24, S["form"], S["form"] + 5,
                  S["spinup"] + 4, S["spinup"] + 10, S["spinup"] + 15, S["spinup"] + 19, S["spinup"] + 22,
                  S["lock"], S["lock"] + 3, S["lock"] + 6):
        if start <= f < start + 6:
            fx.implosion_ring(sc, fx.CORE, (f - start) / 6)
    for i in range(3):
        a0 = S["lock"] + 2 * i
        if a0 <= f < S["fire"]:
            fx.barrel_ring(sc, i, phase(f, a0, a0 + 3), 0.6 + 0.4 * phase(f, a0 + 2, S["iframe"]),
                           f * (0.12 + RATE[f]))
        elif S["fire"] <= f < S["fire"] + 4:
            fx.barrel_ring(sc, i, 1, 1, f * 0.12, burst=f - S["fire"])

    # ---------------- launch / flight / drill
    if n == "fire":
        s_ = [0.10, 0.26][t]
        fx.muzzle_flash(sc, t)
        fx.helix_trail(sc, s_, f, age=0)
        fx.bullet_streak(sc, s_, 0.0)
        fx.vapor_rings(sc, s_, t)
        fx.speed_lines(sc, t)
        pc = fx.path_point(s_)
        spin.vortex(sc, pc, fx.AIM_ANG, 1.55, TURN[f], 1.0, front=False, tail=50)
        spin.drill_bullet(sc, pc, fx.AIM_ANG, 1.55, TURN[f], heat=1.0, blur=1.0, layer="FX", underglow=0, f=f)
        spin.vortex(sc, pc, fx.AIM_ANG, 1.55, TURN[f], 1.0, front=True, tail=50, layer="FX")
    if n == "arrive":
        pc = fx.path_point(0.88)
        fx.helix_trail(sc, 0.88, f, age=0)
        fx.bullet_streak(sc, 0.88, 0.3)
        fx.vapor_rings(sc, 0.88, 3)
        fx.speed_lines(sc, 2)
        spin.vortex(sc, pc, fx.AIM_ANG, 1.55, TURN[f], 1.0, front=False, tail=60)
        spin.drill_bullet(sc, pc, fx.AIM_ANG, 1.55, TURN[f], heat=1.0, blur=1.0, layer="FX", underglow=0, f=f)
        spin.vortex(sc, pc, fx.AIM_ANG, 1.55, TURN[f], 1.0, front=True, tail=60, layer="FX")
    if n == "drill":
        pen = 2 + t * 1.1
        tipc = world.HOLE + fx.AIM_DIR * pen
        pc = tipc - fx.AIM_DIR * spin.TIP * 1.55
        clip = (world.HOLE + fx.AIM_DIR * 1.0, fx.AIM_DIR)
        fx.bullet_streak(sc, 0.95, 0.5 + t * 0.04, fade=0.3 + t * 0.06)
        spin.vortex(sc, pc, fx.AIM_ANG, 1.55, TURN[f], 1.0, front=False, tail=30, clip=clip)
        spin.drill_bullet(sc, pc, fx.AIM_ANG, 1.55, TURN[f], heat=1.0, blur=0.8, layer="FX", underglow=0, f=f, clip=clip)
        spin.vortex(sc, pc, fx.AIM_ANG, 1.55, TURN[f], 1.0, front=True, tail=30, layer="FX", clip=clip)
        sc.put(disc(world.HOLE, 2.0 + 0.3 * t), "mg7", "FX")
        sc.put(outline_of(disc(world.HOLE, 2.0 + 0.3 * t)), "mg5", "FX")
        spin.drill_spray(sc, world.HOLE, fx.AIM_ANG, 6.5, f, 0.6 + 0.04 * t, reach=1.0 + 0.08 * t)
        if t % 3 == 0:
            fx.implosion_ring(sc, world.HOLE, 0.55, rmax=30)
    if S["fire"] <= f < S["fire"] + 16:
        fx.ground_dust(sc, f - S["fire"], x0=p["hip"][0])

    # ---------------- breakthrough
    if S["blast"] <= f < S["sign"]:
        tb = f - S["blast"]
        fx.debris(sc, tb)
        fx.dust_cloud(sc, tb)
        fx.sparks(sc, tb)
        fx.shockwave(sc, tb)
        fx.impact_flash(sc, tb)
        if tb < 3:   # the bullet bursting out of the far side
            a = v2((292, 103)) + fx.AIM_DIR * tb * 18
            xs, ys = line_pts(a - fx.AIM_DIR * 26, a + fx.AIM_DIR * 30)
            sc.put_px(xs, ys, "mg7", "FX")
            sc.put_px(xs, ys + 1, "mg4", "FX")
            sc.put_px(xs, ys - 1, "mg5", "FX")
        if tb > 3:
            fx.embers(sc, f, tb)
    if collapse is not None:
        thin = phase(f, S["reveal"] + 2, S["reveal"] + 20)
        if f < S["face"]:
            base_dust(sc, collapse, thin)
    if suk_on and f < S["face"]:
        # red eyes burn through the dust
        sil = 1.0 - smooth01(phase(f, S["reveal"] + 8, S["reveal"] + 20))
        if sil > 0.2:
            eyes = Scene()
            SK.draw_sukuna(eyes, SUK_X, SUK_FEET, "stand")
            hm = eyes.owner >= 0
            red = hm & (np.abs(eyes.rgb - C["re2"]).sum(-1) < 1)
            sc.put(red, "re2", "FX")
            sc.light(np.exp(-((XS - SUK_X) ** 2 + (YS - (SUK_FEET - 50)) ** 2) / 30.0), C["re2"], 0.9 * sil)

    # ---------------- domain expansion
    if n == "sign":
        title(sc, "DOMAIN EXPANSION", t - 3)
        cr = phase(f, S["sign"], S["expand"])
        for k in range(8):   # red fissures racing out from under him
            a = math.radians(180 + (k - 3.5) * 22)
            pts = [v2((SUK_X, SUK_FEET + 6))]
            rr = np.random.default_rng(k)
            for j in range(int(2 + 6 * cr)):
                a += rr.uniform(-0.4, 0.4)
                pts.append(pts[-1] + v2((math.cos(a) * 9, abs(math.sin(a)) * 2.0 + 0.6)))
            for a_, b_ in zip(pts[:-1], pts[1:]):
                xs, ys = line_pts(a_, b_)
                sc.put_px(xs, ys, "re2", "Back FX")
                sc.put_px(xs, ys + 1, "dm3", "Back FX")
    if n == "expand":
        dom = domain_scene(f, 0.0, dark)
        dom.protect = None
        SK.draw_sukuna(dom, SUK_X, SUK_FEET, "sign")
        draw_characters(dom, f, p, dark, domain=True)
        r = 4 + 380 * smooth01(t / (l - 1)) ** 1.3
        cc = v2((SUK_X, SUK_FEET - 34))
        dd = np.hypot(XS - cc[0], YS - cc[1])
        jag = r + 3 * np.sin(np.arctan2(YS - cc[1], XS - cc[0]) * 9 + f)
        inside = dd < jag
        sc.rgb[inside], sc.owner[inside], sc.dith[inside] = dom.rgb[inside], dom.owner[inside], dom.dith[inside]
        ring = (dd >= jag) & (dd < jag + 3)
        sc.put(ring, "dm6", "FX")
        sc.put((dd >= jag + 3) & (dd < jag + 5) & ((XS + YS) % 2 == 0), "ob0", "FX")
        sc.put((dd >= jag - 2) & (dd < jag), "white", "FX")
        title(sc, "DOMAIN EXPANSION", 5)
    if n == "shrine":
        title(sc, "MALEVOLENT SHRINE", t - 6)
    if f >= S["shrine"] + 8:
        tw = f - (S["shrine"] + 8)
        tcut = f - S["slash"]
        fx.earth_wall(sc, phase(f, S["shrine"] + 8, S["shrine"] + 13), WALL_CUTS, tcut, layer="FX")
    if n == "slash" or n == "finale":
        rate = lambda fb: 3 + 18 * phase(fb, S["slash"], S["finale"])
        SK.slashes(sc, f, rate, S["slash"], layer="FX", x_min=fx.WALL_X0)
        for tc, c, r_, a0, a1 in ((10, (220, 60), 110, 140, 230), (18, (200, 200), 130, 200, 320),
                                  (26, (250, 40), 150, 100, 200)):
            SK.cleave(sc, v2(c), r_, a0, a1, f - (S["slash"] + tc), x_min=fx.WALL_X0)
    if n == "finale" and t == 0:
        SK.cleave(sc, v2((200, 260)), 200, 215, 325, 0, x_min=fx.WALL_X0)
        SK.cleave(sc, v2((200, -80)), 200, 35, 145, 0, x_min=fx.WALL_X0)

    img, owner = sc.finish(shake(f))
    if n == "iframe":
        img = impact_frame(img, owner, inverted=(t == 0), center=fx.CORE)
    if n == "finale" and t in (1, 2):
        img = impact_frame(img, owner, inverted=(t == 1), center=(SUK_X, SUK_FEET - 34),
                           red=np.array(hexc("c0282a"), np.uint8))
    return img, owner


_EYES = []


def suk_eyes():
    """Screen pixels of Sukuna's four red eyes on the wide shot."""
    if not _EYES:
        tmp = Scene()
        SK.draw_sukuna(tmp, SUK_X, SUK_FEET, "sign")
        red = (tmp.owner >= 0) & ((np.abs(tmp.rgb - C["re2"]).sum(-1) < 1) | (np.abs(tmp.rgb - C["re1"]).sum(-1) < 1))
        ys, xs = np.nonzero(red[M:M + H, M:M + W])
        _EYES.extend(zip(xs.tolist(), ys.tolist()))
    return _EYES


# ---------------------------------------------------------------- top level
def render(f):
    n, t, l = seg(f)
    if n == "cutin":
        sc = Scene()
        shots.bullet_closeup(sc, f, t, l, TURN[f] * 1.6, 0.25 if t < l // 2 else 0.85, 0.5 + 0.5 * t / l)
        img, owner = sc.finish((0, 0))
    elif n == "track":
        sc = Scene()
        shots.tracking(sc, f, t, l, TURN[f], 1.0)
        img, owner = sc.finish(shake(f) if t >= l - 3 else (0, 0))
    elif n == "face":
        sc = Scene()
        typing = 6 <= t < 24
        n_chars = int(max(0, (t - 6) * 1.5))
        talk = [0, 1, 2, 1][t % 4] if typing else 0
        grin = smooth01(phase(t, 22, 28))
        eyes = smooth01(phase(t, 28, 30))
        SK.portrait(sc, f, talk=talk, lower_eyes=eyes, grin=grin, sway=math.sin(t * 0.3) * 0.8)
        if t >= 3:
            SK.dialogue(sc, "ORSTED SENDS\nHIS REGARDS.", n_chars, f)
        SK.letterbox(sc, min(1.0, (t + 1) / 2), h=12)
        img, owner = sc.finish((((f % 2) * 2 - 1), 0) if 28 <= t < 32 else (0, 0))
    else:
        img, owner = render_wide(f)
    # fade from / to black around the loop point
    fade = 0.0
    if f < 4:
        fade = 1 - f / 4
    elif n == "finale" and t >= 3:
        fade = min(1.0, (t - 2) / (l - 4))
    if fade > 0:
        bay = BAYER[M:M + H, M:M + W] + 0.47
        m = bay < fade
        img[m] = np.array(hexc("0a0818"), np.uint8)
        owner[m] = L["FX"]
        if n == "finale" and fade < 1.0:
            # the four red eyes are the last thing to go
            for ex, ey in suk_eyes():
                img[ey, ex] = np.array(hexc("ff3a3a"), np.uint8)
                owner[ey, ex] = L["FX"]
    return img, owner

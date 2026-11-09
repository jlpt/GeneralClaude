"""Timeline: which pose and which effects every frame gets.

Frames 0-3 are the idle loop; the cast is 48 frames (c = 0..47):
  0-5 gather, 6-17 charge, 18-20 hush, 21 flash, 22-31 beam, 32-39 starfall, 40-47 settle.
"""
import math
import numpy as np
from canvas import (C, XS, YS, GROUND, HORIZON, SHAPE, CHAR_LAYERS, Scene, dith, v2, unit, lerp, smooth, nb,
                    disc, ellipse, ering, pixels, line_px, polyline)
import backdrop
import fx
from fx import FRONT, BACK
from mage import draw_mage

MAGE_X = 44
GC = (MAGE_X, 99)          # ground circle under the mage
TARGET = (126, 99)         # where the pillar lands
SKY = (126, 15)            # the circle in the sky
N_IDLE, N_CAST = 4, 48

IDLE = dict(hand=(9, -17), staff_ang=90, grip=17, bob=0, lean=0, lift=0, blow=0, float=0, stance=0, braid=-100,
            braid_curl=4, hat_tilt=0, tip=(0, 0), off=(6, -14), off_vis=0)


def _k(**kw):
    d = dict(IDLE)
    d.update(kw)
    return d


_RAISED = _k(hand=(13, -24), staff_ang=70, grip=14, lift=1, float=3, braid=-235, braid_curl=-8, off=(11, -15),
             off_vis=1, tip=(1, -2))
_HIGH = _k(hand=(13, -29), staff_ang=97, grip=12, lift=1, float=4, lean=-2, bob=-1, braid=-245, braid_curl=-6,
           off=(10, -17), off_vis=1, tip=(1, -2))
_STRIKE = _k(hand=(11, -20), staff_ang=15, grip=14, lean=3, bob=2, blow=1, stance=1, braid=-185, braid_curl=3,
             hat_tilt=-0.08, tip=(-3, 1))
KEYS = [
    (-1, IDLE, True),
    (1, _k(hand=(9, -22), bob=-1, braid=-98, tip=(0, -1)), True),
    (2, _k(hand=(9, -16), bob=2, lift=0.2, braid=-108, tip=(0, 1)), False),
    (3, _k(hand=(9, -16), bob=2, lift=0.3, braid=-112, tip=(0, 1)), True),
    (5, _k(hand=(11, -22), staff_ang=80, grip=15, lift=0.5, float=1, braid=-160, braid_curl=-2, off=(9, -14),
           off_vis=1, tip=(0, -1)), True),
    (8, _RAISED, True),
    (17, dict(_RAISED, float=4), True),
    (19, _HIGH, True),
    (20, dict(_HIGH, staff_ang=104), False),
    (21, _k(hand=(13, -24), staff_ang=55, grip=13, lean=1, float=2, lift=0.6, blow=0.3, stance=0.5, braid=-215,
            braid_curl=-2, tip=(-1, 0)), False),
    (22, _STRIKE, False),
    (31, _STRIKE, True),
    (36, _k(hand=(10, -19), staff_ang=62, grip=15, lean=1, bob=1, blow=0.35, stance=0.6, braid=-150, braid_curl=2,
            hat_tilt=-0.03, tip=(-1, 0)), True),
    (41, _k(hand=(9, -18), staff_ang=88, blow=0.05, braid=-108, braid_curl=5), True),
    (43, _k(bob=1, braid=-96), True),
    (47, IDLE, True),
]


def cast_pose(c):
    for (c0, a, _), (c1, b, ease) in zip(KEYS[:-1], KEYS[1:]):
        if c0 <= c <= c1:
            t = (c - c0) / (c1 - c0)
            t = smooth(t) if ease else t
            p = {}
            for key in a:
                if isinstance(a[key], tuple):
                    p[key] = tuple(lerp(x, y, t) for x, y in zip(a[key], b[key]))
                else:
                    p[key] = lerp(a[key], b[key], t)
            break
    p["hand"] = tuple(round(v) for v in p["hand"])
    for key in ("bob", "lean", "float"):
        p[key] = round(p[key])
    p["off_hand"] = tuple(round(v) for v in p.pop("off")) if p.pop("off_vis") > 0.5 else None
    p["wave"] = c * (0.31 if 21 <= c <= 33 else 0.12)
    p["eyes"] = "closed" if 1 <= c <= 4 else ("glow" if 5 <= c <= 38 else "open")
    p["mouth"] = 21 <= c <= 30
    if 22 <= c <= 31:                       # the cloak whips in the blast
        p["blow"] = 1.0 - 0.1 * (c % 2)
        p["tip"] = (-3 + (c % 2), 1)
        p["braid"] += 6 * (c % 2)
    if 8 <= c <= 17:
        p["float"] += (c % 6 >= 3)
    p["x"] = MAGE_X
    return p


def idle_pose(i):
    return dict(x=MAGE_X, bob=(0, 0, 1, 1)[i], wave=i * 0.25, braid=(-100, -98, -100, -102)[i],
                tip=((0, 0), (0, 0), (0, 1), (-1, 1))[i])


# ------------------------------------------------------------------ seeded particle sets
_r = np.random.RandomState(5)
MOTES = [dict(b=_r.randint(3, 31), life=_r.randint(6, 11), dx=_r.uniform(-25, 25), dy=_r.uniform(-1, 1),
              v=_r.uniform(1.6, 3.4), sway=_r.uniform(0, 1.6), ph=_r.uniform(0, 6.28)) for _ in range(40)]
STREAKS = [dict(b=6 + i * 11 // 20, ang=_r.uniform(0, 360), d0=_r.uniform(26, 36)) for i in range(22)]
DEBRIS = [dict(b=22 + i % 4, x0=_r.uniform(-9, 9), vx=_r.uniform(-4.6, 4.6), vy=_r.uniform(-8.5, -4.0),
               big=i % 3 != 0) for i in range(18)]
EMBERS = [dict(b=_r.randint(30, 42), life=_r.randint(6, 11), x0=_r.uniform(-15, 15), y0=_r.uniform(-3, 3),
               v=_r.uniform(0.8, 2.0), sway=_r.uniform(0.5, 2.0), ph=_r.uniform(0, 6.28)) for _ in range(22)]
SHARDS = [(int(_r.randint(-6, 7)), int(_r.randint(18, 94))) for _ in range(60)]
METEORS = [(30, (150, 101)), (31, (100, 95)), (32, (164, 91)), (33, (86, 104)), (34, (136, 106)),
           (35, (114, 89)), (36, (170, 103)), (37, (96, 100))]
METEOR_DIR = v2((-0.42, 1.0)) / math.hypot(0.42, 1.0)
CONSTELLATION = [(92, 31), (104, 13), (122, 5), (143, 8), (160, 21), (146, 35), (118, 37)]
CRACKS = [[(0, 0), (-7, -1), (-12, 1), (-19, 0)], [(0, 0), (6, 1), (13, 0), (20, 2)], [(0, 0), (-4, 2), (-9, 3)],
          [(0, 0), (5, -2), (11, -2), (15, -3)], [(0, 0), (-5, -2), (-10, -3)], [(0, 0), (3, 3), (9, 3)]]
BEAM_HW = {22: 15, 23: 17, 24: 16, 25: 17, 26: 15, 27: 16, 28: 14, 29: 12, 30: 9, 31: 6, 32: 4.5, 33: 3, 34: 2,
           35: 1.2, 36: 0.6}
SHAKE = {22: (-3, 2), 23: (3, -2), 24: (-2, -2), 25: (2, 2), 26: (-2, 1), 27: (2, -1), 28: (-1, 1), 29: (1, -1),
         30: (-1, 0), 31: (0, 1), 33: (1, 0), 34: (0, 1), 35: (-1, 0), 36: (0, -1), 37: (1, 0), 38: (0, 1),
         39: (-1, 0), 40: (0, -1)}
BRIGHT = {"ind_o": "ind_l", "ind_d": "ind_l", "ind_b": "ind_h", "ind_l": "ind_h", "ind_h": "cy_1",
          "gold_o": "gold_b", "gold_d": "gold_l", "gold_b": "gold_h", "gold_l": "gold_h",
          "teal_o": "teal_b", "teal_d": "teal_l", "teal_b": "teal_l", "skin_o": "skin_b", "skin_s": "skin_l",
          "skin_b": "skin_l", "hair_o": "hair_b", "hair_s": "hair_l", "hair_b": "hair_l", "leg_d": "ind_l",
          "leg_b": "ind_l", "boot_d": "boot_l", "boot_b": "boot_l", "wood_o": "wood_b", "wood_d": "wood_l",
          "wood_b": "wood_l", "eye_d": "cy_3"}
_BRIGHT_RGB = {C[a][:3]: C[b] for a, b in BRIGHT.items()}


def rim_light(sc, edge_col, strong=True, silhouette=False):
    """Relight the character from the right: the outline on that side takes `edge_col`, the pixels just
    inside step up their colour ramp. With `silhouette` the whole figure is flattened to one dark tone."""
    sil = sc.alpha(CHAR_LAYERS)
    rim1 = sil & ~nb(sil, 1, 0)
    rim2 = sil & ~rim1 & ~nb(sil, 2, 0)
    if strong:
        rim1 |= sil & ~nb(sil, 0, -1) & ~nb(sil, 1, -1) & (XS > MAGE_X)
    for name in CHAR_LAYERS:
        img = sc.img[name]
        a = img[:, :, 3] > 0
        if silhouette:
            img[a] = C["sil"]
            img[a & (rim1 | rim2)] = C[edge_col]
            continue
        ys, xs = np.nonzero(a & rim2)
        for y, x in zip(ys, xs):
            rgb = tuple(int(v) for v in img[y, x, :3])
            if rgb in _BRIGHT_RGB:
                img[y, x] = _BRIGHT_RGB[rgb]
        img[a & rim1] = C[edge_col]


def dim_at(c):
    if c < 4:
        return 0.0
    if c <= 17:
        return (0.25, 0.5, 0.75)[min((c - 4) // 4, 2)]
    if c <= 20:
        return 1.0
    if c <= 31:
        return 0.5
    return {32: 0.5, 33: 0.5, 34: 0.25, 35: 0.25, 36: 0.25}.get(c, 0.0)


def lit_at(c):
    if 22 <= c <= 27:
        return 1.0
    return {28: 0.9, 29: 0.8, 30: 0.7, 31: 0.6, 32: 0.5, 33: 0.42, 34: 0.34, 35: 0.26, 36: 0.18, 37: 0.1}.get(c, 0.0)


# ------------------------------------------------------------------ frames
def render_idle(i):
    sc = Scene()
    backdrop.draw(sc, i)
    sc.put(ellipse((MAGE_X, GROUND + 1), 9, 2.2) & dith(0.75), "shadow", BACK)
    out = draw_mage(sc, idle_pose(i))
    orb = out["orb"]
    if i == 1:
        fx.sparkle(sc, orb + v2((1, -1)), 2, arm="cy_2", diag=False)
    if i == 2:
        fx.sparkle(sc, orb + v2((1, -1)), 1, arm="cy_1")
    for k, (dx, ph) in enumerate(((-4, 0), (3, 2))):       # two lazy motes circling the orb
        a = (i + ph) % 4
        sc.put(pixels([(orb[0] + dx + (a % 2), orb[1] + 3 - a * 2.5)]), ("cy_1", "cy_2", "cy_3", "cy_4")[a], FRONT)
    return sc, (0, 0)


def render_cast(c):
    sc = Scene()
    rng = np.random.RandomState(100 + c)
    flash = c == 21
    backdrop.draw(sc, c, dim=dim_at(c), lit=lit_at(c), flash=flash)
    pose = cast_pose(c)

    # ---- shadow + ground circle (behind the mage)
    fl = pose["float"]
    if not flash:
        sc.put(ellipse((MAGE_X, GROUND + 1), 9 - fl * 0.8, 2.2 - fl * 0.2) & dith(0.75), "shadow", BACK)
    hot = 21 <= c <= 31
    if 2 <= c <= 42:
        rx = {2: 7, 3: 14, 4: 21, 5: 25}.get(c, 27)
        fade = {40: 0.75, 41: 0.5, 42: 0.25}.get(c, 1.0)
        ring, hotc, star = ("cy_1", "fx_w", "au_1") if hot else ("cy_2", "cy_1", "au_2")
        if flash:
            ring, hotc, star = "cy_4", "cy_5", "cy_4"
        fx.magic_circle(sc, GC, rx, rx * 0.3, c * 8, BACK, fade=fade, ring=ring, hot=hotc, star=star, detail=c >= 3)
        if c >= 4 and not flash:
            h = min(3 + (c - 4) * 1.1, 15) if c <= 20 else (11 if c <= 31 else max(0, 11 - (c - 31) * 1.6))
            fx.light_wall(sc, GC, rx, rx * 0.3, h, False, 0.5 * fade)

    # ---- constellation that draws itself, then the circle in the sky
    if 6 <= c <= 14 and not flash:
        n = min(len(CONSTELLATION), (c - 6) * 2 + 1)
        fade = {13: 0.6, 14: 0.3}.get(c, 1.0)
        for a, b in zip(CONSTELLATION[:n - 1], CONSTELLATION[1:n]):
            sc.put(line_px(a, b) & dith(fade), "cy_4" if c < 11 else "cy_5", BACK)
        if n == len(CONSTELLATION) and c >= 10:
            sc.put(line_px(CONSTELLATION[-1], CONSTELLATION[0]) & dith(fade), "cy_5", BACK)
        for j, q in enumerate(CONSTELLATION[:n]):
            fx.sparkle(sc, q, 2 if j == n - 1 else 1, BACK, arm="cy_2" if fade == 1 else "cy_4", diag=False)
    if 9 <= c <= 36:
        fade = {32: 0.8, 33: 0.6, 34: 0.45, 35: 0.3, 36: 0.15}.get(c, 1.0)
        ring, hotc, star = ("cy_1", "fx_w", "au_1") if 18 <= c <= 31 else ("cy_2", "cy_1", "vi_2")
        if flash:
            ring, hotc, star = "cy_3", "cy_4", "cy_3"
        fx.magic_circle(sc, SKY, 34, 8, c * 6, BACK, progress=min(1.0, (c - 8) / 5.0), fade=fade, ring=ring,
                        hot=hotc, star=star, detail=c >= 12)
        if 22 <= c <= 31:
            k = (c - 22) % 3
            sc.put(ering(SKY, 38 + k * 4, 9.5 + k * 1.2) & dith(1.0 - k * 0.3), "cy_2" if k < 2 else "cy_4", BACK)

    # ---- target sigil on the ground, a guide line, then the hush before the strike
    if 12 <= c <= 20:
        rx = {12: 32, 13: 27, 14: 23, 15: 21}.get(c, 20)
        ring, hotc = ("vi_1", "fx_w") if c >= 18 else ("vi_2", "vi_1")
        fx.magic_circle(sc, TARGET, rx, rx * 0.27, -c * 11, BACK, ring=ring, hot=hotc, star="vi_3", points=4,
                        detail=c >= 14)
    if 15 <= c <= 17:
        col = XS == TARGET[0]
        sc.put(col & (YS > SKY[1] + 8) & (YS < TARGET[1]) & (np.mod(YS - c * 3, 6) < 2), "cy_5", BACK)
    if c == 18:
        sc.put((XS == TARGET[0]) & (YS > SKY[1]) & (YS < 52), "cy_1", BACK)
        fx.sparkle(sc, (TARGET[0], 52), 1, BACK)
    if c == 19:
        sc.put((XS == TARGET[0]) & (YS > SKY[1]) & (YS < TARGET[1]), "fx_w", BACK)
        fx.sparkle(sc, TARGET, 2, FRONT)
    if c == 20:
        colm = (YS > SKY[1]) & (YS < TARGET[1])
        sc.put(colm & (np.abs(XS - TARGET[0]) <= 1) & dith(0.6), "cy_2", BACK)
        sc.put(colm & (XS == TARGET[0]), "fx_w", BACK)
        sc.put(ering(TARGET, 9, 2.6), "fx_w", BACK)
        fx.sparkle(sc, TARGET, 5, FRONT, arm="cy_1")

    # ---- the pillar
    hw = BEAM_HW.get(c, 0)
    if flash:
        colm = (YS <= TARGET[1] + 2)
        d = np.abs(XS - TARGET[0])
        sc.put(colm & (d <= 21), "cy_3", BACK)
        sc.put(colm & (d <= 18), "fx_w", BACK)
        sc.put(ellipse(TARGET, 40, 10), "cy_3", BACK)
        sc.put(ellipse(TARGET, 37, 8.5), "fx_w", BACK)
        sc.put(colm & (d <= 18), "fx_w", BACK)
    if hw:
        if hw >= 1.2:
            fx.beam(sc, TARGET[0], -20, SKY[1], hw * 0.6, c)
            fx.beam(sc, TARGET[0], SKY[1], TARGET[1], hw, c)
        else:
            sc.put((XS == TARGET[0]) & (YS > SKY[1]) & (YS < TARGET[1]) & dith(0.55), "cy_2", BACK)
        if hw >= 6:
            for k in range(3):                                   # rings sliding down the pillar
                yk = SKY[1] + 6 + ((c * 13 + k * 27) % 74)
                rr = ering((TARGET[0], yk), hw + 7 + 2 * (k % 2), 3.2) & (YS >= yk)
                sc.put(rr, "cy_1", FRONT)
                sc.put(rr & (np.abs(XS - TARGET[0]) < hw * 0.8), "cy_2", FRONT)
            for k in range(2):                                   # violet lightning hugging the pillar
                side = -1 if (c + k) % 2 else 1
                x0 = TARGET[0] + side * (hw + rng.uniform(3, 9))
                fx.bolt(sc, (x0, SKY[1] + 6), (x0 + side * rng.uniform(4, 16), TARGET[1] - rng.uniform(0, 5)),
                        rng, 7, "vi_1" if k == 0 else "vi_2", BACK, n=7)
            # base: dome of light on the ground + spikes
            sc.put(ellipse(TARGET, hw + 17, 11) & (YS <= TARGET[1] + 3) & dith(0.6), "cy_2", FRONT)
            sc.put(ellipse(TARGET, hw + 13, 8.5), "cy_1", FRONT)
            sc.put(ellipse(TARGET, hw + 9, 6), "fx_w", FRONT)
            for k, ang in enumerate((22, 38, 58, 122, 142, 158)):
                if (k + c) % 2 == 0:
                    u = unit(ang + rng.uniform(-5, 5))
                    r0, r1 = hw + 12, hw + 22 + rng.uniform(0, 14)
                    sc.put(line_px(v2(TARGET) + u * r0 * v2((1, 0.6)), v2(TARGET) + u * r1 * v2((1, 0.8))),
                           "cy_1", FRONT)
    for c0 in (22, 24, 27):                                      # shockwaves racing over the ground
        a = c - c0
        if 0 <= a <= 8:
            rx = 18 + 13 * a
            col = "fx_w" if a < 2 else ("cy_1" if a < 4 else ("cy_2" if a < 6 else "cy_3"))
            fx.shock(sc, TARGET, rx, rx * 0.2, 2 if a < 3 else 1, col, 1.0 if a < 6 else 0.5)

    # ---- crater with cooling cracks, smoke and embers
    if 26 <= c <= 46:
        fade = {44: 0.75, 45: 0.5, 46: 0.25}.get(c, 1.0)
        crater = ellipse(TARGET, 16, 3.8)
        sc.put(crater & dith(fade), "sil", BACK)
        sc.put(crater & ~nb(crater, 0, 1) & dith(fade), "sm_3", BACK)
        if c <= 44:
            ck = "au_1" if c <= 33 else ("au_2" if c <= 36 else ("au_3" if c <= 39 else ("au_4" if c <= 42 else "vi_4")))
            m = np.zeros(SHAPE, bool)
            for pl in CRACKS:
                m |= polyline([(TARGET[0] + x, TARGET[1] + y) for x, y in pl])
            sc.put(m & dith(1.0 if c <= 42 else 0.6), ck, BACK)
            if c <= 36:
                sc.put(ellipse(TARGET, 4, 1.6), "fx_w" if c <= 33 else "au_1", BACK)
    for k in range(8):
        b = 31 + k * 1.25
        a = c - b
        if a >= 0:
            fade = min(1.0, max(0.0, 1 - (a - 4) / 6.0)) * min(1.0, max(0.0, (47 - c) / 5.0))
            fx.puff(sc, (TARGET[0] + math.sin(k * 2.1 + a * 0.4) * 5 + (k % 3 - 1) * 5, TARGET[1] - 3 - 2.3 * a),
                    min(3.5 + 0.6 * a, 7.5), FRONT, fade=fade, seed=k)
    for e in EMBERS:
        a = c - e["b"]
        if 0 <= a < e["life"] and (a + int(e["ph"] * 3)) % 4 != 3:
            col = ("au_1", "au_2", "au_3", "au_4")[min(3, int(4 * a / e["life"]))]
            sc.put(pixels([(TARGET[0] + e["x0"] + e["sway"] * math.sin(a * 0.9 + e["ph"]),
                            TARGET[1] + e["y0"] - 4 - e["v"] * a - 0.05 * a * a)]), col, FRONT)

    # ---- the mage
    out = draw_mage(sc, pose)
    orb = out["orb"]
    if flash:
        rim_light(sc, "cy_4", silhouette=True)
    elif 22 <= c <= 31:
        rim_light(sc, "cy_1", strong=True)
    elif 32 <= c <= 35:
        rim_light(sc, "cy_2", strong=False)
    elif 18 <= c <= 20:
        rim_light(sc, "cy_2", strong=False)
    elif 10 <= c <= 17:
        rim_light(sc, "cy_3", strong=False)

    # ---- motes rising out of the ground circle
    for m in MOTES:
        a = c - m["b"]
        if 0 <= a < m["life"] and not flash:
            rx = 25
            x = GC[0] + m["dx"]
            y0 = GC[1] + m["dy"] * 7.5 * math.sqrt(max(0.0, 1 - (m["dx"] / rx) ** 2))
            y = y0 - m["v"] * a - 0.08 * a * a
            col = ("fx_w", "cy_1", "cy_2", "cy_3", "cy_4")[min(4, int(5 * a / m["life"]))]
            pts = [(x + m["sway"] * math.sin(a * 0.8 + m["ph"]), y)]
            if m["v"] > 2.6:
                pts.append((pts[0][0], y + 1))
            sc.put(pixels(pts), col, FRONT if m["dy"] > 0 else BACK)
    if 5 <= c <= 39 and not flash:
        h = min(2 + (c - 5) * 0.6, 5) if c <= 31 else max(0, 5 - (c - 31) * 0.8)
        fx.light_wall(sc, GC, 27, 8.1, h, True, 0.3)

    # ---- staff orb: charge, implosion, glint
    if 6 <= c <= 17:
        r = 1.6 + (c - 6) * 0.38
        for s in STREAKS:                                        # light pulled in from all around
            k = c - s["b"]
            if 0 <= k <= 2:
                u = unit(s["ang"])
                d1 = s["d0"] * (1 - k / 3.0) + r
                sc.put(line_px(orb + u * d1, orb + u * max(d1 - 6, r)), ("cy_4", "cy_3", "cy_1")[k], FRONT)
        fx.glow(sc, orb, r)
        fx.sparkle(sc, orb, int(r + 3 + (c % 2) * 2), arm="cy_1", diag=False)
        if c % 2:
            n = int(r * 0.7 + 1)
            sc.put(pixels([(orb[0] + sx * i, orb[1] + sy * i) for i in range(1, n + 1)
                           for sx in (-1, 1) for sy in (-1, 1)]), "cy_2", FRONT)
            sc.put(disc(orb, r * 0.34), "fx_w", FRONT)
        for k in range(3):                                       # orbiting sparks
            a = math.radians(c * 47 + k * 120)
            fx.sparkle(sc, (orb[0] + (r + 6) * math.cos(a), orb[1] + (r + 6) * 0.45 * math.sin(a)), 1, arm="cy_2")
        if c >= 11:                                              # violet static crackling off the orb
            for k in range(2):
                u = unit(rng.uniform(0, 360))
                fx.bolt(sc, orb + u * (r + 1), orb + u * (r + rng.uniform(5, 10)), rng, 2.5,
                        "vi_1" if k else "vi_2", FRONT, n=3)
    if c == 18:
        fx.glow(sc, orb, 6.2)
        sc.put(ering(orb, 14, 14), "cy_2", FRONT)
        fx.sparkle(sc, orb, 9, arm="cy_1", diag=False)
    if c == 19:
        fx.glow(sc, orb, 3.2, halo=False)
        sc.put(ering(orb, 8, 8), "cy_1", FRONT)
    if c == 20:
        fx.glow(sc, orb, 1.6, halo=False)
        fx.sparkle(sc, orb, 8, arm="cy_1")
        sc.put(pixels([(orb[0] + s * i, orb[1] + t * i) for i in (2, 3) for s in (-1, 1) for t in (-1, 1)]),
               "cy_2", FRONT)
    hand = out["hand"]
    if c == 21:
        fx.smear(sc, hand, 25, 39, 106, 55, ("cy_3", "cy_3", "cy_4"))
    if c == 22:
        fx.smear(sc, hand, 26, 39, 78, 15, ("fx_w", "cy_1", "cy_3"))
    if c == 23:
        fx.smear(sc, hand, 30, 39, 46, 15, ("cy_1", "cy_2", "cy_4"), fade=0.5)
    if 22 <= c <= 36:
        r = 3.6 if c <= 30 else max(0.0, 3.6 - (c - 30) * 0.6)
        if r > 1:
            fx.glow(sc, orb, r, halo=c <= 31)
            fx.sparkle(sc, orb, int(r + 2 + (c % 2) * 2), arm="cy_1", diag=False)

    # ---- debris, dust, falling stars
    for d in DEBRIS:
        a = c - d["b"]
        if a >= 0:
            x = TARGET[0] + d["x0"] + d["vx"] * a
            y = TARGET[1] - 2 + d["vy"] * a + 0.5 * a * a
            if y < TARGET[1] + 4:
                pts = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] if d["big"] else [(x, y)]
                sc.put(pixels(pts), "sil", FRONT)
                sc.put(pixels([(x + (1 if d["vx"] < 0 else 0), y)]), "cy_1" if c <= 31 else "sm_2", FRONT)
    for side in (-1, 1):
        for k in range(4):
            a = c - (22 + k)
            if 0 <= a <= 10:
                r = 2.6 + min(a, 3) * 1.0 - max(0, a - 5) * 0.7
                cols = ("cy_1", "sm_1", "sm_2") if c <= 29 else ("sm_1", "sm_2", "sm_3")
                fx.puff(sc, (TARGET[0] + side * (24 + k * 3 + a * 4.6), TARGET[1] - 1 - a * 0.35 + (k % 2) * 4),
                        r * 1.25, FRONT, cols=cols, fade=1.0 if a <= 6 else max(0.0, 1 - (a - 6) / 4.0),
                        seed=k + (side > 0) * 3)
    for i, (s, tgt) in enumerate(METEORS):
        k = c - s
        tgt = v2(tgt)
        ramp = fx.AU if i % 3 == 1 else fx.CY
        if 0 <= k <= 2:
            head = tgt - METEOR_DIR * (3 - k) * 26 + METEOR_DIR * 8
            fx.meteor(sc, head, head - METEOR_DIR * 24, FRONT, ramp)
        j = k - 3
        if j == 0:
            sc.put(ellipse(tgt, 7, 4.2), ramp[2], FRONT)
            sc.put(ellipse(tgt, 5.5, 3.2), "fx_w", FRONT)
            fx.sparkle(sc, tgt, 7, arm=ramp[1])
        if j == 1:
            sc.put(ering(tgt, 10, 3.6), ramp[1], FRONT)
            sc.put(ellipse(tgt + v2((0, -1)), 3.5, 2.5), ramp[2], FRONT)
            fx.sparkle(sc, tgt + v2((0, -2)), 3, arm=ramp[2])
        if j == 2:
            sc.put(ering(tgt, 14, 4.6) & dith(0.6), ramp[3], FRONT)
            fx.puff(sc, tgt + v2((0, -3)), 3.6, FRONT, seed=i)
        if j == 3:
            fx.puff(sc, tgt + v2((1, -6)), 3.2, FRONT, fade=0.5, seed=i)
        if 1 <= j <= 3:
            for sx in (-1, 1):                                    # little stars hopping out of each hit
                fx.sparkle(sc, tgt + v2((sx * (3 + 2.5 * j), -3 - 4.5 * j + j * j * 0.6)), 2 if j < 3 else 1,
                           arm="au_2", core="au_1", diag=False)
    if 33 <= c <= 39:                                            # the pillar breaks up into glitter
        n0 = (c - 33) * 7
        for dx, y in SHARDS[n0:n0 + 9]:
            fx.sparkle(sc, (TARGET[0] + dx, y), 1, FRONT, arm="cy_2" if c < 37 else "cy_4",
                       core="fx_w" if c < 38 else "cy_2")

    # ---- the staff tap that wakes the ground circle
    if c == 2:
        tap = out["staff_bot"] + v2((0, 1))
        fx.sparkle(sc, tap, 4, FRONT, arm="cy_1")
        sc.put(ering(tap, 6, 2), "cy_1", FRONT)
    if c == 3:
        fx.sparkle(sc, out["staff_bot"] + v2((0, 1)), 2, FRONT, arm="cy_2", diag=False)

    # ---- staff planted back on the ground
    if c == 42:
        fx.puff(sc, out["staff_bot"] + v2((2, 0)), 2.2, FRONT)
        fx.sparkle(sc, out["staff_bot"] + v2((-2, -1)), 1, FRONT, arm="au_2", core="au_1")
    if c == 43:
        fx.puff(sc, out["staff_bot"] + v2((4, -1)), 1.8, FRONT, fade=0.5)
    if c == 45:
        fx.sparkle(sc, orb + v2((1, -1)), 4, arm="cy_1")
    if c == 46:
        fx.sparkle(sc, orb + v2((1, -1)), 2, arm="cy_2", diag=False)
    return sc, SHAKE.get(c, (0, 0))


def cast_dur(c):
    if c <= 5:
        return 80
    if c <= 17:
        return 70
    if c <= 20:
        return (70, 70, 110)[c - 18]
    if c == 21:
        return 60
    if c <= 31:
        return 60
    if c <= 39:
        return 70
    return 100 if c < 47 else 160


FRAMES = [("idle", i, 170) for i in range(N_IDLE)] + [("cast", c, cast_dur(c)) for c in range(N_CAST)]


def render(fi):
    kind, i, _ = FRAMES[fi]
    return render_idle(i) if kind == "idle" else render_cast(i)

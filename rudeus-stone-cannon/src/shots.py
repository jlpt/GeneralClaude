"""Insert shots: a close-up of the spinning bullet and a slow-motion tracking shot of its flight."""
import math
import numpy as np
from canvas import C, L, XS, YS, W, H, M, CW, CH, v2, unit, disc, nb, line_pts, smooth01
import world
import fx
import spin
from sukuna import letterbox


def _gradient(sc, c, stops, rmax, layer="Sky", dither=30):
    rad = np.hypot(XS - c[0], YS - c[1])
    t = np.clip(rad / rmax, 0, 1)
    rgb = np.stack([np.interp(t, [s[0] for s in stops], [C[s[1]][k] for s in stops]) for k in range(3)], -1)
    sc.rgb[:] = rgb
    sc.owner[:] = L[layer]
    sc.dith[:] = dither
    return rad


def bullet_closeup(sc, f, t, n, turn, blur, heat):
    """t: frame within the shot (0..n-1)."""
    jit = v2(((f * 7) % 3 - 1, (f * 5) % 3 - 1)) * (0.5 + blur)
    c = v2((156, 90)) + jit
    ang = -4.0
    d, p = unit(ang), v2((-unit(ang)[1], unit(ang)[0]))
    rad = _gradient(sc, c, [(0, "mg2"), (0.25, "mg1"), (0.6, "mg0"), (1.0, "sky0")], 200)
    a = np.degrees(np.arctan2(YS - c[1], XS - c[0]))
    swirl = np.mod(a / 360 * 14 + rad * 0.018 - turn * 2.0, 1.0) < 0.32
    sc.put(swirl & (rad > 40) & (rad < 120), "mg1", "Sky")
    sc.put(swirl & (rad >= 120), "mg0", "Sky")
    # accelerator ring around the bullet's waist, spinning with it
    s = 5.2 + 0.8 * smooth01(t / max(n - 1, 1))
    fx.magic_circle(sc, c - d * 2 * s, p * 15 * s, d * 4.4 * s, turn * 6.0, 1.0, 1.0, layer="Magic", glow=True)
    spin.vortex(sc, c, ang, s, turn, 1.0, front=False, tail=70, layer="Magic", flare=0.05)
    spin.drill_bullet(sc, c, ang, s, turn, heat=heat, blur=blur, layer="Magic", underglow=1.0, f=f)
    spin.spin_rings(sc, c, ang, s, f, 1.0, layer="Magic")
    spin.vortex(sc, c, ang, s, turn, 1.0, front=True, tail=70, layer="Magic", flare=0.05)
    spin.fling(sc, c, ang, s, f, 1.0, n=16, layer="FX", reach=3.0)
    tipc = c + d * spin.TIP * s
    sc.light(np.exp(-((XS - tipc[0]) ** 2 + (YS - tipc[1]) ** 2) / (120 + 60 * heat)), C["mg6"], 0.8 * heat)
    letterbox(sc, min(1.0, (t + 1) / 2), h=16)


FAR = [(37, 8), (17, 4), (7, 1.4), (3.1, 0.6)]
NEAR = [(53, 6), (21, 3), (9, 1.2), (4, 0.5)]
ROCKWALL = None


def _rockwall():
    """A tall slab of the crag's stone texture, used when the target rushes into frame."""
    global ROCKWALL
    if ROCKWALL is None:
        r = np.random.default_rng(9)
        pts = np.stack([r.uniform(-20, 200, 60), r.uniform(-20, 200, 60)], 1)
        d = np.full((CH, CW), 1e9, np.float32)
        idx = np.zeros((CH, CW), np.int32)
        for i, (px, py) in enumerate(pts):
            dd = (XS - px) ** 2 + ((YS - py) * 2.4) ** 2
            closer = dd < d
            d = np.where(closer, dd, d)
            idx = np.where(closer, i, idx)
        tone = r.uniform(1.0, 3.6, len(pts))
        ROCKWALL = (idx, tone)
    return ROCKWALL


def tracking(sc, f, t, n, turn, blur):
    world.draw_sky(sc, f, dark=0.35)
    xi = np.clip((XS + M).astype(int), 0, CW - 1)
    far = YS >= world.ridge(world.XLINE + t * 4.0, 3, 104, 1.0, FAR)[xi]
    sc.put(far, C["mtn0"] * 0.75, "Land")
    sc.put(far & ~nb(far, 0, -1), C["mtn1"] * 0.75, "Land")
    near = YS >= world.ridge(world.XLINE + t * 11.0, 11, 116, 1.0, NEAR)[xi]
    sc.put(near, C["mtn2"] * 0.8, "Land")
    plain = YS >= 128
    tp = np.clip((YS - 128) / 52, 0, 1)
    speed = 30 + 140 * tp
    streak = np.mod(XS + t * speed + (YS * 37 % 23), 18 + 30 * tp) < 6 + 14 * tp
    sc.put(plain, "gr1", "Land")
    sc.put(plain & streak, "gr2", "Land")
    sc.put(plain & streak & (tp > 0.5) & (np.mod(YS, 3) == 0), "gr3", "Land")
    # the bullet, tracked by the camera
    c = v2((150 + t * 1.2, 80 + math.sin(t * 0.7) * 1.0))
    ang = 3.0
    d = unit(ang)
    s = 2.4
    back = c + d * spin.BACK * s
    # energy wake
    wake = (XS < back[0]) & (np.abs(YS - back[1] - (XS - back[0]) * d[1] / d[0]) <
                             1.5 + (back[0] - XS) * 0.025)
    sc.put(wake & ((XS + YS) % 2 == 0), "mg3", "FX")
    sc.put(wake & (np.abs(YS - back[1] - (XS - back[0]) * d[1] / d[0]) < 1.0 + (back[0] - XS) * 0.01), "mg5", "FX")
    sc.light(np.exp(-((XS - c[0]) ** 2 / 900 + (YS - c[1]) ** 2 / 300)), C["mg4"], 0.7)
    # vapour rings peeling off and falling behind
    for k in range(6):
        age = (t + k * 2.4) % 14
        rc = back - d * age * 24
        rr = 9 + age * 2.2
        th = np.linspace(0, 2 * math.pi, int(rr * 6))
        x = rc[0] + np.cos(th) * rr * 0.28
        y = rc[1] + np.sin(th) * rr
        keep = np.ones_like(th, bool) if age < 5 else (np.arange(len(th)) % 2 == 0)
        sc.put_px(x[keep], y[keep], "air2" if age < 4 else "air1", "FX")
    spin.vortex(sc, c, ang, s, turn, 1.0, front=False, tail=150, layer="Magic", flare=0.06, twist=0.16)
    spin.drill_bullet(sc, c, ang, s, turn, heat=1.0, blur=blur, layer="Magic", underglow=0, f=f)
    spin.spin_rings(sc, c, ang, s, f, 1.0, layer="Magic")
    spin.vortex(sc, c, ang, s, turn, 1.0, front=True, tail=150, layer="Magic", flare=0.06, twist=0.16)
    spin.fling(sc, c, ang, s, f, 1.0, n=12, layer="FX", reach=2.2)
    # speed lines rushing past
    r = np.random.default_rng(f)
    for k in range(18):
        y = r.uniform(18, 162)
        x = r.uniform(-40, W)
        ln = r.uniform(30, 110)
        sc.put_px(*line_pts((x, y), (x + ln, y)), "air2" if k % 3 else "white", "FX")
    # the target rushing in for the last frames
    k = t - (n - 4)
    if k >= 0:
        wall_x = 340 - (k + 1) * 34
        idx, tone = _rockwall()
        edge = wall_x + 4 * np.sin(YS * 0.21) + 3 * np.sin(YS * 0.07 + 1)
        wm = XS >= edge
        lvl = np.clip(np.round(tone[idx]), 0, 4).astype(int)
        for i, nm in enumerate(world.RAMP_RK):
            sc.put(wm & (lvl == i), nm, "Crag")
        sc.put(wm & (nb(idx, 0, 1) != idx), "rk0", "Crag")
        sc.put(wm & ~nb(wm, -1, 0), "rk6", "Crag")
        sc.light(np.clip(1 - (XS - edge) / 30, 0, 1) * wm, C["mg4"], 0.6)
    letterbox(sc, 1.0, h=16)

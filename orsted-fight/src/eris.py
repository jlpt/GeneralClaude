"""Eris: port of eris-sword-slash/src/rig.py + render_pose onto local character canvases.

Local space: root (0, 0) is the ground point under her, she faces +x, y points down.
"""
import math
import numpy as np
from px import v2, unit, rot, nb, dilate4, ring4, paint_shaded, blit_sprite, sprite_point, norm
from common import (HAIR, BACK, BODY, HEAD, WPN, FRONT, FXP, torso_frame, prep, solve_arm, auto_smear)
from palette import ERIS as C
from heads import ERIS_HEADS, ERIS_KEY, ERIS_ANCHOR

NAME = "Eris"
CANVAS = (176, 150, -88, -120)   # w, h, x0, y0 of the local canvas
LEG = (13.5, 13.0)
ARM = (8.5, 8.0)
SWORD_LEN = 20
SCAB_ANG = 130

STAND = dict(P=(0, -29), N=(0.5, -43), fn=(-5, -3), ff=(5, -3))


def body_points(P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    return dict(sn=N + right * -4.6 + up * -1.6, sf=N + right * 3.6 + up * -1.6,
                hn=P + right * -2.3, hf=P + right * 2.3)


def scabbard_mouth(P, N):
    return body_points(P, N)["hf"] + rot(v2((2.6, -0.6)), _ang(P, N))


def _ang(P, N):
    d = v2(N) - v2(P)
    return math.degrees(math.atan2(d[1], d[0])) + 90


def hilt_grip(P, N, ang=SCAB_ANG):
    return scabbard_mouth(P, N) + unit(ang + 180) * 3.2


# ---------------------------------------------------------------- parts
def draw_head(cv, neck, variant, offset, k):
    at = v2(neck) + v2(offset)
    blit_sprite(cv, ERIS_HEADS[variant], ERIS_KEY, ERIS_ANCHOR, at, k=k, owner=HEAD)
    return at


def draw_neck(cv, N, head_off, k):
    from px import rot_offset
    dx, dy = rot_offset(0.5, -3, k)
    top = v2(N) + v2(head_off) + v2((dx, dy))
    m = cv.capsule(v2(N) + rot(v2((0.0, 1)), -90 * k), top, 1.1)
    cv.put(ring4(m), C["skin_o"], BODY)
    cv.put(m, C["skin_b"], BODY)


def draw_jacket_tail(cv, P, N, wind=(0.0, 0.0)):
    P, N, up, right, Lt = torso_frame(P, N)
    wx, wy = wind

    def at(u, v):
        return P + right * u + up * (v * Lt)

    pts = [at(-3.0, 0.45), at(-5.2, 0.25), at(-6.5 + wx * 2, -0.25 + wy * 0.2),
           at(-7.5 + wx * 3, -0.55 + wy * 0.4), at(-4.0 + wx * 1.5, -0.42 + wy * 0.2), at(-2.0, -0.1)]
    m = cv.polygon(pts)
    paint_shaded(cv, m, (C["jak_o"], C["jak_d"], C["jak_b"], C["jak_l"]), BODY, shadow=1)


def draw_torso(cv, P, N, twist=0.0):
    P, N, up, right, Lt = torso_frame(P, N)
    X, Y = cv.XS, cv.YS
    dx, dy = X - P[0], Y - P[1]
    v = (dx * up[0] + dy * up[1]) / Lt
    u = dx * right[0] + dy * right[1]
    vs = [-0.32, -0.12, 0.12, 0.32, 0.52, 0.70, 0.86, 1.0, 1.08]
    left = np.interp(v, vs, [-4.6, -5.0, -4.6, -4.0, -4.4, -5.2, -6.2, -6.0, -3.5])
    right_e = np.interp(v, vs, [4.0, 4.6, 4.2, 3.6, 4.6, 5.6, 5.0, 4.6, 2.5])
    body = (v >= -0.32) & (v <= 1.08) & (u >= left) & (u <= right_e)
    body |= (v >= 0.80) & (v <= 1.12) & (u >= left - 0.8) & (u <= right_e + 0.3)

    m = 1.2 + twist
    jn = m - 2.6
    jf = m + 3.2
    shorts = body & (v < 0.10)
    belt = body & (v >= 0.10) & (v < 0.18)
    skin = body & (v >= 0.18) & (v < 0.42)
    top = body & (v >= 0.42) & (v < 0.86)
    collar = (v >= 0.80) & (v <= 1.12) & (u >= left - 0.8) & (u <= right_e + 0.3)
    jacket = body & (v >= 0.06) & ((u < jn) | (u > jf)) & ~collar
    cv.put(ring4(body), C["jak_o"], BODY)
    cv.put(shorts, C["sho_b"], BODY)
    cv.put(shorts & (u > m + 1.5), C["sho_d"], BODY)
    cv.put(shorts & (np.abs(u - (m - 0.4)) < 0.5) & (v < 0.0), C["sho_d"], BODY)
    cv.put(belt, C["belt"], BODY)
    cv.put(belt & (v >= 0.14), C["belt_l"], BODY)
    cv.put(belt & (np.abs(u - (m - 0.5)) < 1.0), C["buckle"], BODY)
    cv.put(skin, C["skin_b"], BODY)
    cv.put(skin & (u > m + 1.8), C["skin_s"], BODY)
    cv.put(skin & (v >= 0.36), C["skin_s"], BODY)
    cv.put(skin & (np.abs(u - (m - 0.3)) < 0.5) & (np.abs(v - 0.26) < 0.04), C["skin_s"], BODY)
    cv.put(top, C["top_b"], BODY)
    cv.put(top & (u < m - 0.5) & (v > 0.56), C["top_l"], BODY)
    cv.put(top & (u > m + 1.5), C["top_s"], BODY)
    cv.put(top & (v < 0.49), C["top_s"], BODY)
    cv.put(top & (np.abs(u - (m + 0.2)) < 0.5) & (v > 0.5), C["top_s"], BODY)
    cv.put(jacket, C["jak_b"], BODY)
    cv.put(jacket & (u < left + 1.2), C["jak_l"], BODY)
    cv.put(jacket & (u > jn - 1.0) & (u < jn), C["jak_d"], BODY)
    cv.put(jacket & (u > jf), C["jak_d"], BODY)
    scallop = (np.floor(u) % 2 == 0)
    fur = collar | ((v >= 0.74) & (v < 0.80) & scallop & (u > left) & (u < right_e))
    cv.put(fur, C["fur_l"], BODY)
    cv.put(fur & ~nb(fur, 0, 1), C["fur_b"], BODY)
    cv.put(fur & (u > m + 2.5), C["fur_b"], BODY)
    cv.put(fur & ~nb(fur, 0, -1) & (u < m), C["fur_h"], BODY)
    cv.put(fur & (u > m + 3.5) & ~nb(fur, 0, 1), C["fur_d"], BODY)
    bow = body & (np.abs(u - (m + 0.3)) < 1.6) & (np.abs(v - 0.84) < 0.05)
    cv.put(bow, C["bow"], BODY)


def draw_leg(cv, hip, ankle, bend, dark=False, inner=1, toe=(1, 0)):
    from px import ik2
    knee, ank = ik2(hip, ankle, LEG[0], LEG[1], bend)
    thigh = cv.capsule(hip, knee, 2.4, 1.7)
    shin = cv.capsule(knee, ank, 1.75, 1.15)
    leg = thigh | shin
    toe = norm(toe)
    perp = v2((-toe[1], toe[0]))
    shoe = cv.capsule(ank + perp * 1.0 - toe * 0.4, ank + perp * 1.6 + toe * 3.6, 1.1, 1.0)
    whole = leg | shoe
    cv.put(ring4(whole), C["leg_o"], BODY)
    rd, rb, rl = (C["leg_d"], C["leg_d"], C["leg_b"]) if dark else (C["leg_d"], C["leg_b"], C["leg_l"])
    cv.put(leg, rb, BODY)
    cv.put(leg & ~nb(leg, 1, 0), rd, BODY)
    cv.put(leg & ~nb(leg, -1, 0) & nb(leg, 1, 0), rl, BODY)
    _, t = cv.seg_t(hip, knee)
    tdir = norm(knee - hip)
    side = ((cv.XS - hip[0]) * -tdir[1] + (cv.YS - hip[1]) * tdir[0]) * inner
    sock = thigh & (((t < 0.62) & (side < -0.2)) | (t < 0.22))
    cv.put(sock, C["sock"], BODY)
    cv.put(sock & ~nb(sock, -1, 0), C["sock_l"], BODY)
    _, t2 = cv.seg_t(knee, ank)
    cv.put(shin & (t2 > 0.86), C["sock"], BODY)
    cv.put(shoe & ~leg, C["shoe_b"], BODY)
    cv.put(shoe & ~leg & ~nb(shoe, 0, -1), C["shoe_l"], BODY)
    return knee, ank


def draw_arm(cv, shoulder, hand_c, bend, owner, dark=False):
    elbow, wrist, hand_c = solve_arm(shoulder, hand_c, bend, *ARM)
    sleeve = cv.capsule(shoulder, elbow, 2.1, 1.8) | cv.capsule(elbow, wrist, 1.8, 2.1)
    hm = cv.disc(hand_c, 1.5) & ~sleeve
    base = C["jak_d"] if dark else C["jak_l"]
    cv.put(ring4(hm), C["skin_o"], owner)
    cv.put(ring4(sleeve), C["jak_o"], owner)
    cv.put(sleeve, base, owner)
    if not dark:
        cv.put(sleeve & ~nb(sleeve, 1, 1), C["jak_b"], owner)
        lt = sleeve & ~(nb(sleeve, -1, 0) & nb(sleeve, 0, -1)) & nb(sleeve, 1, 1)
        cv.put(lt, C["jak_h"], owner)
        _, t = cv.seg_t(elbow, wrist)
        fore = cv.capsule(elbow, wrist, 1.8, 2.1)
        cv.put(fore & (np.abs(t - 0.8) < 0.07), C["stripe"], owner)
    cv.put(hm, C["skin_b"], owner)
    cv.put(hm & ~nb(hm, 1, 1), C["skin_s"], owner)
    return elbow, wrist, hand_c


def draw_blade(cv, guard_c, dirv, length, owner):
    perp = v2((-dirv[1], dirv[0]))
    b0 = guard_c + dirv * 1.0
    tip = guard_c + dirv * length
    blade = cv.capsule(b0, tip, 1.05, 0.55)
    cv.put(ring4(blade), C["bl_o"], owner)
    side = (cv.XS - b0[0]) * perp[0] + (cv.YS - b0[1]) * perp[1]
    lit_sign = -1 if (perp[0] + perp[1]) > 0 else 1
    cv.put(blade, C["bl_b"], owner)
    cv.put(blade & (side * lit_sign > 0), C["bl_l"], owner)
    _, t = cv.seg_t(b0, tip)
    cv.put(blade & (t > 0.9), C["bl_h"], owner)
    return tip


def draw_hilt(cv, guard_c, dirv, owner, grip_len=4.0):
    perp = v2((-dirv[1], dirv[0]))
    pommel = guard_c - dirv * (grip_len + 1.0)
    grip = cv.capsule(guard_c - dirv * 0.5, pommel, 0.75)
    guard = cv.capsule(guard_c - perp * 2.4, guard_c + perp * 2.4, 0.7)
    pm = cv.disc(pommel, 1.0)
    allm = grip | guard | pm
    cv.put(ring4(allm), C["bl_o"], owner)
    cv.put(grip, C["grip"], owner)
    _, t = cv.seg_t(guard_c, pommel)
    cv.put(grip & (np.floor(t * grip_len * 1.4) % 2 == 0), C["grip_l"], owner)
    cv.put(guard | pm, C["guard"], owner)
    cv.put((guard | pm) & ~nb(guard | pm, 1, 1), C["guard_d"], owner)


def sword_geom(grip_pt, ang):
    dirv = unit(ang)
    guard_c = v2(grip_pt) + dirv * 2.2
    return guard_c, guard_c + dirv * SWORD_LEN


def draw_sword(cv, grip_pt, ang, owner=WPN):
    dirv = unit(ang)
    guard_c, tip = sword_geom(grip_pt, ang)
    draw_hilt(cv, guard_c, dirv, owner)
    draw_blade(cv, guard_c, dirv, SWORD_LEN, owner)
    return guard_c, tip


def draw_scabbard(cv, mouth, ang, sheathed):
    dirv = unit(ang)
    mouth = v2(mouth)
    end = mouth + dirv * 21
    m = cv.capsule(mouth, end, 1.2, 0.9)
    cv.put(ring4(m), C["jak_o"], BODY)
    cv.put(m, C["scab"], BODY)
    cv.put(m & ~(nb(m, -1, 0) & nb(m, 0, -1)), C["scab_l"], BODY)
    _, t = cv.seg_t(mouth, end)
    cv.put(m & ((t > 0.94) | (t < 0.06)), C["scab_tip"], BODY)
    if sheathed:
        draw_hilt(cv, mouth - dirv * 0.5, -dirv, BODY)


# ---------------------------------------------------------------- hair
BACK_STRANDS = [
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


def strand_mask(cv, head_at, k, spec, wind, phase):
    (rx, ry), a0, length, r0, _tone, ph, curl = spec
    pos = sprite_point(ERIS_ANCHOR, head_at, (rx, ry), k)
    a0 = a0 - 90 * k
    wind = v2(wind)
    n = 28
    seg = length / n
    m = cv.empty()
    upper = cv.empty()
    for i in range(n + 1):
        t = i / n
        r = r0 * (1.0 - 0.8 * t) * (0.85 + 0.35 * math.sin(math.pi * min(1.0, t * 1.3)))
        dm = cv.disc(pos, max(r, 0.55))
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


def draw_hair_back(cv, head_at, k, wind, phase):
    tones = [
        (C["hair_o"], C["hair_d"], C["hair_d"], C["hair_s"]),
        (C["hair_o"], C["hair_d"], C["hair_s"], C["hair_b"]),
        (C["hair_o"], C["hair_s"], C["hair_b"], C["hair_l"]),
    ]
    for spec in BACK_STRANDS:
        m, upper = strand_mask(cv, head_at, k, spec, wind, phase)
        paint_shaded(cv, m, tones[spec[4]], HAIR, inner_col=C["hair_d"] if spec[4] else C["hair_o"])
        if spec[4] == 2:
            shine = upper & m & nb(m, -2, 0) & ~nb(m, -3, 0)
            cv.put(shine, C["hair_h"], HAIR)


def draw_front_lock(cv, head_at, k, wind, phase):
    m, _ = strand_mask(cv, head_at, k, FRONT_LOCK, v2(wind) * 0.6, phase)
    paint_shaded(cv, m, (C["hair_o"], C["hair_s"], C["hair_b"], C["hair_l"]), HEAD)


# ---------------------------------------------------------------- full render
def render(cv, pose, prev=None):
    """Draw Eris for `pose` (already interpolated) into cv. Returns attachment points (local)."""
    p = prep(pose)
    prev = prep(prev) if prev is not None else None
    P, N = p["P"], p["N"]
    bp = body_points(P, N)
    k = p["hk"]
    head_off = p.get("head", v2((0, 0)))
    head_at = v2(N) + head_off
    wind = v2(p.get("wind", (0, 0)))
    phase = p.get("phase", 0.0)
    spin = p.get("spin", 0.0)
    state = p.get("wstate", "sheathed")
    held = state == "held"
    scab_ang = p.get("scab_ang", SCAB_ANG) + spin
    wz = p.get("wz", "front")
    toe = rot(v2((1, 0)), spin + p.get("toe_rot", 0))

    near_hand = v2(p["hn"])
    far_hand = v2(p["hf"])
    if held and p.get("two"):
        far_hand = near_hand - unit(p["wang"]) * 2.8
    out = {}

    def near_arm():
        out["hn"] = draw_arm(cv, bp["sn"], near_hand, p.get("bn", 1), FRONT)[2]

    def far_arm():
        out["hf"] = draw_arm(cv, bp["sf"], far_hand, p.get("bf", 1), BACK, dark=p.get("zf", "back") == "back")[2]

    def sword():
        g, t = draw_sword(cv, near_hand, p["wang"])
        out["guard"], out["tip"] = g, t

    draw_hair_back(cv, head_at, k, wind, phase)
    if p.get("zf", "back") == "back":
        far_arm()
    draw_jacket_tail(cv, P, N, p.get("coat", (0, 0)))
    if held and wz == "under":
        sword()
    draw_scabbard(cv, scabbard_mouth(P, N), scab_ang, state == "sheathed")
    if held and wz == "back":
        sword()
    draw_leg(cv, bp["hf"], p["ff"], p.get("kf", -1), dark=True, inner=-1, toe=toe)
    draw_leg(cv, bp["hn"], p["fn"], p.get("kn", -1), inner=1, toe=toe)
    draw_neck(cv, N, head_off, k)
    draw_torso(cv, P, N, p.get("twist", 0.0))
    if p.get("zn") == "mid":
        near_arm()
    draw_head(cv, N, p.get("expr", "smirk"), head_off, k)
    draw_front_lock(cv, head_at, k, wind, phase)
    if p.get("zf") == "front":
        far_arm()
    if held:
        guard_c, tip = sword_geom(near_hand, p["wang"])
        auto_smear(cv, p, prev, bp["sn"] + v2((1, 3)), guard_c, tip, warm=p.get("warm", False))
    if held and wz == "front":
        sword()
    if p.get("zn", "front") == "front":
        near_arm()
    if held and "tip" not in out:
        out["guard"], out["tip"] = sword_geom(near_hand, p["wang"])
    out["head"] = head_at + rot(v2((0, -8)), -90 * k)
    out["chest"] = P + (N - P) * 0.7
    out["P"] = P
    return out

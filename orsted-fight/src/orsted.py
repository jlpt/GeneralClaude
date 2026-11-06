"""Orsted: towering, silver hair swept back, golden eyes, long white coat over black. Fights bare-handed."""
import math
import numpy as np
from px import v2, unit, rot, nb, ring4, paint_shaded, blit_sprite, norm, rot_offset, sprite_point
from common import (HAIR, BACK, BODY, HEAD, WPN, FRONT, FXP, torso_frame, prep, draw_arm, draw_leg,
                    coat_back, coat_panel, aura)
from palette import ORSTED as C, FX
from heads import ORSTED_HEADS, ORSTED_KEY, ORSTED_ANCHOR

NAME = "Orsted"
CANVAS = (220, 190, -110, -150)
LEG = (18.6, 18.2)
ARM = (11.2, 10.6)

STAND = dict(P=(0, -40), N=(0.5, -61), fn=(-6, -3), ff=(7, -3))

COAT = (C["co_o"], C["co_s"], C["co_b"], C["co_l"])
COAT_DK = (C["co_o"], C["co_d"], C["co_s"], C["co_b"])
SLEEVE = (C["co_o"], C["co_s"], C["co_b"], C["co_l"], C["co_l"])
SLEEVE_DK = (C["co_o"], C["co_d"], C["co_s"], C["co_b"], C["co_b"])
SKIN = (C["skin_o"], C["skin_s"], C["skin_b"])
PANTS = (C["bk_o"], C["bk_d"], C["bk_b"], C["bk_l"])
BOOT = (C["bt_b"], C["bt_l"])

# short back locks of silver hair that stream with the wind (sprite px, angle, length, radius)
BACK_LOCKS = [((2.5, 7.0), 150, 6, 2.2), ((2.0, 10.0), 135, 7, 2.0), ((3.5, 13.0), 118, 6, 1.8),
              ((4.5, 16.0), 104, 5, 1.6)]


def body_points(P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    return dict(sn=N + right * -5.6 + up * -2.4, sf=N + right * 4.6 + up * -2.4,
                hn=P + right * -2.9, hf=P + right * 2.9)


def draw_torso(cv, P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    X, Y = cv.XS, cv.YS
    dx, dy = X - P[0], Y - P[1]
    v = (dx * up[0] + dy * up[1]) / Lt
    u = dx * right[0] + dy * right[1]
    vs = [-0.18, 0.0, 0.2, 0.45, 0.7, 0.88, 1.0, 1.08]
    left = np.interp(v, vs, [-5.6, -5.8, -5.2, -5.6, -7.0, -7.4, -6.6, -4.0])
    right_e = np.interp(v, vs, [5.0, 5.2, 4.6, 5.0, 6.2, 6.0, 5.2, 3.4])
    body = (v >= -0.18) & (v <= 1.08) & (u >= left) & (u <= right_e)
    m = 1.0
    open_front = (u > m - 1.4 - (v - 0.2) * 1.2) & (u < m + 2.8 + (v - 0.2) * 0.8) & (v < 0.96)
    shirt = body & open_front
    coatm = body & ~open_front
    cv.put(ring4(body), C["co_o"], BODY)
    cv.put(coatm, C["co_b"], BODY)
    cv.put(coatm & (u < left + 1.4), C["co_l"], BODY)
    cv.put(coatm & (u > m + 2.5), C["co_s"], BODY)
    cv.put(coatm & (u > right_e - 1.2), C["co_d"], BODY)
    cv.put(ring4(shirt) & coatm, C["co_trim"], BODY)                # gold edging along the opening
    cv.put(shirt, C["bk_b"], BODY)
    cv.put(shirt & (u > m + 1.2), C["bk_d"], BODY)
    cv.put(shirt & (u < m - 0.2) & (v > 0.45), C["bk_l"], BODY)
    cv.put(shirt & (np.abs(v - 0.58) < 0.03) & (u > m - 0.5), C["bk_d"], BODY)
    belt = shirt & (v >= 0.06) & (v < 0.14)
    cv.put(belt, C["belt"], BODY)
    cv.put(belt & (np.abs(u - (m + 0.6)) < 0.9), C["buckle"], BODY)
    # standing collar, folded open
    col = (v > 0.86) & (v <= 1.16) & (u >= left - 0.6) & (u <= right_e + 0.4) & ~(open_front & (v < 1.0))
    col &= ~((u > m - 0.6) & (u < m + 2.0))
    cv.put(ring4(col) & ~body, C["co_o"], BODY)
    cv.put(col, C["co_l"], BODY)
    cv.put(col & ~nb(col, 0, 1), C["co_s"], BODY)
    cv.put(col & (u > m + 2.0), C["co_s"], BODY)


def draw_back_hair(cv, head_at, k, wind, phase):
    for i, ((sx, sy), a0, ln, r0) in enumerate(BACK_LOCKS):
        pos = sprite_point(ORSTED_ANCHOR, head_at, (sx, sy), k)
        m = cv.empty()
        for j in range(9):
            t = j / 8
            d = norm(unit(a0 - 90 * k) + v2(wind) * 0.9 * t)
            d = rot(d, math.sin(phase + i + t * 3) * 12 * t)
            m |= cv.disc(pos, max(r0 * (1 - 0.75 * t), 0.5))
            pos = pos + d * (ln / 8)
        paint_shaded(cv, m, (C["hair_o"], C["hair_d"], C["hair_s"], C["hair_b"]), HAIR)


def render(cv, pose, prev=None):
    p = prep(pose)
    P, N = p["P"], p["N"]
    bp = body_points(P, N)
    k = p["hk"]
    head_off = p.get("head", v2((0, 0)))
    head_at = N + head_off
    spin = p.get("spin", 0.0)
    toe = rot(v2((1, 0)), spin + p.get("toe_rot", 0))
    coat = v2(p.get("coat", (0, 0))) + v2((-0.4, 0))
    phase = p.get("phase", 0.0)
    out = {}
    hs_n, hs_f = p.get("shape_n", "fist"), p.get("shape_f", "fist")

    draw_back_hair(cv, head_at, k, v2(p.get("wind", (0, 0))), phase)
    coat_back(cv, P, N, 5.6, 1.6, coat * 1.3, COAT_DK, BACK, hem_teeth=4, phase=phase)
    if p.get("zf", "back") == "back":
        out["hf"] = draw_arm(cv, bp["sf"], p["hf"], p.get("bf", 1), *ARM, SLEEVE_DK, SKIN, BACK, dark=True,
                             r_up=(2.6, 2.1), r_fore=(2.1, 2.4), hand_shape=hs_f, hand_scale=1.15)[2]
    kf, _, _, _ = draw_leg(cv, bp["hf"], p["ff"], p.get("kf", -1), *LEG, PANTS, BOOT, BODY, dark=True, toe=toe,
                           r=(2.8, 2.0, 2.0, 1.4), boot_from=0.62, shoe_len=4.2)
    coat_panel(cv, bp["hf"], kf, 3.2, 3.6, 7.0, COAT, BODY, hang=0.55, shadow_side=True)
    kn, _, _, _ = draw_leg(cv, bp["hn"], p["fn"], p.get("kn", -1), *LEG, PANTS, BOOT, BODY, toe=toe,
                           r=(2.8, 2.0, 2.0, 1.4), boot_from=0.62, shoe_len=4.2)
    coat_panel(cv, bp["hn"] + v2((-0.8, 0)), kn, 3.6, 3.8, 7.0, COAT, BODY, hang=0.55, trim=C["co_trim_d"])
    dx, dy = rot_offset(0.5, -3.5, k)
    neck = cv.capsule(N, N + head_off + v2((dx, dy)), 1.7)
    cv.put(ring4(neck), C["skin_o"], BODY)
    cv.put(neck, C["skin_s"], BODY)
    draw_torso(cv, P, N)
    if p.get("zn") == "mid":
        out["hn"] = draw_arm(cv, bp["sn"], p["hn"], p.get("bn", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_up=(2.6, 2.1), r_fore=(2.1, 2.4), hand_shape=hs_n, hand_scale=1.15)[2]
    blit_sprite(cv, ORSTED_HEADS[p.get("expr", "cold")], ORSTED_KEY, ORSTED_ANCHOR, head_at, k=k, owner=HEAD)
    if p.get("zf") == "front":
        out["hf"] = draw_arm(cv, bp["sf"], p["hf"], p.get("bf", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_up=(2.6, 2.1), r_fore=(2.1, 2.4), hand_shape=hs_f, hand_scale=1.15)[2]
    if p.get("zn", "front") == "front":
        out["hn"] = draw_arm(cv, bp["sn"], p["hn"], p.get("bn", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_up=(2.6, 2.1), r_fore=(2.1, 2.4), hand_shape=hs_n, hand_scale=1.15)[2]
    out["eyes"] = sprite_point(ORSTED_ANCHOR, head_at, (14.5, 11), k)
    out["eye_n"] = sprite_point(ORSTED_ANCHOR, head_at, (8.5, 11), k)
    out["head"] = head_at + rot(v2((0, -9)), -90 * k)
    out["chest"] = P + (N - P) * 0.7
    out["P"] = P
    return out

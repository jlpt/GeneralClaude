"""Ruijerd: tall Superd warrior, shaved head with a forehead jewel, slate tunic, white trident."""
import math
import numpy as np
from px import v2, unit, rot, nb, ring4, paint_shaded, blit_sprite, norm, rot_offset
from common import (HAIR, BACK, BODY, HEAD, WPN, FRONT, FXP, torso_frame, prep, draw_arm, draw_leg,
                    coat_back, coat_panel, thrust_lines, auto_smear)
from palette import RUIJERD as C, FX
from heads import RUIJERD_HEADS, RUIJERD_KEY, RUIJERD_ANCHOR

NAME = "Ruijerd"
CANVAS = (220, 170, -110, -140)
LEG = (16.2, 15.8)
ARM = (9.8, 9.2)
SPEAR_FWD, SPEAR_BACK = 36, 17

STAND = dict(P=(0, -35), N=(0.5, -52), fn=(-6, -3), ff=(6, -3))

TUNIC = (C["tu_o"], C["tu_d"], C["tu_b"], C["tu_l"])
TUNIC_DK = (C["tu_o"], C["tu_o"], C["tu_d"], C["tu_b"])
SLEEVE = (C["tu_o"], C["tu_d"], C["tu_b"], C["tu_l"], C["tu_h"])
SLEEVE_DK = (C["tu_o"], C["tu_o"], C["tu_d"], C["tu_b"], C["tu_l"])
SKIN = (C["skin_o"], C["skin_s"], C["skin_b"])
PANTS = (C["tr_o"], C["tr_d"], C["tr_b"], C["tr_l"])
BOOT = (C["sa_d"], C["sa_b"])
SASH = (C["tr_o"], C["sa_d"], C["sa_b"], C["sa_l"])


def body_points(P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    return dict(sn=N + right * -4.8 + up * -2.0, sf=N + right * 4.0 + up * -2.0,
                hn=P + right * -2.5, hf=P + right * 2.5)


def spear_geom(hand, ang):
    d = unit(ang)
    return hand - d * SPEAR_BACK, hand + d * SPEAR_FWD


def draw_spear(cv, hand, ang, owner=WPN):
    d = unit(ang)
    perp = v2((-d[1], d[0]))
    butt, tip = spear_geom(hand, ang)
    base = tip - d * 8.0
    shaft = cv.capsule(butt, base, 0.75)
    cross = cv.capsule(base - perp * 3.2, base + perp * 3.2, 0.7)
    mid = cv.capsule(base, tip, 0.9, 0.35)
    side_a = cv.capsule(base + perp * 3.0, base + perp * 3.0 + d * 5.0, 0.7, 0.3)
    side_b = cv.capsule(base - perp * 3.0, base - perp * 3.0 + d * 5.0, 0.7, 0.3)
    head = cross | mid | side_a | side_b
    allm = shaft | head
    cv.put(ring4(allm), C["sp_o"], owner)
    cv.put(allm, C["sp_b"], owner)
    cv.put(allm & ~nb(allm, 1, 1), C["sp_d"], owner)
    cv.put(allm & ~nb(allm, -1, -1) & nb(allm, 1, 1), C["sp_l"], owner)
    _, t = cv.seg_t(base, tip)
    cv.put(mid & (t > 0.75), C["sp_h"], owner)
    _, ts = cv.seg_t(butt, base)
    cv.put(shaft & (np.abs(ts - 0.9) < 0.04), C["sp_band"], owner)
    cv.put(shaft & (np.abs(ts - 0.03) < 0.04), C["sp_band"], owner)
    return butt, tip


def draw_torso(cv, P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    X, Y = cv.XS, cv.YS
    dx, dy = X - P[0], Y - P[1]
    v = (dx * up[0] + dy * up[1]) / Lt
    u = dx * right[0] + dy * right[1]
    vs = [-0.2, 0.0, 0.2, 0.45, 0.7, 0.9, 1.06]
    left = np.interp(v, vs, [-4.8, -5.0, -4.6, -5.0, -6.0, -6.0, -3.0])
    right_e = np.interp(v, vs, [4.4, 4.6, 4.0, 4.6, 5.4, 5.0, 2.6])
    body = (v >= -0.2) & (v <= 1.06) & (u >= left) & (u <= right_e)
    m = 1.0
    sash = body & (v >= 0.04) & (v < 0.16)
    cv.put(ring4(body), C["tu_o"], BODY)
    cv.put(body, C["tu_b"], BODY)
    cv.put(body & (u < left + 1.3), C["tu_l"], BODY)
    cv.put(body & (u > m + 2.4), C["tu_d"], BODY)
    # wrap-front seam running diagonally from the far shoulder to the near hip
    seam = body & (np.abs(u - (m - 2.0 + (v - 0.2) * 4.0)) < 0.55) & (v > 0.2) & (v < 0.95)
    cv.put(seam, C["tu_d"], BODY)
    # pecs / shoulder plates of shade
    cv.put(body & (v > 0.5) & (v < 0.56) & (u > m - 0.5) & (u < m + 3.5), C["tu_d"], BODY)
    cv.put(sash, C["sa_b"], BODY)
    cv.put(sash & (v < 0.08), C["sa_d"], BODY)
    cv.put(sash & (v > 0.12) & (u < m), C["sa_l"], BODY)
    # high collar
    col = body & (v > 0.92) & ((u < m - 0.5) | (u > m + 1.5))
    cv.put(col, C["tu_l"], BODY)


def draw_sash_tail(cv, P, N, wind, phase):
    P, N, up, right, Lt = torso_frame(P, N)
    root = P + right * -4.2 + up * (0.1 * Lt)
    pos = root
    m = cv.empty()
    for i in range(10):
        t = i / 9
        d = norm(v2((-0.6, 1.0)) + v2(wind) * (0.6 + t))
        wave = math.sin(phase + t * 5) * 0.6
        pos = pos + d * 1.6 + v2((-d[1], d[0])) * wave
        m |= cv.disc(pos, 1.3 - 0.5 * t)
    paint_shaded(cv, m, SASH, BACK)


def render(cv, pose, prev=None):
    p = prep(pose)
    prev = prep(prev) if prev is not None else None
    P, N = p["P"], p["N"]
    bp = body_points(P, N)
    k = p["hk"]
    head_off = p.get("head", v2((0, 0)))
    spin = p.get("spin", 0.0)
    toe = rot(v2((1, 0)), spin + p.get("toe_rot", 0))
    coat = v2(p.get("coat", (0, 0)))
    state = p.get("wstate", "held")
    wz = p.get("wz", "front")
    near_hand = v2(p["hn"])
    far_hand = v2(p["hf"])
    if p.get("two") and "wang" in p:
        far_hand = near_hand + unit(p["wang"]) * p.get("two_gap", 10.0)
    out = {}

    def spear():
        if state != "held":
            return
        out["butt"], out["tip"] = draw_spear(cv, near_hand, p["wang"])

    draw_sash_tail(cv, P, N, v2(p.get("wind", (0, 0))) + coat * 0.5, p.get("phase", 0))
    coat_back(cv, P, N, 4.4, 0.62, coat * 0.6, TUNIC_DK, BACK, top=0.2, hem_teeth=2, phase=p.get("phase", 0))
    if p.get("zf", "back") == "back":
        out["hf"] = draw_arm(cv, bp["sf"], far_hand, p.get("bf", 1), *ARM, SLEEVE_DK, SKIN, BACK, dark=True,
                             r_up=(2.3, 1.9), r_fore=(1.8, 1.6))[2]
    if wz == "back":
        spear()
    kf, _, _, _ = draw_leg(cv, bp["hf"], p["ff"], p.get("kf", -1), *LEG, PANTS, BOOT, BODY, dark=True, toe=toe,
                           r=(2.6, 1.9, 1.9, 1.3), boot_from=0.55, shoe_len=3.8)
    coat_panel(cv, bp["hf"], kf, 2.9, 2.6, -5.0, TUNIC, BODY, hang=0.2, shadow_side=True)
    kn, _, _, _ = draw_leg(cv, bp["hn"], p["fn"], p.get("kn", -1), *LEG, PANTS, BOOT, BODY, toe=toe,
                           r=(2.6, 1.9, 1.9, 1.3), boot_from=0.55, shoe_len=3.8)
    coat_panel(cv, bp["hn"] + v2((-0.5, 0)), kn, 3.0, 2.7, -5.0, TUNIC, BODY, hang=0.2)
    dx, dy = rot_offset(0.5, -3.5, k)
    neck = cv.capsule(N, N + head_off + v2((dx, dy)), 1.5)
    cv.put(ring4(neck), C["skin_o"], BODY)
    cv.put(neck, C["skin_s"], BODY)
    draw_torso(cv, P, N)
    if p.get("zn") == "mid":
        out["hn"] = draw_arm(cv, bp["sn"], near_hand, p.get("bn", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_up=(2.3, 1.9), r_fore=(1.8, 1.6))[2]
    blit_sprite(cv, RUIJERD_HEADS[p.get("expr", "stern")], RUIJERD_KEY, RUIJERD_ANCHOR, N + head_off, k=k,
                owner=HEAD)
    if p.get("zf") == "front":
        out["hf"] = draw_arm(cv, bp["sf"], far_hand, p.get("bf", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_up=(2.3, 1.9), r_fore=(1.8, 1.6))[2]
    if state == "held" and wz == "front":
        butt, tip = spear_geom(near_hand, p["wang"])
        auto_smear(cv, p, prev, bp["sn"] + v2((1, 3)), near_hand + unit(p["wang"]) * 10, tip, threshold=40)
        if p.get("thrust"):
            thrust_lines(cv, butt, tip, n=3, back=18)
        spear()
    if p.get("zn", "front") == "front":
        out["hn"] = draw_arm(cv, bp["sn"], near_hand, p.get("bn", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_up=(2.3, 1.9), r_fore=(1.8, 1.6))[2]
    if state == "held" and "tip" not in out:
        out["butt"], out["tip"] = spear_geom(near_hand, p["wang"])
    # the jewel glows through everything
    gem = N + head_off + v2(rot_offset(10 - RUIJERD_ANCHOR[0], 3.5 - RUIJERD_ANCHOR[1], k))
    out["gem"] = gem
    out["head"] = N + head_off + rot(v2((0, -8)), -90 * k)
    out["chest"] = P + (N - P) * 0.7
    out["P"] = P
    return out

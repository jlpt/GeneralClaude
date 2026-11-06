"""Rudeus: young mage in a grey hooded robe with the staff Aqua Heartia (blue crystal head)."""
import math
import numpy as np
from px import v2, unit, rot, nb, ring4, paint_shaded, blit_sprite, ik2, norm, rot_offset
from common import (HAIR, BACK, BODY, HEAD, WPN, FRONT, FXP, torso_frame, prep, draw_arm, draw_leg,
                    coat_back, coat_panel, auto_smear)
from palette import RUDEUS as C, FX
from heads import RUDEUS_HEADS, RUDEUS_KEY, RUDEUS_ANCHOR

NAME = "Rudeus"
CANVAS = (176, 150, -88, -120)
LEG = (12.2, 11.8)
ARM = (7.6, 7.2)
STAFF_FWD, STAFF_BACK = 25, 12

STAND = dict(P=(0, -26), N=(0.5, -39), fn=(-5, -3), ff=(5, -3))

ROBE = (C["rob_o"], C["rob_d"], C["rob_b"], C["rob_l"])
ROBE_DK = (C["rob_o"], C["rob_in"], C["rob_d"], C["rob_b"])
SLEEVE = (C["rob_o"], C["rob_d"], C["rob_b"], C["rob_l"], C["rob_h"])
SLEEVE_DK = (C["rob_o"], C["rob_in"], C["rob_d"], C["rob_b"], C["rob_l"])
SKIN = (C["skin_o"], C["skin_s"], C["skin_b"])
PANTS = (C["tr_o"], C["tr_d"], C["tr_b"], C["tr_l"])
BOOT = (C["bt_b"], C["bt_l"])


def body_points(P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    return dict(sn=N + right * -4.0 + up * -1.6, sf=N + right * 3.2 + up * -1.6,
                hn=P + right * -2.2, hf=P + right * 2.2)


def staff_geom(hand, ang):
    d = unit(ang)
    return hand - d * STAFF_BACK, hand + d * STAFF_FWD


def draw_staff(cv, hand, ang, glow=0.0, owner=WPN):
    d = unit(ang)
    butt, head = staff_geom(hand, ang)
    perp = v2((-d[1], d[0]))
    shaft = cv.capsule(butt, head - d * 2.0, 0.8)
    # crook: a curl of wood wrapping the crystal
    ring = cv.disc(head + d * 1.5, 3.4) & ~cv.disc(head + d * 1.5, 2.0)
    prong = cv.capsule(head + d * 1.5 + perp * 3.0, head + d * 5.5 + perp * 1.2, 0.7)
    wood = shaft | ring | prong
    cv.put(ring4(wood), C["st_o"], owner)
    cv.put(wood, C["st_b"], owner)
    cv.put(wood & ~nb(wood, 1, 1), C["st_d"], owner)
    cv.put(wood & ~nb(wood, -1, -1) & nb(wood, 1, 1), C["st_l"], owner)
    # binding near the head
    _, t = cv.seg_t(butt, head)
    cv.put(shaft & (np.abs(t - 0.86) < 0.025), C["belt_l"], owner)
    # crystal
    gc = head + d * 1.5
    gem = cv.polygon([gc + d * 2.4, gc + perp * 1.6, gc - d * 2.0, gc - perp * 1.6])
    cv.put(ring4(gem) & ~wood, C["gem_o"], owner)
    cv.put(gem, C["gem_b"], owner)
    cv.put(gem & ~nb(gem, 1, 1), C["gem_d"], owner)
    cv.put(gem & ~nb(gem, -1, -1), C["gem_l"], owner)
    cv.put(cv.point(gc + v2((-0.6, -0.8))), C["gem_h"], owner)
    if glow > 0:
        g = cv.disc(gc, 3.5 + glow * 3) & ~cv.disc(gc, 2.6 + glow * 2) & cv.checker()
        cv.put(g & (cv.rgba[:, :, 3] == 0), FX["c2"], FXP)
        if glow > 0.5:
            cv.put(gem, C["gem_l"], owner)
            cv.put(cv.point(gc) | cv.point(gc + d), FX["w"], owner)
    return butt, head, gc


def draw_torso(cv, P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    X, Y = cv.XS, cv.YS
    dx, dy = X - P[0], Y - P[1]
    v = (dx * up[0] + dy * up[1]) / Lt
    u = dx * right[0] + dy * right[1]
    vs = [-0.25, 0.0, 0.25, 0.5, 0.75, 0.95, 1.08]
    left = np.interp(v, vs, [-4.4, -4.6, -4.2, -4.4, -5.2, -5.4, -3.2])
    right_e = np.interp(v, vs, [4.0, 4.2, 3.8, 4.0, 4.6, 4.2, 2.4])
    body = (v >= -0.25) & (v <= 1.08) & (u >= left) & (u <= right_e)
    m = 1.0
    shirt = body & (u > m - 1.6) & (u < m + 2.6) & (v > 0.12)
    belt = body & (v >= 0.10) & (v < 0.2)
    robe = body & ~shirt
    cv.put(ring4(body), C["rob_o"], BODY)
    cv.put(robe, C["rob_b"], BODY)
    cv.put(robe & (u < left + 1.2), C["rob_l"], BODY)
    cv.put(robe & (u > m + 2.6), C["rob_d"], BODY)
    cv.put(robe & (u > m - 2.6) & (u < m - 1.6), C["rob_d"], BODY)   # inner lapel shadow
    cv.put(shirt, C["sh_b"], BODY)
    cv.put(shirt & (u > m + 1.4), C["sh_s"], BODY)
    cv.put(shirt & (u < m - 0.6) & (v > 0.4), C["sh_l"], BODY)
    cv.put(shirt & (np.abs(u - (m + 0.4)) < 0.5) & (v > 0.55) & (v < 0.95), C["sh_s"], BODY)
    cv.put(belt, C["belt"], BODY)
    cv.put(belt & (v >= 0.15), C["belt_l"], BODY)
    # pouch on the far hip
    pouch = cv.disc(P + right * 3.6 + up * (0.05 * Lt), 1.6)
    cv.put(ring4(pouch), C["tr_o"], BODY)
    cv.put(pouch, C["belt"], BODY)
    # collar of the robe
    col = body & (v > 0.86) & ((u < m - 0.8) | (u > m + 2.0))
    cv.put(col, C["rob_l"], BODY)
    cv.put(col & ~nb(col, 0, -1), C["rob_h"], BODY)


def draw_hood(cv, P, N, k):
    P, N, up, right, Lt = torso_frame(P, N)
    c = N + right * -3.8 + up * 0.5
    m = cv.ellipse(c, 3.6, 2.8) | cv.disc(N + right * -1.5 + up * -0.4, 2.6)
    paint_shaded(cv, m, ROBE, BODY)


def render(cv, pose, prev=None):
    p = prep(pose)
    prev = prep(prev) if prev is not None else None
    P, N = p["P"], p["N"]
    bp = body_points(P, N)
    k = p["hk"]
    head_off = p.get("head", v2((0, 0)))
    spin = p.get("spin", 0.0)
    toe = rot(v2((1, 0)), spin + p.get("toe_rot", 0))
    coat = v2(p.get("coat", (0, 0))) + v2((-0.3, 0))
    state = p.get("wstate", "held")
    wz = p.get("wz", "front")
    near_hand = v2(p["hn"])
    far_hand = v2(p["hf"])
    if p.get("two") and "wang" in p:
        far_hand = near_hand + unit(p["wang"]) * p.get("two_gap", 5.0)
    out = {}

    def staff():
        if state != "held":
            return
        butt, head, gc = draw_staff(cv, near_hand, p["wang"], p.get("glow", 0.0))
        out["butt"], out["staff"], out["gem"] = butt, head, gc

    coat_back(cv, P, N, 4.2, 1.05, coat, ROBE_DK, BACK, phase=p.get("phase", 0))
    if p.get("zf", "back") == "back":
        out["hf"] = draw_arm(cv, bp["sf"], far_hand, p.get("bf", 1), *ARM, SLEEVE_DK, SKIN, BACK, dark=True,
                             r_fore=(1.8, 2.3))[2]
    if wz == "back":
        staff()
    kf, _, _, _ = draw_leg(cv, bp["hf"], p["ff"], p.get("kf", -1), *LEG, PANTS, BOOT, BODY, dark=True, toe=toe,
                           r=(2.3, 1.7, 1.7, 1.2))
    coat_panel(cv, bp["hf"], kf, 2.6, 2.6, 2.5, ROBE, BODY, shadow_side=True)
    kn, _, _, _ = draw_leg(cv, bp["hn"], p["fn"], p.get("kn", -1), *LEG, PANTS, BOOT, BODY, toe=toe,
                           r=(2.3, 1.7, 1.7, 1.2))
    coat_panel(cv, bp["hn"] + v2((-0.6, 0)), kn, 2.8, 2.8, 2.5, ROBE, BODY)
    draw_hood(cv, P, N, k)
    draw_torso(cv, P, N)
    dx, dy = rot_offset(0.5, -2.5, k)
    neck = cv.capsule(N, N + head_off + v2((dx, dy)), 1.2)
    cv.put(ring4(neck), C["skin_o"], BODY)
    cv.put(neck, C["skin_s"], BODY)
    if p.get("zn") == "mid":
        out["hn"] = draw_arm(cv, bp["sn"], near_hand, p.get("bn", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_fore=(1.8, 2.4))[2]
    blit_sprite(cv, RUDEUS_HEADS[p.get("expr", "calm")], RUDEUS_KEY, RUDEUS_ANCHOR, N + head_off, k=k, owner=HEAD)
    if p.get("zf") == "front":
        out["hf"] = draw_arm(cv, bp["sf"], far_hand, p.get("bf", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_fore=(1.8, 2.4))[2]
    if wz == "front" and state == "held":
        butt, head = staff_geom(near_hand, p["wang"])
        auto_smear(cv, p, prev, bp["sn"] + v2((1, 3)), near_hand, head)
        staff()
    if p.get("zn", "front") == "front":
        out["hn"] = draw_arm(cv, bp["sn"], near_hand, p.get("bn", 1), *ARM, SLEEVE, SKIN, FRONT,
                             r_fore=(1.8, 2.4))[2]
    if state == "held" and "gem" not in out:
        butt, head = staff_geom(near_hand, p["wang"])
        out["butt"], out["staff"], out["gem"] = butt, head, head + unit(p["wang"]) * 1.5
    out["head"] = N + head_off + rot(v2((0, -8)), -90 * k)
    out["chest"] = P + (N - P) * 0.7
    out["P"] = P
    return out

"""A small procedural fighter rig.

A pose is a dict of offsets (in px at scale 1, character facing right) for the hands and
feet relative to the hip, plus torso lean, whole-body rotation, weapon angle and so on.
`solve` turns a pose into joints, `draw_fighter` paints the body parts with a character
style, and mirrors the result when the fighter faces left.
"""
import math
import numpy as np
from canvas import (Scene, C, L, XS, YS, W, H, M, CW, CH, v2, rot, unit, disc, ellipse, polygon, capsule, nb,
                    outline_of, dilate4, line_pts)

STAND = 21.0            # hip height above the feet at scale 1

DEFAULT = dict(rot=0.0, lean=0.0, hip_dy=0.0, hn=(-1.5, 3.0), hf=(2.0, 3.0), fn=(2.5, 21.0), ff=(-2.5, 21.0),
               bn=1, bf=1, kn=1, kf=1, w=None, wh="n", wb=0.0, face="idle", cape=0.0, head_dy=0.0)
NUMERIC = ("rot", "lean", "hip_dy", "w", "wb", "cape", "head_dy")
POINTS = ("hn", "hf", "fn", "ff")


def full(pose):
    p = dict(DEFAULT)
    p.update(pose)
    return p


def blend(p0, p1, t):
    """Blend two (full) poses. Numbers and points interpolate; everything else switches at t=0.5."""
    out = {}
    for k in p0:
        a, b = p0[k], p1.get(k, p0[k])
        if k in POINTS:
            out[k] = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        elif k in NUMERIC and a is not None and b is not None:
            out[k] = a + (b - a) * t
        else:
            out[k] = b if t >= 0.5 else a
    return out


def ik2(a, t, l1, l2, bend):
    a, t = v2(a), v2(t)
    d = t - a
    dd = float(np.hypot(*d))
    dirn = d / max(dd, 1e-6)
    dd = min(max(dd, abs(l1 - l2) + 0.05), l1 + l2 - 0.05)
    cosang = (l1 * l1 + dd * dd - l2 * l2) / (2 * l1 * dd)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    return a + rot(dirn, bend * ang) * l1, a + dirn * dd


def solve(pose, st, x, y):
    """Joints for a fighter whose feet would rest at (x, y) when standing; facing right."""
    s = st.scale
    r0 = pose["rot"]
    hip = v2((x, y - STAND * s + pose["hip_dy"] * s))

    def P(off):
        return hip + rot(v2(off) * s, r0)

    up = rot(v2((0, -1)), r0 + pose["lean"])
    right = v2((-up[1], up[0]))
    neck = hip + up * st.torso * s
    J = dict(hip=hip, neck=neck, up=up, right=right, s=s)
    J["sh_n"] = neck - right * 1.2 * s + up * -1.6 * s
    J["sh_f"] = neck + right * 1.6 * s + up * -1.8 * s
    J["hj_n"] = hip - rot(v2((0.8, 0)), r0) * s
    J["hj_f"] = hip + rot(v2((1.2, 0)), r0) * s
    for side in ("n", "f"):
        foot = P(pose["f" + side])
        knee, ank = ik2(J["hj_" + side], foot, st.thigh * s, st.shin * s, pose["k" + side])
        J["knee_" + side], J["foot_" + side] = knee, ank
    J["wdir"] = unit(pose["w"] + r0) if pose["w"] is not None else None
    hn = P(pose["hn"])
    hf = P(pose["hf"])
    if pose["w"] is not None and pose["wh"] == "both":
        hf = hn + J["wdir"] * 2.6 * s
    for side, tgt in (("n", hn), ("f", hf)):
        el, wr = ik2(J["sh_" + side], tgt, st.uarm * s, st.farm * s, pose["b" + side])
        J["elb_" + side], J["hand_" + side] = el, wr
    J["grip"] = J["hand_f"] if pose["wh"] == "f" else J["hand_n"]
    J["head"] = neck + up * (0.5 + pose["head_dy"]) * s
    return J


# ---------------------------------------------------------------- body parts
def shaded(sc, mask, ramp, layer, edge="ol"):
    """ramp = (dark, base, light). Outline, light on the upper-left edge, dark lower-right."""
    d, b, l = ramp
    sc.put(outline_of(mask), edge, layer)
    sc.put(mask, b, layer)
    sc.put(mask & ~nb(mask, 1, 1), d, layer)
    sc.put(mask & ~nb(mask, -1, -1) & nb(mask, 1, 1), l, layer)


def draw_leg(sc, J, side, st, layer, dark=False):
    s = J["s"]
    hj, kn, ft = J["hj_" + side], J["knee_" + side], J["foot_" + side]
    leg = capsule(hj, kn, 2.2 * s * st.limb, 1.9 * s * st.limb) | capsule(kn, ft, 1.9 * s * st.limb, 1.5 * s * st.limb)
    fwd = rot(v2((1, 0)), math.degrees(math.atan2(J["right"][1], J["right"][0])))
    boot = capsule(ft + v2((0, -1.6 * s)), ft, 1.8 * s * st.limb) | capsule(ft - fwd * 0.6 * s, ft + fwd * 3.2 * s, 1.5 * s, 1.2 * s)
    pants = st.pants if not dark else (st.pants[0], st.pants[0], st.pants[1])
    shaded(sc, leg, pants, layer)
    if st.boots:
        bm = boot | (leg & (np.hypot(XS - ft[0], YS - ft[1]) < st.boot_h * s))
        shaded(sc, bm, st.boots if not dark else (st.boots[0], st.boots[0], st.boots[1]), layer)
    if st.leg_hook:
        st.leg_hook(sc, J, side, layer, dark)


def draw_arm(sc, J, side, st, layer, dark=False):
    s = J["s"]
    sh, el, hd = J["sh_" + side], J["elb_" + side], J["hand_" + side]
    upper = capsule(sh, el, 2.0 * s * st.limb, 1.7 * s * st.limb)
    fore = capsule(el, hd, 1.7 * s * st.limb, 1.5 * s * st.limb)
    hand = disc(hd, 1.6 * s)
    ramp_u = st.sleeve if not dark else (st.sleeve[0], st.sleeve[0], st.sleeve[1])
    ramp_f = st.fore if not dark else (st.fore[0], st.fore[0], st.fore[1])
    sc.put(outline_of(upper | fore | hand), "ol", layer)
    shaded(sc, fore, ramp_f, layer)
    shaded(sc, upper, ramp_u, layer)
    sc.put(outline_of(upper) & fore, ramp_u[0], layer)
    shaded(sc, hand & ~upper, st.hand if not dark else (st.hand[0], st.hand[0], st.hand[1]), layer)
    if st.arm_hook:
        st.arm_hook(sc, J, side, layer, dark)


def draw_torso(sc, J, st, layer):
    s = J["s"]
    up, right, hip = J["up"], J["right"], J["hip"]

    def tf(u, v):
        return hip + right * u * s + up * v * s

    pts = [tf(u * st.width, v * st.torso / 15.0) for (u, v) in
           [(-3.4, -2), (-3.6, 3), (-3.4, 8), (-4.0, 12.5), (-2.4, 15.2), (2.6, 15.2), (4.4, 12.5), (4.6, 8),
            (3.8, 3), (3.8, -2)]]
    body = polygon(pts)
    shaded(sc, body, st.top, layer)
    dx, dy = XS - hip[0], YS - hip[1]
    u = (dx * right[0] + dy * right[1]) / s
    v = (dx * up[0] + dy * up[1]) / s
    if st.torso_hook:
        st.torso_hook(sc, J, body, u, v, tf, layer)
    return body


def draw_head(sc, J, st, face, layer):
    rows = st.heads[face]
    ax, ay = st.head_anchor
    # rotate the sprite in 90 degree steps when the body is upside down or sideways
    ang = math.degrees(math.atan2(J["up"][0], -J["up"][1]))
    k = int(round(-ang / 90.0)) % 4
    grid = np.array([list(r) for r in rows])
    hgt, wid = grid.shape
    a = (ax, ay)
    for _ in range(k):
        grid = np.rot90(grid)
        a = (a[1], wid - 1 - a[0])
        hgt, wid = wid, hgt
    ox = int(round(J["head"][0])) - a[0]
    oy = int(round(J["head"][1])) - a[1]
    keys = st.head_key
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            ch = grid[y, x]
            if ch == ".":
                continue
            sc.put_px([ox + x], [oy + y], keys[ch], layer)
    return v2((ox, oy))


def draw_fighter_raw(sc, pose, st, x, y, layer):
    J = solve(pose, st, x, y)
    behind = pose["wb"] > 0.5
    if st.back_hook:
        st.back_hook(sc, J, pose, layer)
    if pose["w"] is not None and behind:
        st.weapon(sc, J, pose, layer)
    draw_arm(sc, J, "f", st, layer, dark=True)
    if pose["w"] is not None and pose["wh"] == "f" and not behind:
        st.weapon(sc, J, pose, layer)
    draw_leg(sc, J, "f", st, layer, dark=True)
    draw_leg(sc, J, "n", st, layer)
    draw_torso(sc, J, st, layer)
    draw_head(sc, J, st, pose["face"], layer)
    if st.head_hook:
        st.head_hook(sc, J, pose, layer)
    if pose["w"] is not None and pose["wh"] in ("n", "both") and not behind:
        st.weapon(sc, J, pose, layer)
    draw_arm(sc, J, "n", st, layer)
    if st.front_hook:
        st.front_hook(sc, J, pose, layer)
    return J


def mirror_pt(p):
    return v2((W - 1 - p[0], p[1]))


def draw_fighter(sc, pose, st, x, y, facing, layer, flash=0.0, tint=None, tint_k=0.0, sil=0.0, glow=None,
                 alpha_mask=None):
    """Draw into sc. Returns (mask of the fighter's pixels, joints in world space)."""
    tmp = Scene()
    xx = x if facing > 0 else W - 1 - x
    J = draw_fighter_raw(tmp, pose, st, xx, y, layer)
    m = tmp.owner >= 0
    rgb = tmp.rgb
    if facing < 0:
        m = m[:, ::-1]
        rgb = rgb[:, ::-1]
        J = {k: (mirror_pt(v) if isinstance(v, np.ndarray) and v.shape == (2,) and k not in ("up", "right", "wdir")
                 else v) for k, v in J.items()}
        for k in ("up", "right", "wdir"):
            if J.get(k) is not None:
                J[k] = v2((-J[k][0], J[k][1]))
    if sil > 0:
        rgb = rgb * (1 - sil) + C["ol"] * sil
    if tint is not None and tint_k > 0:
        rgb = rgb * (1 - tint_k) + C[tint] * tint_k
    if flash > 0:
        rgb = rgb * (1 - flash) + C["white"] * flash
    if alpha_mask is not None:
        m = m & alpha_mask
    sc.put_rgb(m, rgb, layer)
    if glow:
        ring = dilate4(m) & ~m
        sc.put(ring & (sc.owner < L["Back FX"]), glow, "Back FX")
    return m, J


def ghost(sc, pose, st, x, y, facing, color, layer="Back FX", checker=True):
    """Flat-coloured afterimage."""
    tmp = Scene()
    xx = x if facing > 0 else W - 1 - x
    draw_fighter_raw(tmp, pose, st, xx, y, "FX")
    m = tmp.owner >= 0
    if facing < 0:
        m = m[:, ::-1]
    if checker:
        m &= (XS + YS) % 2 == 0
    sc.put(m & (sc.owner < L["Back FX"] + 1), color, layer)
    return m

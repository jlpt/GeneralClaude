"""Rudeus: hand-pixelled head on a small procedural rig (robe, legs, IK arms).

He faces right in 3/4 view. The near arm is the one whose shoulder sits on the
left of the torso (drawn in front); the far arm is drawn behind the torso.
"""
import math
import numpy as np
from canvas import (C, L, XS, YS, v2, rot, capsule, disc, ellipse, polygon, nb, outline_of, seg_dist,
                    M, CW, CH)

# ---------------------------------------------------------------- head sprite
HEAD_KEY = {
    "o": "ol", "1": "hr0", "2": "hr1", "3": "hr2", "4": "hr3", "5": "hr4",
    "k": "sk0", "s": "sk1", "S": "sk2", "L": "sk3",
    "e": "ey0", "g": "ey1", "G": "ey2", "m": "mo", "w": "white",
}

HAIR = [
    ".......ooooo......",
    ".....oo34443oo....",
    "...oo3344554433o..",
    "..o233445555443o..",
    ".o23334455544333o.",
    ".o23333444443333o.",
    "o223333333333333o.",
    "o2233332333323332o",
    "o2233223332233232o",
]
FACES = {
    # brows down, eyes narrowed: concentrating
    "focus": [
        "o2232S23SS3SS3S2o.",
        "o222SSSSSSSSSSSo..",
        "o222kSooSSSSoo2o..",
        "o12kLSegGSSSegSo..",
        "o12kSSeggSSSegSo..",
        ".o21kSSSSSSsSSSSo.",
        ".o21kSSSSSSSSSSo..",
        "..o2kkSSSSSmmSSo..",
        "...oo.kkSSSSSSko..",
        "........kkkkkk....",
    ],
    # mouth open: the release shout
    "shout": [
        "o2232S23SS3SS3S2o.",
        "o222SSSSSSSSSSSo..",
        "o222kSooSSSSoo2o..",
        "o12kLSeoGSSSeoSo..",
        "o12kSSeggSSSegSo..",
        ".o21kSSSSSSsSSSSo.",
        ".o21kSSSSSSmmmSo..",
        "..o2kkSSSSSmwmSo..",
        "...oo.kkSSSSmSko..",
        "........kkkkkk....",
    ],
    # wide eyes, mouth agape
    "shock": [
        "o2232S23SS3SS3S2o.",
        "o222SoSSSSSSSoSo..",
        "o222kSSSSSSSSS2o..",
        "o12kLSeeeSSSeeeo..",
        "o12kSSegGSSSegGo..",
        ".o21kSSeesSSSeeSo.",
        ".o21kSSSSSSSSSSo..",
        "..o2kkSSSSSmmSSo..",
        "...oo.kkSSSmmSko..",
        "........kkkkkk....",
    ],
    # calm, eyes open
    "calm": [
        "o2232S23SS3SS3S2o.",
        "o222SSSSSSSSSSSo..",
        "o222kSSoSSSSSo2o..",
        "o12kLSeoeSSSeoSo..",
        "o12kSSegGSSSegSo..",
        ".o21kSSegsSSsegSo.",
        ".o21kSSSSSSSSSSo..",
        "..o2kkSSSSSSmSSo..",
        "...oo.kkSSSSSSko..",
        "........kkkkkk....",
    ],
}
HEADS = {k: HAIR + v for k, v in FACES.items()}
for _k, _rows in HEADS.items():
    assert all(len(r) == 18 for r in _rows), (_k, [len(r) for r in _rows])
HEAD_ANCHOR = (10, 18)   # head-sprite pixel that sits on top of the neck


def draw_head(sc, neck, variant, layer="Rudeus", hair_lift=0):
    rows = HEADS[variant]
    ox = int(round(neck[0])) - HEAD_ANCHOR[0]
    oy = int(round(neck[1])) - HEAD_ANCHOR[1]
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == ".":
                continue
            sc.put_px([ox + x], [oy + y], HEAD_KEY[ch], layer)
    # a few loose spikes that lift with the magic updraft
    if hair_lift:
        spikes = [((4, 1), -1), ((8, 0), 0), ((13, 1), 1)]
        for (sx, sy), d in spikes:
            n = hair_lift
            xs = [ox + sx + d * (i // 2) for i in range(1, n + 1)]
            ys = [oy + sy - i for i in range(1, n + 1)]
            sc.put_px(xs, ys, "ol", layer)
            sc.put_px([x + 1 for x in xs[:-1]], ys[:-1], "hr2", layer)
    return v2((ox, oy))


# ---------------------------------------------------------------- shading helpers
def shade(sc, mask, ramp, layer, light_dir=(-1, -1)):
    """ramp = (outline, dark, base, light, highlight). Outline outside, light on the
    edge facing light_dir, dark on the opposite edge."""
    o, d, b, l, hi = ramp
    sc.put(outline_of(mask) & (sc.owner != -2), o, layer)
    sc.put(mask, b, layer)
    lx, ly = light_dir
    sc.put(mask & ~nb(mask, -lx, -ly), d, layer)
    lit = mask & ~nb(mask, lx, ly)
    sc.put(lit, l, layer)
    if hi:
        sc.put(lit & ~nb(mask, lx, 0) & ~nb(mask, 0, ly), hi, layer)


def ik2(a, t, l1, l2, bend):
    a, t = v2(a), v2(t)
    d = t - a
    dd = float(np.hypot(*d))
    dirn = d / max(dd, 1e-6)
    dd = min(max(dd, abs(l1 - l2) + 0.05), l1 + l2 - 0.05)
    cosang = (l1 * l1 + dd * dd - l2 * l2) / (2 * l1 * dd)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    elbow = a + rot(dirn, bend * ang) * l1
    return elbow, a + dirn * dd


# ---------------------------------------------------------------- body
def body_frame(hip, lean):
    up = rot(v2((0, -1)), lean)
    right = v2((-up[1], up[0]))
    return v2(hip), up, right


def at(hip, up, right, u, v):
    return hip + right * u + up * v


def draw_legs(sc, hip, up, right, stance, layer="Rudeus"):
    """Legs below the robe hem: shins + boots. stance in px from centre."""
    ground = hip[1] + 21
    out = []
    for side, (hx, fx, bend, tone) in enumerate([(-1.6, -stance, -1, 0), (2.0, stance * 0.9, 1, 1)]):
        h = at(hip, up, right, hx, -1)
        foot = v2((hip[0] + fx, ground - 2.0))
        knee, ank = ik2(h, foot, 10.5, 10.0, bend)
        leg = capsule(h, knee, 2.2, 1.9) | capsule(knee, ank, 1.9, 1.6)
        toe = ank + v2((3.6, 1.2))
        boot = capsule(ank + v2((-0.6, 0.2)), toe, 1.7, 1.3) | capsule(ank + v2((0, -2.2)), ank, 1.8, 1.8)
        whole = leg | boot
        sc.put(outline_of(whole), "ol", layer)
        sc.put(leg, "pa1" if tone else "pa0", layer)
        sc.put(leg & ~nb(leg, -1, 0), "pa2" if tone else "pa1", layer)
        sc.put(boot, "bt1", layer)
        sc.put(boot & ~nb(boot, 0, -1), "bt2", layer)
        sc.put(boot & ~nb(boot, 0, 1), "bt0", layer)
        out.append((knee, ank))
    return out


def robe_poly(hip, up, right, flutter, flare):
    f, fl = flutter, flare

    def p(u, v):
        return at(hip, up, right, u, v)

    back = [p(-1.5, 16.5), p(-4.5, 15.2), p(-5.6, 13.0), p(-5.4, 8.0), p(-5.0, 2.5),
            p(-6.0 - f * 0.4, -3.0), p(-7.2 - f * 0.9 - fl, -8.0), p(-8.4 - f * 1.5 - fl, -11.6)]
    hem = [p(-6.0 - f * 1.0, -12.4), p(-3.0 - f * 0.5, -12.0), p(0.0, -12.6), p(2.5, -12.0)]
    front = [p(5.6 + fl * 0.6 - f * 0.2, -11.4), p(5.2, -6.0), p(4.6, -1.0), p(4.6, 3.0),
             p(5.3, 7.5), p(5.0, 11.5), p(3.6, 14.4), p(1.6, 16.6)]
    return back + hem + front


def draw_body(sc, hip, lean=0.0, flutter=0.0, flare=0.0, stance=6.0, layer="Rudeus", sway=0.0):
    hip, up, right = body_frame(hip, lean)
    draw_legs(sc, hip, up, right, stance, layer)

    def p(u, v):
        return at(hip, up, right, u, v)

    robe = polygon(robe_poly(hip, up, right, flutter, flare))
    # local coords for shading
    dx, dy = XS - hip[0], YS - hip[1]
    u = dx * right[0] + dy * right[1]
    v = dx * up[0] + dy * up[1]
    sc.put(outline_of(robe), "rb0", layer)
    sc.put(robe, "rb2", layer)
    sc.put(robe & (u < -3.2 + 0.25 * np.minimum(v, 0) * 0.3), "rb3", layer)        # lit back edge
    sc.put(robe & ~nb(robe, -1, 0) & (v > -10), "rb4", layer)
    sc.put(robe & (u > 2.6) & (v > -11), "rb1", layer)                               # front in shade
    # skirt folds that swing
    for k, fu in enumerate((-4.0, -1.2, 1.6)):
        off = fu + sway * (0.6 + 0.3 * k) + flutter * 0.3 * (-1 - k * 0.2)
        fold = robe & (v < -3) & (v > -11.8) & (np.abs(u - off - (v + 3) * -0.12 * (k - 1)) < 0.5)
        sc.put(fold, "rb1", layer)
        sc.put(fold & ~nb(robe & ~fold, -1, 0), "rb1", layer)
    # front opening below the belt: legs show through
    open_ = polygon([p(2.6, -1.4), p(1.8, -12.4), p(5.0, -12.1)]) & robe
    sc.put(open_, "pa1", layer)
    sc.put(open_ & ~nb(open_, -1, 0), "pa0", layer)
    # chest V with tunic + collar trim
    vee = polygon([p(0.6, 15.8), p(3.6, 15.0), p(2.6, 9.5)]) & robe
    sc.put(vee, "pa1", layer)
    trim = (outline_of(vee) | outline_of(open_)) & robe
    sc.put(trim, "tr1", layer)
    sc.put(trim & (u < 1.5) & (v > 0), "tr2", layer)
    # hem trim
    hem = robe & ~nb(robe, 0, 1) | (robe & ~nb(robe, 0, 2) & (u < 2))
    sc.put(hem, "tr0", layer)
    sc.put(robe & ~nb(robe, 0, 1) & (u < 0), "tr1", layer)
    # belt + buckle + pouch
    belt = robe & (np.abs(v - 1.0) < 1.0)
    sc.put(belt, "be0", layer)
    sc.put(belt & (v > 1.0), "be1", layer)
    sc.put(belt & (np.abs(u - 2.6) < 0.9), "gold", layer)
    pouch = capsule(p(-4.0, -0.5), p(-4.0, -3.0), 1.4)
    sc.put(outline_of(pouch), "ol", layer)
    sc.put(pouch, "be1", layer)
    sc.put(pouch & ~nb(pouch, 0, 1), "be0", layer)
    # bunched hood behind the neck
    hood_c = p(-3.4, 14.2)
    hood = ellipse(hood_c, 3.4, 2.4) & ~polygon([p(-0.6, 18), p(2, 18), p(2, 13), p(-0.6, 13)])
    sc.put(outline_of(hood) & ~robe, "rb0", layer)
    sc.put(hood, "rb3", layer)
    sc.put(hood & (YS > hood_c[1] + 0.6), "rb2", layer)
    sc.put(hood & (np.abs(YS - hood_c[1]) < 0.6) & (XS > hood_c[0] - 1.5), "rb1", layer)
    sc.put(hood & ~nb(hood, -1, -1), "rb5", layer)
    # neck
    neck = p(0.6, 16.0)
    nk = capsule(neck + v2((0, 1)), neck + v2((0.4, -2.0)), 1.3)
    sc.put(outline_of(nk) & ~robe & ~hood, "sk0", layer)
    sc.put(nk & ~hood, "sk2", layer)
    sc.put(nk & ~hood & (XS > neck[0] + 0.3), "sk1", layer)
    return dict(neck=neck + v2((0.4, -1.5)), sh_near=p(-3.2, 13.0), sh_far=p(3.4, 13.6), up=up, right=right)


# ---------------------------------------------------------------- arms
def draw_arm(sc, shoulder, hand, bend, layer, far=False, sleeve_drop=0.0, spread=False, thumb_side=1):
    """Robe sleeve + bare forearm when raised + hand. sleeve_drop 0..1: how much the
    loose sleeve has slid down toward the elbow (arms overhead)."""
    l1, l2 = 11.5, 11.0
    hand = v2(hand)
    elbow, wrist = ik2(shoulder, hand, l1, l2, bend)
    fdir = (wrist - elbow) / max(np.hypot(*(wrist - elbow)), 1e-6)
    cuff = elbow + (wrist - elbow) * (0.95 - 0.55 * sleeve_drop)
    upper = capsule(shoulder, elbow, 2.2, 2.0)
    sleeve = upper | capsule(elbow, cuff, 2.0, 2.5)
    # loose cloth hanging below the forearm (gravity is +y)
    hang = 1.5 + 2.0 * sleeve_drop
    drape = polygon([cuff + v2((-2.4, 0)), cuff + v2((2.4, 0)),
                     (elbow + cuff) / 2 + v2((1.0, hang)), (elbow + cuff) / 2 + v2((-1.0, hang))]) \
        & (YS > np.minimum(cuff[1], elbow[1]) - 0.5)
    sleeve |= drape
    fore = capsule(elbow, wrist, 1.25, 1.1) & ~sleeve
    hm = disc(hand, 1.5)
    fing = np.zeros_like(hm)
    if spread:
        # open hand: three short fingers fanned out plus a thumb
        for ang in (-38, -8, 22, 70 * thumb_side):
            d = rot(fdir, ang)
            fing |= capsule(hand + d * 1.2, hand + d * (2.9 if ang != 70 * thumb_side else 2.3), 0.45)
    else:
        fing = capsule(hand, hand + fdir * 2.2, 0.8) & ~disc(hand, 0.9)
    skin = (fore | hm | fing) & ~sleeve
    ramp = ("rb0", "rb1", "rb2", "rb3", None) if far else ("rb0", "rb1", "rb3", "rb4", "rb5")
    sc.put(outline_of(skin | sleeve) & ~(sleeve | skin), "ol", layer)
    sc.put(skin, "sk2", layer)
    sc.put(skin & ~nb(skin, 1, 1), "sk1", layer)
    sc.put(skin & ~nb(skin, -1, -1), "sk3", layer)
    sc.put(sleeve, ramp[2], layer)
    sc.put(sleeve & ~nb(sleeve, 1, 1), ramp[1], layer)
    sc.put(sleeve & ~nb(sleeve, -1, -1), ramp[3], layer)
    # cuff trim
    _, t = seg_dist(elbow, cuff)
    sc.put(sleeve & (t > 0.93) & capsule(elbow, cuff, 2.6), "tr0" if far else "tr1", layer)
    return elbow, wrist


def draw_rudeus_full(sc, pose):
    """Draw back arm, body, head, front arm in the right order."""
    from canvas import Scene
    hn, hf = pose["hands"]
    bn, bf = pose.get("bends", (1, -1))
    sl = pose.get("sleeve", 0.0)
    # 1) far arm on its own scene, then body over it
    tmp = Scene()
    b = draw_body(tmp, pose["hip"], pose.get("lean", 0.0), pose.get("flutter", 0.0), pose.get("flare", 0.0),
                  pose.get("stance", 6.0), sway=pose.get("sway", 0.0))
    draw_head(tmp, b["neck"], pose.get("face", "focus"), hair_lift=pose.get("hair_lift", 0))
    body_mask = tmp.owner >= 0
    draw_arm(sc, b["sh_far"], hf, bf, "Back arm", far=True, sleeve_drop=sl, spread=pose.get("spread", False), thumb_side=-1)
    if pose.get("far_in_front"):
        sc.put_rgb(body_mask & ~(sc.owner == L["Back arm"]), tmp.rgb, "Rudeus")
    else:
        sc.put_rgb(body_mask, tmp.rgb, "Rudeus")
    draw_arm(sc, b["sh_near"], hn, bn, "Front arm", far=False, sleeve_drop=sl, spread=pose.get("spread", False), thumb_side=1)
    return b

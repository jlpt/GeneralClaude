"""Procedural rig for the star-caller mage (3/4 view, facing right).

A pose is a dict of a few numbers (hand position, staff angle, how far the cloak is lifted or
blown back, ...). Each body part is rasterised as a pixel mask, shaded from a hand-picked ramp
and given its own coloured outline, on the layer it belongs to.
"""
import math
import numpy as np
from canvas import (C, XS, YS, GROUND, SHAPE, PAD, v2, unit, lerp, nb, dilate4, capsule, disc, ellipse,
                    polygon, pixels, line_px, polyline)

# hand-pixelled head, 13 x 12, centre at (6, 6)
HEAD = [
    "....hhhhhh...",
    "..hhbbllbbhh.",
    ".hbbllbbbbbbh",
    ".hblbbbbbbbbh",
    "hsbbbblbbblbh",
    "hsbbsbSbbSBsO",
    "hsbsSBBBBBBBO",
    "hsbSBttBBBttO",
    "hssSBeiBLBeiO",
    ".hsOSBBBBBLO.",
    ".hs.OSBBmBO..",
    "..h..OOOOO...",
]
HEAD_KEY = {"h": "hair_o", "s": "hair_s", "b": "hair_b", "l": "hair_l", "O": "skin_o", "S": "skin_s",
            "B": "skin_b", "L": "skin_l", "t": "eye_d", "e": "eye_d", "i": "eye_i", "m": "mouth"}

DEFAULT = dict(x=44, float=0.0, bob=0.0, lean=0.0, stance=0.0, hand=(9, -17), staff_ang=90.0, grip=17.0,
               off_hand=None, lift=0.0, blow=0.0, wave=0.0, braid=-100.0, braid_curl=4.0, hat_tilt=0.0,
               tip=(0.0, 0.0), eyes="open", mouth=False, orb=1.0)


def part(sc, layer, mask, outline):
    """Paint a 1px outline ring around `mask` on `layer`; the caller fills the mask afterwards."""
    sc.put(dilate4(mask) & ~mask, outline, layer)


def ik(a, c, l1, l2, bend):
    a, c = v2(a), v2(c)
    d = c - a
    dist = min(max(np.hypot(*d), 1e-6), l1 + l2 - 1e-3)
    dirv = d / max(np.hypot(*d), 1e-6)
    c = a + dirv * dist
    x = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(l1 * l1 - x * x, 0.0))
    perp = v2((-dirv[1], dirv[0]))
    return a + dirv * x + perp * h * bend, c


def bezier(p0, p1, p2, n=8):
    p0, p1, p2 = v2(p0), v2(p1), v2(p2)
    return [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2 for t in np.linspace(0, 1, n)]


def row_param(mask):
    """Per-pixel 0..1 position between the left and right edge of `mask` on that row."""
    any_row = mask.any(axis=1)
    first = mask.argmax(axis=1)
    last = SHAPE[1] - 1 - mask[:, ::-1].argmax(axis=1)
    span = np.maximum(last - first, 1)
    s = (np.arange(SHAPE[1])[None, :] - first[:, None]) / span[:, None]
    return np.where(any_row[:, None], s, 0.0)


def bottom_edge(mask):
    return mask & ~nb(mask, 0, 1)


def hem_points(p, O, scale=1.0):
    """Cloak hem from back (left) to front (right). `scale` < 1 gives the lining's lazier hem."""
    lift, blow = p["lift"] * scale, p["blow"] * scale
    lean = p["lean"]
    xl = -9 - 4 * p["lift"] - 14 * p["blow"] + lean * 0.3
    xr = 7 + 2.5 * p["lift"] - 1.5 * p["blow"] + lean * 0.6
    yh = -3.5 + p["bob"] * 0.4
    amp = 0.5 + 1.3 * max(p["lift"], p["blow"])
    pts = []
    for u in np.linspace(0, 1, 15):
        x = lerp(xl, xr, u)
        rise = (6.5 * lift + 15 * blow) * (1 - u) ** 1.5 + 5.5 * lift * u ** 1.7 + 1.5 * lift
        y = yh - rise + amp * math.sin(2 * math.pi * (1.4 * u + p["wave"])) * (0.35 + 0.65 * (1 - u))
        pts.append(O + v2((x, y)))
    return pts


def draw_mage(sc, pose):
    p = dict(DEFAULT)
    p.update(pose)
    O = v2((p["x"], GROUND - p["float"]))
    bob, lean = p["bob"], p["lean"]

    def P(x, y):
        return O + v2((x, y))

    sh = P(lean, -24 + bob)                      # shoulder centre
    hip = P(lean * 0.4, -12.5 + bob * 0.6)
    hc = np.round(P(lean * 1.25 + 1, -31 + bob))  # head centre
    out = {}

    # ------------------------------------------------ braid
    L = "Braid + lining"
    base = hc + v2((-5, 0.5))
    pts, a = [base], p["braid"]
    for i in range(8):
        a += p["braid_curl"]
        pts.append(pts[-1] + unit(a) * 2.1)
    segs = [disc(q, 1.75 if i < 6 else 1.3) for i, q in enumerate(pts)]
    allb = np.zeros(SHAPE, bool)
    for m in segs:
        allb |= m
    part(sc, L, allb, "hair_o")
    for i, m in enumerate(segs):
        sc.put(m, "hair_b" if i % 2 == 0 else "hair_s", L)
        sc.put(m & ~nb(m, 1, -1) & ~nb(m, 0, -1), "hair_l" if i % 2 == 0 else "hair_b", L)
    tie = disc(pts[6], 1.5)
    sc.put(tie, "gold_b", L)
    sc.put(tie & ~nb(tie, 1, -1), "gold_l", L)

    # ------------------------------------------------ cloak lining (inside of the lifted cloak)
    hem = hem_points(p, O)
    hem_in = hem_points(p, O, 0.45)
    SL, SR = sh + v2((-5.5 - 1.5 * p["blow"], 1)), sh + v2((5.5, 1))
    lining = polygon([SL] + hem_in + [SR])
    part(sc, L, lining, "teal_o")
    sc.put(lining, "teal_d", L)
    s = row_param(lining)
    sc.put(lining & (s > 0.45), "teal_b", L)
    sc.put(bottom_edge(lining), "teal_l", L)

    # ------------------------------------------------ back arm
    if p["off_hand"] is not None:
        L = "Back arm"
        sa = sh + v2((3.5, 2.5))
        hand = P(*p["off_hand"])
        elbow, hand = ik(sa, hand, 6.0, 6.5, 1.0)
        sleeve = capsule(sa, elbow, 2.3) | capsule(elbow, hand - (hand - elbow) * 0.2, 2.3, 3.0)
        hm = disc(hand, 1.7)
        part(sc, L, sleeve | hm, "ind_o")
        sc.put(sleeve, "ind_b", L)
        sc.put(sleeve & ~nb(sleeve, 0, 1), "ind_d", L)
        cuff = sleeve & (np.hypot(XS - hand[0], YS - hand[1]) < 3.4)
        sc.put(cuff, "gold_d", L)
        sc.put(hm, "skin_b", L)
        sc.put(hm & ~nb(hm, 0, 1), "skin_s", L)
        out["off_hand"] = hand

    # ------------------------------------------------ legs, boots, tunic
    L = "Body"
    st = p["stance"]
    for side, fx in ((-1, -3 - 3.5 * st), (1, 3 + 4.5 * st)):
        foot = P(fx + lean * 0.1, 0)
        leg = capsule(hip + v2((side * 2, 0)), foot + v2((0, -3)), 1.9, 1.6)
        boot = ((XS >= foot[0] - 2) & (XS <= foot[0] + 2) & (YS >= foot[1] - 3.5) & (YS <= foot[1])) | \
               ((XS > foot[0] + 2) & (XS <= foot[0] + 3.2) & (YS >= foot[1] - 1.5) & (YS <= foot[1]))
        part(sc, L, leg | boot, "ind_o")
        sc.put(leg, "leg_b" if side > 0 else "leg_d", L)
        sc.put(boot, "boot_b" if side > 0 else "boot_d", L)
        if side > 0:
            sc.put(boot & ~nb(boot, 0, -1), "boot_l", L)
        sc.put(boot & (YS > foot[1] - 0.6), "wood_o", L)
        sc.put(boot & ~nb(boot, 0, -1) & (YS < foot[1] - 2.5), "gold_d" if side < 0 else "gold_b", L)
    torso = capsule(hip, sh + v2((0, 2)), 3.8, 4.2)
    part(sc, L, torso, "teal_o")
    sc.put(torso, "teal_b", L)
    sc.put(torso & (XS < hip[0] - 0.5), "teal_d", L)
    belt = torso & (np.abs(YS - (hip[1] - 2)) <= 0.6)
    sc.put(belt, "gold_b", L)

    # ------------------------------------------------ cloak
    lc = bezier(SL, (lerp(SL[0], hem[0][0], 0.35) - 2.0 - 3 * p["blow"], lerp(SL[1], hem[0][1], 0.55)), hem[0])
    rc = bezier(hem[-1], (hem[-1][0] + 0.8, lerp(SR[1], hem[-1][1], 0.5)), SR)
    cloak = polygon(lc + hem[1:-1] + rc)
    # front opening: an inverted V showing the tunic and legs
    apex = sh + v2((3.2, 5.0))
    o0, o1 = hem[10], hem[13]
    gap = 0.25 + 0.75 * max(p["lift"], p["blow"], st)
    mid = (o0 + o1) / 2
    a0, a1 = mid + (o0 - mid) * gap + v2((0, 3)), mid + (o1 - mid) * gap + v2((0, 3))
    opening = polygon([apex, a0, a1])
    cloak &= ~opening
    part(sc, L, cloak, "ind_o")
    s = row_param(cloak | opening)
    sc.put(cloak, "ind_b", L)
    sc.put(cloak & (s < 0.36), "ind_d", L)
    sc.put(cloak & (s > 0.86), "ind_l", L)
    for u_idx, top in ((4, (-2.5, 6)), (7, (0.5, 8))):
        fold = line_px(sh + v2(top), hem[u_idx] + v2((0, -1))) & cloak
        sc.put(fold & (s >= 0.36), "ind_d", L)
        sc.put(fold & (s < 0.36), "ind_o", L)
        sc.put(nb(fold, -1, 0) & cloak & (s >= 0.36) & (s < 0.86) & (YS > sh[1] + 12) & ~fold, "ind_l", L)
    edge = cloak & nb(opening, 1, 0)             # gold piping down the front opening
    sc.put(edge, "gold_b", L)
    trim = bottom_edge(cloak) & ~nb(opening, 0, 1)
    sc.put(trim, "gold_b", L)
    sc.put(trim & (s < 0.36), "gold_d", L)
    sc.put(trim & (s > 0.8), "gold_l", L)

    # mantle over the shoulders
    bl = p["blow"]
    mantle = polygon([sh + v2((-6.5 - 2 * bl, 0.5)), sh + v2((-8 - 3.5 * bl, 4.5 - 2 * bl)),
                      sh + v2((-5 - 2 * bl, 7.5 - bl)), sh + v2((-0.5, 8.5)), sh + v2((4.5, 7.5)),
                      sh + v2((7, 4.5)), sh + v2((6, 0.5)), sh + v2((3, -1.5)), sh + v2((-3, -1.5))])
    part(sc, L, mantle, "ind_o")
    sm = row_param(mantle)
    sc.put(mantle, "ind_l", L)
    sc.put(mantle & (sm < 0.3), "ind_b", L)
    sc.put(mantle & (sm > 0.72) & (YS < sh[1] + 4), "ind_h", L)
    mt = bottom_edge(mantle)
    sc.put(mt, "gold_b", L)
    sc.put(mt & (sm < 0.3), "gold_d", L)
    sc.put(mt & (sm > 0.72), "gold_l", L)
    clasp = sh + v2((3, 1.5))
    sc.put(pixels([(clasp[0] + dx, clasp[1] + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]),
           "gold_l", L)
    sc.put(pixels([clasp]), "gold_h", L)

    # ------------------------------------------------ head
    L = "Head + hat"
    x0, y0 = int(hc[0]) - 6, int(hc[1]) - 6
    glow = p["eyes"] == "glow"
    for ry, row in enumerate(HEAD):
        for rx, ch in enumerate(row):
            if ch == ".":
                continue
            col = HEAD_KEY[ch]
            if ch in "tei":
                if glow:
                    col = "fx_w" if ch == "i" else "cy_2"
                elif p["eyes"] == "closed":
                    col = "skin_b" if ch == "t" else "skin_o"
            if ch == "m" and p["mouth"]:
                col = "eye_d"
            sc.put(pixels([(x0 + rx, y0 + ry)]), col, L)

    # hat: brim + bent cone + star charm on the tip
    tilt = p["hat_tilt"]
    bc = hc + v2((-0.5, -3.5))
    brim = ((XS - bc[0]) / 10.5) ** 2 + ((YS - bc[1] - (XS - bc[0]) * tilt) / 2.4) ** 2 <= 1.0
    tip = bc + v2((-6.5, -13.5)) + v2(p["tip"])
    cone = polygon([bc + v2((-6.2, 0.5)), bc + v2((-5, -5)), bc + v2((-4.6, -9)), tip,
                    bc + v2((-1.5, -10.5)) + v2(p["tip"]) * 0.5, bc + v2((2.5, -6.5)), bc + v2((6.2, 0.5))])
    part(sc, L, brim | cone, "ind_o")
    sc.put(cone, "ind_b", L)
    sc.put(cone & (XS < bc[0] - 2.5 - (bc[1] - YS) * 0.25), "ind_d", L)
    sc.put(cone & (XS > bc[0] + 1.5 - (bc[1] - YS) * 0.55), "ind_l", L)
    hi = cone & ~nb(cone, 1, 0) & (YS < bc[1] - 3.5)
    sc.put(hi, "ind_h", L)
    band = cone & (YS > bc[1] - 4.4) & (YS <= bc[1] - 1.6)
    sc.put(band, "gold_b", L)
    sc.put(band & (XS < bc[0] - 2.5), "gold_d", L)
    sc.put(band & (XS > bc[0] + 2.5), "gold_l", L)
    sc.put(brim, "ind_b", L)
    sc.put(brim & (YS > bc[1] + (XS - bc[0]) * tilt + 0.4), "ind_d", L)
    sc.put(brim & ~nb(brim, 0, -1) & (XS > bc[0] - 2), "ind_l", L)
    sc.put(brim & ~nb(brim, 0, -1) & (XS > bc[0] + 4), "ind_h", L)
    # charm
    ch_top = tip + v2((-0.5, 1.5))
    ch_c = ch_top + v2((-0.6 + p["tip"][0] * -0.4, 3.0))
    sc.put(line_px(ch_top, ch_c), "gold_d", L)
    star = pixels([(ch_c[0] + dx, ch_c[1] + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))])
    part(sc, L, star, "gold_o")
    sc.put(star, "gold_l", L)
    sc.put(pixels([ch_c]), "gold_h", L)

    # ------------------------------------------------ staff
    L = "Staff"
    hand = P(*p["hand"])
    d = unit(p["staff_ang"])
    n = v2((-d[1], d[0]))
    bot, top = hand - d * p["grip"], hand + d * (50 - p["grip"])
    shaft = capsule(bot, top - d * 5, 0.85)
    cres = disc(top, 4.7) & ~disc(top + d * 2.3, 3.9)
    collar = disc(top - d * 5.2, 1.7)
    part(sc, L, shaft, "wood_o")
    sc.put(shaft, "wood_b", L)
    side = (XS - bot[0]) * n[0] + (YS - bot[1]) * n[1]
    lit_side = side * (1 if (n[0] - n[1]) > 0 else -1)
    sc.put(shaft & (lit_side > 0.25), "wood_l", L)
    sc.put(shaft & (lit_side < -0.45), "wood_d", L)
    part(sc, L, cres | collar, "gold_o")
    sc.put(cres | collar, "gold_b", L)
    g = cres | collar
    sc.put(g & ~nb(g, -1, 1) & ~nb(g, 0, 1), "gold_d", L)
    sc.put(g & ~nb(g, 1, -1), "gold_l", L)
    sc.put(g & ~nb(g, 1, -1) & ~nb(g, 0, -1) & ~nb(g, 1, 0), "gold_h", L)
    orb_c = top + d * 1.6
    if p["orb"] > 0:
        orb = disc(orb_c, 2.1)
        part(sc, L, orb, "cy_5")
        sc.put(orb, "cy_3", L)
        sc.put(disc(orb_c + v2((0.4, -0.4)), 1.3), "cy_2", L)
        sc.put(pixels([orb_c + v2((0.6, -0.8))]), "fx_w", L)
    out.update(orb=orb_c, staff_top=top, staff_bot=bot, staff_dir=d)

    # ------------------------------------------------ front arm
    L = "Front arm"
    sa = sh + v2((-1.5, 3.0))
    elbow, hand = ik(sa, hand, 6.5, 6.5, -1.0)
    upper = capsule(sa, elbow, 2.5)
    fore = capsule(elbow, hand - (hand - elbow) / max(np.hypot(*(hand - elbow)), 1e-6) * 1.5, 2.4, 3.1)
    sleeve = upper | fore
    hm = disc(hand, 1.8)
    part(sc, L, sleeve | hm, "ind_o")
    sc.put(sleeve, "ind_l", L)
    sc.put(sleeve & ~nb(sleeve, 0, 1) & ~nb(sleeve, -1, 0), "ind_b", L)
    sc.put(sleeve & ~nb(sleeve, 0, -1), "ind_h", L)
    cuff = fore & (np.hypot(XS - hand[0], YS - hand[1]) < 3.3)
    sc.put(cuff, "gold_b", L)
    sc.put(cuff & ~nb(sleeve, 0, -1), "gold_l", L)
    sc.put(hm, "skin_b", L)
    sc.put(hm & ~nb(hm, 0, -1), "skin_l", L)
    sc.put(hm & ~nb(hm, 0, 1), "skin_s", L)
    out.update(hand=hand, head=hc, feet=O, shoulder=sh)
    return out

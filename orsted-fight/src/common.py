"""Shared character rig helpers: pose prep (spin / head rotation), limbs, torso frames, smears."""
import math
import numpy as np
from px import v2, rot, unit, ik2, nb, dilate4, ring4, ang_of, norm, paint_shaded
from palette import FX

# owner ids inside a character's local canvas
HAIR, BACK, BODY, HEAD, WPN, FRONT, FXP = range(7)

POINTS = ("N", "hn", "hf", "fn", "ff")


def torso_frame(P, N):
    P, N = v2(P), v2(N)
    up = (N - P) / max(float(np.hypot(*(N - P))), 1e-9)
    right = v2((-up[1], up[0]))
    return P, N, up, right, float(np.hypot(*(N - P)))


def prep(pose):
    """Resolve whole-body spin about the pelvis and work out how many quarter turns the head sprite needs."""
    p = dict(pose)
    for k in ("P",) + POINTS + ("head",):
        if k in p:
            p[k] = v2(p[k])
    spin = p.get("spin", 0.0)
    if spin:
        P = p["P"]
        for k in POINTS:
            p[k] = P + rot(p[k] - P, spin)
        p["head"] = rot(p.get("head", v2((0, 0))), spin)
        if "wang" in p:
            p["wang"] = p["wang"] + spin
    up_ang = ang_of(p["N"] - p["P"])
    if "hk" not in p or p["hk"] is None:
        p["hk"] = int(round(((-90 - up_ang) / 90.0))) % 4
    return p


def solve_arm(shoulder, hand_c, bend, l1, l2, hand_off=1.8):
    """IK that places the centre of the hand (not the wrist) on hand_c."""
    hand_c = v2(hand_c)
    fdir = norm(hand_c - v2(shoulder))
    for _ in range(4):
        elbow, wrist = ik2(shoulder, hand_c - fdir * hand_off, l1, l2, bend)
        fdir = norm(wrist - elbow)
    return elbow, wrist, wrist + fdir * hand_off


def draw_hand(cv, wrist, fdir, shape, skin, owner, scale=1.0):
    """skin = (outline, shade, base). Returns the hand mask."""
    o, s, b = skin
    if shape == "open":
        m = cv.capsule(wrist + fdir * 0.6, wrist + fdir * 3.4 * scale, 1.25 * scale, 0.9 * scale)
        thumb = cv.disc(wrist + fdir * 1.0 + v2((-fdir[1], fdir[0])) * -1.6, 0.7)
        m |= thumb
    elif shape == "point2":
        m = cv.disc(wrist + fdir * 1.2, 1.3 * scale) | cv.capsule(wrist + fdir * 1.5, wrist + fdir * 4.6 * scale,
                                                                   0.6)
    else:
        m = cv.disc(wrist + fdir * 1.6 * scale, 1.55 * scale)
    return m


def draw_arm(cv, shoulder, hand_c, bend, l1, l2, sleeve, skin, owner, r_up=(2.1, 1.8), r_fore=(1.8, 2.1),
             dark=False, hand_shape="fist", cuff=None, bare_fore=False, hand_scale=1.0):
    """sleeve = (outline, dark, base, light, highlight); skin = (outline, shade, base)."""
    elbow, wrist, hand_c = solve_arm(shoulder, hand_c, bend, l1, l2, 1.8 * hand_scale)
    fdir = norm(wrist - elbow)
    upper = cv.capsule(shoulder, elbow, *r_up)
    fore = cv.capsule(elbow, wrist, *r_fore)
    sleeve_m = upper if bare_fore else (upper | fore)
    hm = draw_hand(cv, wrist, fdir, hand_shape, skin, owner, hand_scale) & ~sleeve_m
    if bare_fore:
        hm |= fore & ~upper
    so, sd, sb, sl, sh = sleeve
    ko, ks, kb = skin
    cv.put(ring4(hm), ko, owner)
    cv.put(ring4(sleeve_m), so, owner)
    cv.put(sleeve_m, sd if dark else sb, owner)
    if not dark:
        cv.put(sleeve_m & ~nb(sleeve_m, 1, 1), sd, owner)
        lt = sleeve_m & ~(nb(sleeve_m, -1, 0) & nb(sleeve_m, 0, -1)) & nb(sleeve_m, 1, 1)
        cv.put(lt, sl, owner)
    if cuff is not None and not bare_fore:
        _, t = cv.seg_t(elbow, wrist)
        cv.put(fore & (t > 0.82), cuff, owner)
    cv.put(hm, kb, owner)
    cv.put(hm & ~nb(hm, 1, 1), ks, owner)
    return elbow, wrist, hand_c


def draw_leg(cv, hip, ankle, bend, l1, l2, pants, boot, owner, r=(2.4, 1.7, 1.75, 1.15), dark=False,
             boot_from=0.7, toe=(1, 0), shoe_len=3.6):
    """pants = (outline, dark, base, light); boot = (base, light)."""
    knee, ank = ik2(hip, ankle, l1, l2, bend)
    thigh = cv.capsule(hip, knee, r[0], r[1])
    shin = cv.capsule(knee, ank, r[2], r[3])
    leg = thigh | shin
    toe = norm(toe)
    perp = v2((-toe[1], toe[0]))
    shoe = cv.capsule(ank + perp * 1.0 - toe * 0.4, ank + perp * 1.6 + toe * shoe_len, 1.1 + r[3] * 0.2, 1.0)
    whole = leg | shoe
    po, pd, pb, pl = pants
    cv.put(ring4(whole), po, owner)
    if dark:
        cv.put(leg, pd, owner)
        cv.put(leg & ~nb(leg, -1, 0) & nb(leg, 1, 0), pb, owner)
    else:
        cv.put(leg, pb, owner)
        cv.put(leg & ~nb(leg, 1, 0), pd, owner)
        cv.put(leg & ~nb(leg, -1, 0) & nb(leg, 1, 0), pl, owner)
    _, t2 = cv.seg_t(knee, ank)
    bb, bl = boot
    bm = (shin & (t2 > boot_from)) | (shoe & ~leg)
    cv.put(bm, bb, owner)
    cv.put(bm & ~nb(bm, 0, -1), bl, owner)
    if not dark:
        cv.put(bm & ~nb(bm, -1, 0), bl, owner)
    return knee, ank, thigh, shin


# ---------------------------------------------------------------- weapon smears (inside the character canvas)
def smear(cv, pivot, r_in, r_out, a_from, a_to, warm=False, fade=1.0, owner=FXP, cool=None):
    """Crescent smear swept from a_from to a_to (degrees, screen space).
    Thick and bright at the leading edge (the blade), thinning to a dithered wisp."""
    px_, py_ = pivot
    X, Y = cv.XS, cv.YS
    r = np.hypot(X - px_, Y - py_)
    a = np.degrees(np.arctan2(Y - py_, X - px_))
    span = a_to - a_from
    if abs(span) < 1:
        return
    rel = np.mod((a - a_from) * np.sign(span), 360)
    t = rel / abs(span)
    inside = t <= 1.0
    max_thick = (r_out - r_in) * 0.62 * fade
    thick = max_thick * np.clip(t, 0, 1) ** 1.8
    band = inside & (r <= r_out + 0.5) & (r >= r_out - thick - 0.6)
    core = band & (t > 0.8) & (r > r_out - thick * 0.75)
    mid = band & (t > 0.45) & ~core
    tail = band & (t <= 0.45) & (t > 0.04)
    tail &= ((X + Y) % 2 == 0) | (r > r_out - 1.0)
    edge = band & (r > r_out - 1.0) & (t > 0.15)
    c_tail, c_mid = (FX["o2"], FX["o1"]) if warm else (FX["c2"], FX["c1"])
    if cool == "gold":
        c_tail, c_mid = FX["g3"], FX["g1"]
    cv.put(tail, c_tail, owner)
    cv.put(mid, FX["c2"] if (fade < 0.6 and not warm) else c_mid, owner)
    cv.put(core, FX["w"], owner)
    if warm:
        cv.put(edge & (t <= 0.8), FX["o3"], owner)
        cv.put(edge & (t <= 0.45), FX["o2"], owner)
    else:
        cv.put(edge & ~core & (t > 0.35), FX["w"], owner)


def angle_delta(a_from, a_to):
    """Signed shortest sweep from a_from to a_to, in degrees."""
    return (a_to - a_from + 180) % 360 - 180


def auto_smear(cv, pose, prev, pivot, guard, tip, warm=False, threshold=34, fade=None):
    """Smear from the previous frame's weapon angle when the weapon swung fast enough."""
    if prev is None or not pose.get("smear") or "wang" not in prev:
        return
    d = pose["wang"] - prev["wang"]          # raw: keys choose the swing direction explicitly
    if abs(d) > 330:
        d = angle_delta(prev["wang"], pose["wang"])
    if abs(d) < threshold:
        return
    a_to = ang_of(tip - pivot)
    a_from = a_to - d
    r_out = float(np.hypot(*(tip - pivot)))
    r_in = float(np.hypot(*(guard - pivot)))
    f = fade if fade is not None else min(1.0, 0.45 + abs(d) / 200.0)
    smear(cv, pivot, r_in, r_out + 1.0, a_from, a_to, warm=pose.get("warm", warm), fade=f)


def thrust_lines(cv, base, tip, n=3, owner=FXP, colors=None, spread=2.0, back=14):
    """Streaks trailing behind a thrusting point weapon."""
    colors = colors or (FX["w"], FX["c1"], FX["c2"])
    d = norm(tip - base)
    perp = v2((-d[1], d[0]))
    for i in range(n):
        off = (i - (n - 1) / 2) * spread
        p0 = tip - d * (3 + i % 2 * 2) + perp * off
        p1 = p0 - d * (back - abs(off) * 2)
        cv.put(cv.line(p0, p1), colors[min(i, len(colors) - 1)] if i != (n // 2) else FX["w"], owner)


def silhouette(cv, color):
    m = cv.rgba[:, :, 3] > 0
    cv.rgba[m] = color


# ---------------------------------------------------------------- cloth
def body_frame_pt(P, N, u, v):
    P, N, up, right, Lt = torso_frame(P, N)
    return P + right * u + up * (v * Lt)


def coat_back(cv, P, N, width, length, wind, ramp, owner, top=0.9, hem_teeth=3, phase=0.0):
    """Back flap of a coat / robe that trails behind the character (−u side) and flutters with `wind`."""
    P_, N_, up, right, Lt = torso_frame(P, N)
    wx, wy = wind

    def at(u, v):
        return P_ + right * u + up * (v * Lt)

    pts = [at(-width * 0.55, top), at(-width, top * 0.35), at(-width - 1.0 + wx * 1.6, -length * 0.45 + wy * 0.3)]
    # ragged hem: a few teeth between the back corner and the front of the hem
    back = at(-width - 2.0 + wx * 3.2, -length + wy * 0.7)
    front = at(-width * 0.25 + wx * 1.2, -length * 0.92 + wy * 0.3)
    pts.append(back)
    for i in range(1, hem_teeth):
        t = i / hem_teeth
        q = back + (front - back) * t
        q = q + up * ((1.2 if i % 2 else -0.8) + 0.6 * math.sin(phase * 1.7 + i))
        pts.append(q)
    pts += [front, at(0.0, -0.1)]
    m = cv.polygon(pts)
    paint_shaded(cv, m, ramp, owner, shadow=1)
    return m


def coat_panel(cv, hip, knee, w0, w1, extra, ramp, owner, hang=0.5, trim=None, shadow_side=False):
    """A coat skirt panel hanging over one thigh, split from the other side like a long coat in motion."""
    hip, knee = v2(hip), v2(knee)
    d = norm(knee - hip)
    down = norm(d * (1 - hang) + v2((0, 1)) * hang)
    side = v2((-d[1], d[0]))
    bot = knee + down * extra
    sd = v2((-down[1], down[0]))
    quad = [hip - side * w0, hip + side * w0, bot + sd * w1, bot - sd * w1]
    m = cv.polygon(quad)
    o, dk, b, l = ramp
    cv.put(ring4(m), o, owner)
    cv.put(m, dk if shadow_side else b, owner)
    cv.put(m & ~nb(m, 1, 0), dk, owner)
    if not shadow_side:
        cv.put(m & ~nb(m, -1, 0) & nb(m, 1, 0), l, owner)
    if trim is not None:
        cv.put(m & ~nb(m, 0, 1), trim, owner)
    return m


def aura(cv, colors, strength=1.0, phase=0, owner=FXP):
    """Flickering dithered glow hugging the character silhouette."""
    sil = cv.rgba[:, :, 3] > 0
    r1 = ring4(sil)
    r2 = ring4(sil | r1) & ~r1
    flick = ((cv.XS + cv.YS * 2 + phase) % 3) == 0
    cv.put(r1 & (cv.checker(phase) if strength < 0.7 else ~flick), colors[0], owner)
    if strength >= 0.5:
        cv.put(r2 & flick, colors[1], owner)

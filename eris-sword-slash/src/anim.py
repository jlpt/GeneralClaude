"""Pose description -> layered frame, plus the keyframes for the sword-draw animation."""
import math
import numpy as np
from rig import (Scene, v2, torso_frame, draw_head, HEAD_ANCHOR, draw_hair_back, draw_front_lock,
                 draw_jacket_tail, draw_scabbard, draw_sword, sword_geom, draw_leg, draw_torso, draw_neck,
                 draw_arm, unit)
import fx

SCAB_ANG = 130


def body_points(P, N):
    P, N, up, right, Lt = torso_frame(P, N)
    return dict(
        sn=N + right * -4.6 + up * -1.6,     # near (right) shoulder, image left
        sf=N + right * 3.6 + up * -1.6,      # far (left) shoulder, image right
        hn=P + right * -2.3,
        hf=P + right * 2.3,
    )


def scabbard_mouth(P, N):
    return body_points(P, N)["hf"] + v2((2.6, -0.6))


def hilt_grip(P, N, ang=SCAB_ANG):
    """Where a hand closes on the sheathed sword's grip."""
    return scabbard_mouth(P, N) + unit(ang + 180) * 3.2


def render_pose(p):
    sc = Scene()
    P, N, up, right, Lt = torso_frame(p["P"], p["N"])
    bp = body_points(P, N)
    head_off = v2(p.get("head", (0, 0)))
    head_origin = v2((int(round(N[0] + head_off[0])) - HEAD_ANCHOR[0],
                      int(round(N[1] + head_off[1])) - HEAD_ANCHOR[1]))
    hair = p.get("hair", {})
    wind, phase = hair.get("wind", (0, 0)), hair.get("phase", 0.0)
    an, af = p["arm_n"], p["arm_f"]
    sw = p.get("sword", {"state": "sheathed"})
    held = sw["state"] == "held"
    scab_ang = p.get("scab_ang", SCAB_ANG)
    sz = sw.get("z", "front")

    near_hand = v2(an["hand"])
    far_hand = v2(af["hand"])
    if held and sw.get("two_hand"):
        far_hand = near_hand - unit(sw["ang"]) * 2.8

    def near_arm():
        draw_arm(sc, bp["sn"], near_hand, an.get("bend", 1), "Front arm")

    def far_arm():
        draw_arm(sc, bp["sf"], far_hand, af.get("bend", 1), "Back arm", dark=af.get("z", "back") == "back")

    def sword():
        draw_sword(sc, near_hand, sw["ang"])

    draw_hair_back(sc, head_origin, wind, phase)
    if af.get("z", "back") == "back":
        far_arm()
    draw_jacket_tail(sc, P, N, p.get("coat", (0, 0)))
    if held and sz == "under_scabbard":
        sword()
    draw_scabbard(sc, scabbard_mouth(P, N), scab_ang, sw["state"] == "sheathed")
    if held and sz == "back":
        sword()
    draw_leg(sc, bp["hf"], p["foot_f"], p.get("bend_f", -1), dark=True, inner=-1)
    draw_leg(sc, bp["hn"], p["foot_n"], p.get("bend_n", -1), inner=1)
    draw_neck(sc, N, head_off)
    draw_torso(sc, P, N, p.get("twist", 0.0))
    if an.get("z") == "mid":
        near_arm()
    draw_head(sc, N, p.get("face", "smirk"), head_off)
    draw_front_lock(sc, head_origin, wind, phase)
    if af.get("z") == "front":
        far_arm()

    # smears sit over the body but under the blade and the sword hand
    for f in p.get("fx", []):
        if f[0] == "swing":
            _, a_from, pivot, fade, warm = f
            guard_c, tip = sword_geom(near_hand, sw["ang"])
            pivot = v2(pivot)
            r_out = float(np.hypot(*(tip - pivot)))
            r_in = float(np.hypot(*(guard_c - pivot)))
            a_to = math.degrees(math.atan2(tip[1] - pivot[1], tip[0] - pivot[0]))
            fx.smear(sc, pivot, r_in, r_out + 1.0, a_from, a_to, warm, fade)
    if held and sz == "front":
        sword()
    if an.get("z", "front") == "front":
        near_arm()
    for f in p.get("fx", []):
        if f[0] == "glint":
            _, tip = sword_geom(near_hand, sw["ang"])
            fx.spark(sc, tip, f[1])
        elif f[0] != "swing":
            fx.draw(sc, f)
    return sc


# ---------------------------------------------------------------- keyframes
def frame(**kw):
    base = dict(face="smirk", hair=dict(wind=(0, 0), phase=0.0), coat=(0, 0))
    base.update(kw)
    return base


IDLE_P, IDLE_N = (40, 47), (40.5, 33)

FRAMES = []


def add(dur, **kw):
    f = frame(**kw)
    f["dur"] = dur
    FRAMES.append(f)


# idle: hand behind head, other on the hilt, smirking
for i, (ny, hy, ph) in enumerate([(33, 23, 0.0), (32, 22, 0.8)]):
    P, N = IDLE_P, (40.5, ny)
    add(260, P=P, N=N,
        arm_n=dict(hand=(37, hy), bend=-1, z="mid"),
        arm_f=dict(hand=hilt_grip(P, N) + v2((-0.5, 1.5)), bend=-1, z="back"),
        foot_n=(35, 73), foot_f=(45, 73), hair=dict(wind=(0, 0), phase=ph))
IDLE_FRAMES = len(FRAMES)

# reach across for the hilt
P, N = (41, 48), (42.5, 34.5)
add(110, P=P, N=N, face="fierce",
    arm_n=dict(hand=(45, 41), bend=1, z="front"),
    arm_f=dict(hand=scabbard_mouth(P, N) + v2((0.5, 1.0)), bend=-1, z="back"),
    foot_n=(35, 73), foot_f=(46, 73), hair=dict(wind=(0.15, -0.15), phase=1.6), coat=(-0.2, 0))

# crouch and grip: anticipation
P, N = (40, 51), (43, 37.5)
add(170, P=P, N=N, face="fierce",
    arm_n=dict(hand=hilt_grip(P, N), bend=1, z="front"),
    arm_f=dict(hand=scabbard_mouth(P, N) + v2((0.5, 1.0)), bend=-1, z="back"),
    foot_n=(33, 73), foot_f=(50, 73), hair=dict(wind=(0.25, -0.35), phase=2.2), coat=(0.2, 0.3))

# DRAW-SLASH: rising cut straight out of the scabbard
P, N = (50, 51), (55, 38)
sn = body_points(P, N)["sn"]
add(60, P=P, N=N, face="shout",
    arm_n=dict(hand=(68, 31), bend=1, z="front"),
    arm_f=dict(hand=scabbard_mouth(P, N) + v2((-1.5, 1.0)), bend=-1, z="back"),
    sword=dict(state="held", ang=-52),
    foot_n=(36, 73), foot_f=(62, 73), bend_f=-1,
    hair=dict(wind=(-1.0, -0.5), phase=2.8), coat=(-1.2, 0.4),
    fx=[("swing", 50, sn + v2((0, 4)), 1.0, False), ("dust", (66, 73), 2.0, 0), ("line", (20, 44), (30, 44)),
        ("line", (16, 52), (28, 52)), ("line", (24, 60), (32, 60))])

# follow-through, blade high
P, N = (52, 49), (55.5, 35.5)
sn = body_points(P, N)["sn"]
add(80, P=P, N=N, face="fierce",
    arm_n=dict(hand=(63, 21), bend=1, z="front"),
    arm_f=dict(hand=(53, 50), bend=-1, z="back"),
    sword=dict(state="held", ang=-98),
    foot_n=(38, 73), foot_f=(62, 73),
    hair=dict(wind=(-1.4, -0.6), phase=3.4), coat=(-1.4, 0.6),
    fx=[("swing", 10, sn + v2((0, 4)), 0.45, False)])

# wind-up: both hands up by the near shoulder, blade cocked back behind the head
P, N = (51, 50), (50, 36)
add(150, P=P, N=N, face="shout",
    arm_n=dict(hand=(44.5, 24), bend=1, z="front"),
    arm_f=dict(hand=(0, 0), bend=1, z="back"),
    sword=dict(state="held", ang=-150, two_hand=True, z="back"),
    foot_n=(40, 73), foot_f=(63, 73),
    hair=dict(wind=(-0.4, 0.3), phase=4.0), coat=(-0.5, 0.2))

# BIG SLASH
P, N = (57, 53), (62, 40)
pv = body_points(P, N)["sn"] + v2((2, 0))
add(50, P=P, N=N, face="shout",
    arm_n=dict(hand=(74, 45), bend=1, z="front"),
    arm_f=dict(hand=(0, 0), bend=1, z="front"),
    sword=dict(state="held", ang=38, two_hand=True),
    foot_n=(42, 73), foot_f=(70, 73),
    hair=dict(wind=(0.6, -0.8), phase=4.6), coat=(-0.8, -0.2),
    fx=[("swing", -150, pv, 1.0, True), ("dust", (73, 73), 2.5, 1)])

# impact
P, N = (57, 54), (62.5, 41)
add(70, P=P, N=N, face="shout",
    arm_n=dict(hand=(72, 50), bend=1, z="front"),
    arm_f=dict(hand=(0, 0), bend=1, z="front"),
    sword=dict(state="held", ang=62, two_hand=True),
    foot_n=(42, 73), foot_f=(70, 73),
    hair=dict(wind=(1.0, 0.2), phase=5.2), coat=(0.4, -0.4),
    fx=[("line", (70, 74), (96, 74), "fx_w"), ("line", (74, 75), (90, 75), "fx_2"),
        ("spark", (82, 72), 3), ("spark", (88, 65), 1, "fx_3"), ("spark", (77, 66), 1, "fx_3"),
        ("dust", (86, 73), 2.5, 0), ("dust", (77, 74), 2.0, 1)])
add(150, P=P, N=(62.5, 40.5), face="fierce",
    arm_n=dict(hand=(72, 50), bend=1, z="front"),
    arm_f=dict(hand=(0, 0), bend=1, z="front"),
    sword=dict(state="held", ang=62, two_hand=True),
    foot_n=(42, 73), foot_f=(70, 73),
    hair=dict(wind=(0.6, 0.6), phase=5.8), coat=(0.3, 0.0),
    fx=[("line", (78, 75), (92, 75), "fx_2"), ("spark", (82, 71), 1), ("spark", (91, 61), 1, "fx_3"),
        ("spark", (74, 62), 1, "fx_4"), ("dust", (89, 72), 3.5, 1), ("dust", (75, 73), 3.0, 0)])

# back-step with the sword thrown onto the shoulder
P, N = (47, 44), (47.5, 30)
add(110, P=P, N=N, face="smirk",
    arm_n=dict(hand=(53, 33), bend=1, z="front"),
    arm_f=dict(hand=body_points(P, N)["hf"] + v2((1.0, -1.5)), bend=-1, z="back"),
    sword=dict(state="held", ang=-160, z="back"),
    foot_n=(41, 69), foot_f=(53, 68),
    hair=dict(wind=(0.9, -0.7), phase=6.4), coat=(0.6, -0.5),
    fx=[("dust", (66, 73), 2.5, 0), ("dust", (58, 74), 2.0, 1)])

# land: sword resting on the shoulder, smirk
P, N = IDLE_P, IDLE_N
for i, (ph, w, dur) in enumerate([(7.0, (0.5, 0.2), 140), (7.6, (0.15, 0.0), 420)]):
    add(dur, P=P, N=N, face="smirk",
        arm_n=dict(hand=(45, 35), bend=1, z="front"),
        arm_f=dict(hand=body_points(P, N)["hf"] + v2((1.0, -1.5)), bend=-1, z="back"),
        sword=dict(state="held", ang=-168, z="back"),
        foot_n=(35, 73), foot_f=(46, 73),
        hair=dict(wind=w, phase=ph),
        fx=[("glint", 2)] if i == 1 else [])

# sheathe: blade slides home along the scabbard
P, N = (40, 48), (41.5, 34)
grip = hilt_grip(P, N) + unit(SCAB_ANG + 180) * 7
add(110, P=P, N=N, face="smirk",
    arm_n=dict(hand=grip, bend=1, z="front"),
    arm_f=dict(hand=scabbard_mouth(P, N) + v2((0.5, 1.0)), bend=-1, z="back"),
    sword=dict(state="held", ang=SCAB_ANG, z="under_scabbard"),
    foot_n=(35, 73), foot_f=(45, 73),
    hair=dict(wind=(0.1, 0.0), phase=8.2))
P, N = IDLE_P, IDLE_N
add(200, P=P, N=N, face="smirk",
    arm_n=dict(hand=hilt_grip(P, N), bend=1, z="front"),
    arm_f=dict(hand=scabbard_mouth(P, N) + v2((0.5, 1.0)), bend=-1, z="back"),
    foot_n=(35, 73), foot_f=(45, 73),
    hair=dict(wind=(0, 0), phase=8.8),
    fx=[("spark", scabbard_mouth(P, N) + v2((1, -2)), 1)])
add(140, P=P, N=N, face="smirk",
    arm_n=dict(hand=(38, 30), bend=-1, z="mid"),
    arm_f=dict(hand=hilt_grip(P, N) + v2((-0.5, 1.5)), bend=-1, z="back"),
    foot_n=(35, 73), foot_f=(45, 73),
    hair=dict(wind=(0, 0), phase=9.4))

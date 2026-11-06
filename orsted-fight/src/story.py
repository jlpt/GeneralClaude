"""The fight, frame by frame (15 fps). Positions are world x of each character's root; poses are local
(facing right, y up is negative). Beats and frame ranges match PLAN.md."""
import math
import numpy as np
from px import v2, unit
from timeline import Track
from scene import Scene, Actor, Event
import eris, rudeus, ruijerd, orsted
import fx
from font import text_event

FPS = 15
BEATS = [  # name, first frame, last frame
    ("Standoff", 0, 44), ("Eris opens", 45, 89), ("Ruijerd", 90, 134), ("Stone Cannon", 135, 187),
    ("Team combo", 188, 240), ("Dragon God", 241, 285), ("Last stand", 286, 330), ("Aftermath", 331, 374),
]
LAST = BEATS[-1][2]
SHOT0, SHOT1, SHOT_ARC, SHOT_STOP = (132, 190), (358, 178), 46, 0.9

# ------------------------------------------------------------------ transient defaults for every key
TRANSIENT = dict(flash=False, thrust=False, smear=False)   # smears are opt-in on attack keys


def K(track, f, ease="inout", **kw):
    d = dict(TRANSIENT)
    d.update(kw)
    track.key(f, ease, **d)


# ------------------------------------------------------------------ pose libraries (local space, facing right)
# Eris
E_STANCE = dict(P=(0, -27), N=(2.5, -41), fn=(-8, -3), ff=(8, -3), kn=-1, kf=-1, hn=(10, -31), bn=1, zn="front",
                hf=(6, -31), bf=1, zf="back", two=True, wstate="held", wang=-38, wz="front", expr="fierce",
                breath=1, spin=0, head=(0, 0), wind=(-0.35, -0.1), coat=(-0.3, 0))
E_CROUCH = dict(E_STANCE, P=(-1, -23), N=(0, -37), fn=(-9, -3), ff=(9, -3), hn=(8, -27), wang=-30, breath=0)


def e_run(b):
    if b == 0:
        return dict(P=(3, -26), N=(11, -38), fn=(-11, -4), ff=(12, -7), hn=(-6, -27), bn=-1, wang=168, two=False,
                    hf=(14, -34), bf=1, zf="back", wz="back", expr="fierce", wind=(-1.2, -0.3), coat=(-1.4, 0.3),
                    breath=0, smear=False)
    return dict(P=(3, -27.5), N=(11, -39.5), fn=(10, -8), ff=(-10, -4), hn=(-5, -28), bn=-1, wang=165, two=False,
                hf=(12, -36), bf=1, zf="back", wz="back", expr="fierce", wind=(-1.3, -0.4), coat=(-1.5, 0.2),
                breath=0, smear=False)


E_LEAP = dict(P=(0, -30), N=(-1.5, -44), fn=(-5, -15), ff=(9, -14), hn=(-3, -56), bn=1, two=True, wang=-150,
              wz="back", zf="back", expr="shout", wind=(-0.5, 0.7), coat=(-0.6, 0.8), smear=False)
E_SWING = dict(P=(2, -29), N=(5, -43), fn=(-4, -13), ff=(10, -10), hn=(6, -56), wang=-90, wz="front", zf="front",
               two=True, expr="shout", wind=(-0.6, 0.3), smear=True)
E_CONTACT = dict(P=(4, -28), N=(9, -41), fn=(-6, -10), ff=(11, -8), hn=(14, -42), wang=-20, two=True, zf="front",
                 wz="front", expr="shout", wind=(-0.4, 0.2), smear=True)
E_DEFLECT = dict(P=(1, -29), N=(-2, -43), fn=(-2, -12), ff=(8, -14), hn=(2, -54), wang=-130, two=False, zf="back",
                 hf=(10, -38), expr="hurt", wz="back", wind=(0.6, -0.4), smear=False)
E_TUCK = dict(P=(0, -28), N=(2, -41), fn=(6, -16), ff=(10, -18), hn=(8, -34), wang=-150, two=False, hf=(6, -30),
              zf="back", expr="hurt", wz="back", wind=(0.8, -0.2), smear=False)
E_SKID = dict(P=(-2, -19), N=(3, -32), fn=(-11, -3), ff=(10, -3), hn=(9, -22), bn=1, wang=-15, two=False,
              hf=(-2, -14), bf=-1, zf="back", expr="fierce", wz="front", wind=(0.5, -0.3), spin=0, smear=False)
E_IAI_WIND = dict(P=(1, -24), N=(6, -37), fn=(-10, -3), ff=(12, -3), hn=(-5, -27), bn=-1, wang=140, two=False,
                  hf=(11, -31), zf="back", wz="back", expr="shout", wind=(-1.0, -0.3), smear=False)
E_IAI_CUT = dict(P=(5, -25), N=(12, -37), fn=(-12, -3), ff=(15, -3), hn=(17, -40), bn=1, wang=-50, two=False,
                 hf=(-2, -28), zf="back", wz="front", expr="shout", warm=True, wind=(-1.1, -0.5), smear=True)
E_BLOWN = dict(P=(0, -28), N=(-5, -41), fn=(8, -10), ff=(12, -14), hn=(10, -44), bn=1, wang=-100, hf=(6, -48),
               two=False, zf="back", wz="back", expr="hurt", wind=(1.0, -0.4), smear=False)
E_OVERHEAD = dict(P=(0, -27), N=(-1, -41), fn=(-9, -3), ff=(10, -3), hn=(-3, -54), bn=1, two=True, wang=-150,
                  wz="back", zf="back", expr="shout", wind=(-0.6, -0.4), breath=0, smear=False)
E_LYING = dict(P=(0, -5), N=(-14, -6), fn=(22, -3), ff=(20, -6), kn=-1, kf=-1, hn=(-4, -2), bn=1, hf=(-12, -10),
               bf=1, zf="back", two=False, wang=172, wz="back", expr="ko", breath=0, spin=0, wind=(-0.2, 0.1),
               smear=False)

# Rudeus
R_STANCE = dict(P=(0, -26), N=(1, -39), fn=(-5, -3), ff=(6, -3), kn=-1, kf=-1, hn=(6, -30), bn=1, wang=-78,
                hf=(9, -27), bf=1, zf="back", two=False, wstate="held", wz="front", expr="calm", breath=1,
                breath_off=3, spin=0, glow=0.0, head=(0, 0), wind=(-0.3, 0), coat=(-0.4, 0))
R_CAST = dict(R_STANCE, P=(-1, -25), N=(-1, -38), fn=(-8, -3), ff=(9, -3), hn=(8, -36), wang=-30, two=True,
              two_gap=6, zf="front", expr="cast", glow=1.0, breath=0)
R_RECOIL = dict(R_CAST, P=(-3, -25), N=(-5, -38), hn=(5, -38), wang=-42, glow=0.3)
R_SHOCK = dict(R_STANCE, P=(-2, -26), N=(-2, -39), hn=(6, -31), wang=-40, two=False, hf=(7, -28), zf="back",
               expr="shock", glow=0.0, breath=0)
R_RAISE = dict(R_STANCE, hn=(4, -47), wang=-95, hf=(9, -30), expr="cast", glow=1.0, breath=0)
R_POINT = dict(R_STANCE, P=(1, -24), N=(4, -37), fn=(-7, -3), ff=(10, -3), hn=(11, -32), wang=22, two=True,
               two_gap=6, zf="front", expr="cast", glow=1.0, breath=0, smear=True)
R_BLOWN = dict(R_STANCE, P=(0, -27), N=(-6, -38), fn=(10, -12), ff=(7, -8), hn=(10, -32), wang=-60, hf=(9, -42),
               zf="back", two=False, expr="hurt", glow=0.0, breath=0, wind=(1.2, -0.3), coat=(1.4, -0.2))
R_SLUMP = dict(R_STANCE, P=(0, -8), N=(-3, -21), fn=(13, -3), ff=(11, -4), hn=(8, -5), wang=8, hf=(-2, -9),
               zf="back", two=False, expr="ko", breath=0, head=(0, 1), glow=0.0, wind=(-0.2, 0), coat=(0, 0))

# Ruijerd
J_STANCE = dict(P=(-1, -33), N=(1, -50), fn=(-10, -3), ff=(10, -3), kn=-1, kf=-1, hn=(-3, -32), bn=1, hf=(8, -36),
                bf=1, zf="front", two=True, two_gap=11, wang=-8, wstate="held", wz="front", expr="stern", breath=1,
                breath_off=5, spin=0, head=(0, 0), wind=(-0.3, 0), coat=(-0.4, 0))
J_CROUCH = dict(J_STANCE, P=(-2, -28), N=(1, -45), fn=(-11, -3), ff=(11, -3), hn=(-3, -29), breath=0)


def j_run(b):
    base = dict(J_STANCE, P=(2, -31), N=(9, -46), hn=(2, -34), wang=-4, breath=0, wind=(-1.0, -0.2),
                coat=(-1.3, 0.2))
    if b == 0:
        return dict(base, fn=(-13, -4), ff=(13, -7))
    return dict(base, P=(2, -32.5), N=(9, -47.5), fn=(12, -8), ff=(-12, -4))


J_THRUST = dict(J_STANCE, P=(5, -31), N=(11, -47), fn=(-14, -3), ff=(16, -3), hn=(12, -40), wang=-4, thrust=True,
                expr="shout", breath=0)
J_RETRACT = dict(J_STANCE, P=(1, -32), N=(5, -48), fn=(-12, -3), ff=(13, -3), hn=(-4, -37), wang=-6, breath=0)
J_THRUST_HI = dict(J_THRUST, hn=(11, -45), wang=-14)
J_THRUST_LO = dict(J_THRUST, hn=(13, -36), wang=2)
J_YANKED = dict(J_THRUST_LO, P=(7, -31), N=(15, -45), hn=(16, -40), wang=-4, expr="hurt", thrust=False)
J_FLUNG = dict(J_STANCE, P=(0, -32), N=(-4, -48), fn=(6, -14), ff=(10, -18), hn=(4, -42), wang=-30, expr="hurt",
               breath=0, wind=(0.9, -0.3), coat=(1.0, -0.3), smear=False)
J_TUCK = dict(J_FLUNG, P=(0, -32), N=(3, -47), fn=(6, -18), ff=(10, -20), hn=(6, -38), wang=-20)
J_KNEEL = dict(J_STANCE, P=(0, -21), N=(4, -37), fn=(-11, -2), ff=(9, -3), hn=(2, -26), wang=-10, spin=0,
               expr="stern", breath=0, wind=(0.4, -0.2), coat=(0.6, -0.2))
J_DIVE = dict(J_STANCE, P=(0, -33), N=(3, -49), fn=(-6, -12), ff=(8, -16), hn=(4, -44), wang=55, two=True,
              two_gap=10, zf="front", expr="shout", breath=0, wind=(-0.4, -0.9), coat=(-0.4, -0.9))
J_JUMP = dict(J_DIVE, wang=-30, hn=(0, -46), fn=(-8, -10), ff=(9, -13))
J_LYING = dict(J_STANCE, P=(0, -6), N=(-17, -7), fn=(28, -3), ff=(26, -6), hn=(-6, -2), hf=(-20, -10), zf="back",
               wang=178, two=False, wz="back", expr="hurt", breath=0, spin=0, wind=(0, 0), coat=(0, 0))

# Orsted
O_IDLE = dict(P=(0, -40), N=(-0.5, -61), fn=(-6, -3), ff=(7, -3), kn=-1, kf=-1, hn=(-4, -31), bn=-1, hf=(4, -32),
              bf=-1, zf="back", expr="cold", shape_n="fist", shape_f="fist", breath=1, breath_off=2, head=(0, 0),
              spin=0, aura=0.0, aura_col="dragon", wind=(-0.4, -0.1), coat=(-0.8, 0.0))
O_RAISE = dict(O_IDLE, hn=(12, -60), bn=1, shape_n="open", breath=0)
O_DEFLECT = dict(O_RAISE, hn=(16, -68))
O_SWAT = dict(O_RAISE, hn=(6, -76))
O_SWAY = dict(O_IDLE, P=(-3, -39), N=(-8, -59), hn=(-6, -34), hf=(2, -36), breath=0)
O_DUCK = dict(O_IDLE, P=(1, -36), N=(4, -56), head=(1, 1), breath=0)
O_GRAB = dict(O_IDLE, P=(1, -39), N=(2, -60), hn=(17, -35), bn=1, shape_n="fist", expr="glare", breath=0)
O_PULL = dict(O_GRAB, hn=(10, -42))
O_PUSH = dict(O_GRAB, P=(3, -39), N=(7, -60), fn=(-9, -3), ff=(11, -3), hn=(26, -50), shape_n="open")
O_PALM = dict(O_IDLE, N=(1, -61), hn=(22, -66), bn=1, shape_n="open", expr="glare", breath=0)
O_BURST = dict(O_IDLE, P=(0, -38), N=(0, -60), hn=(12, -76), bn=1, hf=(-6, -77), bf=1, shape_n="open",
               shape_f="open", expr="glare", aura=2.0, breath=0, coat=(-2.0, -0.6), wind=(-1.0, -0.6))
O_CROUCH = dict(O_IDLE, P=(-2, -36), N=(2, -56), fn=(-9, -3), ff=(9, -3), hn=(-2, -36), hf=(6, -38), expr="glare",
                aura=1.0, breath=0)
O_LUNGE = dict(O_IDLE, P=(4, -31), N=(13, -50), fn=(-19, -3), ff=(17, -3), hn=(30, -36), bn=1, shape_n="open",
               hf=(-10, -42), expr="glare", aura=1.0, breath=0, coat=(-2.2, -0.4), wind=(-1.2, -0.3))
O_CATCH = dict(O_IDLE, N=(1, -61), hn=(14, -46), bn=1, shape_n="point2", expr="cold", breath=0)
O_FLICK = dict(O_CATCH, hn=(22, -58), shape_n="open")
O_STRIKE = dict(O_IDLE, P=(2, -38), N=(6, -59), fn=(-8, -3), ff=(11, -3), hn=(22, -66), bn=1, shape_n="open",
                expr="glare", breath=0, aura=1.0)


def build():
    tE = Track("Eris", x=192, face=1, z=2, lift=0, **E_STANCE)
    tR = Track("Rudeus", x=92, face=1, z=0, lift=0, **R_STANCE)
    tJ = Track("Ruijerd", x=142, face=1, z=1, lift=0, **J_STANCE)
    tO = Track("Orsted", x=380, face=-1, z=3, lift=0, **O_IDLE)
    for t in (tE, tR, tJ, tO):
        K(t, 0)
    ev = []

    def E(f0, f1, fn, order="front", **kw):
        ev.append(Event(f0, f1, fn, order, **kw))

    shakes = []      # (f0, f1, amplitude)
    impacts = {}     # frame -> "white" / "black"

    E(0, LAST + 1, fx.motes, "front", n=22)

    # ============================================================ 1. Standoff (0-44)
    K(tE, 44, **E_STANCE)
    K(tR, 44, **R_STANCE)
    K(tJ, 44, **J_STANCE)
    K(tO, 30, **O_IDLE)
    K(tO, 36, **dict(O_IDLE, expr="glare"))
    K(tO, 44, **dict(O_IDLE, expr="glare"))
    E(8, 45, text_event, s="ERIS  RUDEUS  RUIJERD", at=(240, 22), scale=2, cps=1.6, life=37)
    E(20, 45, text_event, s="VS", at=(240, 44), scale=2, fill="o3", top="o2", shade="o4", life=25)
    E(24, 45, text_event, s="ORSTED", at=(240, 64), scale=3, fill="c1", top="w", shade="p3", cps=0.8, life=21)
    E(36, 45, fx.eye_glow)

    # ============================================================ 2. Eris opens (45-89)
    K(tE, 46, **E_CROUCH)
    K(tE, 47, **E_CROUCH)
    xs = [200, 236, 270, 300]
    for i, x in enumerate(xs):
        K(tE, 48 + i * 2, "linear", x=x, **e_run(i % 2))
    K(tE, 55, "out", x=312, lift=12, **E_LEAP)
    K(tE, 57, "out", x=322, lift=24, **E_LEAP)
    K(tE, 58, "in", x=327, lift=22, **E_SWING)
    K(tE, 59, "linear", x=329, lift=20, **E_CONTACT)
    K(tE, 60, "out", x=322, lift=25, **dict(E_DEFLECT, spin=-30))
    K(tE, 62, "linear", x=300, lift=32, **dict(E_TUCK, spin=-120))
    K(tE, 64, "linear", x=276, lift=26, **dict(E_TUCK, spin=-230))
    K(tE, 66, "in", x=256, lift=8, **dict(E_TUCK, spin=-330))
    K(tE, 67, "step", x=248, lift=0, **E_SKID)
    K(tE, 74, "out3", x=230, **E_SKID)
    K(tE, 80, x=228, **dict(E_STANCE, expr="fierce"))
    K(tE, 89, x=228, **E_STANCE)
    E(48, 56, fx.speed_lines, actor="Eris", n=7, seed=1)
    E(48, 58, fx.dust, at=(196, 243), n=3, spread=5, seed=2)
    E(55, 64, fx.dust, at=(306, 243), n=3, spread=6, seed=3)
    E(59, 67, fx.impact, at=(364, 175), size=1.1, seed=4)
    E(67, 76, fx.skid, actor="Eris")
    E(67, 77, fx.dust, at=(246, 243), n=4, spread=9, seed=5)
    shakes += [(59, 63, 3)]

    K(tO, 56, **dict(O_IDLE, expr="glare"))
    K(tO, 57, **O_RAISE)
    K(tO, 58, "out", **O_DEFLECT)
    K(tO, 59, "linear", **O_DEFLECT)
    K(tO, 60, "out", **O_SWAT)
    K(tO, 64, **dict(O_IDLE, hn=(0, -44), bn=-1))
    K(tO, 70, **O_IDLE)
    K(tO, 89, **O_IDLE)

    # ============================================================ 3. Ruijerd (90-134)
    K(tJ, 90, **J_STANCE)
    K(tJ, 92, **J_CROUCH)
    for i, x in enumerate([166, 214, 258, 296]):
        K(tJ, 93 + i * 2, "linear", x=x, **j_run(i % 2))
    K(tJ, 100, "out", x=322, **J_THRUST)
    K(tJ, 102, x=320, **J_RETRACT)
    K(tJ, 103, "out", x=322, **J_THRUST_HI)
    K(tJ, 105, x=320, **J_RETRACT)
    K(tJ, 106, "out", x=322, **J_THRUST_LO)
    K(tJ, 107, "linear", x=322, **dict(J_THRUST_LO, thrust=False))
    K(tJ, 108, "out", x=326, **J_YANKED)
    K(tJ, 109, "out", x=304, lift=18, **dict(J_FLUNG, flash=True, spin=-40))
    K(tJ, 111, "linear", x=258, lift=44, **dict(J_TUCK, spin=-140))
    K(tJ, 114, "linear", x=212, lift=36, **dict(J_TUCK, spin=-250))
    K(tJ, 117, "in", x=176, lift=8, **dict(J_TUCK, spin=-350))
    K(tJ, 118, "step", x=170, lift=0, **J_KNEEL)
    K(tJ, 123, "out3", x=160, **J_KNEEL)
    K(tJ, 127, x=160, **J_KNEEL)
    K(tJ, 133, x=160, **J_STANCE)
    E(93, 100, fx.speed_lines, actor="Ruijerd", n=6, seed=11, height=(-62, -8))
    E(93, 103, fx.dust, at=(162, 243), n=3, spread=6, seed=12)
    E(107, 112, fx.impact, at=(363, 209), size=0.6, seed=13)
    E(109, 116, fx.impact, at=(354, 194), size=1.2, seed=14)
    E(118, 126, fx.skid, actor="Ruijerd")
    E(118, 128, fx.dust, at=(170, 243), n=4, spread=9, seed=15)
    shakes += [(100, 101, 1), (103, 104, 1), (106, 107, 1), (109, 113, 3), (118, 120, 2)]

    K(tO, 99, **O_IDLE)
    K(tO, 100, "out", **O_SWAY)
    K(tO, 102, **dict(O_IDLE, breath=0))
    K(tO, 103, "out", **O_DUCK)
    K(tO, 105, **dict(O_IDLE, breath=0))
    K(tO, 106, "out", **dict(O_SWAY, P=(-2, -39), N=(-5, -60)))
    K(tO, 107, "out", **O_GRAB)
    K(tO, 108, "out", **O_PULL)
    K(tO, 109, "out", **O_PUSH)
    K(tO, 112, **dict(O_PUSH, hn=(20, -52)))
    K(tO, 120, **O_IDLE)
    K(tO, 134, **O_IDLE)

    # ============================================================ 4. Stone Cannon (135-187)
    K(tR, 135, **R_STANCE)
    K(tR, 139, **R_CAST)
    K(tR, 150, **dict(R_CAST, P=(-1.5, -25), N=(-2, -38)))
    K(tR, 151, "out", **R_RECOIL)
    K(tR, 154, **dict(R_RECOIL, glow=0.0))
    K(tR, 160, **R_SHOCK)
    K(tR, 175, **R_SHOCK)
    K(tR, 187, **dict(R_STANCE, expr="cast"))
    E(137, 152, fx.magic_circle, at=("Rudeus", "gem", (3, 0)), r=10, ry=2.4, spin=18, seed=1)
    E(139, 151, fx.stone_charge, at=("Rudeus", "gem", (9, 0)), r_max=5, grow=9)
    E(151, 161, fx.stone_shot, p0=SHOT0, p1=SHOT1, frames=9, r=5, stop=SHOT_STOP, arc=SHOT_ARC)
    E(151, 157, fx.dust, at=("Rudeus", "gem", (6, 0)), n=3, spread=3, rise=0.3, drift=(-0.6, 0), r=2.4, seed=21)
    E(142, 162, text_event, s="STONE CANNON!", at=(112, 158), fill="c2", top="w", shade="c3", cps=1.5, life=20)
    E(153, 170, fx.ranma, at=("Orsted", "hn", (0, 0)), rings=5, period=3, r_max=28)
    E(155, 173, text_event, s="RAN MA", at=(360, 150), fill="g2", top="g1", shade="p3", cps=1.2, life=18)
    E(159, 184, fx.crumble, at=tuple(fx.stone_point(SHOT0, SHOT1, SHOT_STOP, SHOT_ARC)), n=16, dirx=-0.6,
      seed=22)
    shakes += [(151, 153, 2), (159, 162, 2)]

    K(tO, 148, **O_IDLE)
    K(tO, 151, **dict(O_PALM, hn=(16, -56)))
    K(tO, 153, "out", **O_PALM)
    K(tO, 168, **O_PALM)
    K(tO, 176, **O_IDLE)
    K(tO, 187, **O_IDLE)

    # ============================================================ 5. Team combo (188-240)
    K(tR, 190, **R_RAISE)
    K(tR, 192, **R_RAISE)
    K(tR, 194, "out", **R_POINT)
    K(tR, 206, **R_POINT)
    K(tR, 210, **dict(R_STANCE, expr="shock"))
    K(tR, 240, **R_STANCE)
    E(189, 196, fx.magic_circle, at=("Rudeus", "gem", (0, 0)), r=8, ry=1.0, spin=24)
    E(194, 206, fx.magic_circle, "back", at=(380, 244), r=32, ry=0.2, col=("g2", "mud_h", "g1"), spin=10, grow=5,
      layer="FX (back)")
    E(197, 236, fx.quagmire, "back", at=(380, 244), w=28, grow=5, sink_layer="back")
    E(197, 236, fx.quagmire, "front", at=(380, 244), w=28, grow=5, sink_layer="front")
    shakes += [(195, 197, 1)]

    # Eris: dash in, rising iai cut — interrupted by the burst
    K(tE, 194, x=228, **E_CROUCH)
    for i, x in enumerate([252, 284, 312, 332]):
        K(tE, 196 + i * 2, "linear", x=x, **e_run(i % 2))
    K(tE, 204, "out", x=340, **E_IAI_WIND)
    K(tE, 206, "out", x=346, **E_IAI_CUT)
    K(tE, 207, "out", x=332, lift=10, **dict(E_BLOWN, flash=True, spin=-20))
    K(tE, 210, "linear", x=296, lift=26, **dict(E_TUCK, spin=-150))
    K(tE, 213, "linear", x=262, lift=18, **dict(E_TUCK, spin=-280))
    K(tE, 215, "in", x=244, lift=4, **dict(E_TUCK, spin=-350))
    K(tE, 216, "step", x=238, lift=0, **E_SKID)
    K(tE, 222, "out3", x=220, **E_SKID)
    K(tE, 230, x=220, **E_STANCE)
    K(tE, 240, x=220, **E_STANCE)
    E(196, 204, fx.speed_lines, actor="Eris", n=7, seed=31)
    E(196, 206, fx.dust, at=(232, 243), n=3, spread=6, seed=32)
    E(216, 224, fx.skid, actor="Eris")
    E(216, 226, fx.dust, at=(238, 243), n=4, spread=9, seed=33)

    # Ruijerd: leaps over everyone and dives with the trident
    K(tJ, 192, **J_STANCE)
    K(tJ, 195, **J_CROUCH)
    K(tJ, 197, "out", x=200, lift=30, **J_JUMP)
    K(tJ, 200, "out", x=270, lift=62, **J_JUMP)
    K(tJ, 203, "linear", x=320, lift=74, **J_DIVE)
    K(tJ, 206, "in", x=350, lift=56, **dict(J_DIVE, thrust=True))
    K(tJ, 207, "out", x=338, lift=72, **dict(J_FLUNG, flash=True, spin=-30))
    K(tJ, 211, "out", x=300, lift=96, **dict(J_TUCK, spin=-160))
    K(tJ, 215, "in", x=276, lift=50, **dict(J_TUCK, spin=-290))
    K(tJ, 219, "in", x=266, lift=6, **dict(J_TUCK, spin=-355))
    K(tJ, 220, "step", x=264, lift=0, **J_KNEEL)
    K(tJ, 226, "out3", x=258, **J_KNEEL)
    K(tJ, 234, x=258, **J_STANCE)
    K(tJ, 240, x=258, **J_STANCE)
    E(195, 205, fx.dust, at=(160, 243), n=4, spread=7, seed=34)
    E(220, 230, fx.dust, at=(264, 243), n=4, spread=9, seed=35)
    E(220, 226, fx.skid, actor="Ruijerd")

    # Orsted: sinks into the mud, then the mana burst
    K(tO, 196, **dict(O_IDLE, expr="cold"))
    K(tO, 202, lift=-7, **dict(O_IDLE, head=(2, 1), breath=0))
    K(tO, 204, lift=-7, **dict(O_CROUCH, aura=0.6))
    K(tO, 206, "out", lift=-7, **O_BURST)
    K(tO, 212, lift=-7, **O_BURST)
    K(tO, 218, lift=-7, **dict(O_IDLE, aura=0.5))
    K(tO, 226, lift=0, **dict(O_IDLE, aura=0.0))
    K(tO, 240, **O_IDLE)
    E(206, 224, fx.shockwave, at=("Orsted", "chest", (0, 0)), r_max=110, speed=12)
    impacts[206] = "white"
    shakes += [(206, 214, 6), (216, 218, 2), (220, 222, 2)]

    # ============================================================ 6. Dragon God (241-285)
    K(tO, 241, **O_IDLE)
    K(tO, 244, **O_CROUCH)
    K(tO, 245, "step", x=300, hide=True, **O_LUNGE)
    K(tO, 246, "linear", x=210, hide=True, **O_LUNGE)
    K(tO, 247, "linear", x=125, hide=False, **dict(O_LUNGE, hn=(22, -40)))
    K(tO, 248, "out", x=125, **O_LUNGE)
    K(tO, 254, x=125, **dict(O_LUNGE, aura=0.5))
    K(tO, 262, x=125, **dict(O_IDLE, expr="glare"))
    K(tO, 264, "step", x=125, face=1, **dict(O_IDLE, expr="glare"))
    K(tO, 285, x=125, face=1, **dict(O_IDLE, expr="cold"))
    E(242, 251, fx.eye_glow)
    E(245, 252, fx.ghost, "back", actor="Orsted", lags=(1, 2, 3), lo=244, hi=247)
    E(248, 256, fx.palm_burst, at=("Orsted", "hn", (0, 0)), dirx=-1)
    impacts[248] = "black"
    impacts[249] = "white"
    shakes += [(248, 256, 7), (251, 256, 4)]

    K(tR, 247, **dict(R_STANCE, expr="shock"))
    K(tR, 248, **dict(R_SHOCK))
    K(tR, 249, "out", x=80, lift=8, **dict(R_BLOWN, flash=True))
    K(tR, 250, "linear", x=72, lift=12, **R_BLOWN)
    K(tR, 251, "out", x=68, lift=12, **dict(R_BLOWN, P=(0, -27), N=(-2, -40)))
    K(tR, 254, "in", x=68, lift=0, **R_SLUMP)
    K(tR, 285, x=68, **R_SLUMP)
    E(251, 272, fx.crater, at=(60, 222), w=16)

    K(tE, 248, **E_STANCE)
    K(tE, 250, "step", face=-1, **dict(E_STANCE, expr="shout", breath=0))
    K(tE, 270, face=-1, **dict(E_STANCE, expr="shout", aura=1.0, aura_col="rage"))
    K(tE, 285, face=-1, **dict(E_STANCE, expr="fierce", aura=1.0, aura_col="rage"))
    K(tJ, 248, **J_STANCE)
    K(tJ, 251, "step", face=-1, **dict(J_STANCE, expr="shout"))
    K(tJ, 262, face=-1, **dict(J_STANCE, expr="stern"))
    K(tJ, 285, face=-1, **J_STANCE)
    E(256, 274, text_event, s="RUDEUS!!", at=(222, 154), fill="o2", top="o1", shade="o4", cps=1.3, life=18)

    # ============================================================ 7. Last stand (286-330)
    K(tE, 288, x=220, **dict(E_OVERHEAD, aura=1.0, aura_col="rage"))
    for i, x in enumerate([206, 192, 182]):
        K(tE, 289 + i, "linear", x=x, **dict(E_OVERHEAD, aura=1.0, aura_col="rage", fn=((-9, -3), (8, -6), (-8, -3))[i],
                                                ff=((10, -3), (-6, -3), (11, -5))[i]))
    K(tE, 292, "in", x=177, **dict(E_SWING, lift=0, fn=(-10, -3), ff=(12, -3), P=(2, -27), N=(5, -41), aura=1.0,
                                    aura_col="rage", warm=True))
    K(tE, 293, "linear", x=175, **dict(E_CONTACT, wang=-10, fn=(-12, -3), ff=(14, -3), P=(3, -26), N=(8, -39),
                                        aura=1.0, aura_col="rage", warm=True))
    K(tE, 300, "step", x=175, **dict(E_CONTACT, wang=-10, fn=(-12, -3), ff=(14, -3), P=(3, -26), N=(8, -39),
                                      aura=0.4, aura_col="rage", smear=False))
    K(tE, 301, "out", x=196, lift=14, **dict(E_BLOWN, flash=True, spin=-20, aura=0.0))
    K(tE, 304, "linear", x=240, lift=26, **dict(E_TUCK, spin=-160))
    K(tE, 307, "linear", x=278, lift=16, **dict(E_TUCK, spin=-290))
    K(tE, 309, "in", x=298, lift=3, **dict(E_BLOWN, spin=-80))
    K(tE, 310, "step", x=304, lift=0, **E_LYING)
    K(tE, 330, x=304, **E_LYING)
    E(289, 293, fx.speed_lines, actor="Eris", n=6, seed=41)
    E(293, 301, fx.impact, at=(139, 198), size=1.0, warm=True, seed=42)
    E(296, 301, fx.glint, at=(139, 197), size=3)
    E(301, 308, fx.impact, at=(141, 196), size=1.3, seed=43)
    E(310, 320, fx.dust, at=(304, 243), n=5, spread=12, seed=44)
    impacts[293] = "black"
    shakes += [(293, 297, 3), (301, 304, 4), (310, 312, 2)]

    K(tO, 290, face=1, **dict(O_IDLE, expr="cold"))
    K(tO, 292, face=1, **dict(O_CATCH, hn=(12, -52)))
    K(tO, 293, "out", face=1, **O_CATCH)
    K(tO, 300, "step", face=1, **O_CATCH)
    K(tO, 301, "out", face=1, **O_FLICK)
    K(tO, 304, face=1, **dict(O_IDLE, hn=(4, -40), bn=1))
    K(tO, 311, face=1, **dict(O_IDLE, P=(-2, -39), N=(-4, -60), breath=0))
    K(tO, 313, face=1, **dict(O_STRIKE, hn=(14, -58)))
    K(tO, 314, "out", face=1, **O_STRIKE)
    K(tO, 320, face=1, **dict(O_STRIKE, aura=0.5))
    K(tO, 327, face=1, **O_IDLE)
    K(tO, 330, face=1, **O_IDLE)

    K(tJ, 302, face=-1, **dict(J_STANCE, expr="shout"))
    K(tJ, 307, face=-1, **J_CROUCH)
    K(tJ, 309, "out", x=230, lift=36, face=-1, **J_JUMP)
    K(tJ, 311, "linear", x=190, lift=48, face=-1, **J_DIVE)
    K(tJ, 313, "in", x=162, lift=34, face=-1, **dict(J_DIVE, thrust=True))
    K(tJ, 314, "out", x=172, lift=46, face=-1, **dict(J_FLUNG, flash=True, spin=-30))
    K(tJ, 317, "linear", x=196, lift=46, face=-1, **dict(J_TUCK, spin=-170))
    K(tJ, 320, "in", x=212, lift=10, face=-1, **dict(J_FLUNG, spin=-80))
    K(tJ, 321, "step", x=216, lift=0, face=-1, **J_LYING)
    K(tJ, 330, x=216, face=-1, **J_LYING)
    E(307, 316, fx.dust, at=(258, 243), n=4, spread=8, seed=45)
    E(314, 321, fx.impact, at=(147, 176), size=1.4, seed=46)
    E(321, 330, fx.dust, at=(216, 243), n=5, spread=13, seed=47)
    impacts[314] = "white"
    shakes += [(314, 318, 5), (321, 323, 2)]

    # ============================================================ 8. Aftermath (331-374)
    K(tO, 336, face=1, **dict(O_IDLE, coat=(-1.4, -0.1), wind=(-0.8, -0.2)))
    K(tO, 344, face=1, **dict(O_IDLE, expr="glare", coat=(-1.6, -0.2), wind=(-0.9, -0.2)))
    K(tO, LAST, face=1, **dict(O_IDLE, expr="glare", coat=(-1.6, -0.2), wind=(-0.9, -0.2)))
    E(344, 355, fx.eye_glow)
    K(tE, 334, x=304, **E_LYING)
    K(tE, 340, x=304, **dict(E_LYING, N=(-12, -13), hf=(4, -22), bf=-1, zf="front", expr="hurt"))
    K(tE, 346, x=304, **dict(E_LYING, N=(-12, -14), hf=(6, -24), bf=-1, zf="front", expr="hurt"))
    K(tE, 350, "in", x=304, **E_LYING)
    K(tE, LAST, x=304, **E_LYING)
    K(tR, LAST, x=68, **R_SLUMP)
    K(tJ, LAST, x=216, face=-1, **J_LYING)
    E(356, LAST + 1, text_event, s="ORSTED", at=(240, 104), scale=3, fill="c1", top="w", shade="p3", cps=0.7)
    E(362, LAST + 1, text_event, s="THE DRAGON GOD", at=(240, 134), scale=1, fill="g2", top="g1", shade="g4",
      cps=1.4)

    # ------------------------------------------------------------ camera + post
    def cam(f):
        dx = dy = 0.0
        for f0, f1, amp in shakes:
            if f0 <= f < f1:
                k = 1 - (f - f0) / max(f1 - f0, 1)
                a = amp * (0.4 + 0.6 * k)
                rng = np.random.default_rng(f * 31 + 7)
                dx += rng.choice([-1, 1]) * round(a * rng.uniform(0.5, 1.0))
                dy += rng.choice([-1, 1]) * round(a * rng.uniform(0.3, 0.8))
        return (int(max(-9, min(9, dx))), int(max(-9, min(9, dy))))

    def post(f):
        d = {}
        if f < 9:
            d["fade"] = 1 - f / 9
        if f >= 350:
            d["fade"] = min(1.0, (f - 350) / 10)
        if f in impacts:
            d["impact"] = impacts[f]
        return d

    actors = [Actor(rudeus, tR), Actor(ruijerd, tJ), Actor(eris, tE), Actor(orsted, tO)]
    return Scene(actors, ev, cam, post)

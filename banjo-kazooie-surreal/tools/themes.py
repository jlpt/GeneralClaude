"""Dream palettes for the Surreal mod. Imported by recolor.py.

Each theme repaints a group of level models (their textures and vertex colours).
`hue` rotates every colour around the colour wheel (degrees), `sat` scales saturation,
`tint` multiplies RGB (the only way to colour white things such as snow), and `jitter`
lets each individual texture drift a different way so surfaces stop agreeing with
each other. Characters, enemies and items are left alone on purpose: familiar things
in the wrong world.

Asset ids come from sMapModelDescriptions (src/core2/mapModel.c) and D_8036BD40
(src/core2/gc/sky.c) in the decomp.
"""
from recolor import theme, ids

# ---- Spiral Mountain: blue cobbles, golden house, teal cliffs
theme("spiral mountain",   ids(0x14CF, 0x14D0),                 hue=170, sat=1.15, jitter=35)
theme("spiral mtn sky",    ids(0x7C4),                          hue=180, sat=1.25)
theme("banjo's house",     ids(0x14A2),                         hue=95,  sat=1.2,  jitter=30)

# ---- Mumbo's Mountain: golden grass, rose cliffs
theme("mumbo's mountain",  ids(0x14AA, 0x14AB, 0x14AC, 0x14AD), hue=300, sat=1.2,  val=1.12, jitter=30)
theme("mm sky",            ids(0x7BD, 0x7BE),                   hue=270, sat=1.2)
theme("mumbo's skull",     ids(0x14AE),                         hue=180, sat=1.3,  jitter=40)

# ---- Treasure Trove Cove: mint sand, magenta sea
theme("treasure trove",    ids((0x146B, 0x1473)),               hue=100, sat=1.1,  jitter=25)
theme("ttc sky",           ids(0x7BF, 0x7C0),                   hue=175, sat=1.2)

# ---- Clanker's Cavern: violet water, cyan iron
theme("clanker's cavern",  ids((0x14ED, 0x14F2)),               hue=150, sat=1.5,  jitter=30)

# ---- Bubblegloop Swamp: amber swamp, rose wood
theme("bubblegloop swamp", ids((0x14D1, 0x14D5)),               hue=290, sat=1.2,  jitter=30)

# ---- Freezeezy Peak: strawberry snow, orange ice
theme("freezeezy peak",    ids((0x14C8, 0x14CE)),               hue=140, sat=1.2,  jitter=30, tint=(1.0, 0.74, 0.88))
theme("fp sky",            ids(0x7C6, 0x7C7, 0x7C8),            hue=260, sat=1.3)

# ---- Gobi's Valley: blue dunes, golden water
theme("gobi's valley",     ids((0x1474, 0x147E)),               hue=185, sat=1.15, jitter=25)
theme("gv sky",            ids(0x7C1),                          hue=150, sat=1.2)

# ---- Mad Monster Mansion: crimson night, blue lawns
theme("mad monster",       ids((0x147F, 0x149B)),               hue=110, sat=1.25, jitter=35)
theme("mmm sky",           ids(0x7C2, 0x7C3),                   hue=110, sat=1.3)

# ---- Rusty Bucket Bay: violet rust, lime water
theme("rusty bucket bay",  ids((0x14B0, 0x14C7)),               hue=130, sat=1.9,  val=1.25, jitter=35)
theme("rbb sky",           ids(0x7C5),                          hue=120, sat=1.6)

# ---- Click Clock Wood: every season is the wrong season
theme("ccw hub",           ids(0x14D8, 0x14E3),                 hue=60,  sat=1.2,  jitter=30)
theme("ccw spring",        ids(0x14D9, 0x14E4),                 hue=240, sat=1.15, jitter=25)
theme("ccw summer",        ids(0x14DA, 0x14E5),                 hue=160, sat=1.15, jitter=25)
theme("ccw autumn",        ids(0x14DB, 0x14E6),                 hue=90,  sat=1.15, jitter=25)
theme("ccw winter",        ids(0x14DC, 0x14E7),                 hue=300, sat=1.2,  jitter=25, tint=(0.78, 1.0, 0.84))
theme("ccw interiors",     ids((0x14DD, 0x14E2)),               hue=120, sat=1.2,  jitter=35)

# ---- Gruntilda's Lair: green tunnels, orange stone
theme("gruntilda's lair",  ids((0x14F3, 0x150E), (0x1510, 0x1512), 0x1514, 0x1515, 0x14A8), hue=120, sat=1.25, jitter=40)
theme("lair skies",        ids(0x7CA, 0x7CB),                   hue=200, sat=1.3)

# ---- cutscene stages
theme("intro concert",     ids((0x149D, 0x14A1)),               hue=200, sat=1.15, jitter=30)
theme("intro sky",         ids(0x7C9),                          hue=270, sat=1.2)
theme("grunty's lab",      ids(0x150F, 0x1513),                 hue=140, sat=1.3,  jitter=35)
theme("ending beach",      ids(0x14A5, 0x14A9),                 hue=100, sat=1.1,  jitter=25)
theme("ending skies",      ids(0x7CC, 0x7CD),                   hue=175, sat=1.2)

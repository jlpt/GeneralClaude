"""Hand-pixelled head sprites (3/4 view, facing right). Each entry: rows, key -> palette name, anchor (neck)."""
from palette import ERIS, RUDEUS, RUIJERD, ORSTED

# ---------------------------------------------------------------- Eris (from eris-sword-slash/src/faces.py)
_ERIS_TOP = [
    "..........oo...",
    ".........oHo...",
    "......ooooRo...",
    "....ooRRHHRRoo.",
    "...oRRHHhhHRRRo",
    "..oRRHHhhHHRRRo",
    ".oRRRRHHHRRRRRo",
    ".oRRRRRRRRRRrRo",
    "oRRRRRRrRRRrRRo",
    "oRRRrRSrRSrSrRo",
]
_ERIS_FACES = {
    "smirk": [
        "oRRrRSooSSSooRo",
        "oRrRSoeESSSeEko",
        "oRrRSSEeSSSEeSo",
        "oRrRrSSSSsSSSro",
        ".oRrRsSSmSmSkro",
        ".odRo.kSSmSkdro",
        "..oo...kkkk.oo.",
    ],
    "fierce": [
        "oRRrRoSSSSSSoRo",
        "oRrRSSooeSoooko",
        "oRrRSSeESSSeESo",
        "oRrRrSSSSsSSSro",
        ".oRrRsSSSSSSkro",
        ".odRo.kSmmmkdro",
        "..oo...kkkk.oo.",
    ],
    "shout": [
        "oRRrRoSSSSSSoRo",
        "oRrRSSooeSoooko",
        "oRrRSSeESSSeESo",
        "oRrRrSSSSsSSSro",
        ".oRrRsSSmmmSkro",
        ".odRo.kSmEmkdro",
        "..oo...kkmkkoo.",
    ],
    "hurt": [
        "oRRrRSSSSSSSoRo",
        "oRrRSkooSSSookS",
        "oRrRSSSSkSSSSko",
        "oRrRrSSSSsSSSro",
        ".oRrRsSSmmmSkro",
        ".odRo.kSkkkkdro",
        "..oo...kkkk.oo.",
    ],
    "ko": [
        "oRRrRSSSSSSSoRo",
        "oRrRSSSSSSSSSko",
        "oRrRSkkoSSSkkoo",
        "oRrRrSSSSsSSSro",
        ".oRrRsSSSSSSkro",
        ".odRo.kSSmSkdro",
        "..oo...kkkk.oo.",
    ],
}
ERIS_HEADS = {k: _ERIS_TOP + v for k, v in _ERIS_FACES.items()}
ERIS_KEY = {
    "o": ERIS["hair_o"], "d": ERIS["hair_d"], "r": ERIS["hair_s"], "R": ERIS["hair_b"], "H": ERIS["hair_l"],
    "h": ERIS["hair_h"], "k": ERIS["skin_o"], "s": ERIS["skin_s"], "S": ERIS["skin_b"], "L": ERIS["skin_l"],
    "e": ERIS["eye_d"], "E": ERIS["eye_r"], "m": ERIS["mouth"],
}
ERIS_ANCHOR = (8, 19)

# ---------------------------------------------------------------- Rudeus
_RUD_TOP = [
    ".......o.oo....",
    ".....ooHoHRo...",
    "...ooRRHRRRRo..",
    "..oRRHHhhHRRRo.",
    ".oRRHHhhHHRRRRo",
    ".oRRRHHHHRRRRRo",
    "oRRRRRRRRRRRrRo",
    "oRRrRRRrRRRrRro",
    "oRrRRrSRrSSrSRo",
]
_RUD_FACES = {
    "calm": [
        "oRrRSoooSSSoooo",
        "oRrRSeESSSSeEko",
        "orRrSEeSSSSEeSo",
        ".oRrSSSSSsSSSko",
        "..odkSSSSSSSSko",
        "...o.kSSmmSSko.",
        "......kkkkkk...",
    ],
    "cast": [
        "oRrRSSoooSSoooo",
        "oRrRSoeESSSoeEk",
        "orRrSSEeSSSSEeo",
        ".oRrSSSSSsSSSko",
        "..odkSSSSmSSSko",
        "...o.kSSmmmSko.",
        "......kkkkkk...",
    ],
    "shock": [
        "oRrRSoooSSSoooo",
        "oRrRoeEoSSoeEok",
        "orRrSEeSSSSEeSo",
        ".oRrSSSSSsSSSko",
        "..odkSSSmmSSSko",
        "...o.kSSmmSSko.",
        "......kkkkkk...",
    ],
    "hurt": [
        "oRrRSSSSSSSSSSo",
        "oRrRSkooSSSkook",
        "orRrSSSSkSSSSko",
        ".oRrSSSSSsSSSko",
        "..odkSSmmmmSSko",
        "...o.kSkkkkSko.",
        "......kkkkkk...",
    ],
    "ko": [
        "oRrRSSSSSSSSSSo",
        "oRrRSSSSSSSSSSk",
        "orRrSkkoSSSkkoo",
        ".oRrSSSSSsSSSko",
        "..odkSSSSSSSSko",
        "...o.kSSSmSSko.",
        "......kkkkkk...",
    ],
}
RUDEUS_HEADS = {k: _RUD_TOP + v for k, v in _RUD_FACES.items()}
RUDEUS_KEY = {
    "o": RUDEUS["hair_o"], "d": RUDEUS["hair_d"], "r": RUDEUS["hair_s"], "R": RUDEUS["hair_b"],
    "H": RUDEUS["hair_l"], "h": RUDEUS["hair_h"], "k": RUDEUS["skin_o"], "s": RUDEUS["skin_s"],
    "S": RUDEUS["skin_b"], "L": RUDEUS["skin_l"], "e": RUDEUS["eye_d"], "E": RUDEUS["eye_g"],
    "m": RUDEUS["mouth"],
}
RUDEUS_ANCHOR = (8, 18)

# ---------------------------------------------------------------- Ruijerd (shaved head, forehead jewel)
_RUI_TOP = [
    "....ooooo....",
    "..ooLLLSSoo..",
    ".oLLLSSSSSSo.",
    ".oLLSSSSSgGSo",
    "oLLSSSSSSGjSo",
    "osSSSSSSSSSSo",
]
_RUI_FACES = {
    "stern": [
        "osSSSSoooSSoo",
        "ossdkSSeASSeA",
        "osskSSSSSSSSo",
        "ossSSSSSSsSSo",
        ".ossSSSSSSSSo",
        ".osssSSSSmmSo",
        "..ossSSSSSSo.",
        "...oossSSSo..",
        ".....okkk....",
    ],
    "shout": [
        "osSSSSoooSSoo",
        "ossdkSSeASSeA",
        "osskSSSSSSSSo",
        "ossSSSSSSsSSo",
        ".ossSSSSSmmmo",
        ".osssSSSSmeeo",
        "..ossSSSSmmo.",
        "...oossSSSo..",
        ".....okkk....",
    ],
    "hurt": [
        "osSSSSSSSSSSo",
        "ossdkSkooSkoo",
        "osskSSSSSSSSo",
        "ossSSSSSSsSSo",
        ".ossSSSSSSSSo",
        ".osssSSSSdddo",
        "..ossSSSSSSo.",
        "...oossSSSo..",
        ".....okkk....",
    ],
}
RUIJERD_HEADS = {k: _RUI_TOP + v for k, v in _RUI_FACES.items()}
RUIJERD_KEY = {
    "o": RUIJERD["skin_o"], "d": RUIJERD["skin_d"], "k": RUIJERD["skin_d"], "s": RUIJERD["skin_s"],
    "S": RUIJERD["skin_b"], "L": RUIJERD["skin_l"], "e": RUIJERD["eye_d"], "A": RUIJERD["eye_a"],
    "m": RUIJERD["mouth"], "g": RUIJERD["gem_l"], "G": RUIJERD["gem_b"], "j": RUIJERD["gem_d"],
}
RUIJERD_ANCHOR = (7, 16)

# ---------------------------------------------------------------- Orsted (silver hair swept back, golden eyes)
_ORS_TOP = [
    ".......ooooo.....",
    ".....ooRRHHHoo...",
    "...ooRRHHHhhHHo..",
    ".ooRRRRRHHhhHHRo.",
    "oRRRRRRRRHHHHRRRo",
    ".ooRRRRRRRRRRRRRo",
    "oRRRRrRRRRRRrRRRo",
    ".oRRRRrRRRRrRRrRo",
    "..oRRrRRrSRrSRRro",
    ".oRRrRRSSrSSSrSRo",
]
_ORS_FACES = {
    "cold": [
        "..orRrRbbbSSSbbbo",
        "..orRrSSGeSSSSGek",
        "..orRrSSSSSSSSSSo",
        "...orRSSSSSSsSSSk",
        "...orRSSSSSSSSSko",
        "...orrkSSSSmmSSko",
        "....oro.kSSSSSko.",
        ".........kkkkk...",
    ],
    "glare": [
        "..orRrRbbbSSSbbbo",
        "..orRrSSYGSSSSYGk",
        "..orRrSSSSSSSSSSo",
        "...orRSSSSSSsSSSk",
        "...orRSSSSSSSSSko",
        "...orrkSSSmmmSSko",
        "....oro.kSSSSSko.",
        ".........kkkkk...",
    ],
}
ORSTED_HEADS = {k: _ORS_TOP + v for k, v in _ORS_FACES.items()}
ORSTED_KEY = {
    "o": ORSTED["hair_o"], "d": ORSTED["hair_d"], "r": ORSTED["hair_s"], "R": ORSTED["hair_b"],
    "H": ORSTED["hair_l"], "h": ORSTED["hair_h"], "k": ORSTED["skin_o"], "s": ORSTED["skin_s"],
    "S": ORSTED["skin_b"], "L": ORSTED["skin_l"], "e": ORSTED["eye_d"], "G": ORSTED["eye_g"],
    "Y": ORSTED["eye_h"], "m": ORSTED["mouth"], "b": ORSTED["brow"],
}
ORSTED_ANCHOR = (10, 20)

for _name, _heads in (("eris", ERIS_HEADS), ("rudeus", RUDEUS_HEADS), ("ruijerd", RUIJERD_HEADS),
                      ("orsted", ORSTED_HEADS)):
    for _k, _rows in _heads.items():
        _w = {len(r) for r in _rows}
        assert len(_w) == 1, (_name, _k, _w)

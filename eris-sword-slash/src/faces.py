TOP = [
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
FACES = {
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
}
HEADS = {k: TOP + v for k, v in FACES.items()}
for k, rows in HEADS.items():
    assert all(len(r) == 15 for r in rows), k

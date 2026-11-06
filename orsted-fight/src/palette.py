"""Every colour ramp used in the piece, grouped by owner."""
from px import hexc

# ---------------------------------------------------------------- Eris (same ramps as eris-sword-slash)
ERIS = {
    "hair_o": hexc("3b0d16"), "hair_d": hexc("6b1824"), "hair_s": hexc("952a2b"),
    "hair_b": hexc("c23c31"), "hair_l": hexc("e2623f"), "hair_h": hexc("f89a68"),
    "skin_o": hexc("7e3a36"), "skin_s": hexc("e0a084"), "skin_b": hexc("f8d0b4"), "skin_l": hexc("fff0e0"),
    "eye_d": hexc("3a0812"), "eye_r": hexc("d8343a"), "mouth": hexc("a84a4c"),
    "jak_o": hexc("100a10"), "jak_d": hexc("1e161c"), "jak_b": hexc("2e242a"),
    "jak_l": hexc("483b42"), "jak_h": hexc("66565e"), "stripe": hexc("d6ccc8"), "stripe_s": hexc("9a8c8c"),
    "fur_d": hexc("6e5850"), "fur_b": hexc("a98a76"), "fur_l": hexc("d9c2a8"), "fur_h": hexc("f0e2cc"),
    "top_s": hexc("b8a69c"), "top_b": hexc("e8dace"), "top_l": hexc("faf2ea"), "bow": hexc("2a1c1e"),
    "sho_d": hexc("1a1418"), "sho_b": hexc("2c2428"), "sho_l": hexc("40363a"),
    "belt": hexc("6c4632"), "belt_l": hexc("90643f"), "buckle": hexc("c8ccd4"),
    "leg_o": hexc("1a110e"), "leg_d": hexc("362820"), "leg_b": hexc("4f3c33"), "leg_l": hexc("6c5446"),
    "sock": hexc("1e1a1e"), "sock_l": hexc("383038"),
    "shoe_b": hexc("1c181c"), "shoe_l": hexc("3c3440"),
    "bl_o": hexc("262a36"), "bl_d": hexc("6a7688"), "bl_b": hexc("aab6c6"), "bl_l": hexc("e2eaf2"),
    "bl_h": hexc("ffffff"),
    "guard": hexc("c4c8d2"), "guard_d": hexc("7c8090"), "grip": hexc("3a2a28"), "grip_l": hexc("6c4c3c"),
    "scab": hexc("2e2420"), "scab_l": hexc("4a3c34"), "scab_tip": hexc("a8acb6"),
}

# ---------------------------------------------------------------- Rudeus
RUDEUS = {
    "hair_o": hexc("2a170f"), "hair_d": hexc("4a2c1c"), "hair_s": hexc("6b4229"),
    "hair_b": hexc("8a5a36"), "hair_l": hexc("b07a48"), "hair_h": hexc("d6a46a"),
    "skin_o": hexc("7a4234"), "skin_s": hexc("dea486"), "skin_b": hexc("f6d2b6"), "skin_l": hexc("fff0e2"),
    "eye_d": hexc("1c2a22"), "eye_g": hexc("4f9a6a"), "mouth": hexc("a6584e"),
    # robe (grey with a cool tint)
    "rob_o": hexc("1a1a22"), "rob_d": hexc("3a3a48"), "rob_b": hexc("585a6c"), "rob_l": hexc("7c7f92"),
    "rob_h": hexc("a2a6b6"), "rob_in": hexc("2a2a36"),
    # shirt / trousers / boots
    "sh_s": hexc("b4a68e"), "sh_b": hexc("e2d6bc"), "sh_l": hexc("f6eedc"),
    "tr_o": hexc("1c120e"), "tr_d": hexc("3e2a20"), "tr_b": hexc("5a4030"), "tr_l": hexc("765642"),
    "bt_b": hexc("2c1e18"), "bt_l": hexc("4c3628"),
    "belt": hexc("6a4a30"), "belt_l": hexc("94704a"),
    # staff
    "st_o": hexc("24160e"), "st_d": hexc("54361e"), "st_b": hexc("7e5630"), "st_l": hexc("a87c4a"),
    "gem_o": hexc("10284a"), "gem_d": hexc("1e5a9a"), "gem_b": hexc("3a9ae0"), "gem_l": hexc("8ad6ff"),
    "gem_h": hexc("e8fbff"),
}

# ---------------------------------------------------------------- Ruijerd
RUIJERD = {
    "skin_o": hexc("2e2a40"), "skin_d": hexc("6e6888"), "skin_s": hexc("9e98b2"), "skin_b": hexc("c9c4d4"),
    "skin_l": hexc("e6e2ee"),
    "eye_d": hexc("1a1420"), "eye_a": hexc("c0882c"), "mouth": hexc("8a6a74"),
    "gem_d": hexc("7a0c1a"), "gem_b": hexc("d81c30"), "gem_l": hexc("ff6a6a"), "gem_h": hexc("ffd0d0"),
    # tunic (slate teal)
    "tu_o": hexc("0e1a1e"), "tu_d": hexc("1e3a42"), "tu_b": hexc("2e5660"), "tu_l": hexc("467a84"),
    "tu_h": hexc("6aa0a8"),
    # sash / wraps
    "sa_d": hexc("7a6a52"), "sa_b": hexc("b09c7a"), "sa_l": hexc("d8c8a4"),
    # trousers / boots
    "tr_o": hexc("141012"), "tr_d": hexc("2e2628"), "tr_b": hexc("463a3c"), "tr_l": hexc("5e5052"),
    "bt_b": hexc("241a16"), "bt_l": hexc("44342a"),
    # trident (bone white)
    "sp_o": hexc("2a2a34"), "sp_d": hexc("9a9890"), "sp_b": hexc("d8d4c6"), "sp_l": hexc("f4f2e8"),
    "sp_h": hexc("ffffff"), "sp_band": hexc("6a5440"),
}

# ---------------------------------------------------------------- Orsted
ORSTED = {
    "hair_o": hexc("2c2e40"), "hair_d": hexc("5c6078"), "hair_s": hexc("8a8ea6"),
    "hair_b": hexc("b8bccc"), "hair_l": hexc("dcdfea"), "hair_h": hexc("ffffff"),
    "skin_o": hexc("5a3a3e"), "skin_s": hexc("caa092"), "skin_b": hexc("eccab8"), "skin_l": hexc("fbe8dc"),
    "eye_d": hexc("3a2408"), "eye_g": hexc("ffcc3a"), "eye_h": hexc("fff4b0"), "mouth": hexc("7a4a4a"),
    "brow": hexc("3a3c52"),
    # coat (white)
    "co_o": hexc("26222e"), "co_d": hexc("7c768c"), "co_s": hexc("aaa4b8"), "co_b": hexc("d6d2de"),
    "co_l": hexc("f2f0f6"), "co_in": hexc("3a3446"), "co_trim": hexc("c8a44a"), "co_trim_d": hexc("8a6a2a"),
    # black clothes
    "bk_o": hexc("0a090e"), "bk_d": hexc("15131c"), "bk_b": hexc("221f2c"), "bk_l": hexc("363244"),
    "bk_h": hexc("4c4760"),
    "bt_b": hexc("141218"), "bt_l": hexc("302c38"),
    "belt": hexc("3a3040"), "buckle": hexc("c8a44a"),
}

# ---------------------------------------------------------------- effects
FX = {
    "w": hexc("ffffff"), "c1": hexc("e6f6ff"), "c2": hexc("a8dcff"), "c3": hexc("5aa8f0"), "c4": hexc("2a5aa8"),
    "o1": hexc("fff0b0"), "o2": hexc("ffb070"), "o3": hexc("ff6a3c"), "o4": hexc("c8302a"),
    "g1": hexc("fff4c0"), "g2": hexc("ffd45a"), "g3": hexc("e0a02a"), "g4": hexc("9a6a1a"),
    "p1": hexc("e4c8ff"), "p2": hexc("b48aff"), "p3": hexc("7a4ad8"), "p4": hexc("3e2486"),
    "dust_l": hexc("cfc4b8"), "dust_m": hexc("a0958a"), "dust_d": hexc("6e655e"),
    "rock_o": hexc("1e1a1c"), "rock_d": hexc("4a4246"), "rock_b": hexc("6e6468"), "rock_l": hexc("958a8c"),
    "rock_h": hexc("bcb2b0"),
    "mud_o": hexc("1e140e"), "mud_d": hexc("3a2818"), "mud_b": hexc("5a3e24"), "mud_l": hexc("7e5a34"),
    "mud_h": hexc("a07a4c"),
    "shadow": hexc("1a1726"), "black": hexc("000000"), "ink": hexc("0c0a12"),
}

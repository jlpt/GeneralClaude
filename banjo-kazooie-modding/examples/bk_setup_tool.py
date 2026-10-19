#!/usr/bin/env python3
"""
bk_setup_tool.py - inspect and patch Banjo-Kazooie level setup files.

A setup file is the asset at id (map_id + 0x71C). bk_asset_tool extracts it to
assets/lvl_setup/XXXX.lvl_setup.bin. This script reimplements the game's own
parser so you can see what a map contains and retarget actor placements:

    gsworld_load()                    src/core2/gsworld.c
    cubeList_fromFile()               src/core2/gccube.c
    code7AF80_initCubeFromFile()      src/core2/code_A5BC0.c
    ncCameraNodeList_fromFile() etc.  src/core2/nc/cameranodelist.c, code_33*.c
    lightingVectorList_fromFile()     src/core2/gclights.c
    file_isNextByteExpected()         src/core2/file.c  (the "peek" semantics)

Usage:
    bk_setup_tool.py dump <setup.bin> [--all]
    bk_setup_tool.py replace-actor <setup.bin> <old_id> <new_id> [--limit N] [-o out.bin]
    bk_setup_tool.py selftest

The layout was derived from the decomp source (commit e8d31fa5). `selftest`
checks the reader against the writer below. Run `dump` on an unmodified file
first to make sure the output looks sane.
"""
import argparse
import struct
import sys

NODE_SIZE = 20          # sizeof(NodeProp)
OTHER_NODE_SIZE = 12    # sizeof(OtherNode)
PROP_SIZE = 12          # sizeof(Prop)

NODE_CATEGORIES = {
    3: "warp",           # func_80334448: sWarpFunctions[actorId]
    4: "trigger",        # func_80334448: sRadiusTriggers[actorId]
    6: "actor",          # cube_spawnActors: spawn actorId
    7: "enemy-boundary",
    8: "path",
    9: "camera-trigger",
    10: "flag",
}


class SetupFormatError(Exception):
    pass


class Reader:
    """Mirrors File + file_isNextByteExpected() from src/core2/file.c."""

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.stash = None   # File::last_expected

    def _take(self, n):
        if self.pos + n > len(self.data):
            raise SetupFormatError(f"unexpected end of file at 0x{self.pos:X}")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def byte(self):
        return self._take(1)[0]

    def expect(self, value):
        if self.stash is None:
            b = self.byte()
            if b == value:
                return True
            self.stash = b
            return False
        if self.stash == value:
            self.stash = None
            return True
        return False

    def words(self, n):
        return list(struct.unpack(f">{n}i", self._take(4 * n)))

    def floats(self, n):
        return list(struct.unpack(f">{n}f", self._take(4 * n)))

    def short(self):
        return struct.unpack(">h", self._take(2))[0]

    def raw(self, n):
        start = self.pos
        return start, self._take(n)


def decode_node(buf, offset):
    x, y, z, w6, actor_id, marker_id, _pad, w12, w16 = struct.unpack_from(">hhhHHBBII", buf, offset)
    return {
        "offset": offset,
        "pos": (x, y, z),
        "radius": w6 >> 7,                 # selector_or_radius:9
        "category": (w6 >> 1) & 0x3F,      # category:6
        "bit0": w6 & 1,
        "actor_id": actor_id,
        "marker_id": marker_id,
        "yaw": w12 >> 23,                  # yaw:9
        "scale": w12 & 0x7FFFFF,           # scale:23 (percent, 0 = 100)
        "unk10_31": w16 >> 20,             # -> Actor::secondaryId
        "unk10_19": (w16 >> 8) & 0xFFF,
    }


def decode_prop(buf, offset):
    w0, x, y, z, w10 = struct.unpack_from(">IhhhH", buf, offset)
    is_actor = w10 & 1
    is_model = (w10 >> 1) & 1
    prop = {"offset": offset, "pos": (x, y, z)}
    if is_actor:
        prop["kind"] = "actor"
    elif is_model:
        prop["kind"] = "model"
        prop["asset_id"] = (w0 >> 20) + 0x2D1      # ModelProp.modelId:12 + MODEL_ASSET_OFFSET
        prop["yaw"] = (w0 >> 8) & 0xFF
        prop["roll"] = w0 & 0xFF
        prop["scale"] = buf[offset + 10]
    else:
        prop["kind"] = "sprite"
        prop["asset_id"] = (w0 >> 20) + 0x572      # SpriteProp.spriteId:12 + SPRITE_ASSET_OFFSET
        prop["scale"] = (w0 >> 2) & 0xFF
        prop["frame"] = w10 >> 11
    return prop


def parse_cube_contents(r, cube):
    # code7AF80_initCubeFromFile
    if r.expect(0x0A):
        count = r.byte()
        if r.expect(0x0B):
            start, blob = r.raw(count * NODE_SIZE)
            cube["nodes"] = [decode_node(r.data, start + i * NODE_SIZE) for i in range(count)]
    elif r.expect(0x06):
        count = r.byte()
        if r.expect(0x07):
            start, _ = r.raw(count * OTHER_NODE_SIZE)
            cube["other_nodes"] = count
    if r.expect(0x08):
        count = r.byte()
        if r.expect(0x09):
            start, _ = r.raw(count * PROP_SIZE)
            cube["props"] = [decode_prop(r.data, start + i * PROP_SIZE) for i in range(count)]


def parse_cube(r, cube):
    # __code7AF80_initCubeFromFile
    guard = 0
    while not r.expect(0x01):
        guard += 1
        if guard > 100000:
            raise SetupFormatError(f"stuck parsing cube at 0x{r.pos:X}")
        if r.expect(0x00):
            r.words(3)
            r.words(3)
        elif r.expect(0x02):
            r.words(3)
        elif r.expect(0x03):
            parse_cube_contents(r, cube)
        else:
            raise SetupFormatError(f"unknown cube byte 0x{r.stash:02X} near 0x{r.pos:X}")


def parse_cubes(r, out):
    # cubeList_fromFile
    if not r.expect(0x01):
        raise SetupFormatError("cube section without dimensions marker")
    lo = r.words(3)
    hi = r.words(3)
    out["cube_min"], out["cube_max"] = lo, hi
    cubes = []
    for x in range(lo[0], hi[0] + 1):
        for y in range(lo[1], hi[1] + 1):
            for z in range(lo[2], hi[2] + 1):
                cube = {"index": (x, y, z), "nodes": [], "props": []}
                parse_cube(r, cube)
                cubes.append(cube)
    r.expect(0x00)
    out["cubes"] = cubes


def parse_camera_node(r, node_type):
    data = {}
    if node_type in (1, 3):     # pivot / zoom
        while not r.expect(0x00):
            if r.expect(0x01):
                data["position"] = r.floats(3)
            elif r.expect(0x02):
                data["hv_speed"] = r.floats(2)
            elif r.expect(0x03):
                data["turn"] = r.floats(2)
            elif node_type == 3 and r.expect(0x06):
                data["distance"] = r.floats(2)
            elif r.expect(0x04):
                data["pitch_yaw_roll"] = r.floats(3)
            elif r.expect(0x05):
                data["flag"] = r.words(1)[0]
            else:
                raise SetupFormatError(f"bad camera byte near 0x{r.pos:X}")
    elif node_type == 2:        # static
        while not r.expect(0x00):
            if r.expect(0x01):
                data["position"] = r.floats(3)
            elif r.expect(0x02):
                data["pitch_yaw_roll"] = r.floats(3)
            else:
                raise SetupFormatError(f"bad static camera byte near 0x{r.pos:X}")
    elif node_type == 4:        # random
        while not r.expect(0x00):
            if r.expect(0x01):
                data["flag"] = r.words(1)[0]
            else:
                raise SetupFormatError(f"bad random camera byte near 0x{r.pos:X}")
    return data


def parse_cameras(r, out):
    cams = []
    while not r.expect(0x00):
        if r.expect(0x01):
            index = r.short()
            node_type = r.byte() if r.expect(0x02) else 0
            cams.append({"index": index, "type": node_type, **parse_camera_node(r, node_type)})
        else:
            raise SetupFormatError(f"bad camera list byte near 0x{r.pos:X}")
    out["cameras"] = cams


def parse_lights(r, out):
    lights = []
    while not r.expect(0x00):
        if r.expect(0x01) and r.expect(0x02):
            pos = r.floats(3)
            if r.expect(0x03):
                radii = r.floats(2)
                if r.expect(0x04):
                    rgb = r.words(3)
                    lights.append({"position": pos, "fade_radii": radii, "rgb": rgb})
        elif r.stash not in (None, 0x00):
            raise SetupFormatError(f"bad lighting byte near 0x{r.pos:X}")
    out["lights"] = lights


def parse_setup(data):
    # gsworld_load
    r = Reader(data)
    out = {"cubes": [], "cameras": [], "lights": []}
    guard = 0
    while not r.expect(0x00):
        guard += 1
        if guard > 100:
            raise SetupFormatError("too many sections")
        if r.expect(0x02):
            pass
        elif r.expect(0x01):
            parse_cubes(r, out)
        elif r.expect(0x03):
            parse_cameras(r, out)
        elif r.expect(0x04):
            parse_lights(r, out)
        else:
            raise SetupFormatError(f"unknown section byte 0x{r.stash:02X} at 0x{r.pos:X}")
    out["end"] = r.pos
    return out


# ---------------------------------------------------------------- commands

def cmd_dump(args):
    data = open(args.file, "rb").read()
    s = parse_setup(data)
    print(f"cubes {s.get('cube_min')} .. {s.get('cube_max')}  "
          f"({len(s['cubes'])} cubes, parsed 0x{s['end']:X} of 0x{len(data):X} bytes)")
    counts = {}
    for cube in s["cubes"]:
        if not (cube["nodes"] or cube["props"]) and not args.all:
            continue
        print(f"\ncube {cube['index']}: {len(cube['nodes'])} nodes, {len(cube['props'])} props")
        for n in cube["nodes"]:
            cat = NODE_CATEGORIES.get(n["category"], f"cat{n['category']}")
            counts[cat] = counts.get(cat, 0) + 1
            extra = ""
            if n["category"] == 6:
                extra = f" actor=0x{n['actor_id']:03X} yaw={n['yaw']} scale={n['scale'] or 100}% secondaryId={n['unk10_31']} param={n['radius']}"
            elif n["category"] in (3, 4):
                extra = f" func_index=0x{n['actor_id']:X} half_size={n['radius']} marker_bit={n['marker_id']}"
            else:
                extra = f" id=0x{n['actor_id']:X} radius={n['radius']}"
            print(f"  [@0x{n['offset']:05X}] {cat:15s} pos={n['pos']}{extra}")
        if args.all or args.props:
            for p in cube["props"]:
                desc = p["kind"] + (f" asset=0x{p['asset_id']:X}" if "asset_id" in p else "")
                print(f"  [@0x{p['offset']:05X}] prop {desc} pos={p['pos']}")
    print("\nnode totals:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"camera nodes: {len(s['cameras'])}, lights: {len(s['lights'])}")


def cmd_replace_actor(args):
    data = bytearray(open(args.file, "rb").read())
    s = parse_setup(bytes(data))
    old, new = int(args.old_id, 0), int(args.new_id, 0)
    done = 0
    for cube in s["cubes"]:
        for n in cube["nodes"]:
            if n["category"] == 6 and n["actor_id"] == old:
                if args.limit is not None and done >= args.limit:
                    break
                struct.pack_into(">H", data, n["offset"] + 8, new)
                print(f"cube {cube['index']} node @0x{n['offset']:05X} pos={n['pos']}: 0x{old:X} -> 0x{new:X}")
                done += 1
    out = args.output or args.file
    open(out, "wb").write(data)
    print(f"replaced {done} placement(s); wrote {out}")


# ---------------------------------------------------------------- writer / selftest

def encode_node(pos, category, actor_id, yaw=0, scale=0, radius=0, marker_id=0, secondary=0):
    w6 = ((radius & 0x1FF) << 7) | ((category & 0x3F) << 1)
    w12 = ((yaw & 0x1FF) << 23) | (scale & 0x7FFFFF)
    w16 = (secondary & 0xFFF) << 20
    return struct.pack(">hhhHHBBII", *pos, w6, actor_id, marker_id, 0, w12, w16)


def build_setup(cube_min, cube_max, cube_nodes, cameras=(), lights=()):
    """cube_nodes: {(x,y,z): [encoded node bytes, ...]}"""
    out = bytearray([0x01, 0x01])
    out += struct.pack(">3i", *cube_min) + struct.pack(">3i", *cube_max)
    for x in range(cube_min[0], cube_max[0] + 1):
        for y in range(cube_min[1], cube_max[1] + 1):
            for z in range(cube_min[2], cube_max[2] + 1):
                nodes = cube_nodes.get((x, y, z), [])
                if nodes:
                    out += bytes([0x03, 0x0A, len(nodes), 0x0B]) + b"".join(nodes)
                    out += bytes([0x08, 0x00, 0x09])
                out.append(0x01)
    out.append(0x00)
    if cameras:
        out.append(0x03)
        for index, position in cameras:
            out += bytes([0x01]) + struct.pack(">h", index) + bytes([0x02, 0x02])
            out += bytes([0x01]) + struct.pack(">3f", *position) + bytes([0x00])
        out.append(0x00)
    if lights:
        out.append(0x04)
        for position, radii, rgb in lights:
            out += bytes([0x01, 0x02]) + struct.pack(">3f", *position)
            out += bytes([0x03]) + struct.pack(">2f", *radii)
            out += bytes([0x04]) + struct.pack(">3i", *rgb)
        out.append(0x00)
    out.append(0x00)
    return bytes(out)


def cmd_selftest(_args):
    nodes = {
        (0, 0, 0): [encode_node((100, 0, -50), 6, 0x6, yaw=90, secondary=3),
                    encode_node((10, 20, 30), 3, 0x17, radius=150)],
        (1, 0, -1): [encode_node((1500, 10, -900), 6, 0x6, scale=150)],
    }
    blob = build_setup((0, 0, -1), (1, 0, 0), nodes, cameras=[(1, (0.0, 500.0, 0.0))],
                       lights=[((0.0, 0.0, 0.0), (100.0, 200.0), (255, 128, 0))])
    s = parse_setup(blob)
    actors = [n for c in s["cubes"] for n in c["nodes"] if n["category"] == 6]
    assert len(s["cubes"]) == 4, s["cubes"]
    assert len(actors) == 2 and actors[0]["yaw"] == 90 and actors[0]["unk10_31"] == 3
    assert actors[1]["scale"] == 150 and actors[1]["pos"] == (1500, 10, -900)
    warp = [n for c in s["cubes"] for n in c["nodes"] if n["category"] == 3][0]
    assert warp["radius"] == 150 and warp["actor_id"] == 0x17
    assert s["cameras"][0]["position"] == [0.0, 500.0, 0.0]
    assert s["lights"][0]["rgb"] == [255, 128, 0]
    assert s["end"] == len(blob)
    print("selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="list the contents of a setup file")
    d.add_argument("file")
    d.add_argument("--all", action="store_true", help="include empty cubes and props")
    d.add_argument("--props", action="store_true", help="also list sprite/model props")
    d.set_defaults(func=cmd_dump)
    ra = sub.add_parser("replace-actor", help="change the actor id of placed actors")
    ra.add_argument("file")
    ra.add_argument("old_id")
    ra.add_argument("new_id")
    ra.add_argument("--limit", type=int)
    ra.add_argument("-o", "--output")
    ra.set_defaults(func=cmd_replace_actor)
    st = sub.add_parser("selftest", help="round-trip a synthetic setup file")
    st.set_defaults(func=cmd_selftest)
    args = ap.parse_args(argv)
    try:
        args.func(args)
    except SetupFormatError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()

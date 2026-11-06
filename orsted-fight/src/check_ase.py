"""Independent reader for the generated .aseprite file: rebuilds every frame from its layer cels (including
linked cels) and prints the layer / tag summary. With a .npy of the flat frames it checks they match.

    python3 check_ase.py ../orsted_fight.aseprite [frames.npy]
"""
import struct
import sys
import zlib
import numpy as np


def read(path):
    data = open(path, "rb").read()
    size, magic, nframes, w, h, depth = struct.unpack_from("<IHHHHH", data, 0)
    assert magic == 0xA5E0 and depth == 32, (hex(magic), depth)
    assert size == len(data), (size, len(data))
    off = 128
    layers, tags, frames, durs = [], [], [], []
    cels_by_frame = []
    for fi in range(nframes):
        fsize, fmagic, _old, dur, _r, nchunks = struct.unpack_from("<IHHH2sI", data, off)
        assert fmagic == 0xF1FA
        p = off + 16
        cels = {}
        for _ in range(nchunks):
            csize, ctype = struct.unpack_from("<IH", data, p)
            body = data[p + 6:p + csize]
            if ctype == 0x2004:
                n = struct.unpack_from("<H", body, 16)[0]
                layers.append(body[18:18 + n].decode())
            elif ctype == 0x2005:
                li, x, y, _op, ctyp, _z = struct.unpack_from("<HhhBHh", body, 0)
                if ctyp == 2:
                    cw, ch = struct.unpack_from("<HH", body, 16)
                    px = np.frombuffer(zlib.decompress(body[20:]), np.uint8).reshape(ch, cw, 4)
                    cels[li] = (x, y, px)
                elif ctyp == 1:
                    link = struct.unpack_from("<H", body, 16)[0]
                    cels[li] = cels_by_frame[link][li]
                else:
                    raise ValueError(ctyp)
            elif ctype == 0x2018:
                nt = struct.unpack_from("<H", body, 0)[0]
                q = 10
                for _t in range(nt):
                    a, b = struct.unpack_from("<HH", body, q)
                    n = struct.unpack_from("<H", body, q + 17)[0]
                    tags.append((body[q + 19:q + 19 + n].decode(), a, b))
                    q += 19 + n
            p += csize
        cels_by_frame.append(cels)
        img = np.zeros((h, w, 4), np.uint8)
        for li in range(len(layers)):
            if li in cels:
                x, y, px = cels[li]
                reg = img[y:y + px.shape[0], x:x + px.shape[1]]
                m = px[:, :, 3] > 0
                reg[m] = px[m]
        frames.append(img)
        durs.append(dur)
        off += fsize
    return dict(w=w, h=h, layers=layers, tags=tags, frames=frames, durs=durs)


if __name__ == "__main__":
    a = read(sys.argv[1])
    print(f"{a['w']}x{a['h']}, {len(a['frames'])} frames, {sum(a['durs']) / 1000:.2f}s")
    print("layers:", ", ".join(a["layers"]))
    print("tags:", "; ".join(f"{n} {x}-{y}" for n, x, y in a["tags"]))
    if len(sys.argv) > 2:
        ref = np.load(sys.argv[2])
        assert len(ref) == len(a["frames"]), (len(ref), len(a["frames"]))
        bad = [i for i, (r, f) in enumerate(zip(ref, a["frames"])) if not np.array_equal(r, f)]
        print("frames matching the render:", len(ref) - len(bad), "/", len(ref), "" if not bad else bad[:10])

"""Minimal native .aseprite (v1.3 format) writer.

Implements the subset of the spec at
https://github.com/aseprite/aseprite/blob/main/docs/ase-file-specs.md
needed for an RGBA sprite with layers, per-frame durations, tags and a palette.
"""
import struct
import zlib


def _string(s):
    b = s.encode("utf-8")
    return struct.pack("<H", len(b)) + b


def _chunk(ctype, data):
    return struct.pack("<IH", len(data) + 6, ctype) + data


def _color_profile_chunk():
    # type 1 = sRGB, no flags, gamma unused, 8 reserved bytes
    return _chunk(0x2007, struct.pack("<HHi", 1, 0, 0) + bytes(8))


def _old_palette_chunk(palette):
    # 0x0004: kept for readers that predate the new palette chunk
    n = len(palette)
    data = struct.pack("<HBB", 1, 0, 0 if n == 256 else n)
    for r, g, b, _a in palette:
        data += struct.pack("<BBB", r, g, b)
    return _chunk(0x0004, data)


def _palette_chunk(palette, names=None):
    n = len(palette)
    data = struct.pack("<III", n, 0, n - 1) + bytes(8)
    for i, (r, g, b, a) in enumerate(palette):
        name = names[i] if names else None
        if name:
            data += struct.pack("<HBBBB", 1, r, g, b, a) + _string(name)
        else:
            data += struct.pack("<HBBBB", 0, r, g, b, a)
    return _chunk(0x2019, data)


def _layer_chunk(name, flags=3, opacity=255):
    data = struct.pack("<HHHHHHB", flags, 0, 0, 0, 0, 0, opacity) + bytes(3)
    data += _string(name)
    return _chunk(0x2004, data)


def _tags_chunk(tags):
    data = struct.pack("<H", len(tags)) + bytes(8)
    for t in tags:
        r, g, b = t.get("color", (0, 0, 0))
        data += struct.pack("<HHBH", t["from"], t["to"], t.get("direction", 0), t.get("repeat", 0))
        data += bytes(6) + struct.pack("<BBBB", r, g, b, 0) + _string(t["name"])
    return _chunk(0x2018, data)


def _user_data_color_chunk(rgb):
    # User data with only a color (flag 2); assigned to the preceding tag
    r, g, b = rgb
    return _chunk(0x2020, struct.pack("<IBBBB", 2, r, g, b, 255))


def _cel_chunk(layer_index, x, y, w, h, rgba_bytes, opacity=255):
    data = struct.pack("<HhhBHh", layer_index, x, y, opacity, 2, 0) + bytes(5)
    data += struct.pack("<HH", w, h) + zlib.compress(rgba_bytes, 9)
    return _chunk(0x2005, data)


def write_aseprite(path, width, height, layer_names, frames, tags=(), palette=None, palette_names=None):
    """
    layer_names: bottom-to-top list of layer names.
    frames: list of dicts {"duration": ms, "layers": [numpy uint8 array (H, W, 4) or None per layer]}
    tags: list of dicts {"name", "from", "to", "direction", "repeat", "color"}
    palette: list of (r, g, b, a)
    """
    palette = list(palette or [(0, 0, 0, 255)])
    body = b""
    for fi, frame in enumerate(frames):
        chunks = []
        if fi == 0:
            chunks.append(_color_profile_chunk())
            chunks.append(_old_palette_chunk(palette[:256]))
            chunks.append(_palette_chunk(palette, palette_names))
            for name in layer_names:
                chunks.append(_layer_chunk(name))
            if tags:
                chunks.append(_tags_chunk(tags))
                for t in tags:
                    chunks.append(_user_data_color_chunk(t.get("color", (0, 0, 0))))
        for li, img in enumerate(frame["layers"]):
            if img is None:
                continue
            alpha = img[:, :, 3]
            ys, xs = alpha.nonzero()
            if len(xs) == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            crop = img[y0:y1 + 1, x0:x1 + 1].copy()
            crop[crop[:, :, 3] == 0] = 0
            chunks.append(_cel_chunk(li, int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1), crop.tobytes()))
        n = len(chunks)
        payload = b"".join(chunks)
        header = struct.pack("<IHHHBBI", 16 + len(payload), 0xF1FA, min(n, 0xFFFF), frame["duration"], 0, 0, n)
        body += header + payload

    header = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH84s",
        128 + len(body),  # file size
        0xA5E0,
        len(frames),
        width,
        height,
        32,               # RGBA
        1,                # layer opacity field is valid
        100,              # deprecated speed
        0, 0,
        0, bytes(3),
        len(palette) if len(palette) < 256 else 0,
        1, 1,             # pixel ratio
        0, 0, 16, 16,     # grid
        bytes(84),
    )
    assert len(header) == 128
    with open(path, "wb") as f:
        f.write(header + body)

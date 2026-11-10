"""Minimal Banjo-Kazooie model (.model.bin) reader, per include/core2/model.h"""
import struct

class Model:
    def __init__(self, data):
        self.d = bytearray(data)
        d = self.d
        (self.magic, self.geo_off, self.tex_off, self.geo_type, self.gfx_off, self.vtx_off,
         self.unk14_off, self.anim_off, self.coll_off, self.camarea_off, self.mesh_off,
         self.animvtx_off, self.animtex_off, self.tri_count, self.vtx_count) = struct.unpack_from(">IihhiiiiiiiiiHH", d, 0)
        assert self.magic == 0xB, hex(self.magic)
        # textures
        self.textures = []
        if self.tex_off:
            size, count = struct.unpack_from(">ih", d, self.tex_off)
            self.tex_list_size = size
            base = self.tex_off + 8 + 16 * count
            self.tex_data_base = base
            for i in range(count):
                off, typ, _, w, h = struct.unpack_from(">ihHBB", d, self.tex_off + 8 + 16 * i)
                self.textures.append(dict(offset=base + off, rel=off, type=typ, w=w, h=h))
            self.tex_end = self.tex_off + size
        # vertices
        self.vtx_base = None
        if self.vtx_off:
            self.vtx_count2 = struct.unpack_from(">h", d, self.vtx_off + 0x14)[0]
            self.vtx_base = self.vtx_off + 0x18
    def vtx(self, i):
        return struct.unpack_from(">hhhHhhBBBB", self.d, self.vtx_base + 16 * i)
    def set_vtx_color(self, i, r, g, b, a=None):
        o = self.vtx_base + 16 * i + 12
        self.d[o], self.d[o+1], self.d[o+2] = r, g, b
        if a is not None: self.d[o+3] = a
    def has_bones(self):
        return self.anim_off != 0

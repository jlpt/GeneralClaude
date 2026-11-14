"""
Minimal headless libretro frontend (ctypes) used to test the ROMs.

    emu = Retro("/usr/lib/x86_64-linux-gnu/libretro/nestopia_libretro.so", "rom.nes")
    emu.run(60); emu.press("start", 3); emu.run(600)
    emu.save_png("shot.png"); wram = emu.memory(Retro.SAVE_RAM)
"""
import ctypes as C
import os
import tempfile

ENV_GET_CAN_DUPE = 3
ENV_SET_PIXEL_FORMAT = 10
ENV_GET_SYSTEM_DIRECTORY = 9
ENV_GET_VARIABLE = 15
ENV_GET_VARIABLE_UPDATE = 17
ENV_GET_SAVE_DIRECTORY = 31

BUTTONS = {"b": 0, "y": 1, "select": 2, "start": 3, "up": 4, "down": 5, "left": 6,
           "right": 7, "a": 8, "x": 9, "l": 10, "r": 11}


class GameInfo(C.Structure):
    _fields_ = [("path", C.c_char_p), ("data", C.c_void_p), ("size", C.c_size_t), ("meta", C.c_char_p)]


class Variable(C.Structure):
    _fields_ = [("key", C.c_char_p), ("value", C.c_char_p)]


ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AUDIO_BATCH_CB = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
POLL_CB = C.CFUNCTYPE(None)
STATE_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


class Retro:
    SAVE_RAM = 0
    SYSTEM_RAM = 2

    def __init__(self, core, rom, variables=None):
        self.lib = C.CDLL(core)
        self.held = set()
        self.frame = None
        self.fmt = 0
        self.vars = {k.encode(): v.encode() for k, v in (variables or {}).items()}
        self._dir = tempfile.mkdtemp().encode()
        self._keep = []
        self._cbs = [ENV_CB(self._env), VIDEO_CB(self._video), AUDIO_CB(lambda l, r: None),
                     AUDIO_BATCH_CB(lambda d, n: n), POLL_CB(lambda: None), STATE_CB(self._state)]
        lib = self.lib
        lib.retro_set_environment(self._cbs[0])
        lib.retro_set_video_refresh(self._cbs[1])
        lib.retro_set_audio_sample(self._cbs[2])
        lib.retro_set_audio_sample_batch(self._cbs[3])
        lib.retro_set_input_poll(self._cbs[4])
        lib.retro_set_input_state(self._cbs[5])
        lib.retro_init()
        data = open(rom, "rb").read()
        self._rom = C.create_string_buffer(data, len(data))
        gi = GameInfo(os.path.abspath(rom).encode(), C.cast(self._rom, C.c_void_p), len(data), None)
        if not lib.retro_load_game(C.byref(gi)):
            raise RuntimeError("core refused the ROM")
        for port in (0, 1):
            lib.retro_set_controller_port_device(port, 1)       # RETRO_DEVICE_JOYPAD
        lib.retro_get_memory_data.restype = C.c_void_p
        lib.retro_get_memory_size.restype = C.c_size_t

    # ---- callbacks
    def _env(self, cmd, data):
        cmd &= 0xFFFF
        if cmd == ENV_SET_PIXEL_FORMAT:
            self.fmt = C.cast(data, C.POINTER(C.c_int))[0]
            return True
        if cmd in (ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY):
            C.cast(data, C.POINTER(C.c_char_p))[0] = self._dir
            return True
        if cmd == ENV_GET_CAN_DUPE:
            C.cast(data, C.POINTER(C.c_bool))[0] = True
            return True
        if cmd == ENV_GET_VARIABLE:
            v = C.cast(data, C.POINTER(Variable))[0]
            if v.key in self.vars:
                buf = C.c_char_p(self.vars[v.key])
                self._keep.append(buf)
                C.cast(data, C.POINTER(Variable))[0].value = buf.value
                return True
            return False
        if cmd == ENV_GET_VARIABLE_UPDATE:
            C.cast(data, C.POINTER(C.c_bool))[0] = False
            return True
        return False

    def _video(self, data, w, h, pitch):
        if data:
            self.frame = (C.string_at(data, pitch * h), w, h, pitch)

    def _state(self, port, device, index, id_):
        return 1 if port == 0 and device == 1 and id_ in self.held else 0

    # ---- API
    def run(self, frames=1):
        for _ in range(frames):
            self.lib.retro_run()

    def press(self, button, frames=2):
        self.held.add(BUTTONS[button])
        self.run(frames)
        self.held.discard(BUTTONS[button])
        self.run(1)

    def memory(self, kind):
        size = self.lib.retro_get_memory_size(kind)
        ptr = self.lib.retro_get_memory_data(kind)
        if not ptr or not size:
            return b""
        return C.string_at(ptr, size)

    def image(self):
        from PIL import Image
        raw, w, h, pitch = self.frame
        if self.fmt == 1:
            img = Image.frombuffer("RGBX", (w, h), raw, "raw", "BGRX", pitch, 1)
        elif self.fmt == 2:
            img = Image.frombuffer("RGB", (w, h), raw, "raw", "BGR;16", pitch, 1)
        else:
            img = Image.frombuffer("RGB", (w, h), raw, "raw", "BGR;15", pitch, 1)
        return img.convert("RGB")

    def save_png(self, path, scale=2):
        img = self.image()
        img = img.resize((img.width * scale, img.height * scale), 0)
        img.save(path)

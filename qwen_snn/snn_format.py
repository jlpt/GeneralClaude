"""The .snn network file: neurons, dendrite slots and packed synapses.

Layout (little endian):

    8 bytes   magic  b"QWENSNN1"
    8 bytes   uint64 length of the JSON header
    HEADER_SPACE bytes  JSON header (zero padded)
    arrays, each 64-byte aligned, at the offsets the header lists

The synapses use the same packing as GPU-NETWORK-GOOD's kernel.cu:

    uint32 = (dest << 8) | uint8(weight)      dest: 24 bits, weight: int8 / 127

stored CSR-style by source neuron: neuron i's outgoing synapses are
syn.packed[syn.offsets[i] : syn.offsets[i+1]].  `dest` indexes the dendrite
slot buffer (the equivalent of d_activations, the thing atomicAdd writes into).
Each slot has a float gain so a synapse's real weight is

    (int8 / 127) * slot.gain[dest]

which is the int8 per-output-channel scale Qwen's weights need; the /127 is
kept exactly as in kernel.cu.
"""
import json
import struct

import numpy as np

MAGIC = b"QWENSNN1"
HEADER_SPACE = 1 << 20
ALIGN = 64
MAX_DEST = (1 << 24) - 1


def pack(dest, weight_i8):
    """dest (int array, < 2^24) and int8 weights -> uint32 packed synapses."""
    dest = np.asarray(dest, dtype=np.uint32)
    w = np.asarray(weight_i8, dtype=np.int8).view(np.uint8).astype(np.uint32)
    return (dest << 8) | w


def unpack(packed):
    """uint32 packed synapses -> (dest int64, weight int8)."""
    packed = np.asarray(packed, dtype=np.uint32)
    dest = (packed >> 8).astype(np.int64)
    w = (packed & 0xFF).astype(np.uint8).view(np.int8)
    return dest, w


class Writer:
    """Collect array specs, then stream them to disk after the header.

    add(name, dtype, shape, producer): producer() yields numpy chunks whose
    concatenation (row-major) is the array.
    """

    def __init__(self):
        self.specs = []

    def add(self, name, dtype, shape, producer):
        self.specs.append((name, np.dtype(dtype), tuple(int(s) for s in shape), producer))

    def write(self, path, header):
        arrays = {}
        off = 16 + HEADER_SPACE
        for name, dt, shape, _ in self.specs:
            off = (off + ALIGN - 1) // ALIGN * ALIGN
            nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
            arrays[name] = {"offset": off, "dtype": dt.str, "shape": list(shape)}
            off += nbytes
        header = dict(header)
        header["arrays"] = arrays
        blob = json.dumps(header).encode()
        if len(blob) > HEADER_SPACE:
            raise ValueError(f"header too large ({len(blob)} bytes)")
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for name, dt, shape, producer in self.specs:
                start = arrays[name]["offset"]
                f.write(b"\0" * (start - f.tell()))
                want = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
                got = 0
                for chunk in producer():
                    chunk = np.ascontiguousarray(chunk, dtype=dt)
                    f.write(chunk.tobytes())
                    got += chunk.nbytes
                if got != want:
                    raise ValueError(f"{name}: wrote {got} bytes, expected {want}")
        return header


def read_header(path):
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path} is not a .snn file")
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def update_header(path, header):
    """Rewrite the JSON header in place (array offsets must not change)."""
    blob = json.dumps(header).encode()
    if len(blob) > HEADER_SPACE:
        raise ValueError("header too large")
    with open(path, "r+b") as f:
        f.seek(8)
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob + b"\0" * (HEADER_SPACE - len(blob)))


def open_array(path, header, name, mode="r"):
    a = header["arrays"][name]
    return np.memmap(path, dtype=np.dtype(a["dtype"]), mode=mode,
                     offset=a["offset"], shape=tuple(a["shape"]))

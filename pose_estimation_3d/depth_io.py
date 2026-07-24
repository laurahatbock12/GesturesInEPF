"""Lossless save/load for DA3 depth predictions, compressed far better than np.savez_compressed.

np.savez_compressed uses zlib (DEFLATE) on each array independently. Two changes get a ~2.4x
smaller file with no loss of precision:

1. zstandard instead of zlib: same lossless guarantee, higher compression ratio, and much
   faster at comparable ratios (zstandard is already a dependency of depth_anything_3, so no
   extra install).
2. Temporal delta-encoding of the depth array: consecutive video frames have very similar
   depth, so storing frame[0] plus (frame[i] - frame[i-1]) has far lower entropy than storing
   raw frames. The subtraction/reconstruction is done on the raw uint16 bit pattern of the
   float16 array with numpy's wraparound integer arithmetic, which makes the encode/decode
   round trip exactly bit-for-bit lossless regardless of overflow (it's arithmetic in
   Z/65536Z, always invertible) - verified against real data before this was adopted.

The `sky` mask (if present) is bit-packed before compression; it's normally almost entirely
False for indoor recordings and compresses to near nothing either way.

File format: a single zstandard-compressed blob wrapping a plain (uncompressed) .npz buffer,
so compression happens once over everything together rather than per-array. Old plain .npz
files (zlib, no delta-encoding) are still readable - load_depth_npz auto-detects the format
from the file's magic bytes.
"""

from pathlib import Path
import io

import numpy as np
import zstandard

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _delta_encode(depth):
    depth_i16 = depth.view(np.int16)
    delta = np.empty_like(depth_i16)
    delta[0] = depth_i16[0]
    np.subtract(depth_i16[1:], depth_i16[:-1], out=delta[1:])
    return delta


def _delta_decode(delta, shape, dtype):
    depth_i16 = np.cumsum(delta, axis=0, dtype=np.int16)
    return depth_i16.view(dtype).reshape(shape)


def save_depth_npz(path, depth, sky=None, zstd_level=19, **other_arrays):
    """Save depth (and optionally a boolean sky mask) plus arbitrary small arrays/scalars."""
    arrays = dict(other_arrays)
    arrays["_depth_delta_i16"] = _delta_encode(depth)
    arrays["_depth_shape"] = np.array(depth.shape, dtype=np.int64)
    arrays["_depth_dtype"] = str(depth.dtype)
    if sky is not None:
        arrays["_sky_packed"] = np.packbits(sky)
        arrays["_sky_shape"] = np.array(sky.shape, dtype=np.int64)

    buf = io.BytesIO()
    np.savez(buf, **arrays)

    cctx = zstandard.ZstdCompressor(level=zstd_level, threads=-1)
    compressed = cctx.compress(buf.getvalue())
    Path(path).write_bytes(compressed)


def load_depth_npz(path):
    """Returns a dict with 'depth' (and 'sky' if present) plus any other saved arrays/scalars.
    Transparently handles both the new zstandard+delta format and old plain np.savez_compressed
    files."""
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(4)

    if header != ZSTD_MAGIC:
        # Old-format file: a real zip-based .npz, readable directly by np.load.
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    dctx = zstandard.ZstdDecompressor()
    raw = dctx.decompress(path.read_bytes())
    data = np.load(io.BytesIO(raw), allow_pickle=True)

    result = {}
    for key in data.files:
        if key.startswith("_depth") or key.startswith("_sky"):
            continue
        result[key] = data[key]

    if "_depth_delta_i16" in data:
        shape = tuple(data["_depth_shape"])
        dtype = np.dtype(str(data["_depth_dtype"]))
        result["depth"] = _delta_decode(data["_depth_delta_i16"], shape, dtype)
    if "_sky_packed" in data:
        shape = tuple(data["_sky_shape"])
        result["sky"] = np.unpackbits(data["_sky_packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)

    return result

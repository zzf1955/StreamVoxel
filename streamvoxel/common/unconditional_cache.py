"""Compact tile storage geometry shared by data builder and train loader."""
import numpy as np


def pack_tiles(array):
    y,z,x=array.shape
    if y!=256 or z%16 or x%16:
        raise ValueError("Raw training view must tile into16-square full-height targets")
    return array.reshape(y,z//16,16,x//16,16).transpose(1,3,0,2,4).reshape(-1,256,16,16)


def tile_index(z,x,*,raw_z,raw_x,world_x):
    if z%16 or x%16:
        raise ValueError("Target must align with tile16")
    per_group=(raw_z//16)*(raw_x//16)
    group=(z//raw_z)*(world_x//raw_x)+x//raw_x
    local=((z%raw_z)//16)*(raw_x//16)+(x%raw_x)//16
    return group*per_group+local


ZSTD_FORMAT="java26_compact_tile16_zstd_v1"
TILE_SHAPE=(256,16,16)
TILE_BYTES=256*16*16*2


def compress_tiles(tiles,*,level=3):
    """[N,256,16,16] uint16 -> (concatenated independent zstd frames, uint64 offsets[N+1])."""
    import zstandard
    if tiles.dtype!=np.uint16 or tiles.shape[1:]!=TILE_SHAPE:
        raise ValueError("Expected uint16 [N,256,16,16] compact tiles")
    compressor=zstandard.ZstdCompressor(level=level,write_content_size=True)
    frames=[compressor.compress(np.ascontiguousarray(tile).tobytes()) for tile in tiles]
    offsets=np.zeros(len(frames)+1,dtype=np.uint64)
    np.cumsum([len(f) for f in frames],out=offsets[1:])
    return b"".join(frames),offsets


def decode_tile(decompressor,frame):
    """One zstd frame -> read-only uint16 [256,16,16]."""
    raw=decompressor.decompress(frame)
    if len(raw)!=TILE_BYTES:
        raise ValueError("Compressed tile has the wrong size")
    return np.frombuffer(raw,dtype="<u2").reshape(TILE_SHAPE)

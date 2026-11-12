"""Wire format and tensor layout of ControlNet generation requests.

numpy only, so the client can import it without torch. Coordinates are voxel XZ in any common frame: the server
re-expresses every item relative to its own target tile, which is exact because all attention is rotary (relative).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import json

import numpy as np

PROTOCOL = 1


class ProtocolError(ValueError):
    """The request is malformed; reported to the caller as HTTP 400."""


@dataclass(frozen=True)
class ModelSpec:
    height: int            # voxel_shape[0]
    tile: int              # horizontal tile side (16)
    num_classes: int       # compact classes; the MASK token id equals num_classes
    biome_classes: int
    biome_halo: int        # 4 -> request biomes [H/4,6,6]; 16 -> [H/4,12,12]
    y_origin: float        # backbone.y_range[0]
    trained_l1: tuple = (1, 3)

    @property
    def biome_shape(self):
        side = 6 if self.biome_halo == 4 else 12
        return (self.height // 4, side, side)


class Mapping:
    """Global (981) <-> compact (304) class ids; the identity when no mapping file is given."""

    def __init__(self, global_to_compact, compact_to_global):
        self.to_compact = np.asarray(global_to_compact, np.int32)
        self.to_global = np.asarray(compact_to_global, np.int64)

    @classmethod
    def from_json(cls, path):
        value = json.loads(open(path).read())
        return cls(value["global_to_compact"], value["compact_to_global"])

    @classmethod
    def identity(cls, n):
        return cls(np.arange(n), np.arange(n))

    def compact(self, values, space, classes):
        values = np.asarray(values)
        if values.size and (values.min() < 0 or values.max() >= (len(self.to_compact) if space == "global" else classes)):
            raise ProtocolError(f"block id outside the {space} vocabulary")
        if space == "compact":
            return values.astype(np.int16)
        result = self.to_compact[values]
        if result.size and result.min() < 0:
            raise ProtocolError("global block id has no compact class (not in the training vocabulary)")
        return result.astype(np.int16)

    def export(self, compact, space):
        return compact.astype(np.int16) if space == "compact" else self.to_global[compact].astype(np.uint16)


@dataclass
class Item:
    """One validated generation request; arrays are compact-id numpy, XZ relative to the target tile."""
    id: str
    seed: int
    steps: int
    temperature: float
    l1_blocks: np.ndarray    # [n1,H,T,T] int16
    l1_xz: np.ndarray        # [n1,2] int64 voxels
    l2_blocks: np.ndarray    # [n2,H,T,T]
    l2_xz: np.ndarray        # [n2,2]
    biomes: np.ndarray       # [H/4,12,12] int64, zero padded outside the supplied window
    known: np.ndarray | None  # [H,T,T] int16, -1 = to generate
    warnings: list = field(default_factory=list)

    @property
    def key(self):
        return (len(self.l1_blocks), len(self.l2_blocks), self.steps, self.temperature)


def _array(archive, name):
    if name not in archive.files:
        raise ProtocolError(f"missing array {name}")
    return archive[name]


def _int_xz(value, what):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != 2 or not np.issubdtype(value.dtype, np.integer):
        raise ProtocolError(f"{what} must be an integer array [n,2] of voxel (x,z)")
    return value.astype(np.int64)


def parse_request(body, spec, mapping, *, max_items=4096, max_steps=8192, max_tiles=4096):
    """Decode and validate an npz request. Returns (items, id_space, return_space)."""
    try:
        archive = np.load(io.BytesIO(body), allow_pickle=False)
        meta = json.loads(str(archive["meta"]))
    except Exception as error:
        raise ProtocolError(f"unreadable request: {error}") from error
    if meta.get("version") != PROTOCOL:
        raise ProtocolError(f"unsupported protocol version {meta.get('version')}")
    space = meta.get("id_space", "global")
    ret = meta.get("return_space", space)
    if space not in ("global", "compact") or ret not in ("global", "compact"):
        raise ProtocolError("id_space/return_space must be 'global' or 'compact'")
    rows = meta.get("items")
    if not isinstance(rows, list) or not 0 < len(rows) <= max_items:
        raise ProtocolError(f"items must be a list of 1..{max_items} entries")
    default = dict(steps=1024, temperature=1.0, seed=0)
    default.update(meta.get("defaults", {}))
    H, T = spec.height, spec.tile
    items, seen = [], set()
    for index, row in enumerate(rows):
        prefix = f"i{index}."
        name = str(row.get("id", f"item{index}"))
        if name in seen:
            raise ProtocolError(f"duplicate item id {name}")
        seen.add(name)
        steps = int(row.get("steps", default["steps"]))
        temperature = float(row.get("temperature", default["temperature"]))
        seed = int(row.get("seed", int(default["seed"]) + index))
        if not 1 <= steps <= max_steps:
            raise ProtocolError(f"{name}: steps must be in 1..{max_steps}")
        if not np.isfinite(temperature) or temperature < 0:
            raise ProtocolError(f"{name}: temperature must be >= 0 (0 = greedy)")
        target = np.asarray(row.get("target_xz"))
        if target.shape != (2,) or not np.issubdtype(target.dtype, np.integer):
            raise ProtocolError(f"{name}: target_xz must be two integer voxel coordinates")
        levels = {}
        for level in ("l1", "l2"):
            blocks = _array(archive, prefix + level + "_blocks")
            xz = _int_xz(_array(archive, prefix + level + "_xz"), f"{name}.{level}_xz")
            if blocks.shape != (len(xz), H, T, T):
                raise ProtocolError(f"{name}.{level}_blocks must be [{len(xz)},{H},{T},{T}], got {list(blocks.shape)}")
            if len(xz) > max_tiles:
                raise ProtocolError(f"{name}: more than {max_tiles} {level} tiles")
            relative = xz - target
            if (relative == 0).all(1).any():
                raise ProtocolError(f"{name}: a {level} tile coincides with the target tile")
            levels[level] = (mapping.compact(blocks, space, spec.num_classes), relative)
        biomes = _array(archive, prefix + "biomes")
        if biomes.shape != spec.biome_shape:
            raise ProtocolError(f"{name}.biomes must be {list(spec.biome_shape)}, got {list(biomes.shape)}")
        if biomes.size and (biomes.min() < 0 or biomes.max() >= spec.biome_classes):
            raise ProtocolError(f"{name}.biomes outside 0..{spec.biome_classes - 1}")
        padded = np.zeros((H // 4, 12, 12), np.int64)
        lo = (12 - biomes.shape[1]) // 2
        padded[:, lo:lo + biomes.shape[1], lo:lo + biomes.shape[2]] = biomes
        known = None
        if prefix + "known" in archive.files:
            raw = archive[prefix + "known"]
            if raw.shape != (H, T, T):
                raise ProtocolError(f"{name}.known must be [{H},{T},{T}]")
            unknown = raw < 0
            known = np.full(raw.shape, -1, np.int16)
            if (~unknown).any():
                known[~unknown] = mapping.compact(raw[~unknown], space, spec.num_classes)
            if not unknown.any():
                raise ProtocolError(f"{name}: known has no unknown voxel to generate")
        warnings = []
        n1 = len(levels["l1"][1])
        if not spec.trained_l1[0] <= n1 <= spec.trained_l1[1]:
            warnings.append(f"{n1} L1 tiles; the model was trained with {spec.trained_l1[0]}..{spec.trained_l1[1]}")
        items.append(Item(name, seed, steps, temperature, *levels["l1"], *levels["l2"], padded, known, warnings))
    return items, space, ret


def encode_request(items, *, id_space="global", return_space=None, defaults=None):
    """Build the npz body. Each item is a dict, see client.make_item."""
    meta_items, arrays = [], {}
    for index, item in enumerate(items):
        row = {k: item[k] for k in ("id", "seed", "steps", "temperature") if item.get(k) is not None}
        row["target_xz"] = [int(item["target_xz"][0]), int(item["target_xz"][1])]
        meta_items.append(row)
        prefix = f"i{index}."
        for level in ("l1", "l2"):
            tiles = item.get(level) or []
            shape = (item["height"], item.get("tile", 16), item.get("tile", 16))
            arrays[prefix + level + "_xz"] = np.asarray([xz for xz, _ in tiles], np.int64).reshape(-1, 2)
            arrays[prefix + level + "_blocks"] = (np.stack([np.asarray(b) for _, b in tiles]) if tiles
                                                  else np.zeros((0, *shape), np.uint16))
        arrays[prefix + "biomes"] = np.asarray(item["biomes"])
        if item.get("known") is not None:
            arrays[prefix + "known"] = np.asarray(item["known"])
    meta = dict(version=PROTOCOL, id_space=id_space, return_space=return_space or id_space,
                defaults=defaults or {}, items=meta_items)
    arrays["meta"] = np.asarray(json.dumps(meta))
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _centers(shape, size):
    """Patch centers in (x, y, z) voxels, grid order (y, z, x) like loaders.streaming_controlnet.centers."""
    grid = np.stack(np.meshgrid(*(np.arange(s) for s in shape), indexing="ij"), -1)
    return (grid[..., [2, 0, 1]] * size + size / 2).reshape(-1, 3).astype(np.float32)


def build_group(item, spec):
    """The group dict StreamingControlNet expects, in the item's own frame (target tile at x=z=0)."""
    import torch
    H, T = spec.height, spec.tile
    offsets = lambda xz: np.stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]), -1).astype(np.float32)
    pos1 = (_centers((H // 4, T // 4, T // 4), 4)[None] + offsets(item.l1_xz)[:, None]) / 4
    pos2 = ((_centers((H // 16, 1, 1), 16)[None] + offsets(item.l2_xz)[:, None]) / 4).reshape(-1, 3)
    bio_positions = (_centers((H // 4, 12, 12), 4) + np.asarray([-T, 0, -T], np.float32)) / 4
    origin = np.asarray([0, spec.y_origin, 0], np.float32)
    group = dict(
        l1_blocks=item.l1_blocks.astype(np.int64), l2_blocks=item.l2_blocks.astype(np.int64),
        l1_positions=pos1, prefix_positions=pos2, l2_count=len(pos2),
        l1_indices=np.arange(len(item.l1_blocks), dtype=np.int64)[None], n=np.asarray([len(item.l1_blocks)], np.int64),
        biomes=item.biomes.reshape(1, -1), biome_positions=bio_positions[None],
        struct_labels=np.zeros((0,), np.int64), struct_sizes=np.zeros((0, 3), np.float32),
        target_origins=origin[None], group_origin=origin)
    return {k: torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v for k, v in group.items()}

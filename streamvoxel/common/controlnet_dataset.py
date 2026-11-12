"""Streaming ControlNet's immutable spatial and dataset contracts (no torch)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from streamvoxel.common.io import digest
from streamvoxel.common.unconditional_tiles import contained

FORMAT = "java26_streaming_controlnet_v1"


def ring_targets(side):
    if side < 2:
        raise ValueError("Known region needs at least two tile16 columns")
    z, x = np.indices((side + 2, side + 2))
    keep = (z == 0) | (x == 0) | (z == side + 1) | (x == side + 1)
    return np.stack((z[keep] - 1, x[keep] - 1), -1).astype(np.int32)


def known_neighbors(target, side):
    z, x = map(int, target)
    return np.asarray([(z + dz, x + dx) for dz in (-1, 0, 1) for dx in (-1, 0, 1)
                       if (dz or dx) and 0 <= z + dz < side and 0 <= x + dx < side],
                      dtype=np.int32).reshape(-1, 2)


def context_plan(world, origin, *, side=32, targets=128, seed=0, balanced=False):
    candidates = ring_targets(side)
    n = np.asarray([len(known_neighbors(t, side)) for t in candidates], dtype=np.int32)
    rng = np.random.default_rng(seed)
    if balanced:
        if targets % 3:
            raise ValueError("Balanced panel must divide three N buckets")
        chosen = np.concatenate([rng.choice(np.flatnonzero(n == k), targets // 3,
                                           replace=False) for k in (1, 2, 3)])
    else:
        chosen = rng.choice(len(candidates), targets, replace=False)
    return dict(world=int(world), origin_zx=list(map(int, origin)), side=int(side),
                targets=candidates[chosen].tolist(), n=n[chosen].tolist())


def validation_plans(worlds, *, contexts=64, world_side=128, side=32, seed=20261011):
    plans = []
    for world in worlds:
        rng = np.random.default_rng([seed, int(world)])
        grid = np.arange(2, world_side - side - 1)
        coords = np.stack(np.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
        for i in rng.choice(len(coords), contexts, replace=False):
            plans.append(context_plan(world, coords[i], side=side, targets=12,
                                      seed=int(rng.integers(2**31)), balanced=True))
    return plans


def full_mask_plans(plans, *, contexts_per_world=4, per_n=2):
    result, counts = [], {}
    for plan in plans:
        w = plan["world"]
        if counts.get(w, 0) >= contexts_per_world:
            continue
        chosen = [i for k in (1, 2, 3) for i in np.flatnonzero(np.asarray(plan["n"]) == k)[:per_n]]
        if len(chosen) != 3 * per_n:
            raise ValueError("Full-mask panel exceeds ordinary N support")
        result.append(dict(plan, targets=[plan["targets"][i] for i in chosen],
                           n=[plan["n"][i] for i in chosen]))
        counts[w] = counts.get(w, 0) + 1
    return result


def training_validation_plans(manifest, packaged, *, side, seed=20261011):
    """Align validation geometry to training without modifying the dataset package."""
    if side < 3:
        raise ValueError('Balanced validation requires N1/N2/N3 support')
    if all(p['side'] == side for kind in ('ordinary', 'full_mask') for p in packaged[kind]):
        return packaged
    ordinary = validation_plans(manifest['splits']['validation'], side=side, seed=seed)
    return dict(ordinary=ordinary, full_mask=full_mask_plans(ordinary))


def validate_dataset(directory, *, verify_arrays=False, require_complete=True):
    root = Path(directory).resolve()
    m = json.loads((root / "manifest.json").read_text())
    if m.get("format") != FORMAT or (require_complete and (root / "COMPLETE").read_text().strip() != digest(root / "manifest.json")):
        raise ValueError("Incomplete ControlNet package")
    if m["num_classes"] != 304 or m["target_shape_yzx"] != [256, 16, 16]:
        raise ValueError("Unexpected target contract")
    assigned = []
    for split in ("train", "validation", "test"):
        assigned.extend(m["splits"][split])
    if sorted(assigned) != sorted(map(int, m["worlds"])) or len(set(assigned)) != len(assigned):
        raise ValueError("World split overlap or missing World")
    for row in m["files"].values():
        p = contained(root, row["file"])
        if p.stat().st_size != row["bytes"]:
            raise ValueError(f"File size mismatch: {p}")
        if (verify_arrays or row.get("kind") != "blocks") and digest(p) != row["sha256"]:
            raise ValueError(f"File hash mismatch: {p}")
    mapping = json.loads(contained(root, m["mapping_file"]).read_text())
    inverse, forward = mapping["compact_to_global"], mapping["global_to_compact"]
    if len(inverse) != 304 or any(forward[g] != i for i, g in enumerate(inverse)):
        raise ValueError("Invalid reversible compact mapping")
    for wi, w in m["worlds"].items():
        a = np.load(contained(root, w["biomes"]), mmap_mode="r", allow_pickle=False)
        if w.get('storage')=='lossless_tile32_zstd':
            index=json.loads(contained(root,w['tile_index']).read_text())
            packed=contained(root,w['blocks'])
            if (index['codec']!='zstd' or index['tileShapeYZX']!=[384,32,32]
                    or index['shapeYZX']!=[384,2048,2048] or len(index['entries'])!=4096
                    or packed.stat().st_size!=index['bytes'] or a.dtype!=np.uint16 or a.shape!=(64,512,512)):
                raise ValueError(f'Invalid lossless cached World: {wi}')
            continue
        b = np.load(contained(root, w["blocks"]), mmap_mode="r", allow_pickle=False)
        if b.dtype != np.uint16 or list(b.shape) != w["block_shape"] or a.dtype != np.uint16 or a.shape != (64, 512, 512):
            raise ValueError(f"Invalid World array headers: {wi}")
    return m


def publish(directory, cache, source=None):
    """Place existing immutable cache files; validate before publishing COMPLETE."""
    import os
    from streamvoxel.common.io import atomic_json
    root, cache = Path(directory), Path(cache)
    m = json.loads((root / "manifest.json").read_text())
    if digest(cache / "manifest.json") != m["source"]["cache_manifest_sha256"]:
        raise ValueError("Deployment cache differs from prepared manifest")
    for wi, w in m["worlds"].items():
        src = contained(cache, w["cache_file"]) if w.get("cache_file") else contained(source, w["source_file"])
        dst = contained(root, w["blocks"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)
        elif not dst.samefile(src):
            raise ValueError("Refusing to overwrite an existing block file")
    validate_dataset(root, require_complete=False)
    (root / "COMPLETE").write_text(digest(root / "manifest.json") + "\n")
    atomic_json(root / "deployment_receipt.json", dict(manifest_sha256=digest(root / "manifest.json"),
        arrays_rehashed=False, blocks_placed_by="byte-identical hardlinks", metadata_and_labels_rehashed=True))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--publish-from-cache")
    p.add_argument("--source-corpus")
    p.add_argument("--verify-arrays", action="store_true")
    a = p.parse_args()
    if a.publish_from_cache:
        publish(a.directory, a.publish_from_cache, a.source_corpus)
    else:
        validate_dataset(a.directory, verify_arrays=a.verify_arrays)
    print(json.dumps(dict(valid=True, manifest_sha256=digest(Path(a.directory) / "manifest.json"))))

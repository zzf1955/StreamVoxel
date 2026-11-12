"""Immutable index-only full-height tile contract (no train/data imports)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from streamvoxel.common.io import sha256_file

FORMAT = "java26_unconditional_v4_tiles_v1"


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Path escapes dataset root")
    return path


def validate_tiles(directory, *, source_dir=None, check_sources=True):
    directory = Path(directory).resolve()
    m = json.loads((directory / "manifest.json").read_text())
    if (m.get("format") != FORMAT or m.get("axisOrder") != "YZX"
            or m.get("dtype") != "uint16" or m.get("targetShapeYZX") != [256,16,16]
            or m.get("worldY") != [-64,191] or m.get("strideZX") != [16,16]
            or m.get("globalNumClasses") != 981 or m.get("numClasses",0) <= 0
            or m.get("conditions") != []):
        raise ValueError("Unexpected unconditional tile contract")
    vocab = contained(directory, m["vocabulary"]["file"])
    if sha256_file(vocab) != m["vocabulary"]["sha256"]:
        raise ValueError("Vocabulary hash mismatch")
    v = json.loads(vocab.read_text())
    if v.get("cleanClassCount") != m["globalNumClasses"]:
        raise ValueError("Vocabulary class count mismatch")
    mapping_path = contained(directory,m["tokenMapping"]["file"])
    if sha256_file(mapping_path) != m["tokenMapping"]["sha256"]:
        raise ValueError("Compact mapping hash mismatch")
    mapping = json.loads(mapping_path.read_text())
    inverse = mapping["compact_to_global"]
    forward = mapping["global_to_compact"]
    if (mapping.get("format") != "java26_v4_compact_mapping_v1"
            or mapping.get("source_manifest_sha256") != m["source"]["manifestSha256"]
            or mapping.get("global_vocabulary_sha256") != m["vocabulary"]["sha256"]
            or mapping.get("num_classes") != m["numClasses"]
            or len(inverse)!=m["numClasses"] or len(forward)!=m["globalNumClasses"]
            or len(set(inverse))!=len(inverse)
            or any(not 0<=g<len(forward) or forward[g]!=i for i,g in enumerate(inverse))
            or any(c!=-1 and (not 0<=c<len(inverse) or inverse[c]!=g) for g,c in enumerate(forward))):
        raise ValueError("Compact mapping identity mismatch")
    worlds = m["worlds"]
    if len({w["seed"] for w in worlds}) != len(worlds):
        raise ValueError("Duplicate World seed")
    assigned = set()
    gy, gz, gx = m["sourceShapeYZX"]
    if gy != 256 or gz % 16 or gx % 16:
        raise ValueError("Source dimensions do not tile exactly")
    origins_z, origins_x = np.arange(0,gz,16), np.arange(0,gx,16)
    expected_z = np.repeat(origins_z, len(origins_x))
    expected_x = np.tile(origins_x, len(origins_z))
    per_world = len(expected_z)
    for name, split in m["splits"].items():
        members = split["worldIndices"]
        if (len(set(members)) != len(members) or assigned.intersection(members)
                or any(not 0 <= i < len(worlds) for i in members)):
            raise ValueError("Splits overlap or contain invalid Worlds")
        assigned.update(members)
        p = contained(directory, split["indicesFile"])
        if sha256_file(p) != split["sha256"]:
            raise ValueError(f"Index hash mismatch: {name}")
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        if (a.dtype != np.int32 or a.shape != (len(members)*per_world,3)
                or len(a) != split["tiles"]):
            raise ValueError("Invalid tile index dimensions")
        for j, wi in enumerate(members):
            rows = a[j*per_world:(j+1)*per_world]
            if (not np.all(rows[:,0] == wi) or not np.array_equal(rows[:,1],expected_z)
                    or not np.array_equal(rows[:,2],expected_x)):
                raise ValueError("Tile coverage or split identity mismatch")
    if assigned != set(range(len(worlds))):
        raise ValueError("Some Worlds have no split")
    if check_sources:
        root = Path(source_dir or m["source"]["directory"]).resolve()
        if sha256_file(root / "manifest.json") != m["source"]["manifestSha256"]:
            raise ValueError("Source corpus manifest changed")
        for w in worlds:
            p = contained(root,w["manifest"])
            if sha256_file(p) != w["manifestSha256"]:
                raise ValueError("Source World manifest changed")
            p = contained(root,w["file"])
            if p.stat().st_size != w["bytes"]:
                raise ValueError("Source array size changed")
            a = np.load(p,mmap_mode="r",allow_pickle=False)
            if a.dtype != np.uint16 or list(a.shape) != m["sourceShapeYZX"]:
                raise ValueError("Source array header changed")
    return m

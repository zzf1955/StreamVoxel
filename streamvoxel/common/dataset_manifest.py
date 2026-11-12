"""Validators for the stream-neighbor dataset packages (windows, tile cache, portable cache).

Only validation lives here; formats, field names and error messages are unchanged from the original.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from streamvoxel.common.io import sha256_file

WINDOWS_FORMAT = "java26_stream_neighbor_windows_v1"
TILE_CACHE_FORMAT = "java26_stream_neighbor_tile8_cache_v1"
PORTABLE_FORMAT = "java26_stream_neighbor_portable_cache_v1"
WINDOW_SIDE = 24
TARGET_SIDE = 8
STRIDE = 8
WORLD_Y = [-64, 191]


def window_origins(side: int) -> np.ndarray:
    if side < WINDOW_SIDE or (side - WINDOW_SIDE) % STRIDE:
        raise ValueError("World side must fit complete stride-8 24-square windows")
    return np.arange(0, side - WINDOW_SIDE + 1, STRIDE, dtype=np.int32)


def window_coordinates(world_indices: list[int], origins: np.ndarray) -> np.ndarray:
    z, x = np.meshgrid(origins, origins, indexing="ij")
    locations = np.stack((z.ravel(), x.ravel()), axis=1)
    rows = np.empty((len(world_indices) * len(locations), 3), dtype=np.int32)
    for position, world_index in enumerate(world_indices):
        section = rows[position * len(locations):(position + 1) * len(locations)]
        section[:, 0] = world_index
        section[:, 1:] = locations
    return rows


def validate_dataset(directory: Path, *, verify_sources: bool = False) -> dict:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("format") != WINDOWS_FORMAT or manifest.get("windowShapeYZX") != [256, 24, 24] or
            manifest.get("targetShapeYZX") != [256, 8, 8] or manifest.get("strideZX") != [8, 8] or
            manifest.get("worldY") != WORLD_Y or set(manifest.get("splits", {})) != {"train", "validation"}):
        raise ValueError("Unexpected stream neighbor dataset contract")
    source = manifest["source"]
    source_dir = Path(source["directory"])
    if sha256_file(source_dir / "manifest.json") != source["manifestSha256"]:
        raise ValueError("Source manifest identity changed")
    if sha256_file(source_dir / source["vocabulary"]) != source["vocabularySha256"]:
        raise ValueError("Source vocabulary identity changed")
    originals = json.loads((source_dir / "manifest.json").read_text())
    if originals["shapeYZX"] != manifest["sourceShapeYZX"]:
        raise ValueError("Source shape changed")
    origins = window_origins(originals["shapeYZX"][1])
    expected_per_world = len(origins) ** 2
    if expected_per_world != manifest["windowsPerWorld"]:
        raise ValueError("Window count contract changed")
    worlds = manifest["source"]["worlds"]
    if len(worlds) != len(originals["worlds"]):
        raise ValueError("Source World count changed")
    for row in worlds:
        path = source_dir / row["file"]
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Source World size changed")
        if verify_sources and sha256_file(path) != row["sha256"]:
            raise ValueError("Source World identity changed")
    for name, split in manifest["splits"].items():
        file_path = directory / split["indicesFile"]
        if sha256_file(file_path) != split["sha256"]:
            raise ValueError(f"{name} index identity changed")
        records = np.load(file_path, mmap_mode="r", allow_pickle=False)
        world_indices = split["worldIndices"]
        if (records.dtype != np.int32 or records.shape != (len(world_indices) * expected_per_world, 3) or
                split["windows"] != len(records) or
                split["worldIds"] != [worlds[i]["id"] for i in world_indices]):
            raise ValueError(f"{name} index contract mismatch")
        expected = window_coordinates(world_indices, origins)
        if not np.array_equal(records, expected):
            raise ValueError(f"{name} window coordinates changed")
    if (set(manifest["splits"]["train"]["worldIndices"]) &
            set(manifest["splits"]["validation"]["worldIndices"])):
        raise ValueError("Train and validation Worlds overlap")
    if set(manifest["splits"]["train"]["worldIndices"] +
           manifest["splits"]["validation"]["worldIndices"]) != set(range(len(worlds))):
        raise ValueError("Not all source Worlds are assigned")
    return manifest


def validate_cache(directory, data_dir, *, verify_arrays=False):
    directory, data_dir = Path(directory).resolve(), Path(data_dir).resolve()
    cache = json.loads((directory / "manifest.json").read_text())
    source = validate_dataset(data_dir, verify_sources=False)
    shape = source["sourceShapeYZX"]
    expected = [shape[1]//8, shape[2]//8, shape[0], 8, 8]
    if (cache.get("format") != TILE_CACHE_FORMAT or cache.get("sourceManifestSha256") != sha256_file(data_dir/"manifest.json")
            or cache.get("sourceVocabularySha256") != source["source"]["vocabularySha256"]
            or cache.get("shape") != expected or cache.get("dtype") != "uint16"):
        raise ValueError("Tile cache/source manifest mismatch")
    worlds = source["source"]["worlds"]
    if len(cache["files"]) != len(worlds):
        raise ValueError("Tile cache World count mismatch")
    for record, world in zip(cache["files"], worlds):
        file = directory / record["file"]
        if (record["worldId"] != world["id"] or record["sourceSha256"] != world["sha256"]
                or file.stat().st_size != record["bytes"]):
            raise ValueError("Tile cache file/source mismatch")
        values = np.load(file, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.uint16 or list(values.shape) != expected:
            raise ValueError("Invalid tile cache array")
        if verify_arrays and sha256_file(file) != record["sha256"]:
            raise ValueError("Tile cache hash mismatch")
    return cache


def validate_portable_cache(directory, *, verify_arrays=False):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    shape = manifest.get("tileShape")
    source_shape = manifest.get("sourceShapeYZX")
    if (manifest.get("format") != PORTABLE_FORMAT or manifest.get("axisOrder") != "YZX"
            or manifest.get("dtype") != "uint16" or manifest.get("targetShapeYZX") != [256, 8, 8]
            or not isinstance(shape, list) or len(shape) != 5 or shape[-3:] != [256, 8, 8]
            or not isinstance(source_shape, list) or len(source_shape) != 3
            or shape[:2] != [source_shape[1]//8, source_shape[2]//8]
            or manifest.get("numClasses", 0) <= 0):
        raise ValueError("Invalid portable neighbor Cache contract")
    vocabulary = manifest["vocabulary"]
    vocab_path = directory / vocabulary["file"]
    if sha256_file(vocab_path) != vocabulary["sha256"]:
        raise ValueError("Portable Cache vocabulary hash mismatch")
    names = json.loads(vocab_path.read_text())["idToCleanName"]
    if len(names) != manifest["numClasses"]:
        raise ValueError("Portable Cache vocabulary size mismatch")
    worlds = manifest["worlds"]
    if not worlds or len({row["id"] for row in worlds}) != len(worlds):
        raise ValueError("Invalid portable World IDs")
    for row in worlds:
        path = directory / row["file"]
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Portable Cache World file size mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != shape or values.dtype != np.uint16:
            raise ValueError("Invalid portable Cache tile array")
        if verify_arrays and sha256_file(path) != row["sha256"]:
            raise ValueError("Portable Cache World hash mismatch")
    splits = manifest["splits"]
    if set(splits) != {"train", "validation"}:
        raise ValueError("Portable Cache requires train and validation")
    memberships = []
    for split in splits.values():
        path = directory / split["indicesFile"]
        if sha256_file(path) != split["sha256"]:
            raise ValueError("Portable Cache index hash mismatch")
        records = np.load(path, mmap_mode="r", allow_pickle=False)
        members = split["worldIndices"]
        if (records.dtype != np.int32 or records.shape != (split["windows"], 3)
                or not members or any(not 0 <= i < len(worlds) for i in members)
                or split["worldIds"] != [worlds[i]["id"] for i in members]
                or not np.isin(records[:, 0], members).all()
                or (records[:, 1:] < 0).any()
                or (records[:, 1:] % 8).any()
                or (records[:, 1:] > source_shape[1]-24).any()):
            raise ValueError("Invalid portable Cache indices")
        memberships.extend(members)
    if sorted(memberships) != list(range(len(worlds))):
        raise ValueError("Portable Cache World split leakage or omission")
    return manifest

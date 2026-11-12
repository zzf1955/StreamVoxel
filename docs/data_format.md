# Data formats

The data generation pipeline is not part of this repository. This document describes what training, streaming and evaluation read. The authoritative checks are `streamvoxel/common/unconditional_tiles.py` (`validate_tiles`), `streamvoxel/common/controlnet_dataset.py` (`validate_dataset`), `streamvoxel/common/unconditional_cache.py` and `streamvoxel/stream/session.py` (`Store`, `prepare`).

## Conventions

- Block arrays hold global token ids as `uint16`, axis order **YZX**: `tokens[y, z, x]`.
- Height is 256 layers (world Y -64 to 191). A tile is `[256, 16, 16]` (one 16x16 column, the unit that is generated and committed).
- Global vocabulary: 981 clean block classes. Training uses a compact vocabulary of 304 classes; the compact id exists only inside training and the streaming backend. Inputs and outputs of evaluation and the viewer are global ids.
- Biomes are stored as quart labels (4x4 columns, 64 layers of 4 blocks): `[64, Z/4, X/4]`.

## Vocabulary and compact mapping

`vocabulary.json` (global vocabulary) is read through these keys: `cleanClassCount` (must be 981), `idToCleanName` (global id to block name; air is `minecraft:air`, `minecraft:cave_air`, `minecraft:void_air`) and `rawStateIdToCleanId` (raw block-state id to global id, used when reading raw source tiles).

`compact_mapping.json` is a reversible mapping with keys `format` (`java26_v4_compact_mapping_v1`), `num_classes` (304), `compact_to_global` (length 304, unique) and `global_to_compact` (length 981, `-1` for classes outside the training vocabulary). Loading code verifies `global_to_compact[compact_to_global[i]] == i`. A global id without a compact class is rejected when conditions are built (`Mapping.compact`).

Two registries ship in `streamvoxel/common/` for the viewer and the world export: `block_state_vocab.json` (all block states of the target game version) and `block_defaults.json` (default state per block, game version 26.2, data version 4903). `streamvoxel/common/preview_states.py` turns global tokens into block states with them.

## Unconditional tile dataset (stages 1 and 2)

`--data-dir` points to an immutable index directory. `manifest.json` must declare `format = java26_unconditional_v4_tiles_v1`, `axisOrder = YZX`, `dtype = uint16`, `targetShapeYZX = [256, 16, 16]`, `worldY = [-64, 191]`, `strideZX = [16, 16]`, `globalNumClasses = 981`, `conditions = []`, and refer to the vocabulary and token mapping by file name and SHA256 (`vocabulary`, `tokenMapping`).

- `worlds`: one entry per world (unique `seed`, `originBlockXYZ`, source file and manifest with hashes).
- `splits`: `train`, `validation`, ... each with `worldIndices` (disjoint between splits; every world has a split), `indicesFile` (an `int32` npy of shape `[tiles, 3]` with rows `(world_index, z, x)`, covering every aligned 16-cell of each world exactly once) and its SHA256.
- `sourceShapeYZX`: `[256, Z, X]` with `Z`, `X` multiples of 16.

Read backends (`--read-backend`):

| Backend | Source |
| --- | --- |
| `raw_tiles` | raw source tiles under the source corpus, hash checked per tile |
| `world` | one `uint16` npy array per world |
| `compact_cache` | per-world npy of compact tiles `[N, 256, 16, 16]`, format `java26_compact_tile16_cache_v1` |
| `zstd_cache` | per-world payload of independent zstd frames (one per tile, `uint16` compact `[256, 16, 16]`) plus a `uint64` offsets array of length `N + 1`, format `java26_compact_tile16_zstd_v1` |

The cache backends need `--tile-cache-dir` with a `manifest.json` and a `COMPLETE.json` whose `manifestSha256` matches. The cache manifest must carry `indexManifestSha256` (hash of the index `manifest.json`), `mappingSha256` (hash of the compact mapping file) and the matching `format`, plus per-world rows (`file`, `bytes`, `shape` or `offsets`/`tiles`, `rawTileShapeZX`, `worldShapeYZX`). Tile `N` of a world is located by `tile_index` in `unconditional_cache.py`. Worlds that are not in the cache (for example test worlds) fall back to reading their source arrays.

## ControlNet dataset (stage 3)

`--dataset` points to a directory with `manifest.json` (`format = java26_streaming_controlnet_v1`) and a `COMPLETE` file holding the SHA256 of `manifest.json`. Required fields: `num_classes` (304), `target_shape_yzx` (`[256, 16, 16]`), `splits` (`train`, `validation`, `test`, disjoint lists of world ids), `worlds` (per world: `blocks`, `biomes`, `block_shape`, optional tile-cache fields), `files` (each with `file`, `bytes`, `sha256`, `kind`), `mapping_file` (compact mapping) and `vocabulary_file`. Per world, `biomes` is a `uint16` npy of shape `[64, 512, 512]` and `blocks` a `uint16` npy of compact ids `[256, Z, X]`; worlds stored as `lossless_tile32_zstd` carry a tile index (`[384, 32, 32]` zstd tiles of a `[384, 2048, 2048]` world). Training samples are groups of `side` x `side` known tiles plus ring targets around them (`ring_targets`, `known_neighbors`). `validate_dataset(dir, verify_arrays=True)` re-hashes every file.

## Panel for streaming and evaluation

A panel (`parent_panel` in the streaming config) is a directory with `manifest.json`, a `COMPLETE` file with the SHA256 of that manifest, and one folder per case. `manifest.json` has `cases` (each with `world`, `directory`, and for the viewer export `initial_box_zx`, plus `gt_canvas_box_zx` when the ground truth is stored cropped; the viewer export reads `case00`) and `files` (SHA256 of every `<directory>/gt_global.npy` and `<directory>/biomes.npy`). Per case:

- `gt_global.npy`: `uint16` `[256, Z, X]`, global ids, YZX, `Z` and `X` multiples of 16. Only the initial box (`initial_box_zx`) is read as known context; the rest is ground truth for scoring and for the viewer export.
- `biomes.npy`: quart labels `[64, Z/4 + 2, X/4 + 2]` (one-quart halo). Biome crops wrap periodically with the world's 64 quarts.

The evaluation entries take a single case directory with the same two files (`--case`) plus `--mapping` and `--vocabulary`.

`session prepare` links the compact mapping and vocabulary into `<run_dir>/panel` and records `air_compact_ids` from the ControlNet checkpoint config.

## Contract modules

Three small modules in `streamvoxel/common/` hold the format contracts; they need only numpy and are CPU-only.

- `world_format.py`: `AXIS_ORDER = "YZX"`, `validate_world_array(array)` (3-D `[Y,Z,X]` or 4-D `[N,Y,Z,X]`, `uint16`, no empty dimension) and `load_world(path)` (loads a `blocks.npy`, and checks `axis_order` in a sibling `metadata.json` if present).
- `vocab.py`: `load_global_vocab(path)`, which requires a JSON object with a `token_to_block` mapping. This is the preview/world-array vocabulary layout, not the `vocabulary.json` of the training packages described above (`idToCleanName`).
- `dataset_manifest.py`: validators for the stream-neighbor packages (`[256, 8, 8]` targets in 24x24 windows, stride 8): `validate_dataset`, `validate_cache` and `validate_portable_cache`. They check format strings, shapes, dtypes, SHA256 of index/vocabulary files, window coordinates and train/validation disjointness. `verify_arrays=True` additionally re-hashes every world array. There is no command line entry point.

```python
from pathlib import Path
import numpy as np
from streamvoxel.common.world_format import load_world
from streamvoxel.common.dataset_manifest import validate_portable_cache

blocks, metadata = load_world(Path("my_world/blocks.npy"))  # raises ValueError unless uint16 YZX
manifest = validate_portable_cache("my_cache", verify_arrays=True)
```

These validators cover the stream-neighbor package format only. The tile index, ControlNet dataset and panel layouts of this document are checked by `validate_tiles` and `validate_dataset` in `unconditional_tiles.py` and `controlnet_dataset.py`. Tests: `tests/common/test_contracts.py`.

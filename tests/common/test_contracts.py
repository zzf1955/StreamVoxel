from __future__ import annotations

import json

import numpy as np
import pytest

from streamvoxel.common.dataset_manifest import validate_portable_cache, window_origins
from streamvoxel.common.vocab import load_global_vocab
from streamvoxel.common.world_format import AXIS_ORDER, load_world, validate_world_array


def test_world_array_requires_uint16_yzx_volume() -> None:
    value = np.zeros((12, 12, 12), dtype=np.uint16)
    assert validate_world_array(value).shape == (12, 12, 12)
    with pytest.raises(ValueError, match="uint16"):
        validate_world_array(value.astype(np.uint8))
    with pytest.raises(ValueError, match="shape"):
        validate_world_array(np.zeros((12, 12), dtype=np.uint16))


def test_world_metadata_axis_order_is_checked(tmp_path) -> None:
    blocks = tmp_path / "blocks.npy"
    np.save(blocks, np.zeros((2, 2, 2), dtype=np.uint16))
    (tmp_path / "metadata.json").write_text(json.dumps({"axis_order": AXIS_ORDER}))
    loaded, metadata = load_world(blocks)
    assert loaded.dtype == np.uint16
    assert metadata == {"axis_order": AXIS_ORDER}


def test_global_vocab_requires_token_to_block(tmp_path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"token_to_block": {"0": "minecraft:air"}}))
    assert load_global_vocab(path)["token_to_block"] == {"0": "minecraft:air"}
    path.write_text(json.dumps({"token_to_block": []}))
    with pytest.raises(ValueError, match="invalid global vocabulary"):
        load_global_vocab(path)


def test_portable_cache_rejects_wrong_format(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError, match="portable neighbor Cache contract"):
        validate_portable_cache(tmp_path)


def test_window_origins_need_complete_windows() -> None:
    assert window_origins(40).tolist() == [0, 8, 16]
    with pytest.raises(ValueError, match="stride-8"):
        window_origins(30)

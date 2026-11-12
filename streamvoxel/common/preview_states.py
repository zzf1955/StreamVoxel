"""One context-free engine-default block-state policy for every Java preview."""
from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
import numpy as np

from streamvoxel.common.paths import ROOT
REGISTRY_PATH = ROOT / 'streamvoxel/common/block_state_vocab.json'
DEFAULTS_PATH = ROOT / 'streamvoxel/common/block_defaults.json'
POLICY = 'java26_engine_default_v1'

@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))

@lru_cache(maxsize=1)
def load_defaults() -> dict[str, Any]:
    return json.loads(DEFAULTS_PATH.read_text(encoding='utf-8'))

def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    defaults = load_defaults()
    encoded = json.dumps(registry['idToState'], ensure_ascii=False, separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != defaults['registryStatesSha256']:
        raise ValueError('Java preview registry differs from the engine-default table')
    if int(registry.get('dataVersion', defaults['dataVersion'])) != defaults['dataVersion']:
        raise ValueError('Java preview DataVersion differs from engine defaults')
    return defaults

def default_ids_for_names(names: Iterable[str], registry: dict[str, Any] | None = None) -> np.ndarray:
    registry = load_registry() if registry is None else registry
    blocks = validate_registry(registry)['blocks']
    result = []
    for value in names:
        name = str(value).split('[', 1)[0]
        if name.startswith('clean:'):
            name = 'minecraft:white_' + name.split(':', 1)[1]
        if name not in blocks:
            raise ValueError(f'No engine default for preview block {name!r}')
        result.append(blocks[name]['rawStateId'])
    return np.asarray(result, dtype=np.uint16)

def preview_lookup(token_space: str, vocabulary: dict[str, Any] | None = None,
                   registry: dict[str, Any] | None = None) -> np.ndarray:
    registry = load_registry() if registry is None else registry
    vocabulary = {} if vocabulary is None else vocabulary
    if token_space == 'raw':
        names = registry['idToState']
    elif token_space == 'clean':
        names = vocabulary['idToCleanName']
    elif token_space == 'compact':
        names = vocabulary.get('compact_to_clean_name') or vocabulary.get('compact_to_state')
        if names is None:
            ids = vocabulary['compact_to_global']
            if isinstance(ids, dict):
                ids = [ids[str(i)] for i in range(len(ids))]
            if any(not isinstance(i, int) or i < 0 or i >= len(registry['idToState']) for i in ids):
                raise ValueError('compact_to_global contains an invalid raw state ID')
            names = [registry['idToState'][i] for i in ids]
    else:
        raise ValueError(f'Unknown preview token space: {token_space}')
    return default_ids_for_names(names, registry)

def apply_preview_lookup(tokens: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens)
    if values.dtype.kind not in 'iu':
        raise ValueError('Preview tokens must have integer dtype')
    if values.size and (int(values.min()) < 0 or int(values.max()) >= len(lookup)):
        raise ValueError('Preview token outside vocabulary, possibly an unresolved MASK')
    return lookup[values.astype(np.int64, copy=False)]

def to_preview_states(tokens: np.ndarray, *, token_space: str = 'clean',
                      vocabulary: dict[str, Any] | None = None,
                      registry: dict[str, Any] | None = None) -> np.ndarray:
    """GT and generated tokens use exactly the same table; no provenance or neighbors."""
    return apply_preview_lookup(tokens, preview_lookup(token_space, vocabulary, registry))

"""Repository-relative paths. ROOT is the checkout root on whichever machine runs the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = Path(__file__).resolve().parent
BLOCK_STATE_VOCAB = COMMON / "block_state_vocab.json"
BLOCK_DEFAULTS = COMMON / "block_defaults.json"

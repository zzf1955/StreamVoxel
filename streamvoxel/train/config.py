"""YAML defaults for argparse trainers: the command line wins over the file."""
import argparse
from pathlib import Path

import yaml


def parse_with_config(parser, argv=None):
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config")
    path = early.parse_known_args(argv)[0].config
    parser.add_argument("--config", help="YAML file of argparse defaults; command-line flags override it")
    if path:
        values = yaml.safe_load(Path(path).read_text()) or {}
        values = {str(key).replace("-", "_"): value for key, value in values.items()}
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(values) - set(actions))
        if unknown:
            parser.error(f"Unknown keys in {path}: {unknown}")
        for key, value in list(values.items()):
            action = actions[key]
            action.required = False
            if isinstance(value, str) and action.type is not None:
                values[key] = action.type(value)
        parser.set_defaults(**values)
    return parser.parse_args(argv)

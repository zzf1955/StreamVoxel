from pathlib import Path

import torch

from streamvoxel.common.io import digest


def resolve_control(path):
    path = Path(path)
    if path.is_dir():
        for name in ("checkpoint_best.pt", "checkpoint_latest.pt"):
            if (path / name).exists():
                return path / name
        raise FileNotFoundError(f"no checkpoint_best.pt / checkpoint_latest.pt in {path}")
    return path


def load_model(base, control, *, device, weights="raw"):
    """Rebuild the training architecture from the checkpoint's own config and load its control tensors."""
    from streamvoxel.models.compact_controlnet import CompactStreamingControlNet
    from streamvoxel.models.streaming_controlnet import from_base_checkpoint
    from streamvoxel.train.train_controlnet import FORMAT, load_control_state
    saved = torch.load(control, map_location="cpu", mmap=True, weights_only=False)
    if saved.get("format") != FORMAT:
        raise ValueError(f"{control} is not a ControlNet checkpoint")
    config = saved["config"]
    base_sha = digest(base)
    if config.get("identity", {}).get("base") not in (None, base_sha):
        raise ValueError("base weights differ from the ones this ControlNet was trained on")
    mc = config["model"]
    model = from_base_checkpoint(
        base, model_class=CompactStreamingControlNet, control_depth=mc["control_depth"],
        control_hidden_dim=mc["control_hidden_dim"], checkpointing=False, cell_chunk=2,
        biome_patch=mc.get("biome_patch", 1), biome_halo=mc.get("biome_halo", 16),
        biome_classes=mc["biome_classes"], struct_classes=0)   # struct tensors of legacy checkpoints are pruned
    state = dict(saved["model"])
    if weights == "ema":
        state.update(saved["ema"])
    elif weights != "raw":
        raise ValueError("weights must be 'raw' or 'ema'")
    load_control_state(model, state)
    model.set_runtime(target_checkpoint="none", attention_backend="concat")
    model.eval().requires_grad_(False).to(device)
    identity = dict(control_checkpoint=str(control), control_sha256=digest(control), step=int(saved["step"]),
                    weights=weights, base_checkpoint=str(base), base_sha256=base_sha,
                    half_mask=config.get("mask"), model_config=model.config())
    return model, identity

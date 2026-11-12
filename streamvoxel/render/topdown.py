"""Top-down surface images and LPIPS for generated windows (CPU numpy; LPIPS needs the `lpips` package).

Block colours are the mean of the texture in the user's own Minecraft client jar (`--client-jar` or the `STREAMVOXEL_MC_JAR`
environment variable). No texture or jar is shipped with this repository.
"""
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np

AIR = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
TINT = {"grass": (145, 189, 89), "leaves": (89, 174, 48), "water": (63, 118, 228),
        "vine": (89, 174, 48), "fern": (145, 189, 89), "lily": (32, 128, 48)}


def client_jar(path=None):
    """The jar given on the command line, else `STREAMVOXEL_MC_JAR`."""
    path = path or os.environ.get('STREAMVOXEL_MC_JAR')
    if not path:
        raise ValueError('Pass --client-jar or set STREAMVOXEL_MC_JAR to your local Minecraft client jar')
    return Path(path)


def palette(vocabulary, jar):
    """Per-class mean texture colour [classes,3] and the boolean air flag, from the vocabulary's `idToCleanName`."""
    from PIL import Image
    names = json.loads(Path(vocabulary).read_text())["idToCleanName"]
    colors = np.full((len(names), 3), 128, dtype=np.float32)
    with zipfile.ZipFile(jar) as archive:
        have = set(archive.namelist())
        for i, name in enumerate(names):
            base = name.split(":")[-1]
            candidates = [base + suffix for suffix in ("_top", "", "_side", "_still", "_0", "_bottom")]
            found = next(("assets/minecraft/textures/block/" + candidate + ".png"
                          for candidate in candidates
                          if "assets/minecraft/textures/block/" + candidate + ".png" in have), None)
            if found:
                pixels = np.asarray(Image.open(io.BytesIO(archive.read(found))).convert("RGBA"), dtype=np.float32)
                weights = pixels[..., 3:] / 255
                if weights.sum() > 0:
                    colors[i] = (pixels[..., :3] * weights).sum((0, 1)) / weights.sum()
            for key, tint in TINT.items():
                if (key in base and "block" not in base.replace("grass_block", "")) or base == "grass_block":
                    colors[i] *= np.asarray(tint) / 255
                    break
            if base == "water":
                colors[i] = TINT["water"]
    return colors, np.asarray([name in AIR for name in names], dtype=bool)


def surface_rgb(compact, mapping, colors, air):
    """Highest non-air block per column with fixed height shading; the same rule is used for GT and generated. Returns rgb, height, top."""
    blocks = mapping[compact]
    solid = ~air[blocks]
    height = 255 - np.argmax(solid[::-1], axis=0)
    has = solid.any(0)
    top = np.take_along_axis(blocks, height[None], axis=0)[0]
    dz, dx = np.gradient(height.astype(np.float32))
    shade = np.clip(1 + .3 * (dx + dz) / np.sqrt(1 + dx * dx + dz * dz), .72, 1.25)
    rgb = np.clip(colors[top] * shade[..., None], 0, 255).astype(np.uint8)
    rgb[~has] = 0
    return rgb, height, top


def image_tensor(rgb):
    import torch
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 127.5 - 1


def lpips_net(device):
    """AlexNet LPIPS v0.1, frozen."""
    import lpips
    return lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().requires_grad_(False).to(device)


def masked_lpips(net, gt_rgb, pred_rgb, selected, device):
    """LPIPS of the two images with everything outside `selected` (bool [H,W]) black in both."""
    import torch
    a, b = gt_rgb.copy(), pred_rgb.copy()
    a[~selected] = 0
    b[~selected] = 0
    with torch.no_grad():
        return float(net(image_tensor(a).to(device), image_tensor(b).to(device)).item())

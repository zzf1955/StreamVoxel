"""Fixed-capacity slot forward. Conditions are copied into padded key buffers.

One CUDA graph per bucket size (1, 2, 4, ... batch) runs only the leading slots, so a half-empty slot table
does not pay for its idle slots. Callers keep active windows in the lowest slot indexes.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from streamvoxel.models.streaming_controlnet import RotaryCrossAttention
from streamvoxel.train.runtime import autocast
from streamvoxel.stream.condition import Item, build_group
from streamvoxel.stream.model_loading import load_model, resolve_control


class SlotCrossAttention(RotaryCrossAttention):
    """`RotaryCrossAttention` plus a boolean key mask over the padded prefix and the padded local keys.

    A True mask entry is allowed to attend. With both masks left at None the original forward is unchanged.
    """

    prefix_valid = None
    local_valid = None

    def forward(self, value, positions, prefix_kv, local, local_positions):
        prefix_valid, local_valid = self.prefix_valid, self.local_valid
        if prefix_valid is None and local_valid is None:
            return super().forward(value, positions, prefix_kv, local, local_positions)
        batch, queries, width = value.shape
        query = self.query(self.norm(value)).reshape(batch, queries, self.heads, width // self.heads).transpose(1, 2)
        query = self.rope(query, positions)
        key, val = self.key_value(local).reshape(batch, local.shape[1], 2, self.heads, width // self.heads).unbind(2)
        key, val = key.transpose(1, 2), val.transpose(1, 2)
        key = self.rope(key, local_positions)
        prefix_k, prefix_v = prefix_kv
        if prefix_k.shape[0] != batch or prefix_k.ndim != 4:
            raise ValueError('Slot prefix K/V must be [batch, heads, keys, dim]')
        key = torch.cat((prefix_k, key), 2)
        val = torch.cat((prefix_v, val), 2)
        if prefix_valid.shape != (batch, prefix_k.shape[2]) or local_valid.shape != (batch, local.shape[1]):
            raise ValueError('Slot key mask does not match the padded keys')
        mask = torch.cat((prefix_valid, local_valid), 1)
        out = torch.nn.functional.scaled_dot_product_attention(query, key, val, attn_mask=mask[:, None, None, :], dropout_p=0.)
        return self.proj(out.transpose(1, 2).reshape(batch, queries, width))


def install_slot_attention(model):
    for cross in model.cross:
        if type(cross) not in (RotaryCrossAttention, SlotCrossAttention):
            raise TypeError(f'Unexpected cross attention {type(cross)}')
        cross.__class__ = SlotCrossAttention
        cross.prefix_valid = None
        cross.local_valid = None
    return model


def condition_widths(model, spec):
    """Tokens contributed by one L1 tile, one L2 tile, and the biome window."""
    height, tile = spec.height, spec.tile
    l1 = (height // 4) * (tile // 4) * (tile // 4)
    l2 = (height // 16) * (tile // 16) * (tile // 16)
    if model.biome_halo == 16 and model.biome_patch == 1:
        biome = (height // 4) * 12 * 12
    else:
        side = 6 if model.biome_halo == 4 else 12
        patch = int(model.biome_patch)
        side += (-side) % patch
        biome = (height // 4 // patch) * (side // patch) * (side // patch)
    return l1, l2, biome


def encode_item(model, item, spec, device):
    """Exact (unpadded) prefix K/V and local keys for one item."""
    index = torch.zeros(1, dtype=torch.long, device=device)
    index.condition_n = len(item.l1_blocks)
    group = {k: v.to(device) if torch.is_tensor(v) else v for k, v in build_group(item, spec).items()}
    with autocast(device):
        encoded = model.encode_group(group)
        prefix = model.prefix_kv(encoded, group)
        local, positions = model.local_condition(encoded, group, index)
    return prefix, local, positions


def unpadded_logits(model, item, spec, tokens, timestep):
    """Reference forward with the original attention, masks disabled."""
    saved = [(cross.prefix_valid, cross.local_valid) for cross in model.cross]
    for cross in model.cross:
        cross.prefix_valid = None
        cross.local_valid = None
    try:
        prefix, local, positions = encode_item(model, item, spec, tokens.device)
        stacked = [(key.unsqueeze(0), value.unsqueeze(0)) for key, value in prefix]
        origin = torch.tensor([[0., spec.y_origin, 0.]], device=tokens.device)
        with autocast(tokens.device):
            return model(tokens, timestep=timestep, origin_xyz=origin, group_origin_xyz=origin[0],
                         prefix_kv=stacked, local=local, local_positions=positions)
    finally:
        for cross, pair in zip(model.cross, saved):
            cross.prefix_valid, cross.local_valid = pair


class SlotForward:
    """B slots, one captured graph. `load_slot` encodes on a side stream; `wait_loads` joins it before replay."""

    def __init__(self, model, spec, device, batch, l1_cap, l2_cap):
        self.model, self.spec = model, spec
        self.device = torch.device(device)
        self.batch, self.l1_cap, self.l2_cap = int(batch), int(l1_cap), int(l2_cap)
        self.classes = spec.num_classes
        self.shape = (spec.height, spec.tile, spec.tile)
        install_slot_attention(model)
        l1, l2, biome = condition_widths(model, spec)
        self.l1_tokens, self.l2_tokens, self.biome_tokens = l1, l2, biome
        self.prefix_len = l2_cap * l2
        self.local_len = l1_cap * l1 + biome
        probe = encode_item(model, _probe_item(spec), spec, self.device)
        self._check_probe(probe)
        dtype, pos_dtype, heads, head_dim, hidden, depth = _probe_layout(probe, model)
        self.tokens = torch.full((self.batch, *self.shape), self.classes, dtype=torch.long, device=self.device)
        self.timestep = torch.ones(self.batch, device=self.device)
        origin = torch.tensor([0., spec.y_origin, 0.], device=self.device)
        self.origin = origin.expand(self.batch, -1).contiguous()
        self.group_origin = origin
        self.prefix = []
        for _ in range(depth):
            key = torch.zeros(self.batch, heads, self.prefix_len, head_dim, device=self.device, dtype=dtype)
            self.prefix.append((key, torch.zeros_like(key)))
        self.local = torch.zeros(self.batch, self.local_len, hidden, device=self.device, dtype=dtype)
        self.local_positions = torch.zeros(self.batch, self.local_len, 3, device=self.device, dtype=pos_dtype)
        self.prefix_valid = torch.zeros(self.batch, self.prefix_len, dtype=torch.bool, device=self.device)
        self.local_valid = torch.zeros(self.batch, self.local_len, dtype=torch.bool, device=self.device)
        for cross in model.cross:
            cross.prefix_valid = self.prefix_valid
            cross.local_valid = self.local_valid
        for slot in range(self.batch):
            self._write(slot, *probe)
        self.side = torch.cuda.Stream(device=self.device)
        self.sizes = sorted({min(self.batch, 2 ** k) for k in range(self.batch.bit_length() + 1)})
        self.graphs, self.outputs = {}, {}
        self._capture()

    def _check_probe(self, probe):
        prefix, local, _ = probe
        if prefix[0][0].shape[1] != self.l2_tokens:
            raise RuntimeError(f'L2 tokens {prefix[0][0].shape[1]} != {self.l2_tokens}')
        if local.shape[1] != self.l1_tokens + self.biome_tokens:
            raise RuntimeError(f'Local tokens {local.shape[1]} != {self.l1_tokens}+{self.biome_tokens}')

    def _write(self, index, prefix, local, positions):
        keys = prefix[0][0].shape[1]
        queries = local.shape[1]
        if keys > self.prefix_len or queries > self.local_len:
            raise ValueError(f'Condition ({keys} prefix, {queries} local) exceeds the captured capacity')
        for layer, (key, value) in enumerate(prefix):
            stored_k, stored_v = self.prefix[layer]
            stored_k[index, :, :keys].copy_(key)
            stored_v[index, :, :keys].copy_(value)
            stored_k[index, :, keys:].zero_()
            stored_v[index, :, keys:].zero_()
        self.prefix_valid[index, :keys] = True
        self.prefix_valid[index, keys:] = False
        self.local[index, :queries].copy_(local[0])
        self.local[index, queries:].zero_()
        self.local_positions[index, :queries].copy_(positions[0])
        self.local_positions[index, queries:].zero_()
        self.local_valid[index, :queries] = True
        self.local_valid[index, queries:] = False

    def bucket(self, count):
        """Smallest captured size that covers the first `count` slots."""
        return next(size for size in self.sizes if size >= max(1, count))

    def _call(self, size=None):
        n = self.batch if size is None else size
        for cross in self.model.cross:
            cross.prefix_valid = self.prefix_valid[:n]
            cross.local_valid = self.local_valid[:n]
        with autocast(self.device):
            return self.model(self.tokens[:n], timestep=self.timestep[:n], origin_xyz=self.origin[:n],
                              group_origin_xyz=self.group_origin, prefix_kv=[(k[:n], v[:n]) for k, v in self.prefix],
                              local=self.local[:n], local_positions=self.local_positions[:n])

    def _capture(self):
        current = torch.cuda.current_stream(self.device)
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(current)
        with torch.cuda.stream(warm):
            for size in self.sizes:
                for _ in range(3):
                    self._call(size)
        current.wait_stream(warm)
        torch.cuda.synchronize(self.device)
        # torch.cuda.graph's default capture stream is a class attribute made on the first device that
        # captures; a second GPU must bring its own stream or the capture is invalidated.
        stream = torch.cuda.Stream(device=self.device)
        pool = torch.cuda.graph_pool_handle()
        try:
            for size in reversed(self.sizes):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode='thread_local'):
                    self.outputs[size] = self._call(size)
                self.graphs[size] = graph
            self.graph_ok = True
        except Exception as error:
            self.graphs, self.outputs = {}, {}
            self.graph_ok = False
            print(json.dumps(dict(event='cuda_graph_fallback', device=str(self.device), error=f'{type(error).__name__}: {error}')), flush=True)
        self.graph = self.graphs.get(self.batch)

    def load_slot(self, index, item):
        """Enqueue an encode+copy on the side stream. It overlaps CPU work until `wait_loads`."""
        def move():
            self._write(index, *encode_item(self.model, item, self.spec, self.device))
            self.tokens[index].fill_(self.classes)
        with torch.cuda.stream(self.side):
            move()

    def wait_loads(self):
        torch.cuda.current_stream(self.device).wait_stream(self.side)

    def clear_tokens(self):
        self.wait_loads()
        self.tokens.fill_(self.classes)

    def replay(self, timestep, count=None):
        """Logits for the first `bucket(count)` slots (all slots when `count` is None)."""
        self.timestep.copy_(torch.as_tensor(timestep, dtype=self.timestep.dtype, device=self.device))
        size = self.batch if count is None else self.bucket(count)
        if size not in self.graphs:
            return self._call(size)
        self.graphs[size].replay()
        return self.outputs[size]

    def eager(self, timestep):
        self.timestep.copy_(torch.as_tensor(timestep, dtype=self.timestep.dtype, device=self.device))
        return self._call()

    @torch.inference_mode()
    def numerical_gate(self, item, other=None):
        """Padded graph vs eager (exact) and padded vs the unpadded attention (bf16 close)."""
        reference_tokens = torch.full((1, *self.shape), self.classes, dtype=torch.long, device=self.device)
        reference_time = torch.ones(1, device=self.device)
        reference = unpadded_logits(self.model, item, self.spec, reference_tokens, reference_time).clone()
        self.load_slot(0, item)
        if other is not None and self.batch > 1:
            self.load_slot(1, other)
        self.wait_loads()
        time = torch.ones(self.batch, device=self.device)
        eager = self.eager(time).clone()
        if self.graph is None:
            replayed = eager
            equal = False
        else:
            replayed = self.replay(time).clone()
            equal = bool(torch.equal(eager, replayed))
        gap = (eager[0].float() - reference[0].float()).abs()
        match = float((eager[0].argmax(0) == reference[0].argmax(0)).float().mean())
        mean_abs = float(gap.mean())
        # Masked and unmasked SDPA use different bf16 kernels; the class decision is the contract.
        close = match >= 0.999 and mean_abs <= 0.05
        # A smaller bucket runs other GEMM shapes, so it is held to the same bf16 contract on the loaded slots.
        lead = min(2, self.batch)
        small = self.replay(time, lead)[:lead].float()
        bucket_match = float((small.argmax(1) == eager[:lead].argmax(1)).float().mean())
        bucket_close = bucket_match >= 0.999 and float((small - eager[:lead].float()).abs().mean()) <= 0.05
        return dict(graph_equal=equal, graph_used=self.graph is not None, padded_close=bool(close),
                    bucket_close=bool(bucket_close), bucket_match=bucket_match, buckets=sorted(self.graphs) or [],
                    max_abs=float(gap.max()), mean_abs=mean_abs, argmax_match=match,
                    batch=self.batch, prefix_len=self.prefix_len, local_len=self.local_len)


def _probe_item(spec):
    height, tile = spec.height, spec.tile
    blocks = np.zeros((1, height, tile, tile), np.int16)
    biomes = np.zeros((height // 4, 12, 12), np.int64)
    return Item('probe', 0, 1, 1.0, blocks, np.asarray([[-tile, 0]], np.int64), blocks.copy(),
                np.asarray([[-2 * tile, 0]], np.int64), biomes, None)


def _probe_layout(probe, model):
    prefix, local, positions = probe
    key = prefix[0][0]
    return key.dtype, positions.dtype, key.shape[0], key.shape[2], local.shape[-1], len(prefix)


def open_model(base, control, device, weights='raw'):
    model, identity = load_model(base, resolve_control(control), device=torch.device(device), weights=weights)
    from streamvoxel.stream.condition import ModelSpec
    backbone = model.backbone
    spec = ModelSpec(height=backbone.voxel_shape[0], tile=backbone.voxel_shape[1], num_classes=backbone.num_classes,
                     biome_classes=model.biome_embed.num_embeddings, biome_halo=model.biome_halo,
                     y_origin=float(backbone.y_range[0]))
    return model, spec, identity

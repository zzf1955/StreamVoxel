"""Faster live slot forward: same model, same weights, same public API as `engine.SlotForward`.

What changes against `SlotForward` (each switchable, see `FastSlotForward(..., patchify=, attention=, compile=)`):

* Local (L1 + biome) cross K/V, RoPE included, is computed once per window at `load_slot`, exactly the way
  `SlotCrossAttention` computes it every forward. Prefix and local keys are packed into one buffer per layer,
  valid keys first, so a slot of length L occupies keys [0, L).
* `attention='flex'`: FlexAttention over the packed keys with a per-slot block mask (128-key blocks). Blocks past a
  slot's length are skipped, so each slot pays for its own keys only and no key-length buckets or extra graphs are
  needed: the block-mask tensors are static buffers rewritten in place by `load_slot`. `attention='sdpa'` keeps
  the masked SDPA over the packed buffer (packing only).
* `patchify='gather'`: the one-hot Conv3d is replaced by `embedding_bag` over conv weight slices (exactly the 64
  non-zero products per patch, FP32 accumulation, then the conv bias in BF16 as cuDNN adds it).
* `compile='emulate'` (recommended) / `True`: transformer blocks and the cross-attention query path run through
  `torch.compile` (Inductor), which fuses LayerNorm/modulation/RoPE/gating element-wise work. 'emulate' sets
  `inductor.config.emulate_precision_casts` so fused intermediates round to BF16 like eager autocast (plain True
  doubles the mean |dlogits| to ~0.054, over the 0.05 contract). Graphs are still captured per slot bucket.

Sampling: `FastLowEntropyLastSlot` + `SelectGraphs` + `consume_batch` reproduce `LowEntropyLastSlot.consume` bit for bit (same chosen
voxels, same generator stream) with one host wait per forward instead of ~13 per slot.

The model modules are not modified, so `engine.unpadded_logits` and `SlotForward` remain valid references.
Measured on an RTX 4090 D by `scripts/bench_ops.py`.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from streamvoxel.stream.engine import _probe_item, condition_widths, encode_item, install_slot_attention, unpadded_logits
from streamvoxel.train.runtime import autocast

KEY_BLOCK = 128
RECOMMENDED = dict(patchify='gather', attention='flex', compile='emulate')
_FLEX = None


def flex_attention():
    """Compiled FlexAttention (compiled once per process; recompiles per batch shape)."""
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention as raw
        _FLEX = torch.compile(raw, dynamic=False)
    return _FLEX


# ------------------------------------------------------------------------------------------------ patchify

def patch_offsets(tokens, patch):
    """[B,Y,Z,X] -> [B*patches, patch^3] voxel tokens in Conv3d patch order (patches YZX, kernel offsets YZX)."""
    batch, height, depth, width = tokens.shape
    view = tokens.reshape(batch, height // patch, patch, depth // patch, patch, width // patch, patch)
    return view.permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, patch ** 3)


class GatherPatch:
    """Embedding-bag form of `categorical_patch`: conv(onehot(t)) == bias + sum_o W[:, t_o, o]."""

    def __init__(self, conv, dtype=torch.bfloat16):
        weight = conv.weight.detach()
        hidden, inputs, patch = weight.shape[0], weight.shape[1], weight.shape[2]
        self.patch, self.hidden, self.inputs = patch, hidden, inputs
        self.table = weight.permute(1, 2, 3, 4, 0).reshape(inputs * patch ** 3, hidden).to(dtype).contiguous()
        self.bias = None if conv.bias is None else conv.bias.detach().to(dtype)
        self.offsets = torch.arange(patch ** 3, device=weight.device)

    def __call__(self, tokens):
        cells = patch_offsets(tokens.long(), self.patch)
        index = cells * self.patch ** 3 + self.offsets
        out = F.embedding_bag(index, self.table, mode='sum')
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(len(tokens), -1, self.hidden)


def conv_patch(conv, classes):
    """The original one-hot conv patchify (`UnconditionalMD4DiT.patchify`)."""
    def run(tokens):
        dtype = torch.get_autocast_dtype('cuda') if tokens.is_cuda and torch.is_autocast_enabled('cuda') else conv.weight.dtype
        x = torch.zeros((len(tokens), classes + 1, *tokens.shape[1:]), device=tokens.device, dtype=dtype)
        x.scatter_(1, tokens.long()[:, None], 1.)
        return conv(x).flatten(2).transpose(1, 2)
    return run


# ------------------------------------------------------------------------------------------- cross attention

def local_kv(cross, local, positions):
    """`SlotCrossAttention`'s per-forward local K/V for one unpadded item: [1,L,W] -> ([1,H,L,D], [1,H,L,D])."""
    batch, keys, width = local.shape
    key, value = cross.key_value(local).reshape(batch, keys, 2, cross.heads, width // cross.heads).unbind(2)
    key, value = key.transpose(1, 2), value.transpose(1, 2)
    return cross.rope(key, positions), value


def cross_query(cross, value, positions):
    batch, queries, width = value.shape
    query = cross.query(cross.norm(value)).reshape(batch, queries, cross.heads, width // cross.heads).transpose(1, 2)
    return cross.rope(query, positions)


def cross_flex(cross, value, positions, key, val, block_mask):
    batch, queries, width = value.shape
    query = cross_query(cross, value, positions)
    out = flex_attention()(query, key, val, block_mask=block_mask)
    return cross.proj(out.transpose(1, 2).reshape(batch, queries, width))


def cross_sdpa(cross, value, positions, key, val, valid):
    batch, queries, width = value.shape
    query = cross_query(cross, value, positions)
    out = F.scaled_dot_product_attention(query, key, val, attn_mask=valid[:, None, None, :], dropout_p=0.)
    return cross.proj(out.transpose(1, 2).reshape(batch, queries, width))


def run_block(block, x, time, positions):
    return block(x, positions) if time is None else block(x, time, positions)


class PackedKeys:
    """Per-layer packed cross K/V `[B,H,K,D]` plus the per-slot lengths and FlexAttention block-mask buffers."""

    def __init__(self, depth, batch, heads, keys, head_dim, queries, dtype, device):
        self.batch, self.keys, self.queries = batch, keys, queries
        self.kv = [(torch.zeros(batch, heads, keys, head_dim, device=device, dtype=dtype),
                    torch.zeros(batch, heads, keys, head_dim, device=device, dtype=dtype)) for _ in range(depth)]
        self.lengths = torch.zeros(batch, dtype=torch.int32, device=device)
        self.valid = torch.zeros(batch, keys, dtype=torch.bool, device=device)
        q_blocks, k_blocks = -(-queries // KEY_BLOCK), keys // KEY_BLOCK
        self.q_blocks, self.k_blocks = q_blocks, k_blocks
        self.partial_count = torch.zeros(batch, 1, q_blocks, dtype=torch.int32, device=device)
        self.partial_index = torch.zeros(batch, 1, q_blocks, k_blocks, dtype=torch.int32, device=device)
        self.full_count = torch.zeros(batch, 1, q_blocks, dtype=torch.int32, device=device)
        self.full_index = torch.arange(k_blocks, dtype=torch.int32, device=device).expand(batch, 1, q_blocks, k_blocks).contiguous()
        self._masks = {}

    def set_length(self, index, length):
        """Host-side ints only: no device read."""
        full, rest = divmod(int(length), KEY_BLOCK)
        self.lengths[index] = int(length)
        self.valid[index, :length] = True
        self.valid[index, length:] = False
        self.full_count[index] = full
        self.partial_count[index] = 1 if rest else 0
        self.partial_index[index] = min(full, self.k_blocks - 1)

    def block_mask(self, n):
        if n not in self._masks:
            from torch.nn.attention.flex_attention import BlockMask
            lengths = self.lengths

            def inside(b, h, q, k):
                return k < lengths[b]
            self._masks[n] = BlockMask.from_kv_blocks(
                self.partial_count[:n], self.partial_index[:n], self.full_count[:n], self.full_index[:n],
                BLOCK_SIZE=(KEY_BLOCK, KEY_BLOCK), mask_mod=inside, seq_lengths=(self.queries, self.keys))
        return self._masks[n]


# ---------------------------------------------------------------------------------------------- slot forward

class FastSlotForward:
    """Drop-in for `engine.SlotForward` (same attributes used by the worker: tokens, shape, batch, sizes, graphs,
    graph, load_slot, wait_loads, clear_tokens, replay, eager, bucket, numerical_gate)."""

    def __init__(self, model, spec, device, batch, l1_cap, l2_cap, *, patchify='gather', attention='flex',
                 compile=False, capture=True):
        if patchify not in ('gather', 'conv') or attention not in ('flex', 'sdpa'):
            raise ValueError('patchify must be gather|conv and attention flex|sdpa')
        self.model, self.spec = model, spec
        self.device = torch.device(device)
        self.batch, self.l1_cap, self.l2_cap = int(batch), int(l1_cap), int(l2_cap)
        self.classes = spec.num_classes
        self.shape = (spec.height, spec.tile, spec.tile)
        self.options = dict(patchify=patchify, attention=attention, compile=compile)
        bb, cc = model.backbone, model.control
        # Only so `engine.unpadded_logits` (numerical gate reference) works; the fast path never calls these modules.
        install_slot_attention(model)
        if not hasattr(model, 'layer_map'):
            model.layer_map = {i: i for i in range(bb.depth)}
        l1, l2, biome = condition_widths(model, spec)
        self.l1_tokens, self.l2_tokens, self.biome_tokens = l1, l2, biome
        self.prefix_len = l2_cap * l2
        self.local_len = l1_cap * l1 + biome
        keys = self.prefix_len + self.local_len
        keys += (-keys) % KEY_BLOCK
        cross = model.cross[0]
        queries = len(bb.patch_centers_xyz)
        weight_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.packed = PackedKeys(len(model.cross), self.batch, cross.heads, keys, cross.width // cross.heads, queries,
                                 weight_dtype, self.device)
        self.timed = hasattr(bb, 'time_encoding')
        if patchify == 'gather':
            self.patch_b, self.patch_c = GatherPatch(bb.target_patch, weight_dtype), GatherPatch(cc.target_patch, weight_dtype)
        else:
            self.patch_b, self.patch_c = conv_patch(bb.target_patch, bb.num_classes), conv_patch(cc.target_patch, cc.num_classes)
        self.block_fn, self.flex_fn, self.sdpa_fn = run_block, cross_flex, cross_sdpa
        if compile not in (False, True, 'emulate'):
            raise ValueError("compile must be False, True or 'emulate'")
        if compile == 'emulate':
            # Inductor then rounds fused intermediates to BF16 where eager autocast would (process-wide setting).
            import torch._inductor.config as inductor_config
            inductor_config.emulate_precision_casts = True
        if compile:
            self.block_fn = torch.compile(run_block, dynamic=False)
            self.flex_fn = torch.compile(cross_flex, dynamic=False)
            self.sdpa_fn = torch.compile(cross_sdpa, dynamic=False)
        self.tokens = torch.full((self.batch, *self.shape), self.classes, dtype=torch.long, device=self.device)
        self.timestep = torch.ones(self.batch, device=self.device)
        origin = torch.tensor([0., spec.y_origin, 0.], device=self.device)
        self.origin = origin.expand(self.batch, -1).contiguous()
        self.group_origin = origin
        probe = self.encode(_probe_item(spec))
        for slot in range(self.batch):
            self._write(slot, probe)
        self.side = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.sizes = sorted({min(self.batch, 2 ** k) for k in range(self.batch.bit_length() + 1)})
        self.graphs, self.outputs = {}, {}
        self.graph_ok = False
        if capture and self.device.type == 'cuda':
            self._capture()
        self.graph = self.graphs.get(self.batch)

    # -- conditions
    @torch.inference_mode()
    def encode(self, item):
        """Per-layer packed (K, V) `[1,H,P+L,D]` for one item: exact prefix K/V then local K/V."""
        model = self.model
        prefix, local, positions = encode_item(model, item, self.spec, self.device)
        if prefix[0][0].shape[1] > self.prefix_len or local.shape[1] > self.local_len:
            raise ValueError('Condition exceeds the slot capacity')
        layers = []
        with autocast(self.device):
            for cross, (pk, pv) in zip(model.cross, prefix):
                key, value = local_kv(cross, local, positions)
                layers.append((torch.cat((pk[None].to(key.dtype), key), 2), torch.cat((pv[None].to(value.dtype), value), 2)))
        return layers

    def _write(self, index, layers):
        length = layers[0][0].shape[2]
        if length > self.packed.keys:
            raise ValueError('Condition exceeds the captured capacity')
        for (stored_k, stored_v), (key, value) in zip(self.packed.kv, layers):
            stored_k[index, :, :length].copy_(key[0])
            stored_v[index, :, :length].copy_(value[0])
            stored_k[index, :, length:].zero_()
            stored_v[index, :, length:].zero_()
        self.packed.set_length(index, length)

    def load_slot(self, index, item):
        """Enqueue encode + copy on the side stream; `wait_loads` joins it before the next replay."""
        if self.side is None:
            self._write(index, self.encode(item))
            self.tokens[index].fill_(self.classes)
            return
        with torch.cuda.stream(self.side):
            self._write(index, self.encode(item))
            self.tokens[index].fill_(self.classes)

    def wait_loads(self):
        if self.side is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.side)

    def clear_tokens(self):
        self.wait_loads()
        self.tokens.fill_(self.classes)

    def bucket(self, count):
        return next(size for size in self.sizes if size >= max(1, count))

    # -- forward
    def forward(self, n):
        model = self.model
        bb, cc = model.backbone, model.control
        tokens, timestep, origin = self.tokens[:n], self.timestep[:n], self.origin[:n]
        centers = bb.patch_centers_xyz[None] + origin[:, None]
        positions = (centers - self.group_origin[None, None]) / 4
        sa_positions = bb.patch_centers_xyz[None].expand(n, -1, -1) / 4
        y = 2 * (centers[..., 1:2] - bb.y_range[0]) / (bb.y_range[1] - bb.y_range[0]) - 1
        b = self.patch_b(tokens) + bb.y_embed(y)
        c = self.patch_c(tokens) + cc.y_embed(y)
        bt = bb.time_encoding(timestep) if self.timed else None
        ct = cc.time_encoding(timestep) if self.timed else None
        flex = self.options['attention'] == 'flex'
        mask = self.packed.block_mask(n) if flex else self.packed.valid[:n]
        for i in range(bb.depth):
            j = model.layer_map.get(i)
            if j is None:
                b = self.block_fn(bb.blocks[i], b, bt, sa_positions)
                continue
            c = self.block_fn(cc.blocks[j], c, ct, sa_positions)
            key, value = self.packed.kv[j]
            attend = self.flex_fn if flex else self.sdpa_fn
            c = c + attend(model.cross[j], c, positions, key[:n], value[:n], mask)
            # The reference multiplies by scale=1., an exact no-op.
            b = self.block_fn(bb.blocks[i], b, bt, sa_positions) + model.zero[j](c)
        if self.timed:
            from streamvoxel.models.unconditional_md4_timefree import modulate
            shift, gain = bb.final_ada(bt).chunk(2, -1)
            patches = bb.class_head(modulate(bb.final_norm(b), shift, gain))
        else:
            patches = bb.class_head(bb.final_norm(b))
        gy, gz, gx = bb.patch_shape
        p = bb.patch_size
        return patches.reshape(n, gy, gz, gx, p, p, p, bb.num_classes).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            n, bb.num_classes, *bb.voxel_shape)

    def _call(self, size=None):
        n = self.batch if size is None else size
        with torch.inference_mode(), autocast(self.device):
            return self.forward(n)

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

    def replay(self, timestep, count=None):
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
        # Note: like `SlotForward.numerical_gate`, slot 0 (and 1) keep these items' conditions afterwards.
        """Same report keys as `SlotForward.numerical_gate`: fast eager vs graph, and vs the unpadded reference."""
        reference_tokens = torch.full((1, *self.shape), self.classes, dtype=torch.long, device=self.device)
        reference = unpadded_logits(self.model, item, self.spec, reference_tokens, torch.ones(1, device=self.device)).clone()
        self.load_slot(0, item)
        if other is not None and self.batch > 1:
            self.load_slot(1, other)
        self.wait_loads()
        time = torch.ones(self.batch, device=self.device)
        eager = self.eager(time).clone()
        replayed = self.replay(time).clone() if self.graph is not None else eager
        equal = self.graph is not None and bool(torch.equal(eager, replayed))
        gap = (eager[0].float() - reference[0].float()).abs()
        match = float((eager[0].argmax(0) == reference[0].argmax(0)).float().mean())
        mean_abs = float(gap.mean())
        lead = min(2, self.batch)
        small = self.replay(time, lead)[:lead].float()
        bucket_match = float((small.argmax(1) == eager[:lead].argmax(1)).float().mean())
        bucket_close = bucket_match >= 0.999 and float((small - eager[:lead].float()).abs().mean()) <= 0.05
        return dict(graph_equal=equal, graph_used=self.graph is not None, padded_close=bool(match >= 0.999 and mean_abs <= 0.05),
                    bucket_close=bool(bucket_close), bucket_match=bucket_match, buckets=sorted(self.graphs) or [],
                    max_abs=float(gap.max()), mean_abs=mean_abs, argmax_match=match, batch=self.batch,
                    prefix_len=self.prefix_len, local_len=self.local_len, options=self.options)


# ------------------------------------------------------------------------------------------------- sampling

def entropy_torch(logits):
    from streamvoxel.sampling.entropy import entropy_scores
    return entropy_scores(logits)


_ENTROPY_KERNEL = None


def _entropy_kernel():
    global _ENTROPY_KERNEL
    if _ENTROPY_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(x_ptr, out_ptr, C, N, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
            inside = offs < N
            peak = tl.full([BLOCK_N], -float('inf'), tl.float32)
            for start in range(0, C, BLOCK_C):
                rows = start + tl.arange(0, BLOCK_C)
                keep = (rows[:, None] < C) & inside[None, :]
                tile = tl.load(x_ptr + rows[:, None] * N + offs[None, :], mask=keep, other=-float('inf')).to(tl.float32)
                peak = tl.maximum(peak, tl.max(tile, 0))
            total = tl.zeros([BLOCK_N], tl.float32)
            weighted = tl.zeros([BLOCK_N], tl.float32)
            for start in range(0, C, BLOCK_C):
                rows = start + tl.arange(0, BLOCK_C)
                keep = (rows[:, None] < C) & inside[None, :]
                tile = tl.load(x_ptr + rows[:, None] * N + offs[None, :], mask=keep, other=0.).to(tl.float32)
                shifted = tile - peak[None, :]
                e = tl.where(keep, tl.exp(shifted), 0.)
                total += tl.sum(e, 0)
                weighted += tl.sum(e * shifted, 0)
            tl.store(out_ptr + offs, weighted / total - tl.log(total), mask=inside)
        _ENTROPY_KERNEL = kernel
    return _ENTROPY_KERNEL


def entropy_triton(logits, block_n=256, block_c=16):
    """Fused `entropy_scores` (sum p log p over the class axis) reading BF16 logits [C,...] once per pass, FP32 math.
    Not bitwise equal to the PyTorch reduction order."""
    classes = logits.shape[0]
    flat = logits.reshape(classes, -1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    n = flat.shape[1]
    out = torch.empty(n, device=logits.device, dtype=torch.float32)
    _entropy_kernel()[(-(-n // block_n),)](flat, out, classes, n, BLOCK_N=block_n, BLOCK_C=block_c)
    return out.reshape(logits.shape[1:])


def top_mask_nosync(scores, eligible, count):
    """`entropy_spatial_decode.top_mask` without a host read; `count` may be a device scalar. Scores must be finite."""
    flat = scores.reshape(-1).masked_fill(~eligible.reshape(-1), -float('inf'))
    order = torch.argsort(flat, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order), device=order.device)
    return ((rank < count) & eligible.reshape(-1)).reshape(eligible.shape)


def quota_select_nosync(score, mask, k, b, pos, frac, n_total, committed):
    """`scheduler_grid.block_quota_select` with no device->host read.

    `pos` is the inverse of the block order (as the original builds it), `committed` the count of committed
    non-last voxels: under the low-entropy-last invariant (each step commits its quota) it equals `already.sum()`.
    The trim / top-up branches are both evaluated and picked with `torch.where`. `k`, `frac` and `committed` may be
    host numbers or 0-dim device tensors (long, double, long), so the function can be captured in a CUDA graph.
    """
    from streamvoxel.sampling.block_quota import block_view
    mb = block_view(mask, b)
    nbk = mb.shape[0]
    legal = mb.sum(1)
    already = n_total - legal
    ideal = frac * n_total.double()
    base = torch.floor(ideal + 1e-9).long()
    goal = committed + k
    extra = (goal - base.sum()).clamp(min=0)
    frac_rank = torch.argsort(-(ideal - base.double()) * 1e6 + pos.double())
    bump = torch.zeros_like(base)
    bump[frac_rank] = (torch.arange(nbk, device=mask.device) < extra).long()
    target = base + bump
    quota = torch.minimum((target - already).clamp(min=0), legal)
    sm = block_view(score, b).masked_fill(~mb, -float('inf'))
    ranks = torch.argsort(sm, dim=1, descending=True, stable=True)
    bc = torch.zeros_like(mb)
    bc.scatter_(1, ranks, torch.arange(b ** 3, device=mask.device)[None, :] < quota[:, None])
    Y, Z, X = mask.shape
    sel = bc.reshape(Y // b, Z // b, X // b, b, b, b).permute(0, 3, 1, 4, 2, 5).reshape(mask.shape) & mask
    got = sel.sum()
    trim = top_mask_nosync(score, sel, k)
    up = sel | top_mask_nosync(score, mask & ~sel, k - got)
    return torch.where(got > k, trim, torch.where(got < k, up, sel))


def _low_entropy_last_slot():
    from streamvoxel.stream.slot import LowEntropyLastSlot
    return LowEntropyLastSlot


class FastLowEntropyLastSlot(_low_entropy_last_slot()):
    """`LowEntropyLastSlot` whose solid steps do not read the device until `collect` (one D2H copy per step).

    `stage` runs `select` (entropy + quota selection into the static `chosen` buffer; capturable, see
    `SelectGraphs`) and then the eager tail: softmax of the chosen voxels, the multinomial draw on the slot
    generator and the token write. `collect` waits for the copy and does the host bookkeeping. With
    `entropy='torch'` the outputs and the generator stream equal `LowEntropyLastSlot.consume` exactly. Marking and finishing
    steps fall back to `LowEntropyLastSlot.consume`.
    """

    def __init__(self, *args, entropy='torch', **kwargs):
        super().__init__(*args, **kwargs)
        if entropy not in ('torch', 'triton'):
            raise ValueError('entropy must be torch or triton')
        self.entropy_name = entropy
        self.entropy = entropy_torch if entropy == 'torch' else entropy_triton
        blocks = self.total // self.block_size ** 3
        device = self.device
        # Static buffers, so a captured `select` stays valid across windows.
        # `self.last` is re-allocated by every `arm`; `last_static` is its copy at a fixed address (same values).
        self.last_static =torch.zeros(self.shape, dtype=torch.bool, device=device)
        self.n_total = torch.zeros(blocks, dtype=torch.long, device=device)
        self.pos = torch.zeros(blocks, dtype=torch.long, device=device)
        self.k = torch.zeros((), dtype=torch.long, device=device)
        self.committed = torch.zeros((), dtype=torch.long, device=device)
        self.frac = torch.zeros((), dtype=torch.float64, device=device)
        self.chosen = torch.zeros(self.shape, dtype=torch.bool, device=device)
        self.bad = torch.zeros((), dtype=torch.bool, device=device)
        self._bound = False
        self._host = torch.empty(2 * self.total + 1, dtype=torch.long, pin_memory=self.device.type == 'cuda')

    def arm(self, seed, cap, target):
        super().arm(seed, cap, target)
        self._bound = False

    def stageable(self):
        return self.phase == 'solid' and self.quotas is not None and self.solid_calls < len(self.quotas)

    def prepare(self):
        """Per-step scalars (kernel fills, no H2D copy) and, once per window, the last mask and block constants."""
        from streamvoxel.sampling.block_quota import block_view
        k = self.quotas[self.solid_calls]
        committed = sum(self.quotas[:self.solid_calls])
        if not self._bound:
            self.last_static.copy_(self.last)
            self.n_total.copy_(block_view((~self.last).contiguous(), self.block_size).sum(1))
            self.pos[self.order] = torch.arange(len(self.order), device=self.order.device)
            self._bound = True
        self.k.fill_(k)
        self.committed.fill_(committed)
        self.frac.fill_((committed + k) / (self.total - self.last_count))
        return k

    def select(self, logits, tokens):
        """Graph-capturable: reads logits/tokens/static buffers, writes `chosen` and `bad`."""
        selectable = (tokens == self.classes) & ~self.last_static
        chosen = quota_select_nosync(self.entropy(logits), selectable, self.k, self.block_size, self.pos, self.frac,
                                     self.n_total, self.committed)
        self.chosen.copy_(chosen)
        self.bad.copy_((chosen.sum() != self.k) | (chosen & ~selectable).any())

    def stage(self, logits, tokens, graph=None):
        k = self.prepare()
        if graph is None:
            self.select(logits, tokens)
        else:
            graph.replay()
        index = torch.nonzero_static(self.chosen.reshape(-1), size=k).squeeze(1)
        probabilities = logits.reshape(self.classes, -1).index_select(1, index).t().contiguous().float().softmax(-1)
        values = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(-1)
        tokens.view(-1).index_copy_(0, index, values)
        host = self._host[:2 * k + 1]
        host.copy_(torch.cat((index, values, self.bad.long().reshape(1))), non_blocking=True)
        event = None
        if tokens.is_cuda:
            event = torch.cuda.Event()
            event.record()
        return k, host, event, tokens

    def collect(self, staged):
        k, host, event, tokens = staged
        if event is not None:
            event.synchronize()
        if int(host[2 * k]):
            raise AssertionError('low-entropy-last solid step did not meet its quota')
        positions = host[:k].clone()
        values_cpu = host[k:2 * k].to(torch.int16)
        self.solid_calls += 1
        self.actual += 1
        self.nfe += 1
        self.step_map.reshape(-1)[positions] = self.nfe
        self.values.reshape(-1)[positions] = values_cpu
        commit = dict(positions=positions.numpy().astype('int32'), values=values_cpu.numpy(), step=int(self.nfe))
        if self.solid_calls >= len(self.quotas):
            return self._finish(tokens), commit
        return None, commit

    def consume(self, logits, tokens):
        if not self.stageable():
            return super().consume(logits, tokens)
        return self.collect(self.stage(logits, tokens))


class SelectGraphs:
    """One captured `FastLowEntropyLastSlot.select` per (bucket, slot) over the engine's static outputs and token rows.

    Build after the engine captured its graphs, with the slots that will consume its outputs. `lookup(logits)`
    returns the bucket whose output tensor `logits` is (None for eager logits, which then select eagerly)."""

    def __init__(self, engine, slots):
        self.engine, self.slots = engine, slots
        self.graphs = {}
        if not getattr(engine, 'graphs', None):
            return
        device = engine.device
        current = torch.cuda.current_stream(device)
        warm = torch.cuda.Stream(device=device)
        warm.wait_stream(current)
        with torch.cuda.stream(warm), torch.inference_mode():
            for size, out in engine.outputs.items():
                for i in range(size):
                    slots[i].select(out[i], engine.tokens[i])
        current.wait_stream(warm)
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode():
            for size, out in sorted(engine.outputs.items()):
                for i in range(size):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode='thread_local'):
                        slots[i].select(out[i], engine.tokens[i])
                    self.graphs[size, i] = graph
        # Leave the static buffers in a clean state.
        for slot in slots:
            slot.chosen.zero_()
            slot.bad.zero_()

    def lookup(self, logits):
        for size, out in self.engine.outputs.items():
            if logits.data_ptr() == out.data_ptr() and len(logits) == len(out):
                return size
        return None


def consume_batch(slots, logits, tokens, count, graphs=None):
    """`[slot.consume(logits[i], tokens[i]) for i < count]` with every staged solid step enqueued before the first
    host wait. Returns {index: (finished, commit)} for the non-idle slots. Results equal per-slot `consume`."""
    size = graphs.lookup(logits) if graphs is not None else None
    staged, results = {}, {}
    for index in range(count):
        slot = slots[index]
        if slot.idle:
            continue
        if isinstance(slot, FastLowEntropyLastSlot) and slot.stageable():
            graph = graphs.graphs.get((size, index)) if size is not None else None
            staged[index] = slot.stage(logits[index], tokens[index], graph)
        else:
            results[index] = slot.consume(logits[index], tokens[index])
    for index, item in staged.items():
        results[index] = slots[index].collect(item)
    return dict(sorted(results.items()))

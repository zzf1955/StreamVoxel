"""GPU benchmark and numerics for `fast_ops` (GPU machine only; never on the Mac).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/bench_ops.py <mode> --run <prepared run dir> [--out <json>]

`--run` is a run directory after `python -m streamvoxel.stream.session prepare` (it holds `panel/` and `config.json`).
`--base` / `--control` default to the checkpoints named in that `config.json`; `--world` and `--box` pick the panel world
and the initial box z0 x0 z1 x1 the conditions are cut from.

Modes (the data behind the system ablation):
  profile  SlotForward (or FastSlotForward with --config) graph time per batch bucket 1/2/4/8, per-module CUDA-event
           split and top kernels at b4/b8.
  speed    Interleaved A/B graph timing of SlotForward against the --configs list of FastSlotForward variants
           (min/median over --rounds); this is the 281 -> 44 ms forward comparison.
  forward  Numerics and timing of every FastSlotForward config: patchify gather vs one-hot conv, logits against the
           unpadded exact reference (argmax agreement, |dlogits|), bucket-4 vs bucket-8, local K/V cache bitwise check.
  sample   End-to-end sampling of --windows windows: SlotForward/FastSlotForward x LowEntropyLastSlot/FastLowEntropyLastSlot
           (plain, batched, CUDA-graph select, Triton entropy); wall time per forward and per consume, token agreement
           against the reference run, repeat/other-seed controls and accuracy against the ground truth.
  gate     The worker's numerical gate (padded CUDA graph vs unpadded forward) on the run's own initial frontier,
           for SlotForward and the --configs variants.
  fp8      Measurement only: FP8 (E4M3) backbone GEMMs against the BF16 fast path; --configs mlp|all picks the scope.

Conditions are real: GT cells of one panel world (read only) and `make_item` targets with 2..8 L1 tiles.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

RUN = BASE = CONTROL = WORLD = BOX = None   # set from the command line in `main`
# (target, L1 neighbours kept): spread over the 8x8 GT canvas, so each slot sees a different key count.
TARGETS = [((5, 5), 3), ((5, 9), 5), ((9, 5), 8), ((9, 9), 2), ((7, 7), 4), ((5, 7), 6), ((9, 7), 3), ((7, 5), 5)]


def build_items(spec, *, cap=64, l2_cap=256, targets=TARGETS, truth=False):
    from streamvoxel.stream.scheduler import make_item
    from streamvoxel.stream.session import Store
    from streamvoxel.stream.condition import Mapping
    mapping = Mapping.from_json(RUN / 'panel' / 'compact_mapping.json')
    cells, biomes = Store(RUN, mapping).load(WORLD, BOX)
    rng = np.random.default_rng(0)
    items, gts = [], []
    for (z, x), keep in targets:
        near = [(a, b) for a in range(z - 1, z + 2) for b in range(x - 1, x + 2) if (a, b) != (z, x)]
        drop = [near[i] for i in rng.permutation(8)[:8 - keep]]
        view = {c: v for c, v in cells.items() if c != (z, x) and c not in drop}
        items.append(make_item(view, (z, x), biomes, spec, WORLD, cap, l2_cap))
        gts.append(np.asarray(cells[(z, x)], np.int64))
    return (items, gts) if truth else items


def half_masked(gts, classes, device, seed=1):
    """GT targets with half the voxels masked: exercises every patchify class, not only the MASK token."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for gt in gts:
        t = torch.as_tensor(gt)
        t = torch.where(torch.rand(t.shape, generator=g) < .5, torch.full_like(t, classes), t)
        out.append(t)
    return torch.stack(out).to(device)


def load(device, batch=8):
    from streamvoxel.stream.engine import open_model
    torch.backends.cuda.matmul.allow_tf32 = True
    model, spec, _ = open_model(BASE, CONTROL, device)
    return model, spec


def cuda_ms(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return dict(median=round(times[len(times) // 2], 3), min=round(times[0], 3))


# ----------------------------------------------------------------------------------------------- profile

def module_profile(engine, size, iters=5):
    """Eager forward with CUDA events around modules. Inner categories are subsets of their parent."""
    model = engine.model
    bb, cc = model.backbone, model.control
    groups = {}

    def tag(module, name):
        def pre(_m, _a):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            _m._ev = event

        def post(_m, _a, _o):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            groups.setdefault(name, []).append((_m._ev, event))
        return [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]

    handles = []
    handles += tag(bb.target_patch, 'patchify.backbone_conv')
    handles += tag(cc.target_patch, 'patchify.control_conv')
    handles += tag(bb.class_head, 'head.class_head')
    for block in bb.blocks:
        handles += tag(block, 'backbone.block')
        handles += tag(block.attn, 'backbone.block.attn')
        handles += tag(block.attn.qkv, 'backbone.block.attn.qkv')
        handles += tag(block.attn.proj, 'backbone.block.attn.proj')
        handles += tag(block.mlp, 'backbone.block.mlp')
    for block in cc.blocks:
        handles += tag(block, 'control.block')
        handles += tag(block.attn, 'control.block.attn')
        handles += tag(block.mlp, 'control.block.mlp')
    for cross in model.cross:
        handles += tag(cross, 'cross')
        handles += tag(cross.query, 'cross.query')
        handles += tag(cross.key_value, 'cross.key_value')
        handles += tag(cross.proj, 'cross.proj')
    for zero in model.zero:
        handles += tag(zero, 'zero')
    totals = {}
    try:
        with torch.inference_mode():
            for it in range(iters + 1):
                groups.clear()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                engine._call(size)
                b.record()
                torch.cuda.synchronize()
                if it == 0:
                    continue
                totals.setdefault('total', []).append(a.elapsed_time(b))
                for name, pairs in groups.items():
                    totals.setdefault(name, []).append(sum(s.elapsed_time(e) for s, e in pairs))
    finally:
        for h in handles:
            h.remove()
    return {k: round(float(np.median(v)), 3) for k, v in totals.items()}


def kernel_profile(fn, top=25):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        t = getattr(event, 'device_time_total', None) or getattr(event, 'cuda_time_total', 0)
        if t:
            rows.append((t / 1e3, event.count, event.key[:110]))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    return dict(total_ms=round(total, 3), top=[dict(ms=round(a, 3), share=round(a / total, 3), count=c, name=n) for a, c, n in rows[:top]])


def cmd_profile(args):
    from streamvoxel.stream.engine import SlotForward
    from streamvoxel.stream.fast_ops import FastSlotForward
    device = torch.device('cuda:0')
    model, spec = load(device)
    items = build_items(spec)
    if args.config:
        engine = FastSlotForward(model, spec, device, 8, 8, 256, **CONFIGS[args.config])
    else:
        engine = SlotForward(model, spec, device, 8, 8, 256)
    for i, item in enumerate(items):
        engine.load_slot(i, item)
    engine.wait_loads()
    torch.cuda.synchronize()
    out = dict(padded_keys=engine.prefix_len + engine.local_len)
    if not args.config:
        out['valid_keys'] = [int(engine.prefix_valid[i].sum() + engine.local_valid[i].sum()) for i in range(8)]
    for size in (1, 2, 4, 8):
        out[f'graph_b{size}'] = cuda_ms(lambda: engine.replay(torch.ones(8), size))
    for size in (4, 8):
        if not (args.config and CONFIGS[args.config]['compile']):   # hooks would break the compiled blocks
            out[f'modules_b{size}'] = module_profile(engine, size)
        with torch.inference_mode():
            out[f'kernels_b{size}'] = kernel_profile(lambda: engine.replay(torch.ones(8), size))
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------- forward

CONFIGS = {
    'kv_sdpa': dict(patchify='conv', attention='sdpa', compile=False),
    'kv_flex': dict(patchify='conv', attention='flex', compile=False),
    'kv_sdpa_gather': dict(patchify='gather', attention='sdpa', compile=False),
    'kv_flex_gather': dict(patchify='gather', attention='flex', compile=False),
    'kv_flex_gather_compile': dict(patchify='gather', attention='flex', compile=True),
    'kv_flex_gather_compile_emulate': dict(patchify='gather', attention='flex', compile='emulate'),
}


def cmd_speed(args):
    """Interleaved graph timing (GPU 3 is shared: A/B/A/B rounds, min and median over rounds)."""
    from streamvoxel.stream.engine import SlotForward
    from streamvoxel.stream.fast_ops import FastSlotForward
    device = torch.device('cuda:0')
    model, spec = load(device)
    items = build_items(spec)
    engines = {'slot_forward': SlotForward(model, spec, device, 8, 8, 256)}
    for name in args.configs.split(','):
        engines[name] = FastSlotForward(model, spec, device, 8, 8, 256, **CONFIGS[name])
    for engine in engines.values():
        for i, item in enumerate(items):
            engine.load_slot(i, item)
        engine.wait_loads()
    torch.cuda.synchronize()
    samples = {name: {s: [] for s in (1, 2, 4, 8)} for name in engines}
    with torch.inference_mode():
        for _ in range(args.rounds):
            for name, engine in engines.items():
                for s in (1, 2, 4, 8):
                    samples[name][s].append(cuda_ms(lambda: engine.replay(torch.ones(8), s), iters=5, warm=1)['median'])
    report = {name: {f'b{s}': dict(min=round(min(v), 2), median=round(float(np.median(v)), 2)) for s, v in per.items()}
              for name, per in samples.items()}
    report['mem_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


def compare(ref, new, margin=0.125):
    """Per-slot logits contract against the reference eager logits [n,C,Y,Z,X].

    `argmax_clear` restricts the class match to voxels whose reference top-2 gap is >= `margin` logits (two BF16
    ulps at |logit|~8): mismatches below it are near-ties that any re-ordered BF16 reduction can flip."""
    rows = []
    for a, b in zip(ref, new):
        gap = (a.float() - b.float()).abs()
        same = a.argmax(0) == b.argmax(0)
        top = a.float().topk(2, 0).values
        clear = (top[0] - top[1]) >= margin
        rows.append(dict(argmax=round(float(same.float().mean()), 6),
                         argmax_clear=round(float(same[clear].float().mean()), 6),
                         mean_abs=round(float(gap.mean()), 5), max_abs=round(float(gap.max()), 4),
                         equal=bool(torch.equal(a, b))))
    return dict(worst_argmax=min(r['argmax'] for r in rows), mean_argmax=round(float(np.mean([r['argmax'] for r in rows])), 6),
                worst_argmax_clear=min(r['argmax_clear'] for r in rows),
                worst_mean_abs=max(r['mean_abs'] for r in rows), all_equal=all(r['equal'] for r in rows), slots=rows)


def exact_reference(model, spec, items, tokens):
    """`engine.unpadded_logits` per item: unmasked attention over the exact keys (no padding)."""
    from streamvoxel.stream.engine import unpadded_logits
    outs = []
    with torch.inference_mode():
        for i, item in enumerate(items):
            outs.append(unpadded_logits(model, item, spec, tokens[i:i + 1], torch.ones(1, device=tokens.device))[0].clone())
    return torch.stack(outs)


def timings(engine, sizes=(1, 2, 4, 8)):
    return {f'b{s}': cuda_ms(lambda s=s: engine.replay(torch.ones(engine.batch), s)) for s in sizes}


def cmd_forward(args):
    from streamvoxel.stream.engine import SlotForward
    from streamvoxel.stream.fast_ops import FastSlotForward, GatherPatch, conv_patch
    device = torch.device('cuda:0')
    model, spec = load(device)
    items, gts = build_items(spec, truth=True)
    tokens = half_masked(gts, spec.num_classes, device)
    report = dict(gpu=torch.cuda.get_device_name(0))
    # patchify alone, on half-masked GT tokens
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for name, conv in (('backbone', model.backbone.target_patch), ('control', model.control.target_patch)):
            a = conv_patch(conv, spec.num_classes)(tokens)
            b = GatherPatch(conv)(tokens)
            report[f'patchify_{name}'] = dict(equal_frac=float((a == b).float().mean()), max_abs=float((a.float() - b.float()).abs().max()),
                                              conv_ms=cuda_ms(lambda: conv_patch(conv, spec.num_classes)(tokens)),
                                              gather_ms=cuda_ms(lambda g=GatherPatch(conv): g(tokens)))
    print(json.dumps(report), flush=True)
    base = SlotForward(model, spec, device, 8, 8, 256)
    for i, item in enumerate(items):
        base.load_slot(i, item)
    base.wait_loads()
    base.tokens.copy_(tokens)
    with torch.inference_mode():
        ref = base.eager(torch.ones(8)).clone()
        ref_masked = None
        report['baseline'] = dict(timing=timings(base), graph_equal=bool(torch.equal(base.replay(torch.ones(8)), ref)))
        base.tokens.fill_(spec.num_classes)
        ref_masked = base.eager(torch.ones(8)).clone()
        masked = torch.full_like(tokens, spec.num_classes)
        exact_half = exact_reference(model, spec, items, tokens)
        exact_masked = exact_reference(model, spec, items, masked)
        floor = dict(half_masked=compare(exact_half, ref), all_masked=compare(exact_masked, ref_masked))
        for key in floor:
            floor[key].pop('slots') if not args.verbose else None
        report['baseline']['vs_unpadded'] = floor
        # Is the per-window local K/V cache bitwise what the slot forward recomputes on its padded batch?
        from streamvoxel.stream.fast_ops import local_kv
        from streamvoxel.stream.engine import encode_item
        same = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for j, cross in enumerate(model.cross):
                k_all, v_all = local_kv(cross, base.local, base.local_positions)
                for i, item in enumerate(items):
                    prefix, local, positions = encode_item(model, item, spec, device)
                    k1, v1 = local_kv(cross, local, positions)
                    n = local.shape[1]
                    same.append(bool(torch.equal(k1[0], k_all[i, :, :n]) and torch.equal(v1[0], v_all[i, :, :n])))
                    pk = base.prefix[j][0][i, :, :prefix[j][0].shape[1]]
                    same.append(bool(torch.equal(pk, prefix[j][0])))
        report['baseline']['kv_cache_bitwise'] = all(same)
    print(json.dumps(report['baseline']), flush=True)
    del base
    torch.cuda.empty_cache()
    for name in (args.configs.split(',') if args.configs else CONFIGS):
        t0 = time.time()
        engine = FastSlotForward(model, spec, device, 8, 8, 256, **CONFIGS[name])
        build_s = time.time() - t0
        for i, item in enumerate(items):
            engine.load_slot(i, item)
        engine.wait_loads()
        engine.tokens.copy_(tokens)
        with torch.inference_mode():
            eager = engine.eager(torch.ones(8)).clone()
            graph = engine.replay(torch.ones(8)).clone()
            small = engine.replay(torch.ones(8), 4)[:4].clone()
            row = dict(build_s=round(build_s, 1), graph_ok=engine.graph_ok, graph_equal_eager=bool(torch.equal(eager, graph)),
                       half_masked=compare(ref, graph), bucket4=compare(ref[:4], small), timing=timings(engine),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            row['half_masked_vs_unpadded'] = compare(exact_half, graph)
            engine.tokens.fill_(spec.num_classes)
            new_masked = engine.replay(torch.ones(8)).clone()
            row['all_masked'] = compare(ref_masked, new_masked)
            row['all_masked_vs_unpadded'] = compare(exact_masked, new_masked)
        for key in ('half_masked', 'bucket4', 'all_masked', 'half_masked_vs_unpadded', 'all_masked_vs_unpadded'):
            row[key].pop('slots') if not args.verbose else None
        report[name] = row
        print(json.dumps({name: row}), flush=True)
        del engine
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


# ----------------------------------------------------------------------------------------------- sampling

def run_windows(engine, spec, items, slot_kind, *, cap=64, batched=False, air=None, entropy='torch', graphs=False):
    """All items sampled together (fixed batch) through `engine` with LowEntropyLastSlot or FastLowEntropyLastSlot. Returns final values
    [n,Y,Z,X] int16, per-forward forward/consume wall ms."""
    from streamvoxel.stream.fast_ops import FastLowEntropyLastSlot, SelectGraphs, consume_batch
    from streamvoxel.stream.slot import LowEntropyLastSlot
    device = engine.device
    n = len(items)
    kwargs = dict(air_token=0, last_eps=1e-4, block_size=4, device=device)
    slots = [(FastLowEntropyLastSlot(engine.shape, spec.num_classes, air, entropy=entropy, **kwargs) if slot_kind == 'fast'
              else LowEntropyLastSlot(engine.shape, spec.num_classes, air, **kwargs)) for _ in range(n)]
    select = None
    if graphs:
        slots += [FastLowEntropyLastSlot(engine.shape, spec.num_classes, air, entropy=entropy, **kwargs) for _ in range(engine.batch - n)]
        select = SelectGraphs(engine, slots)
    engine.clear_tokens()
    for index, item in enumerate(items):
        engine.load_slot(index, item)
        slots[index].arm(item.seed, cap, (index, 0))
    finished = {}
    fwd, con = [], []
    with torch.inference_mode():
        while len(finished) < n:
            if len(fwd) > 4 * cap:
                raise RuntimeError('window did not finish')
            engine.wait_loads()
            count = 1 + max(i for i, s in enumerate(slots) if not s.idle)
            times = [s.time() if i < count else 1.0 for i, s in enumerate(slots)] + [1.0] * (engine.batch - n)
            t0 = time.perf_counter()
            logits = engine.replay(times, count)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if batched:
                results = consume_batch(slots, logits, engine.tokens, count, select)
            else:
                results = {i: slots[i].consume(logits[i], engine.tokens[i]) for i in range(count) if not slots[i].idle}
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for i, (done, _commit) in results.items():
                if done is not None:
                    finished[i] = done
                    engine.tokens[i].fill_(spec.num_classes)
            fwd.append((t1 - t0) * 1e3)
            con.append((t2 - t1) * 1e3)
    values = torch.stack([finished[i]['values'] for i in range(n)])
    nfe = [finished[i]['nfe'] for i in range(n)]
    return values, dict(forwards=len(fwd), forward_ms=round(float(np.mean(fwd)), 2), consume_ms=round(float(np.mean(con)), 2),
                        consume_ms_per_slot=round(float(np.sum(con)) / sum(nfe), 3), nfe=nfe)


def agreement(a, b):
    same = (a == b).float()
    return dict(token_agreement=round(float(same.mean()), 6), worst_window=round(float(same.flatten(1).mean(1).min()), 6),
                identical=bool(torch.equal(a, b)))


def cmd_sample(args):
    from streamvoxel.stream.engine import SlotForward
    from streamvoxel.stream.fast_ops import FastSlotForward, entropy_torch, entropy_triton, quota_select_nosync
    from streamvoxel.stream.session import Store
    from streamvoxel.stream.condition import Mapping
    device = torch.device('cuda:0')
    model, spec = load(device)
    air = tuple(Store(RUN, Mapping.from_json(RUN / 'panel' / 'compact_mapping.json')).manifest['air_compact_ids'])
    items = build_items(spec, cap=args.cap)[:args.windows]
    report = dict(cap=args.cap, windows=len(items))
    base = SlotForward(model, spec, device, 8, 8, 256)
    # entropy kernel alone, on real logits
    for i, item in enumerate(items):
        base.load_slot(i, item)
    base.wait_loads()
    with torch.inference_mode():
        logits = base.replay(torch.ones(8)).clone()
        a, b = entropy_torch(logits[0]), entropy_triton(logits[0])
        report['entropy'] = dict(torch_ms=cuda_ms(lambda: entropy_torch(logits[0])), triton_ms=cuda_ms(lambda: entropy_triton(logits[0])),
                                 max_abs=float((a - b).abs().max()), max_rel=float(((a - b).abs() / a.abs().clamp(min=1e-6)).max()),
                                 equal_frac=float((a == b).float().mean()))
    del logits
    print(json.dumps(report), flush=True)
    import dataclasses
    reseeded = [dataclasses.replace(item, seed=item.seed + 1) for item in items]
    truth = torch.stack([torch.as_tensor(g) for g in build_items(spec, cap=args.cap, truth=True)[1][:len(items)]])
    runs = {}
    kw = dict(cap=args.cap, air=air)
    runs['old'] = run_windows(base, spec, items, 'old', **kw)
    print('old', json.dumps(runs['old'][1]), flush=True)
    runs['old_repeat'] = run_windows(base, spec, items, 'old', **kw)
    runs['old_other_seed'] = run_windows(base, spec, reseeded, 'old', **kw)
    half = run_windows(base, spec, items[:4], 'old', **kw)
    report['old_bucket4_vs_bucket8'] = agreement(runs['old'][0][:4], half[0])
    runs['old_fwd_fast_sample'] = run_windows(base, spec, items, 'fast', batched=True, **kw)
    runs['old_fwd_fast_sample_graph'] = run_windows(base, spec, items, 'fast', batched=True, graphs=True, **kw)
    runs['old_fwd_fast_sample_graph_triton'] = run_windows(base, spec, items, 'fast', batched=True, graphs=True, entropy='triton', **kw)
    del base
    torch.cuda.empty_cache()
    fast = FastSlotForward(model, spec, device, 8, 8, 256, **CONFIGS[args.config or 'kv_flex_gather'])
    runs['fast_fwd_old_sample'] = run_windows(fast, spec, items, 'old', **kw)
    runs['fast_fwd_fast_sample_graph'] = run_windows(fast, spec, items, 'fast', batched=True, graphs=True, **kw)
    runs['fast_fwd_fast_sample_graph_triton'] = run_windows(fast, spec, items, 'fast', batched=True, graphs=True, entropy='triton', **kw)
    runs['fast_fwd_fast_sample_repeat'] = run_windows(fast, spec, items, 'fast', batched=True, graphs=True, **kw)
    runs['fast_fwd_other_seed'] = run_windows(fast, spec, reseeded, 'fast', batched=True, graphs=True, **kw)
    ref = runs['old'][0]
    for name, (values, stats) in runs.items():
        report[name] = dict(stats, vs_old=agreement(ref, values),
                            gt_accuracy=round(float((values.long() == truth).float().mean()), 5))
        print(name, json.dumps(report[name]), flush=True)
    print('old_bucket4_vs_bucket8', json.dumps(report['old_bucket4_vs_bucket8']), flush=True)
    report['fast_vs_fast_other_seed'] = agreement(runs['fast_fwd_fast_sample_graph'][0], runs['fast_fwd_other_seed'][0])
    report['fast_fwd_repeat_vs_fast_fwd'] = agreement(runs['fast_fwd_fast_sample_graph'][0], runs['fast_fwd_fast_sample_repeat'][0])
    report['fast_fwd_fast_vs_fast_fwd_old_sample'] = agreement(runs['fast_fwd_old_sample'][0], runs['fast_fwd_fast_sample_graph'][0])
    print(json.dumps({k: report[k] for k in ('fast_fwd_repeat_vs_fast_fwd', 'fast_fwd_fast_vs_fast_fwd_old_sample', 'fast_vs_fast_other_seed')}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


def cmd_gate(args):
    """The worker's `_gate` on the run's own initial frontier, for SlotForward and FastSlotForward."""
    from streamvoxel.stream.engine import SlotForward
    from streamvoxel.stream.fast_ops import FastSlotForward
    from streamvoxel.stream.scheduler import make_item, pick_window
    from streamvoxel.stream.session import Store
    from streamvoxel.stream.condition import Mapping
    device = torch.device('cuda:0')
    model, spec = load(device)
    cfg = json.loads((RUN / 'config.json').read_text())
    box = tuple(cfg['initial_box_zx'])
    cells, biomes = Store(RUN, Mapping.from_json(RUN / 'panel' / 'compact_mapping.json')).load(cfg['world'], box)
    fallback = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    first = pick_window(set(cells), set(), fallback)
    second = pick_window(set(cells), {first}, fallback)
    cap = int(cfg['defaults']['steps'])
    item = make_item(cells, first, biomes, spec, cfg['world'], cap, cfg['l2_cap'])
    other = make_item(cells, second, biomes, spec, cfg['world'], cap, cfg['l2_cap'])
    report = {}
    base = SlotForward(model, spec, device, 8, cfg['l1_cap'], cfg['l2_cap'])
    report['slot_forward'] = base.numerical_gate(item, other)
    del base
    torch.cuda.empty_cache()
    for name in (args.configs.split(',') if args.configs else ['kv_flex_gather']):
        engine = FastSlotForward(model, spec, device, 8, cfg['l1_cap'], cfg['l2_cap'], **CONFIGS[name])
        report[name] = engine.numerical_gate(item, other)
        del engine
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


class FP8Linear(torch.nn.Module):
    """Measurement-only: dynamic tensorwise E4M3 activations x per-tensor E4M3 weights via `torch._scaled_mm`."""

    def __init__(self, linear):
        super().__init__()
        w = linear.weight.detach()
        self.w_scale = (w.abs().max().float() / 448.).clamp(min=1e-12)
        self.w = (w.float() / self.w_scale).to(torch.float8_e4m3fn).t()
        self.bias = linear.bias.detach().to(torch.bfloat16)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1]).to(torch.bfloat16)
        scale = (x.abs().max().float() / 448.).clamp(min=1e-12)
        q = (x.float() / scale).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(q, self.w, scale_a=scale, scale_b=self.w_scale, bias=self.bias, out_dtype=torch.bfloat16)
        return out.reshape(*shape[:-1], -1)


def cmd_fp8(args):
    """FP8 backbone GEMMs (optional item d): numerics against the BF16 fast path; scope via --configs mlp|all."""
    from streamvoxel.stream.fast_ops import FastSlotForward
    device = torch.device('cuda:0')
    model, spec = load(device)
    items, gts = build_items(spec, truth=True)
    tokens = half_masked(gts, spec.num_classes, device)
    report = {}
    engine = FastSlotForward(model, spec, device, 8, 8, 256, **CONFIGS['kv_flex_gather'])
    for i, item in enumerate(items):
        engine.load_slot(i, item)
    engine.wait_loads()
    engine.tokens.copy_(tokens)
    with torch.inference_mode():
        ref = engine.replay(torch.ones(8)).clone()
        report['bf16_ms'] = timings(engine, (4, 8))
    del engine
    torch.cuda.empty_cache()
    for scope in (args.configs or 'mlp,all').split(','):
        for block in model.backbone.blocks:
            names = ['mlp.0', 'mlp.2'] + (['attn.qkv', 'attn.proj'] if scope == 'all' else [])
            for name in names:
                parent, child = name.split('.')
                module = getattr(block, parent)
                original = getattr(module, '_bf16_' + child, None) or module._modules[child]
                setattr(module, '_bf16_' + child, original)
                module._modules[child] = FP8Linear(original)
        engine = FastSlotForward(model, spec, device, 8, 8, 256, **CONFIGS['kv_flex_gather'])
        for i, item in enumerate(items):
            engine.load_slot(i, item)
        engine.wait_loads()
        engine.tokens.copy_(tokens)
        with torch.inference_mode():
            out = engine.replay(torch.ones(8)).clone()
            row = compare(ref, out)
            row.pop('slots')
            row['ms'] = timings(engine, (4, 8))
        report[f'fp8_{scope}'] = row
        print(json.dumps({scope: row}), flush=True)
        del engine
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


def main():
    global RUN, BASE, CONTROL, WORLD, BOX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('profile', 'forward', 'sample', 'gate', 'speed', 'fp8'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--base')
    parser.add_argument('--control')
    parser.add_argument('--world', type=int, default=161)
    parser.add_argument('--box', type=int, nargs=4, default=(64, 64, 192, 192), metavar=('Z0', 'X0', 'Z1', 'X1'))
    parser.add_argument('--out')
    parser.add_argument('--configs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cap', type=int, default=64)
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--config', default=None)
    parser.add_argument('--rounds', type=int, default=10)
    args = parser.parse_args()
    RUN, WORLD, BOX = args.run, args.world, tuple(args.box)
    cfg = json.loads((RUN / 'config.json').read_text())
    BASE, CONTROL = args.base or cfg['base'], args.control or cfg['control']
    dict(profile=cmd_profile, forward=cmd_forward, sample=cmd_sample, gate=cmd_gate, speed=cmd_speed, fp8=cmd_fp8)[args.mode](args)


if __name__ == '__main__':
    main()

"""Paired four-condition validation and exactly-budgeted full-MASK generation."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from streamvoxel.common.distributed import all_reduce_sum
from streamvoxel.common.io import atomic_json
from streamvoxel.data.streaming_controlnet import microbatches, to_device
from streamvoxel.objectives.md4 import md4_mask, strata_mask, weighted_masked_ce_sums
from streamvoxel.train.runtime import autocast, rng_state, restore_rng


@torch.no_grad()
def sample_full_mask(model, group, encoded, indices, *, steps=1024, seed=20262011, case_ids=None, progress=None):
    device = group["targets"].device
    shape = tuple(group["targets"].shape[1:])
    classes = model.backbone.num_classes
    tokens = torch.full((len(indices), *shape), classes, dtype=torch.long, device=device)
    case_ids = list(range(len(indices))) if case_ids is None else case_ids
    generators = [torch.Generator(device=device).manual_seed(seed + int(i)) for i in case_ids]
    prefix = model.prefix_kv(encoded, group)
    local, lp = model.local_condition(encoded, group, indices)
    ce = None
    for i in range(steps, 0, -1):
        with autocast(device):
            logits = model(tokens, timestep=torch.full((len(indices),), i / steps, device=device),
                origin_xyz=group["target_origins"][indices], group_origin_xyz=group["group_origin"],
                prefix_kv=prefix, local=local, local_positions=lp)
        if i == steps:
            ce = F.cross_entropy(logits.float(), group["targets"][indices].long(), reduction="none").flatten(1).mean(1)
        probability = 1. if i == 1 else (math.sin(math.pi*i/(2*steps))-math.sin(math.pi*(i-1)/(2*steps))) / math.sin(math.pi*i/(2*steps))
        for j, generator in enumerate(generators):
            selected = (tokens[j] == classes) & (torch.rand(shape, device=device, generator=generator) < probability)
            if selected.any():
                distribution = logits[j].permute(1,2,3,0)[selected].float().softmax(-1)
                tokens[j][selected] = torch.multinomial(distribution,1,generator=generator).squeeze(-1)
        del logits
        if progress and (steps-i+1) % 64 == 0: progress(steps-i+1)
    if (tokens == classes).any(): raise AssertionError("Sampler left MASK")
    return tokens, ce, steps


def confusion_metrics(matrix, air_ids):
    m = matrix.double()
    gt, pred, tp = m.sum(1), m.sum(0), m.diag()
    support = gt + pred > 0
    f1 = 2 * tp / (gt + pred).clamp_min(1)
    nonair = support.clone()
    nonair[air_ids] = False
    return dict(acc=float(tp.sum()/gt.sum().clamp_min(1)),
        macro_f1=float(f1[support].mean()) if support.any() else None,
        macro_f1_nonair=float(f1[nonair].mean()) if nonair.any() else None,
        supported_classes=int(support.sum()), voxels=int(gt.sum()))


@torch.no_grad()
def evaluate(model, dataset, plans, env, *, microbatch, full_steps=1024,
             ordinary_limit=None, full_limit=None, output=None, step=0, air_ids=None):
    previous = model.training
    state = rng_state(env.device)
    model.eval()
    started = time.perf_counter()
    # variant, N, [weighted CE numerator, correct, masked, sample count].
    totals = torch.zeros((4,3,4), dtype=torch.float64, device=env.device)
    strata_totals = torch.zeros((3,4), dtype=torch.float64, device=env.device)
    worlds = dataset.manifest["splits"]["validation"]
    matrices = torch.zeros((1+3+len(worlds),304,304),dtype=torch.float64,device=env.device)
    summaries = torch.zeros((1+3+len(worlds),2),dtype=torch.float64,device=env.device)
    records = []
    try:
        ordinary = plans["ordinary"] if ordinary_limit is None else plans["ordinary"][:ordinary_limit]
        for pi in range(env.rank, len(ordinary), env.world_size):
            group = to_device(dataset.gather(ordinary[pi]), env.device)
            with autocast(env.device): encoded = model.encode_group(group)
            corruptions = [md4_mask(t[None].long(),304,
                generator=torch.Generator().manual_seed(20261711+pi*1000+j)) for j,t in enumerate(group["targets"].cpu())]
            x, mask, t, weight = [torch.cat([c[k] for c in corruptions]).to(env.device) for k in range(4)]
            for vi, drop in enumerate((None,"L1","L2","biome")):
                with autocast(env.device): prefix = model.prefix_kv(encoded,group,drop=drop)
                for ids in microbatches(group,microbatch):
                    with autocast(env.device):
                        local,lp=model.local_condition(encoded,group,ids,drop=drop)
                        logits=model(x[ids],timestep=t[ids],origin_xyz=group["target_origins"][ids],
                            group_origin_xyz=group["group_origin"],prefix_kv=prefix,local=local,local_positions=lp)
                    loss,correct,count=weighted_masked_ce_sums(logits,group["targets"][ids].long(),mask[ids],weight[ids])
                    n=int(group["n"][ids[0]])-1
                    totals[vi,n]+=torch.stack((loss.double(),correct.double(),count.double(),count.new_tensor(len(ids)).double()))
                    del logits
                del prefix
            if air_ids is not None:
                # Same fixed targets, height-strata corruption, full conditioning only.
                strata = [strata_mask(t[None].long(),304,air_ids,band=8,
                    generator=torch.Generator().manual_seed(20261811+pi*1000+j)) for j,t in enumerate(group["targets"].cpu())]
                xs, ms, ts, ws = [torch.cat([c[k] for c in strata]).to(env.device) for k in range(4)]
                with autocast(env.device): prefix = model.prefix_kv(encoded,group,drop=None)
                for ids in microbatches(group,microbatch):
                    with autocast(env.device):
                        local,lp=model.local_condition(encoded,group,ids,drop=None)
                        logits=model(xs[ids],timestep=ts[ids],origin_xyz=group["target_origins"][ids],
                            group_origin_xyz=group["group_origin"],prefix_kv=prefix,local=local,local_positions=lp)
                    loss,correct,count=weighted_masked_ce_sums(logits,group["targets"][ids].long(),ms[ids],ws[ids])
                    n=int(group["n"][ids[0]])-1
                    strata_totals[n]+=torch.stack((loss.double(),correct.double(),count.double(),count.new_tensor(len(ids)).double()))
                    del logits
                del prefix
            if output and env.is_main:
                atomic_json(Path(output)/"validation_progress.json",dict(step=step,phase="ordinary",context=pi+1,contexts=len(ordinary)))
        all_reduce_sum(totals,env);all_reduce_sum(strata_totals,env)
        ordinary_seconds=time.perf_counter()-started
        start_full=time.perf_counter()
        full=plans["full_mask"] if full_limit is None else plans["full_mask"][:full_limit]
        for pi in range(env.rank,len(full),env.world_size):
            plan=full[pi];group=to_device(dataset.gather(plan),env.device)
            with autocast(env.device):encoded=model.encode_group(group)
            for ids in microbatches(group,microbatch):
                case_ids=[pi*1000+int(i) for i in ids.cpu()]
                def report(call):
                    if output and env.is_main: atomic_json(Path(output)/"validation_progress.json",
                        dict(step=step,phase="full_mask",context=pi+1,contexts=len(full),calls=call,call_budget=full_steps))
                pred,ce,calls=sample_full_mask(model,group,encoded,ids,steps=full_steps,case_ids=case_ids,progress=report)
                for j,idx in enumerate(ids.cpu().tolist()):
                    matrix=torch.bincount((group["targets"][idx].long()*304+pred[j]).flatten(),minlength=304**2).reshape(304,304).double()
                    n=int(group["n"][idx]);wi=plan["world"]
                    rows=[0,n,4+worlds.index(wi)]
                    for row in rows:
                        matrices[row]+=matrix
                        summaries[row]+=torch.stack((ce[j].double(),ce.new_tensor(1).double()))
                    records.append(dict(world=wi,n=n,case_id=case_ids[j],acc=float(matrix.diag().sum()/matrix.sum()),
                                        ce=float(ce[j]),forward_calls=calls))
                if output:
                    folder=Path(output)/f"samples_step{step:06d}";folder.mkdir(exist_ok=True)
                    # uint16 global IDs meet the preview contract, never export compact IDs.
                    for j,cid in enumerate(case_ids):
                        np.save(folder/f"case_{cid:06d}.npy",np.asarray(dataset.mapping["compact_to_global"],dtype=np.uint16)[pred[j].cpu().numpy()])
        all_reduce_sum(matrices,env);all_reduce_sum(summaries,env)
        air=[]
        names=json.loads((dataset.root/dataset.manifest["vocabulary_file"]).read_text())["idToCleanName"]
        for i,g in enumerate(dataset.mapping["compact_to_global"]):
            name=names[g] if isinstance(names,list) else names[str(g)]
            if name in {"air","cave_air","void_air","minecraft:air","minecraft:cave_air","minecraft:void_air"}:air.append(i)
        ordinary_result={}
        for vi,name in enumerate(("full","no_L1","no_L2","no_biome")):
            ordinary_result[name]={str(n+1):dict(loss=float(row[0]/row[2].clamp_min(1)),acc=float(row[1]/row[2].clamp_min(1)),
                samples=int(row[3]),masked_voxels=int(row[2])) for n,row in enumerate(totals[vi])}
        strata_result={str(n+1):dict(loss=float(row[0]/row[2].clamp_min(1)),acc=float(row[1]/row[2].clamp_min(1)),
            samples=int(row[3]),masked_voxels=int(row[2])) for n,row in enumerate(strata_totals)} if air_ids is not None else None
        full_result={}
        for row,name in enumerate(["all","N1","N2","N3"]+[f"world_{w:03d}" for w in worlds]):
            full_result[name]=dict(**confusion_metrics(matrices[row],air),ce=float(summaries[row,0]/summaries[row,1].clamp_min(1)),
                samples=int(summaries[row,1]))
        if env.enabled:
            all_records=[None]*env.world_size
            torch.distributed.all_gather_object(all_records,records)
            records=sum(all_records,[])
        timing=torch.tensor([ordinary_seconds,time.perf_counter()-start_full,time.perf_counter()-started],
                            device=env.device,dtype=torch.float64)
        if env.enabled:torch.distributed.all_reduce(timing,op=torch.distributed.ReduceOp.MAX)
        result=dict(step=step,ordinary=ordinary_result,full_mask=full_result,full_mask_steps=full_steps,
                    ordinary_seconds=float(timing[0]),full_mask_seconds=float(timing[1]),seconds=float(timing[2]),records=records)
        if strata_result is not None:result['ordinary_strata']=strata_result
        if output and env.is_main:
            atomic_json(Path(output)/f"validation_step{step:06d}.json",result)
            atomic_json(Path(output)/"validation_latest.json",result)
            np.save(Path(output)/f"fullmask_confusion_step{step:06d}.npy",matrices.cpu().numpy())
        return result
    finally:
        model.train(previous);restore_rng(state,env.device)

"""Fresh/resumable blocks-only MD4 training and synchronized real-data DDP timing."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import itertools
import json
import os
from pathlib import Path
import random
import signal
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from streamvoxel.common.distributed import init_distributed, all_reduce_sum, barrier, cleanup
from streamvoxel.common.io import atomic_json, append, sha256_file
from streamvoxel.common.unconditional_tiles import validate_tiles
from streamvoxel.data.unconditional_tiles import UnconditionalTileDataset
from streamvoxel.models.unconditional_md4_timefree import UnconditionalMD4DiT
from streamvoxel.objectives.md4 import weighted_masked_ce_sums
from streamvoxel.train.runtime import ShuffledCursor, autocast, rng_state, restore_rng
from streamvoxel.data.batch_prefetch import BatchPrefetcher,prepare_batch
from streamvoxel.train.config import parse_with_config
from streamvoxel.train.ema_control import read_ema_control
from streamvoxel.train.lr_control import read_lr_control
from streamvoxel.train.validation_base import evaluate, select_eval_worlds

FORMAT = "java26_unconditional_md4_patch4_preln_v1"
_cluster_stop = False


def _request_cluster_stop(_signum, _frame):
    global _cluster_stop
    _cluster_stop = True


def apply_lr_control(optimizer, run, env, fallback):
    """Rank 0 reads lr_control.json; every rank steps with that rate. Moments stay."""
    requested = read_lr_control(run, fallback=fallback) if env.is_main else None
    if env.enabled:
        box = [requested]
        dist.broadcast_object_list(box, src=0)
        requested = box[0]
    current = optimizer.param_groups[0]["lr"]
    if requested == current:
        return False
    for group in optimizer.param_groups:
        group["lr"] = requested
    return True


def validation_due(step,*,first_step=1,every=1000):
    return step>0 and (step==first_step or (every>0 and step%every==0))


def training_steps(start, stop):
    """stop=0 keeps training until a stop request or a wall-clock budget."""
    if start<0 or stop<0:raise ValueError("Negative training step")
    return itertools.count(start+1) if stop==0 else range(start+1,stop+1)


def _weight_bucket(key):
    parts=key.split(".")
    if parts[0]=="blocks" and len(parts)>2:
        return f"blocks.{parts[2]}"
    return parts[0]


def apply_init_weights(model,path):
    """Copy same-shaped tensors. Drop removed modules and keep new parameters at initialization."""
    saved=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    state=saved["model"] if isinstance(saved,dict) and "model" in saved else saved
    current={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    skipped=[]
    for key,value in state.items():
        if key not in current:
            skipped.append(key)
            continue
        if tuple(current[key].shape)!=tuple(value.shape):
            raise ValueError(f"Init weight shape mismatch for {key}: checkpoint {tuple(value.shape)} model {tuple(current[key].shape)}")
        current[key].copy_(value.detach().cpu())
    model.load_state_dict(current,strict=True)
    fresh=sorted(set(current)-set(state))
    return dict(copied=len(current)-len(fresh),
                skipped_tensors=len(skipped),
                skipped_buckets=sorted({_weight_bucket(key) for key in skipped}),
                fresh_buckets=sorted({_weight_bucket(key) for key in fresh}))


def restore_optimizer(optimizer,state,*,lr,previous_lr,override_lr=False):
    if lr!=previous_lr and not override_lr:
        raise ValueError("Resume LR differs; explicit --override-resume-lr required")
    optimizer.load_state_dict(state)
    if override_lr:
        for group in optimizer.param_groups:
            group['lr']=lr
            if 'initial_lr' in group:group['initial_lr']=lr


def half_mask_plan(global_batch,microbatch,world_size):
    """Half of the ranks draw random masks, the other half height-strata masks; every rank has equal work."""
    if world_size==1:
        # One rank alternates the role every optimizer step.
        if microbatch<=0 or global_batch%microbatch:raise ValueError("Global batch must split into microbatches")
        return [global_batch//microbatch],["alternate"]
    if world_size%2:
        raise ValueError("Half-mask needs one rank or an even number of ranks")
    if microbatch<=0 or global_batch%(microbatch*world_size):
        raise ValueError("Global batch must split evenly into microbatches across every rank")
    per_rank=global_batch//(microbatch*world_size)
    roles=["random"]*(world_size//2)+["strata"]*(world_size//2)
    return [per_rank]*world_size,roles


def air_compact_ids(dataset):
    mapping=json.loads((dataset.directory/dataset.manifest["tokenMapping"]["file"]).read_text())
    names=mapping["compact_to_clean_name"]
    ids=[i for i,name in enumerate(names) if name in ("minecraft:air","minecraft:cave_air","minecraft:void_air")]
    if not ids:
        raise ValueError("Compact mapping has no air class")
    return ids


class RankBatches:
    def __init__(self,size,seed,rank,world_size,microbatch,global_batch,start_step=0):
        if microbatch<=0 or global_batch%(microbatch*world_size):
            raise ValueError("Global batch must divide microbatch * world size")
        self.size,self.seed,self.rank,self.world_size = size,seed,rank,world_size
        self.microbatch,self.global_batch = microbatch,global_batch
        consumed = start_step*global_batch
        self.cursor = ShuffledCursor(size,seed,dict(size=size,seed=seed,
            epoch=consumed//size,offset=consumed%size))

    def __iter__(self):
        while True:
            indices = self.cursor.take(self.global_batch)
            unit=self.world_size*self.microbatch
            for j in range(self.global_batch//unit):
                start = (j*self.world_size+self.rank)*self.microbatch
                yield indices[start:start+self.microbatch]


def synchronized(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_step(model,optimizer,loader,env,*,accumulation,num_classes,grad_clip,prefetcher=None,mask_role=None,air_ids=None,strata_band=8):
    # Determine the exact global masked-voxel denominator before backpropagation.
    before = time.perf_counter()
    data_start_unix=time.time()
    ready=(prefetcher.take() if prefetcher is not None else prepare_batch(loader,
        accumulation=accumulation,num_classes=num_classes,
        mask_role=mask_role,air_ids=air_ids,strata_band=strata_band))
    prepared,denominator=ready["prepared"],ready["count"]
    load_seconds,mask_seconds=ready["loader_seconds"],ready["mask_seconds"]
    data_seconds = time.perf_counter()-before
    data_end_unix=time.time()
    denominator = torch.tensor(denominator,device=env.device,dtype=torch.float64)
    all_reduce_sum(denominator,env)
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3,device=env.device,dtype=torch.float64)
    # Rows are random, then height-strata: weighted CE, correct voxels, masked voxels.
    role=ready.get("role",mask_role)
    role_index={"random":0,"strata":1}.get(role)
    split = torch.zeros((2,3),device=env.device,dtype=torch.float64)
    split_samples = torch.zeros(2,device=env.device,dtype=torch.float64)
    for i,row in enumerate(prepared):
        batch,corrupted,mask,t,weight=row
        target = batch["target"].to(env.device,non_blocking=True)
        origin = batch["origin_xyz"].to(env.device,non_blocking=True)
        corrupted,mask,t,weight=[x.to(env.device,non_blocking=True) for x in (corrupted,mask,t,weight)]
        sync = model.no_sync() if env.enabled and i<accumulation-1 else nullcontext()
        with sync:
            with autocast(env.device):
                logits = model(corrupted,timestep=t,origin_xyz=origin)
            loss,correct,count = weighted_masked_ce_sums(logits,target,mask,weight)
            scale = 1/denominator
            (loss*(env.world_size*scale)).backward()
        values = torch.stack((loss.detach().double(),correct.double(),count.double()))
        totals += values
        if role_index is not None:
            split[role_index]+=values
            split_samples[role_index]+=len(target)
        del logits,loss
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip,error_if_nonfinite=True)
    optimizer.step()
    all_reduce_sum(totals,env)
    if role_index is not None:
        all_reduce_sum(split,env);all_reduce_sum(split_samples,env)
    data_max = torch.tensor(data_seconds,device=env.device,dtype=torch.float64)
    if env.enabled:
        dist.all_reduce(data_max,op=dist.ReduceOp.MAX)
    result = dict(train_md4_loss=float(totals[0]/totals[2]),
        train_masked_accuracy=float(totals[1]/totals[2]),masked_voxels=int(totals[2]),
        grad_norm=float(norm),data_seconds_max=float(data_max),
        rank0_data_start_unix=data_start_unix,rank0_data_end_unix=data_end_unix,
        rank0_loader_seconds=load_seconds,rank0_cpu_mask_seconds=mask_seconds)
    if role_index is not None:
        for prefix,row,samples in zip(("random","strata"),split,split_samples):
            result[f"train_{prefix}_samples"]=int(samples)
            result[f"train_{prefix}_masked_voxels"]=int(row[2])
            if row[2]>0:
                result[f"train_{prefix}_md4_loss"]=float(row[0]/row[2])
                result[f"train_{prefix}_masked_accuracy"]=float(row[1]/row[2])
    return result


@torch.no_grad()
def sample_unconditional(model,*,batch_size=1,steps=8,generator=None):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.full((batch_size,*model.voxel_shape),model.num_classes,device=device,dtype=torch.long)
    try:
        for i in range(steps,0,-1):
            unknown = tokens==model.num_classes
            with autocast(device):
                logits = model(tokens,timestep=torch.full((batch_size,),i/steps,device=device))
            probability = 1. if i==1 else (np.sin(np.pi*i/(2*steps))-np.sin(np.pi*(i-1)/(2*steps)))/np.sin(np.pi*i/(2*steps))
            selected = unknown & (torch.rand(tokens.shape,device=device,generator=generator)<probability)
            if selected.any():
                probs = logits.permute(0,2,3,4,1)[selected].float().softmax(-1)
                tokens[selected] = torch.multinomial(probs,1,generator=generator).squeeze(-1)
        if (tokens==model.num_classes).any():
            raise AssertionError("Sampling left MASK")
        return tokens
    finally:
        model.train(was_training)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",required=True)
    p.add_argument("--source-dir")
    p.add_argument("--read-backend",choices=("raw_tiles","world","compact_cache","zstd_cache"),default="raw_tiles")
    p.add_argument("--tile-cache-dir")
    p.add_argument("--batch-prefetch",type=int,default=0)
    p.add_argument("--loader-prefetch",type=int,default=2)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--resume")
    p.add_argument("--resume-reshard",action="store_true",
                   help="Allow resuming on a different world size; per-rank RNG is reseeded from the step")
    p.add_argument("--override-resume-lr",action="store_true",
                   help="Preserve optimizer moments but replace checkpoint learning rate")
    p.add_argument("--steps",type=int,default=15)
    p.add_argument("--max-wall-hours",type=float,default=0,
                   help="Stop, validate and checkpoint after this wall budget; 0 disables it")
    p.add_argument("--warmup-timing-steps",type=int,default=5)
    p.add_argument("--global-batch",type=int,default=512)
    p.add_argument("--microbatch",type=int,default=8)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--depth",type=int,default=32)
    p.add_argument("--hidden-dim",type=int,default=768)
    p.add_argument("--patch-size",type=int,choices=(2,4),default=4)
    p.add_argument("--mlp-ratio",type=float,default=4.)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--gradient-checkpointing",action="store_true")
    p.add_argument("--seed",type=int,default=20261009)
    p.add_argument("--eval-limit",type=int,default=0,help="0 evaluates the complete validation split")
    p.add_argument("--eval-every",type=int,default=1000,help="0 disables intermediate validation")
    p.add_argument("--eval-batch-size",type=int,default=32)
    p.add_argument("--validate-on-start",action="store_true")
    p.add_argument("--first-validation-step",type=int,default=1,
                   help="First validation after this optimizer step in a fresh run")
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--ema-decay",type=float,default=.99)
    p.add_argument("--ema-update-every",type=int,default=10)
    p.add_argument("--eval-worlds",type=int,default=16)
    p.add_argument("--init-weights",help="Load model weights only; fresh optimizer and EMA")
    p.add_argument("--time-conditioned",action="store_true",
                   help="Time-conditioned AdaLN backbone (stage 1)")
    p.add_argument("--timefree-arch",action="store_true",
                   help="Time-free AdaLN backbone with a constant time embedding (stage 2)")
    p.add_argument("--freeze-time-at",type=float,
                   help="With --timefree-arch and a time-conditioned --init-weights: write the time MLP at t into the constant")
    p.add_argument("--half-mask",action="store_true",
                   help="Half of the ranks use random masks, the other half height-strata masks")
    p.add_argument("--strata-band",type=int,default=8)
    p.add_argument("--eval-include-test",action="store_true",
                   help="Fill --eval-worlds from the test split after the validation Worlds")
    p.add_argument("--sample-steps",type=int,default=8)
    a = parse_with_config(p,argv)
    if (a.steps<0 or a.microbatch<=0 or a.global_batch<=0 or a.workers<0 or a.lr<=0
            or not np.isfinite(a.max_wall_hours) or a.max_wall_hours<0
            or not np.isfinite(a.mlp_ratio) or a.mlp_ratio<=0
            or a.warmup_timing_steps<0 or a.save_every<=0 or a.eval_limit<0
            or a.eval_every<0 or a.eval_batch_size<1 or a.eval_worlds<1 or a.first_validation_step<1
            or a.batch_prefetch<0 or a.loader_prefetch<1
            or not 0<a.ema_decay<1 or a.ema_update_every<=0 or a.sample_steps<=0
            or a.strata_band<1):
        p.error("Invalid positive training settings")
    if a.time_conditioned==a.timefree_arch:
        p.error("Choose exactly one of --time-conditioned or --timefree-arch")
    if a.freeze_time_at is not None and not (a.timefree_arch and a.init_weights):
        p.error("--freeze-time-at needs --timefree-arch and --init-weights")
    if a.init_weights and a.resume:
        p.error("Use either --init-weights or --resume")
    return a


def main():
    started_unix=time.time()
    args = parse_args()
    env = init_distributed()
    signal.signal(signal.SIGTERM, _request_cluster_stop)
    signal.signal(signal.SIGINT, _request_cluster_stop)
    prefetcher=None
    try:
        mask_counts=mask_roles=None
        if args.half_mask:
            mask_counts,mask_roles=half_mask_plan(args.global_batch,args.microbatch,env.world_size)
            accumulation=mask_counts[env.rank]
        elif args.global_batch%(args.microbatch*env.world_size):
            raise ValueError("Effective batch must divide microbatch * world size")
        else:
            accumulation = args.global_batch//(args.microbatch*env.world_size)
        run = Path(args.run_dir).resolve()
        if env.is_main:
            if run.exists():
                raise FileExistsError("Use a new run directory, including for resumed branches")
            run.mkdir(parents=True)
        barrier(env)
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        manifest = validate_tiles(args.data_dir,source_dir=args.source_dir,check_sources=env.is_main and args.read_backend!="zstd_cache")
        data = UnconditionalTileDataset(args.data_dir,"train",source_dir=args.source_dir,manifest=manifest,
                                        read_backend=args.read_backend,tile_cache_dir=args.tile_cache_dir)
        validation = UnconditionalTileDataset(args.data_dir,"validation",source_dir=args.source_dir,manifest=manifest,
                                              read_backend=args.read_backend,tile_cache_dir=args.tile_cache_dir)
        if args.strata_band and data.sample_shape[0]%args.strata_band:
            raise ValueError("Strata band must divide the column height")
        air_ids=air_compact_ids(data) if args.half_mask else None
        architecture=dict(num_classes=data.num_classes,hidden_dim=args.hidden_dim,patch_size=args.patch_size,
            depth=args.depth,num_heads=args.heads,mlp_ratio=args.mlp_ratio,
            gradient_checkpointing=args.gradient_checkpointing)
        raw = UnconditionalMD4DiT(**architecture,time_free=args.timefree_arch and args.freeze_time_at is None).to(env.device)
        init_transfer=None
        if args.init_weights:
            init_transfer=apply_init_weights(raw,args.init_weights)
        if args.freeze_time_at is not None:
            raw.freeze_time_at(args.freeze_time_at)
        ema = copy.deepcopy(raw).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(raw.parameters(),lr=args.lr,betas=(.9,.99),weight_decay=.01)
        identity = dict(manifestSha256=sha256_file(Path(args.data_dir)/"manifest.json"),
                        vocabularySha256=manifest["vocabulary"]["sha256"],
                        tokenMappingSha256=manifest["tokenMapping"]["sha256"])
        start = 0
        if args.resume:
            saved = torch.load(args.resume,map_location="cpu",weights_only=False)
            saved_training=saved["config"]["training"]
            # Keys of removed experiment switches in the saved config are ignored.
            mask_changed=any(saved_training.get(key,default)!=getattr(args,key)
                for key,default in (("global_batch",args.global_batch),("half_mask",False),("strata_band",8)))
            if (saved["format"]!=FORMAT or saved["dataset_identity"]!=identity
                    or saved["config"]["model"]!=raw.config()
                    or (saved_training["world_size"]!=env.world_size and not args.resume_reshard)
                    or saved_training["seed"]!=args.seed
                    or mask_changed):
                raise ValueError("Resume identity/config mismatch")
            raw.load_state_dict(saved["model"]); ema.load_state_dict(saved["ema"])
            restore_optimizer(optimizer,saved["optimizer"],lr=args.lr,
                previous_lr=saved["config"]["training"]["lr"],override_lr=args.override_resume_lr)
            start=saved["step"]
            if args.resume_reshard:
                reseed=args.seed+7919*start+1000003*env.rank
                torch.manual_seed(reseed);np.random.seed(reseed%2**32);random.seed(reseed)
            else:
                restore_rng(saved["rng_by_rank"][env.rank],env.device)
            del saved
        config = dict(format=FORMAT,model=raw.config(),dataset_identity=identity,
            training=dict(vars(args),world_size=env.world_size,accumulation=accumulation,
                lr_schedule="constant",warmup_steps=0,optimizer="AdamW",betas=[.9,.99],
                weight_decay=.01,grad_clip=1.,precision="bf16",start_step=start,
                objective="half_mask random + strata per rank" if args.half_mask else "column_MD4",
                lr_control_file="lr_control.json",
                parameters=sum(p.numel() for p in raw.parameters()),weights_source="raw",
                source_snapshot=str(Path.cwd()),torch_version=torch.__version__,
                started_unix=started_unix,
                deadline_unix=started_unix+args.max_wall_hours*3600 if args.max_wall_hours else None,
                source_manifest_sha256=sha256_file(Path.cwd()/"source_manifest.json")
                    if (Path.cwd()/"source_manifest.json").is_file() else None))
        if mask_counts:
            config["training"].update(accumulation_by_rank=mask_counts,mask_role_by_rank=mask_roles,
                air_compact_ids=air_ids)
        test_data=UnconditionalTileDataset(args.data_dir,"test",source_dir=args.source_dir,manifest=manifest,
            read_backend=args.read_backend,tile_cache_dir=args.tile_cache_dir) if args.eval_include_test else None
        eval_world_indices=select_eval_worlds(validation,args.eval_worlds,test=test_data)
        if args.init_weights:
            receipt=dict(path=str(Path(args.init_weights).resolve()),sha256=sha256_file(args.init_weights)) if env.is_main else None
            if env.enabled:
                box=[receipt];dist.broadcast_object_list(box,src=0);receipt=box[0]
            config["training"]["initialized_from"]=receipt["path"]
            config["training"]["initialized_from_sha256"]=receipt["sha256"]
            config["training"]["initialized_transfer"]=init_transfer
        if args.freeze_time_at is not None:
            config["training"]["time_free_from"]=args.freeze_time_at
        config["training"]["eval_world_ids"]=[manifest["worlds"][i]["id"] for i in eval_world_indices]
        if env.is_main:
            atomic_json(run/"config.json",config)
            print(json.dumps(dict(event="initialized",config=config)),flush=True)
        model = DDP(raw,device_ids=[env.local_rank],broadcast_buffers=False,
                    gradient_as_bucket_view=True) if env.enabled else raw
        def validate(step):
            barrier(env)
            if env.is_main:
                atomic_json(run/"progress.json",dict(status="validating",step=step,
                    cases=min(args.eval_limit,len(validation)) if args.eval_limit else len(validation)))
                print(json.dumps(dict(event="validation_started",step=step)),flush=True)
            def report(update):
                if env.is_main:
                    atomic_json(run/"validation_progress.json",dict(step=step,**update))
            result=evaluate(raw,validation,env,seed=args.seed+700,limit=args.eval_limit,
                batch_size=args.eval_batch_size,workers=args.workers,
                loader_prefetch=args.loader_prefetch,on_progress=report)
            if args.half_mask:
                strata_result=evaluate(raw,validation,env,seed=args.seed+700,limit=args.eval_limit,
                    batch_size=args.eval_batch_size,workers=args.workers,
                    loader_prefetch=args.loader_prefetch,on_progress=report,
                    air_ids=air_ids,strata_band=args.strata_band)
                random_result=result
                result=dict(random=random_result,strata=strata_result,cases=random_result['cases'],
                    validation_md4_loss=.5*(random_result['validation_md4_loss']+strata_result['validation_md4_loss']),
                    validation_masked_accuracy=.5*(random_result['validation_masked_accuracy']+strata_result['validation_masked_accuracy']),
                    validation_random_md4_loss=random_result['validation_md4_loss'],
                    validation_random_masked_accuracy=random_result['validation_masked_accuracy'],
                    validation_strata_md4_loss=strata_result['validation_md4_loss'],
                    validation_strata_masked_accuracy=strata_result['validation_masked_accuracy'])
            result.update(step=step)
            if env.is_main:
                atomic_json(run/f"validation_step{step:06d}.json",result)
                atomic_json(run/"validation.json",result)
                append(run/"validation_metrics.jsonl",result)
                atomic_json(run/"validation_progress.json",dict(step=step,status="complete",cases=result["cases"]))
                print(json.dumps(dict(event="validation_complete",**result)),flush=True)
            barrier(env)
            return result
        if start==0:validate(0)
        elif args.resume and args.validate_on_start:validate(start)
        if not args.resume:
            # Identical initialization, independent per-rank diffusion masks.
            torch.manual_seed(args.seed+1000003*env.rank)
            np.random.seed((args.seed+1000003*env.rank)%2**32)
            random.seed(args.seed+1000003*env.rank)
        sampler = RankBatches(len(data),args.seed,env.rank,env.world_size,args.microbatch,args.global_batch,start)
        options = dict(batch_sampler=sampler,num_workers=args.workers,pin_memory=env.device.type=="cuda",
                       generator=torch.Generator().manual_seed(args.seed+env.rank))
        if args.workers:
            options.update(persistent_workers=True,prefetch_factor=args.loader_prefetch)
        loader = iter(DataLoader(data,**options))
        if args.batch_prefetch and mask_roles is not None and mask_roles[env.rank]=="alternate":
            raise ValueError("Single-rank alternating masks need --batch-prefetch 0")
        fixed_role=None if mask_roles is None else mask_roles[env.rank]
        if args.batch_prefetch:
            prefetcher=BatchPrefetcher(loader,accumulation=accumulation,num_classes=data.num_classes,
                depth=args.batch_prefetch,pin=env.device.type=="cuda",device=env.device,
                mask_role=fixed_role,air_ids=air_ids,strata_band=args.strata_band)
        if env.device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(env.device)
        timings=[]

        def save(step):
            state = rng_state(env.device)
            states = [None]*env.world_size
            if env.enabled:
                dist.all_gather_object(states,state)
            else:
                states=[state]
            if env.is_main:
                seen=step*args.global_batch
                cursor=dict(size=len(data),seed=args.seed,epoch=seen//len(data),offset=seen%len(data))
                payload=dict(format=FORMAT,step=step,model=raw.state_dict(),ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(),config=config,dataset_identity=identity,
                    cursor=cursor,
                    rng_by_rank=states,weights_source="raw",uses_ema=True)
                p=run/f"checkpoint_step{step:06d}.pt"; tmp=p.with_suffix(".tmp")
                torch.save(payload,tmp);tmp.replace(p)
                alias=run/"checkpoint_latest.pt"; temp=run/"checkpoint_latest.tmp"
                os.link(p,temp);temp.replace(alias)
            barrier(env)

        step=start
        stop_reason=None
        ema_default=dict(enabled=True,decay=args.ema_decay)
        for step in training_steps(start,args.steps):
            barrier(env);synchronized(env.device)
            before=time.perf_counter()
            step_start_unix=time.time()
            if apply_lr_control(optimizer,run,env,args.lr) and env.is_main:
                print(json.dumps(dict(event="lr_updated",step=step,lr=optimizer.param_groups[0]["lr"])),flush=True)
            if mask_roles is None:
                step_role=None
            elif mask_roles[env.rank]=="alternate":
                step_role="random" if step%2 else "strata"
            else:
                step_role=mask_roles[env.rank]
            metrics=train_step(model,optimizer,loader,env,accumulation=accumulation,
                               num_classes=data.num_classes,grad_clip=1.,prefetcher=prefetcher,
                               mask_role=step_role,
                               air_ids=air_ids,strata_band=args.strata_band)
            ema_state=read_ema_control(run,fallback=ema_default) if env.is_main else None
            if env.enabled:
                box=[ema_state];dist.broadcast_object_list(box,src=0);ema_state=box[0]
            if ema_state["enabled"] and step%args.ema_update_every==0:
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),raw.parameters(),strict=True):
                        a.lerp_(b.detach(),1-ema_state["decay"])
                    for a,b in zip(ema.buffers(),raw.buffers(),strict=True):
                        a.copy_(b)
            synchronized(env.device)
            step_end_unix=time.time()
            seconds=torch.tensor(time.perf_counter()-before,device=env.device,dtype=torch.float64)
            if env.enabled: dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
            metrics.update(step=step,seconds=float(seconds),samples_per_second=args.global_batch/float(seconds),
                lr=optimizer.param_groups[0]["lr"],ema_enabled=ema_state["enabled"],ema_decay=ema_state["decay"],
                timing_warmup=step<=start+args.warmup_timing_steps,
                rank0_step_start_unix=step_start_unix,rank0_step_end_unix=step_end_unix)
            if step>start+args.warmup_timing_steps:timings.append(metrics)
            if env.is_main:
                append(run/"metrics.jsonl",metrics)
                atomic_json(run/"progress.json",dict(status="training",**metrics))
                print(json.dumps(metrics),flush=True)
            requested=torch.tensor(int(_cluster_stop or (env.is_main and (run/"stop_request.json").is_file())),device=env.device)
            all_reduce_sum(requested,env)
            if int(requested):
                stop_reason="signaled" if _cluster_stop else "requested"
                if env.is_main:print(json.dumps(dict(event="stop_requested",step=step,stop_reason=stop_reason)),flush=True)
                save(step)
                break
            stop=torch.tensor(int(env.is_main and args.max_wall_hours>0 and
                time.time()-started_unix>=args.max_wall_hours*3600),device=env.device)
            if args.max_wall_hours>0:
                all_reduce_sum(stop,env)
                if int(stop):
                    if env.is_main:print(json.dumps(dict(event="wall_budget_reached",step=step)),flush=True)
                    stop_reason="wall_budget"
                    break
            if step%args.save_every==0 and step!=args.steps:save(step)
            if step!=args.steps and validation_due(step,
                    first_step=args.first_validation_step if start==0 else -1,
                    every=args.eval_every):validate(step)
        if stop_reason in ("requested","signaled"):
            if env.is_main:
                summary=dict(status="stopped",step=step,requested_steps=args.steps,stop_reason=stop_reason,
                    checkpoint=f"checkpoint_step{step:06d}.pt",
                    resume="new run directory with --resume pointing at this checkpoint")
                atomic_json(run/"summary.json",summary)
                atomic_json(run/"progress.json",dict(status="stopped",step=step,stop_reason=stop_reason))
                print(json.dumps(summary),flush=True)
            barrier(env)
            return
        if not timings:raise ValueError("Need measured steps after timing warmup")
        if prefetcher is not None:prefetcher.close();prefetcher=None
        final_step=step
        result=validate(final_step)
        sample=sample_unconditional(raw,steps=args.sample_steps,
            generator=torch.Generator(device=env.device).manual_seed(args.seed+env.rank+900))
        if env.is_main:
            np.save(run/"sample_compact.npy",sample.cpu().numpy().astype(np.uint16))
            np.save(run/"sample_blocks.npy",data.decode_compact(sample.cpu().numpy()))
            atomic_json(run/"validation.json",result)
        save(final_step)
        memory=dict(rank=env.rank,gpu_name=torch.cuda.get_device_name(env.device) if env.device.type=="cuda" else "cpu",
            peak_allocated_gib=torch.cuda.max_memory_allocated(env.device)/2**30 if env.device.type=="cuda" else 0,
            peak_reserved_gib=torch.cuda.max_memory_reserved(env.device)/2**30 if env.device.type=="cuda" else 0)
        memories=[None]*env.world_size
        if env.enabled:dist.all_gather_object(memories,memory)
        else:memories=[memory]
        if env.is_main:
            mean=statistics.mean(r["seconds"] for r in timings)
            summary=dict(status="complete",steps=final_step,requested_steps=args.steps,
                stop_reason="wall_budget" if final_step<args.steps else "step_target",measured_steps=len(timings),
                mean_step_seconds=mean,median_step_seconds=statistics.median(r["seconds"] for r in timings),
                samples_per_second=args.global_batch/mean,
                target_voxels_per_second=args.global_batch*65536/mean,
                mean_data_seconds_max=statistics.mean(r["data_seconds_max"] for r in timings),
                timing_includes="real DataLoader, mask, forward, backward, DDP, optimizer, scheduled EMA",
                timing_excludes="initialization, validation, sampling, checkpoint writes",gpus=memories,
                validation=result,sample_steps=args.sample_steps,sample_quality_claim=False)
            atomic_json(run/"summary.json",summary)
            atomic_json(run/"progress.json",dict(status="complete",step=final_step,
                stop_reason=summary["stop_reason"]))
            print(json.dumps(summary),flush=True)
    finally:
        if prefetcher is not None:prefetcher.close()
        cleanup(env)


if __name__=="__main__":
    main()

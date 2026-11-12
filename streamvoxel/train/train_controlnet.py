"""Train shared-context ControlNet, preserving frozen base and exact global loss."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from streamvoxel.common.distributed import init_distributed, all_reduce_sum, barrier, cleanup
from streamvoxel.common.io import atomic_json, append, digest
from streamvoxel.common.controlnet_dataset import validate_dataset, training_validation_plans
from streamvoxel.data.streaming_controlnet import ControlDataset, GroupPrefetch, microbatches, to_device
from streamvoxel.models.streaming_controlnet import from_base_checkpoint
from streamvoxel.models.compact_controlnet import CompactStreamingControlNet
from streamvoxel.objectives.md4 import md4_mask, strata_mask, weighted_masked_ce_sums
from streamvoxel.objectives.controlnet_compiled_ce import compiled_masked_ce_sums
from streamvoxel.objectives.controlnet_schedule import learning_rate, validate_resume_schedule
from streamvoxel.train.runtime import autocast, rng_state, restore_rng
from streamvoxel.train.config import parse_with_config
from streamvoxel.train.validation_controlnet import evaluate

FORMAT = 'java26_streaming_controlnet_checkpoint_v1'
STRATA_BAND = 8
AIR_NAMES = ('minecraft:air', 'minecraft:cave_air', 'minecraft:void_air')


def air_compact_ids(mapping):
    names = mapping.get('compact_to_clean_name')
    if not names:
        raise ValueError('ControlNet mapping has no compact_to_clean_name')
    ids = [i for i, name in enumerate(names) if name in AIR_NAMES]
    found = {names[i] for i in ids}
    if 'minecraft:air' not in found or 'minecraft:cave_air' not in found:
        raise ValueError('Compact mapping is missing an air class')
    return ids


def half_mask_role(rank, world_size, step):
    """Half the ranks draw i.i.d. MD4 masks, the other half height-strata masks; one rank alternates by step."""
    if world_size==1:return 'random' if step%2 else 'strata'
    if world_size%2:raise ValueError('Half-mask needs one rank or an even number of ranks')
    return 'random' if rank<world_size//2 else 'strata'


def training_steps(start, stop):
    """stop=0 is an explicitly requested, checkpointable unbounded run."""
    if start<0 or stop<0:raise ValueError('Negative training step')
    return itertools.count(start+1) if stop==0 else range(start+1,stop+1)


def rank_batch_plan(global_batch,microbatch,world_size,rank,side):
    """Distribute nominal accumulation slots while keeping exact global samples."""
    if global_batch%microbatch or not 0<=rank<world_size:raise ValueError('Invalid rank batch allocation')
    slots=global_batch//microbatch
    per_rank=[len(range(r,slots,world_size)) for r in range(world_size)]
    if min(per_rank)<=0:raise ValueError('Every rank must receive samples')
    targets=per_rank[rank]*microbatch;limit=min(4*side,targets)
    groups=[min(limit,targets-i) for i in range(0,targets,limit)]
    return per_rank,[n*microbatch for n in per_rank],groups


def reduce_gradients(model, env, *, asynchronous=False):
    # Bounded buckets; exactly one SUM (not DDP averaging) per gradient value.
    bucket, size, pending = [], 0, []
    def flush():
        if not bucket:return
        flat=torch.cat([p.grad.flatten() for p in bucket])
        if env.enabled and asynchronous:
            pending.append((dist.all_reduce(flat,op=dist.ReduceOp.SUM,async_op=True),flat,list(bucket)))
            return
        if env.enabled:dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        offset=0
        for p in bucket:
            count=p.numel();p.grad.copy_(flat[offset:offset+count].view_as(p));offset+=count
    for p in model.trainable_parameters():
        if p.grad is None:p.grad=torch.zeros_like(p)
        bucket.append(p);size+=p.numel()*p.element_size()
        if size>=32*1024**2:flush();bucket=[];size=0
    flush()
    for work,flat,parameters in pending:
        work.wait();offset=0
        for p in parameters:
            count=p.numel();p.grad.copy_(flat[offset:offset+count].view_as(p));offset+=count


def _group_backward(model,group,env,prepared,denominator,*,microbatch,compile_loss=False):
    target,x,mask,t,weight=prepared
    start=time.perf_counter()
    with autocast(env.device):original=model.encode_group(group)
    encoded={k:v.detach().requires_grad_() for k,v in original.items()}
    with autocast(env.device):projected=model.prefix_kv(encoded,group)
    prefix=[tuple(v.detach().requires_grad_() for v in kv) for kv in projected]
    encoding_seconds=time.perf_counter()-start
    totals=torch.zeros((3,3),device=env.device,dtype=torch.float64)
    for ids in microbatches(group,microbatch):
        with autocast(env.device):
            local,lp=model.local_condition(encoded,group,ids)
            logits=model(x[ids],timestep=t[ids],origin_xyz=group['target_origins'][ids],
                group_origin_xyz=group['group_origin'],prefix_kv=prefix,local=local,local_positions=lp)
        objective=compiled_masked_ce_sums if compile_loss else weighted_masked_ce_sums
        loss,correct,count=objective(logits,target[ids],mask[ids],weight[ids])
        if env.device.type=='cuda':torch._assert_async(torch.isfinite(loss),'Nonfinite ControlNet loss')
        elif not torch.isfinite(loss):raise FloatingPointError('Nonfinite ControlNet loss')
        (loss/denominator).backward()
        n=ids.condition_n-1
        totals[n]+=torch.stack((loss.detach().double(),correct.double(),count.double()))
        del logits,loss
    # VJP through per-layer shared projection, then through shared encoders exactly once.
    outputs=[v for kv in projected for v in kv]
    grads=[v.grad if v.grad is not None else torch.zeros_like(v) for kv in prefix for v in kv]
    torch.autograd.backward(outputs,grads)
    torch.autograd.backward(list(original.values()),[encoded[k].grad if encoded[k].grad is not None
        else torch.zeros_like(v) for k,v in original.items()])
    return totals,encoding_seconds


def _train_step_impl(model, optimizer, group, env, *, microbatch=4, asynchronous_reduce=False,compile_loss=False,
                     air_ids=(0,), strata_band=STRATA_BAND, mask_role='strata'):
    groups=group if isinstance(group,(list,tuple)) else [group]
    optimizer.zero_grad(set_to_none=True)
    prepared=[]
    denominator=torch.zeros((),device=env.device,dtype=torch.float64)
    for item in groups:
        target=item['targets'].long()
        if mask_role=='random':x,mask,t,weight=md4_mask(target,model.backbone.num_classes)
        else:x,mask,t,weight=strata_mask(target,model.backbone.num_classes,air_ids,band=strata_band)
        prepared.append((target,x,mask,t,weight));denominator+=mask.sum().double()
    all_reduce_sum(denominator,env)
    totals=torch.zeros((3,3),device=env.device,dtype=torch.float64)
    encoding_seconds=0.
    for item,batch in zip(groups,prepared):
        partial,seconds=_group_backward(model,item,env,batch,denominator,microbatch=microbatch,compile_loss=compile_loss)
        totals+=partial;encoding_seconds+=seconds
    reduce_gradients(model,env,asynchronous=asynchronous_reduce)
    norm=torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()),1.,error_if_nonfinite=True)
    optimizer.step()
    role_totals=torch.zeros((2,3),device=env.device,dtype=torch.float64)
    role_totals[0 if mask_role=='random' else 1]+=totals.sum(0)
    all_reduce_sum(totals,env);all_reduce_sum(role_totals,env)
    row=totals.sum(0)
    split={}
    for name,part in zip(('random','strata'),role_totals):
        if part[2]>0:split.update({f'train_{name}_loss':float(part[0]/part[2]),f'train_{name}_acc':float(part[1]/part[2]),
                                   f'train_{name}_masked_voxels':int(part[2])})
    return dict(train_loss=float(row[0]/row[2]),train_acc=float(row[1]/row[2]),masked_voxels=int(row[2]),**split,
        grad_norm=float(norm),encoding_seconds=encoding_seconds,
        by_n={str(i+1):dict(loss=float(r[0]/r[2].clamp_min(1)),acc=float(r[1]/r[2].clamp_min(1)),
                            masked_voxels=int(r[2])) for i,r in enumerate(totals)})


def train_step(model,optimizer,group,env,*,microbatch=4,asynchronous_reduce=False,persistent_autocast=False,compile_loss=False,
               air_ids=(0,),strata_band=STRATA_BAND,mask_role='strata'):
    kwargs=dict(microbatch=microbatch,asynchronous_reduce=asynchronous_reduce,compile_loss=compile_loss,
                air_ids=air_ids,strata_band=strata_band,mask_role=mask_role)
    if persistent_autocast:
        # One update shares BF16 parameter casts; the cache is cleared before any
        # subsequent optimizer update, so no stale cast of changed parameters survives.
        with autocast(env.device):
            return _train_step_impl(model,optimizer,group,env,**kwargs)
    return _train_step_impl(model,optimizer,group,env,**kwargs)


def control_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('backbone.')}


def load_control_state(model, state, *, allow_new_biome=False):
    if any(k.startswith('backbone.') for k in state):
        raise ValueError('Control weights must not replace the frozen backbone')
    if model.struct_classes == 0:
        # Explicit architecture removal: preserve all remaining trained weights.
        state = {k:v for k,v in state.items() if not k.startswith(('struct_embed.', 'struct_size.'))}
        if state['type_embed.weight'].shape[0] == 4:
            state['type_embed.weight'] = state['type_embed.weight'][:3].clone()
    missing,unexpected=model.load_state_dict(state,strict=False)
    # A time-conditioned control branch has no constant_time; the time-free copy keeps the base's t=0.5 value.
    allowed={'biome_down.weight','biome_down.bias','control.constant_time'} if allow_new_biome else {'control.constant_time'}
    if allow_new_biome and model.biome_patch==2:allowed.add('biome_size8')
    bad=[k for k in missing if not k.startswith('backbone.') and k not in allowed]
    if unexpected or bad:
        raise ValueError(f'Incomplete control state missing={bad[:40]} unexpected={list(unexpected)[:40]}')


def save_checkpoint(run, model, optimizer, ema, config, step, env, *, numbered=True):
    states=[None]*env.world_size
    state=rng_state(env.device)
    if env.enabled:dist.all_gather_object(states,state)
    else:states[0]=state
    if env.is_main:
        payload=dict(format=FORMAT,step=step,config=config,model=control_state(model),
            optimizer=optimizer.state_dict(),ema=ema,rng_by_rank=states,consumed_targets=step*config['global_batch'])
        import os
        if numbered:
            tmp=run/f'checkpoint_step{step:06d}.tmp';dest=tmp.with_suffix('.pt')
            torch.save(payload,tmp);tmp.replace(dest)
            alias=run/'checkpoint_latest.tmp'
            if alias.exists():raise FileExistsError(alias)
            os.link(dest,alias);alias.replace(run/'checkpoint_latest.pt')
        else:
            # One rolling file. A hardlink taken earlier keeps that inode after this replace.
            tmp=run/'checkpoint_latest.tmp'
            if tmp.exists():raise FileExistsError(tmp)
            torch.save(payload,tmp);tmp.replace(run/'checkpoint_latest.pt')
    barrier(env)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','base','run-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--resume');p.add_argument('--steps',type=int,default=10000,help='0: train until explicitly stopped')
    p.add_argument('--skip-dataset-validation',action='store_true',help='Explicitly reuse prepared data without startup full-file reverification')
    p.add_argument('--allow-world-size-change',action='store_true')
    p.add_argument('--prefetch-depth',type=int,default=2)
    p.add_argument('--serial-model-load',action='store_true',help='Bound CPU peak memory during multi-rank initialization')
    p.add_argument('--global-batch',type=int,default=512)
    p.add_argument('--side',type=int,default=32)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--cell-chunk',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--warmup-steps',type=int,default=200)
    p.add_argument('--lr-schedule',choices=('constant','cosine'),default='constant')
    p.add_argument('--cosine-start-step',type=int,default=0)
    p.add_argument('--cosine-steps',type=int,default=0)
    p.add_argument('--min-lr',type=float,default=0.)
    p.add_argument('--allow-schedule-change',action='store_true')
    p.add_argument('--allow-lr-change',action='store_true')
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--eval-on-start',action='store_true')
    p.add_argument('--eval-ordinary-contexts',type=int)
    p.add_argument('--eval-full-contexts',type=int)
    p.add_argument('--sample-steps',type=int,default=1024)
    p.add_argument('--target-checkpoint',choices=('all','none','blocks'),default='all')
    p.add_argument('--attention-backend',choices=('concat','shared'),default='concat')
    p.add_argument('--async-reduce',action='store_true')
    p.add_argument('--allow-runtime-change',action='store_true')
    p.add_argument('--compile-blocks',action='store_true')
    p.add_argument('--persistent-autocast',action='store_true')
    p.add_argument('--compile-loss',action='store_true')
    p.add_argument('--biome-patch',type=int,choices=(1,2,4),default=1)
    p.add_argument('--biome-halo',type=int,choices=(4,16),default=16)
    p.add_argument('--control-depth',type=int)
    p.add_argument('--control-hidden-dim',type=int)
    p.add_argument('--max-wall-hours',type=float,default=0)
    p.add_argument('--half-mask',action='store_true',help='Half random MD4 masks, half height-strata masks (by rank; one rank alternates by step)')
    a=parse_with_config(p);env=init_distributed();prefetch=None
    # Control and backbone blocks share one forward; grad / no-grad inputs and shape buckets each recompile it.
    import torch._dynamo
    torch._dynamo.config.recompile_limit=64
    try:
        if a.steps<0 or min(a.microbatch,a.cell_chunk,a.save_every,a.sample_steps,a.prefetch_depth)<=0:
            raise ValueError('Invalid global batch/runtime')
        accumulation_by_rank,targets_by_rank,group_sizes=rank_batch_plan(a.global_batch,a.microbatch,env.world_size,env.rank,a.side)
        group_targets=max(group_sizes);groups_per_rank=len(group_sizes)
        run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=bool(a.resume)) if env.is_main else None
        barrier(env)
        if env.is_main and not a.skip_dataset_validation:validate_dataset(a.dataset)
        barrier(env)
        data=ControlDataset(a.dataset,validate=False,use_struct=False)
        packaged_plans=json.loads((data.root/data.manifest['plans_file']).read_text())
        plans=training_validation_plans(data.manifest,packaged_plans,side=a.side)
        if env.is_main:
            plan_path=run/'validation_plans.json'
            if plan_path.exists():
                if json.loads(plan_path.read_text())!=plans:raise ValueError('Pinned validation plans differ')
            else:atomic_json(plan_path,plans)
        barrier(env)
        identity=None
        if env.is_main:identity=dict(dataset=digest(data.root/'manifest.json'),base=digest(a.base),
            validation_plans=digest(run/'validation_plans.json'))
        if env.enabled:
            obj=[identity];dist.broadcast_object_list(obj,src=0);identity=obj[0]
        torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True
        control_options = (dict(model_class=CompactStreamingControlNet,control_depth=a.control_depth,
            control_hidden_dim=a.control_hidden_dim,control_initialization='copied')
            if a.control_depth is not None or a.control_hidden_dim is not None else {})
        def construct():
            result=from_base_checkpoint(a.base,biome_classes=len(data.biome_names),struct_classes=0,
                checkpointing=True,cell_chunk=a.cell_chunk,biome_patch=a.biome_patch,
                biome_halo=a.biome_halo,**control_options).to(env.device)
            result.set_runtime(target_checkpoint=a.target_checkpoint,attention_backend=a.attention_backend)
            return result
        if a.serial_model_load and env.enabled:
            for rank in range(env.world_size):
                if rank==env.rank:model=construct()
                barrier(env)
        else:model=construct()
        if a.compile_blocks:
            modules=list(model.backbone.blocks)+list(model.control.blocks)
            if a.attention_backend=='concat':modules+=list(model.cross)
            for block in modules:block.compile(dynamic=True)
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=a.lr,betas=(.9,.99),weight_decay=.01,
                                    fused=env.device.type=='cuda')
        ema={n:v.detach().clone() for n,v in model.named_parameters() if v.requires_grad}
        config=dict(vars(a),format=FORMAT,identity=identity,model=model.config(),world_size=env.world_size,
            trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),
            frozen_parameters=sum(p.numel() for p in model.backbone.parameters()),
            snapshot=str(Path.cwd()),source_manifest_sha256=digest('source_manifest.json') if Path('source_manifest.json').exists() else None,
            condition_levels=['L1','L2','biome'],validation_plan_side=a.side,
            accumulation_by_rank=accumulation_by_rank,targets_by_rank=targets_by_rank,
            local_group_sizes=group_sizes,
            initialization='resume' if a.resume else 'fresh_controlnet_from_frozen_base',
            control_transfer=None,mask='half_random_strata' if a.half_mask else 'strata',strata_band=STRATA_BAND,
            air_compact_ids=air_compact_ids(data.mapping))
        start=0
        if a.resume:
            saved=torch.load(a.resume,map_location='cpu',mmap=True,weights_only=False)
            validate_resume_schedule(saved['config'],config,saved['step'],allow_change=a.allow_schedule_change)
            for key in ('identity','model','world_size','global_batch','seed','lr','warmup_steps',
                        'side','sample_steps','eval_every','eval_ordinary_contexts','eval_full_contexts',
                        'mask','strata_band','air_compact_ids'):
                if key in ('mask','strata_band','air_compact_ids'):
                    old,new=saved['config'].get(key),config[key]
                else:
                    old,new=saved['config'][key],config[key]
                if key=='world_size' and a.allow_world_size_change:continue
                if key=='lr' and a.allow_lr_change:
                    if a.lr_schedule!='constant' or not 0<a.lr<=old:
                        raise ValueError('Explicit LR continuation must use a positive lower constant LR')
                    continue
                if key=='model':
                    old=dict(old,biome_patch=old.get('biome_patch',1),biome_halo=old.get('biome_halo',16))
                if key=='identity':
                    # Saved identities of the removed VAE metric are ignored.
                    old={k:v for k,v in old.items() if k not in ('vae','vae_mapping')}
                if key=='identity' and 'validation_plans' not in old:
                    if packaged_plans!=plans:raise ValueError('Legacy checkpoint validation geometry differs')
                    old=dict(old,validation_plans=new['validation_plans'])
                if key=='model' and a.allow_runtime_change:
                    old={k:v for k,v in old.items() if k not in ('cell_chunk','checkpointing')}
                    new={k:v for k,v in new.items() if k not in ('cell_chunk','checkpointing')}
                if old!=new:raise ValueError(f'Resume differs: {key}')
            for key,default in [('target_checkpoint','all'),('attention_backend','concat'),('async_reduce',False),
                                ('compile_blocks',False),('persistent_autocast',False),('compile_loss',False)]:
                if saved['config'].get(key,default)!=config[key] and not a.allow_runtime_change:
                    raise ValueError(f'Resume runtime differs: {key}')
            load_control_state(model,saved['model']);optimizer.load_state_dict(saved['optimizer'])
            ema={k:v.to(env.device) for k,v in saved['ema'].items()};start=saved['step']
            if env.rank<len(saved['rng_by_rank']):restore_rng(saved['rng_by_rank'][env.rank],env.device)
            else:torch.manual_seed(a.seed+env.rank*1337)
            config['control_transfer']=saved['config'].get('control_transfer')
            config['resume_repartition']=dict(previous_world_size=saved['config']['world_size'],world_size=env.world_size,
                accumulation_by_rank=accumulation_by_rank,targets_by_rank=targets_by_rank,
                bitwise_same_trajectory=saved['config']['world_size']==env.world_size)
            del saved
        else:
            torch.manual_seed(a.seed+env.rank*1337)
        if env.is_main:
            atomic_json(run/'config.json',config)
            print(json.dumps(dict(event='initialized',start_step=start,trainable_parameters=config['trainable_parameters'],
                                  frozen_parameters=config['frozen_parameters'],identity=identity)),flush=True)
        def validation(step):
            barrier(env)
            if env.is_main:atomic_json(run/'progress.json',dict(status='validating',step=step))
            result=evaluate(model,data,plans,env,microbatch=a.microbatch,full_steps=a.sample_steps,
                ordinary_limit=a.eval_ordinary_contexts,full_limit=a.eval_full_contexts,output=run,step=step,
                air_ids=config['air_compact_ids'] if a.half_mask else None)
            if env.is_main:
                buckets=list(result['ordinary']['full'].values())
                count=sum(b['masked_voxels'] for b in buckets)
                score=sum(b['loss']*b['masked_voxels'] for b in buckets)/max(count,1)
                if 'ordinary_strata' in result:
                    sb=list(result['ordinary_strata'].values())
                    score=.5*(score+sum(b['loss']*b['masked_voxels'] for b in sb)/max(sum(b['masked_voxels'] for b in sb),1))
                append(run/'validation_metrics.jsonl',{**{k:v for k,v in result.items() if k!='records'},'selection_loss':score})
                old=json.loads((run/'best_validation.json').read_text()) if (run/'best_validation.json').exists() else None
                current=run/'checkpoint_latest.pt'
                if step%a.save_every==0 and current.exists() and (old is None or score<old['loss']):
                    import os
                    os.link(current,run/'checkpoint_best.tmp')
                    (run/'checkpoint_best.tmp').replace(run/'checkpoint_best.pt')
                    atomic_json(run/'best_validation.json',dict(step=step,loss=score,
                        full_mask=result['full_mask'],identity=identity))
                print(json.dumps(dict(event='validation_complete',step=step,seconds=result['seconds'],
                    ordinary=result['ordinary'],full_mask=result['full_mask'])),flush=True)
            barrier(env)
        if a.eval_on_start:validation(start)
        # The producer and evaluator must not mutate the same mmap/LRU dictionaries.
        loader_data=ControlDataset(a.dataset,validate=False,use_struct=False)
        prefetch=GroupPrefetch(loader_data,start+1,env.rank,env.world_size,
                              a.seed,side=a.side,targets=group_targets,groups_per_step=groups_per_rank,
                              group_sizes=group_sizes,depth=a.prefetch_depth,device=env.device)
        started=time.monotonic();stop_requested=False
        def request_stop(*_):
            nonlocal stop_requested
            stop_requested=True
        signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
        model.train()
        last_saved=start
        for step in training_steps(start,a.steps):
            begin=time.perf_counter();plan,cpu=prefetch.take(step)
            group=to_device(cpu,env.device)
            if env.device.type=='cuda':torch.cuda.synchronize(env.device)
            data_seconds=time.perf_counter()-begin
            lr=learning_rate(step,peak_lr=a.lr,warmup_steps=a.warmup_steps,schedule=a.lr_schedule,
                cosine_start_step=a.cosine_start_step,cosine_steps=a.cosine_steps,min_lr=a.min_lr)
            for g in optimizer.param_groups:g['lr']=lr
            values=train_step(model,optimizer,group,env,microbatch=a.microbatch,
                              asynchronous_reduce=a.async_reduce,persistent_autocast=a.persistent_autocast,compile_loss=a.compile_loss,
                              air_ids=config['air_compact_ids'],strata_band=config['strata_band'],
                              mask_role=half_mask_role(env.rank,env.world_size,step) if a.half_mask else 'strata')
            if step%10==0:
                with torch.no_grad():
                    for n,v in model.named_parameters():
                        if v.requires_grad:ema[n].lerp_(v,.1)
            if env.device.type=='cuda':torch.cuda.synchronize(env.device)
            seconds=torch.tensor(time.perf_counter()-begin,device=env.device,dtype=torch.float64)
            if env.enabled:dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
            wait=torch.tensor(data_seconds,device=env.device,dtype=torch.float64)
            if env.enabled:dist.all_reduce(wait,op=dist.ReduceOp.MAX)
            values.update(step=step,lr=lr,seconds=float(seconds),samples_per_second=a.global_batch/float(seconds),
                data_seconds=data_seconds,world=(plan[0] if isinstance(plan,list) else plan)['world'],
                data_seconds_max=float(wait),accumulation_by_rank=accumulation_by_rank,targets_by_rank=targets_by_rank,
                loader_io=dict(loader_data.io_stats),
                origin_zx=(plan[0] if isinstance(plan,list) else plan)['origin_zx'],groups_per_rank=groups_per_rank,
                peak_allocated_gib=torch.cuda.max_memory_allocated(env.device)/2**30 if env.device.type=='cuda' else 0)
            if env.is_main:
                append(run/'metrics.jsonl',values);atomic_json(run/'progress.json',dict(status='training',**values))
                print(json.dumps(values),flush=True)
            end=torch.tensor(int(stop_requested or (run/'stop_request.json').is_file() or (run/'STOP_REQUESTED').exists()
                or (a.max_wall_hours and time.monotonic()-started>=a.max_wall_hours*3600)),device=env.device)
            if env.enabled:dist.all_reduce(end,op=dist.ReduceOp.MAX)
            if step%a.save_every==0 or end or step==a.steps:
                save_checkpoint(run,model,optimizer,ema,config,step,env,numbered=False);last_saved=step
            if a.eval_every and step%a.eval_every==0:validation(step)
            if end:break
        if env.is_main:atomic_json(run/('training_complete.json' if last_saved==a.steps else 'training_stopped.json'),
                                  dict(step=last_saved,config_identity=identity))
    finally:
        if prefetch:prefetch.close()
        cleanup(env)


if __name__=='__main__':main()

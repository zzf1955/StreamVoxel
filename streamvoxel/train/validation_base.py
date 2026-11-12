"""Batched fixed-corruption validation, with exact coverage and global totals."""
from functools import partial
import time

import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate

from streamvoxel.common.distributed import all_reduce_sum
from streamvoxel.objectives.md4 import md4_mask,strata_mask,weighted_masked_ce_sums
from streamvoxel.train.runtime import autocast,rng_state,restore_rng


def _world_order(records):
    order=[];seen=set()
    for wi in map(int,records[:,0]):
        if wi not in seen:
            seen.add(wi);order.append(wi)
    return order


def first_worlds(dataset,count):
    """Keep records from the first `count` Worlds in index order."""
    if count<1:raise ValueError('Evaluation World count must be positive')
    order=_world_order(dataset.records)
    if len(order)<count:
        raise ValueError(f'Validation split has {len(order)} Worlds; evaluation requires {count}')
    keep=np.asarray(order[:count],dtype=dataset.records.dtype)
    dataset.records=np.ascontiguousarray(dataset.records[np.isin(dataset.records[:,0],keep)])
    return order[:count]


def select_eval_worlds(validation,count,*,test=None):
    """Keep the validation Worlds, then append whole test Worlds until `count`."""
    if test is None:
        return first_worlds(validation,count)
    if count<1:raise ValueError('Evaluation World count must be positive')
    order=_world_order(validation.records)
    if len(order)>count:
        return first_worlds(validation,count)
    if len(order)==count:
        return order
    extra=[]
    seen=set(order)
    for wi in _world_order(test.records):
        if wi not in seen:
            seen.add(wi);extra.append(wi)
        if len(order)+len(extra)==count:
            break
    if len(order)+len(extra)<count:
        raise ValueError(f'Validation plus test has {len(order)+len(_world_order(test.records))} Worlds; evaluation requires {count}')
    keep=np.asarray(extra,dtype=test.records.dtype)
    added=test.records[np.isin(test.records[:,0],keep)]
    validation.records=np.ascontiguousarray(np.concatenate([validation.records,added]))
    return order+extra


def validation_ids(size,seed,limit=0):
    return (np.arange(size,dtype=np.int64) if limit==0 else
            np.random.default_rng(seed).choice(size,min(limit,size),replace=False))


def collate_validation(records,*,seed,num_classes,air_ids=None,strata_band=8):
    batch=default_collate(records)
    if air_ids is None:
        corruptions=[md4_mask(r['target'][None],num_classes,
            generator=torch.Generator().manual_seed(seed+int(r['record_id']))) for r in records]
    else:
        corruptions=[strata_mask(r['target'][None],num_classes,air_ids,band=strata_band,
            generator=torch.Generator().manual_seed(seed+int(r['record_id']))) for r in records]
    batch.update(zip(('corrupted','mask','timestep','weight'),
        (torch.cat([c[j] for c in corruptions],dim=0) for j in range(4))))
    return batch


@torch.no_grad()
def evaluate(model,dataset,env,*,seed,limit=0,batch_size=32,workers=4,loader_prefetch=8,on_progress=None,air_ids=None,strata_band=8):
    if limit<0 or batch_size<1 or workers<0:raise ValueError('Invalid validation settings')
    ids=validation_ids(len(dataset),seed,limit)
    worlds=sorted(set(map(int,dataset.records[ids,0])))
    world_rows={wi:i+1 for i,wi in enumerate(worlds)}
    totals=torch.zeros((len(worlds)+1,4),device=env.device,dtype=torch.float64)
    previous_mode=model.training
    previous_rng=rng_state(env.device)
    before=time.perf_counter()
    options=dict(batch_size=batch_size,sampler=ids[env.rank::env.world_size].tolist(),
        num_workers=workers,pin_memory=env.device.type=='cuda',
        collate_fn=partial(collate_validation,seed=seed,num_classes=dataset.num_classes,
            air_ids=air_ids,strata_band=strata_band),
        generator=torch.Generator().manual_seed(seed+env.rank))
    if workers:options['prefetch_factor']=loader_prefetch
    model.eval()
    try:
        for batch_number,batch in enumerate(DataLoader(dataset,**options),1):
            target=batch['target'].to(env.device,non_blocking=True)
            x,mask,t,w=(batch[k].to(env.device,non_blocking=True)
                for k in ('corrupted','mask','timestep','weight'))
            with autocast(env.device):
                logits=model(x,timestep=t,origin_xyz=batch['origin_xyz'].to(env.device,non_blocking=True))
            loss,correct,count,losses,successes,counts=weighted_masked_ce_sums(
                logits,target,mask,w,by_sample=True)
            totals[0,:3]+=torch.stack((loss.double(),correct.double(),count.double()))
            totals[0,3]+=len(target)
            for wi in batch['world_index'].unique().tolist():
                selected=(batch['world_index']==wi).to(env.device)
                totals[world_rows[int(wi)],:3]+=torch.stack((losses[selected].double().sum(),
                    successes[selected].double().sum(),counts[selected].double().sum()))
                totals[world_rows[int(wi)],3]+=int(selected.sum())
            del logits,target,x,mask,t,w
            if on_progress is not None and batch_number%32==0:
                on_progress(dict(local_cases=int(totals[0,3]),local_total=len(ids[env.rank::env.world_size])))
        all_reduce_sum(totals,env)
        if env.device.type=='cuda':torch.cuda.synchronize(env.device)
        seconds=torch.tensor(time.perf_counter()-before,device=env.device,dtype=torch.float64)
        if env.enabled:torch.distributed.all_reduce(seconds,op=torch.distributed.ReduceOp.MAX)
        if int(totals[0,3])!=len(ids):raise AssertionError('Validation coverage mismatch')
        def metrics(row):
            return dict(validation_md4_loss=float(row[0]/row[2]),
                validation_masked_accuracy=float(row[1]/row[2]),
                masked_voxels=int(row[2]),cases=int(row[3]))
        result=dict(**metrics(totals[0]),seed=seed,weights_source='raw',
            coverage='full' if limit==0 else 'subset',
            corruption='fixed_cpu_per_record_seed_plus_record_id'+('' if air_ids is None else '_strata'),
            batch_size_per_rank=batch_size,world_size=env.world_size,
            seconds=float(seconds),samples_per_second=len(ids)/float(seconds),
            per_world={dataset.manifest['worlds'][wi]['id']:metrics(totals[j])
                for wi,j in world_rows.items()})
        if limit:result['record_ids']=ids.tolist()
        return result
    finally:
        model.train(previous_mode)
        restore_rng(previous_rng,env.device)

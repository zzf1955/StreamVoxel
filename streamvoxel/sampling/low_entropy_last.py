import torch

from streamvoxel.sampling.schedule import counts

STRATA_BAND=8


def low_entropy_last_update(last,mask,air,solid,eps,band,col=False):
    """last: bool [Y] deferred layers (monotone). A unit (layer or 8-layer band) is clean if all its masked voxels have P(air) > 1-eps and no
    committed non-air voxel (solid [Y,Z,X]); `last` gains the maximal top run of clean units (largest y = top)."""
    bad=(mask&(air<=1-eps))|solid
    if col:     # per-column variant: `last` is a voxel mask [Y,Z,X]; every column defers its own top run of clean voxels (follows terrain height)
        return last|torch.cumprod((~bad).flip(0).long(),0).flip(0).bool()
    clean=~bad.flatten(1).any(1)           # per layer
    if band:clean=clean.reshape(-1,STRATA_BAND).all(1).repeat_interleave(STRATA_BAND)
    top_run=torch.cumprod(clean.flip(0).long(),0).flip(0).bool()
    return last|top_run


def low_entropy_last_quotas(total,steps,n_last,schedule='quartercos'):
    """Quarter-cosine quotas of length `steps`, with `n_last` voxels assigned from the tail.

    Returns the positive active quotas only. Their sum is total-n_last, so a cap of `steps`
    runs fewer forwards once whole tail steps are consumed by the deferred set."""
    n_last=int(n_last)
    if steps<1 or not 0<=n_last<=total:raise ValueError((total,steps,n_last))
    quotas=counts(total,dict(steps=int(steps),schedule=schedule,rounding='positive'))
    air=n_last;solid=[]
    for q in reversed(quotas):
        take=min(q,air);air-=take;solid.append(q-take)
    solid.reverse()
    assert air==0 and sum(solid)==total-n_last
    return [q for q in solid if q>0]

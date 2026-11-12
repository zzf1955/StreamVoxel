import torch


def top_mask(scores, eligible, count):
    result = torch.zeros_like(eligible)
    indices = torch.where(eligible.flatten())[0]
    count = min(int(count), len(indices))
    if count:
        # Stable tie rule: lower YZX flat index first, including CPU tests.
        order = torch.argsort(scores.flatten()[indices], descending=True, stable=True)
        result.flatten()[indices[order[:count]]] = True
    return result


def block_view(a,b):
    y,z,x=a.shape
    return a.reshape(y//b,b,z//b,b,x//b,b).permute(0,2,4,1,3,5).reshape(-1,b**3)


def block_quota_select(score,mask,k,b,order,frac,n_total):
    """k voxels of `mask` (the non-deferred masked voxels), balanced over b^3 blocks: block j should hold ~frac*N_j committed voxels
    (N_j = its non-deferred voxel count = masked_j + already committed_j, given by `n_total` [blocks]); seeded order breaks remainder ties;
    inside a block the lowest-entropy voxels (largest score) are taken. Exactly k are returned (global entropy top-up/trim if clamped)."""
    mb=block_view(mask,b);nbk=mb.shape[0];legal=mb.sum(1);already=n_total-legal
    ideal=frac*n_total.double();base=torch.floor(ideal+1e-9).long();goal=int(already.sum())+k
    pos=torch.empty_like(order);pos[order]=torch.arange(nbk,device=mask.device)
    extra=max(goal-int(base.sum()),0)
    frac_rank=torch.argsort(-(ideal-base.double())*1e6+pos.double())   # largest fractional part first, seeded tie-break
    target=base.clone();target[frac_rank[:extra]]+=1
    quota=torch.minimum((target-already).clamp(min=0),legal)
    sm=block_view(score,b).masked_fill(~mb,-float('inf'));ranks=torch.argsort(sm,dim=1,descending=True,stable=True)
    bc=torch.zeros_like(mb);bc.scatter_(1,ranks,torch.arange(b**3,device=mask.device)[None,:]<quota[:,None])
    Y,Z,X=mask.shape
    sel=bc.reshape(Y//b,Z//b,X//b,b,b,b).permute(0,3,1,4,2,5).reshape(mask.shape)&mask
    got=int(sel.sum())
    if got>k:sel=top_mask(score,sel,k)
    elif got<k:sel=sel|top_mask(score,mask&~sel,k-got)
    return sel

"""Existing PyTorch categorical loss compiled with static masks; no custom kernel."""
import torch
from torch.nn import functional as F

_COMPILED=None


def _broadcast_weight(weight,targets):
    if weight.shape==targets.shape:return weight
    if weight.shape==(targets.shape[0],):return weight[:,None,None,None]
    raise ValueError('MD4 weight must be one value per sample or one value per voxel')


def static_masked_ce_sums(logits,targets,mask,weight):
    ce=F.cross_entropy(logits.float(),targets.long(),reduction='none')
    loss=(ce*mask*_broadcast_weight(weight,targets)).sum()
    correct=((logits.detach().argmax(1)==targets)&mask).sum()
    return loss,correct,mask.sum()


def compiled_masked_ce_sums(logits,targets,mask,weight):
    global _COMPILED
    if not logits.is_cuda:return static_masked_ce_sums(logits,targets,mask,weight)
    if _COMPILED is None:_COMPILED=torch.compile(static_masked_ce_sums,dynamic=True,fullgraph=True)
    return _COMPILED(logits,targets,mask,weight)

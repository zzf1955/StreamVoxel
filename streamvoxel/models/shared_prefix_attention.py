"""Shared-prefix storage using existing PyTorch FlexAttention, no custom kernels."""
from __future__ import annotations
import torch
from torch.nn import functional as F

_MASKS={}
_FLEX=None


def shared_prefix_attention(q,pk,pv,lk,lv):
    b,h,n,d=q.shape
    prefix,local=pk.shape[1],lk.shape[2]
    if not prefix:return F.scaled_dot_product_attention(q,lk,lv,dropout_p=0.)
    pq=q.permute(1,0,2,3).reshape(1,h,b*n,d)
    k=torch.cat((pk[None],lk.permute(1,0,2,3).reshape(1,h,b*local,d)),2)
    v=torch.cat((pv[None],lv.permute(1,0,2,3).reshape(1,h,b*local,d)),2)
    def visible(batch,head,query,key):
        return (key<prefix)|(((key-prefix)//local)==(query//n))
    if q.is_cuda:
        from torch.nn.attention.flex_attention import flex_attention,create_block_mask
        global _FLEX
        if _FLEX is None:_FLEX=torch.compile(flex_attention,dynamic=False)
        identity=(b,n,prefix,local,str(q.device))
        if identity not in _MASKS:
            _MASKS[identity]=create_block_mask(visible,1,1,b*n,prefix+b*local,device=q.device,_compile=True)
            if len(_MASKS)>24:_MASKS.pop(next(iter(_MASKS)))
        # FlexAttention requires power-of-two head width; zero padding preserves
        # the original 96-channel dot product when the original scale is explicit.
        width=1<<(d-1).bit_length()
        if width!=d:
            pq=F.pad(pq,(0,width-d));k=F.pad(k,(0,width-d));v=F.pad(v,(0,width-d))
        out=_FLEX(pq,k,v,block_mask=_MASKS[identity],scale=d**-.5)[...,:d]
    else:
        qi=torch.arange(b*n,device=q.device)[:,None]
        ki=torch.arange(prefix+b*local,device=q.device)[None]
        mask=(ki<prefix)|(((ki-prefix)//local)==(qi//n))
        out=F.scaled_dot_product_attention(pq,k,v,attn_mask=mask)
    return out.reshape(h,b,n,d).permute(1,0,2,3).contiguous()

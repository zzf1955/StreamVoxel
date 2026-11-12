"""Patch2/4 blocks-only DiT: absolute Y features, XYZ Q/K RoPE, pre-norm residuals."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AxialRoPE(nn.Module):
    def __init__(self,head_dim,base=10000.):
        super().__init__()
        if head_dim % 6:
            raise ValueError("Each head must allocate an even width to each XYZ axis")
        self.axis_dim = head_dim//3
        self.register_buffer("frequencies",base**(-torch.arange(0,self.axis_dim,2).float()/self.axis_dim))

    def forward(self,x,positions_xyz):
        # x [B,H,N,D], positions [B,N,3]; compute phases in fp32.
        phase = positions_xyz.float()[...,None]*self.frequencies
        sin,cos = phase.sin()[:,None].to(x.dtype),phase.cos()[:,None].to(x.dtype)
        pairs = x.reshape(*x.shape[:-1],3,self.axis_dim//2,2)
        a,b = pairs.unbind(-1)
        return torch.stack((a*cos-b*sin,a*sin+b*cos),-1).flatten(-3)


class RotaryAttention(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width,3*width)
        self.proj = nn.Linear(width,width)
        self.rope = AxialRoPE(width//heads)

    def forward(self,x,positions):
        b,n,d = x.shape
        q,k,v = self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).unbind(2)
        q,k,v = (v.transpose(1,2) for v in (q,k,v))
        q,k = self.rope(q,positions),self.rope(k,positions)
        v = F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        return self.proj(v.transpose(1,2).reshape(b,n,d))


class UnconditionalBlock(nn.Module):
    def __init__(self,width,heads,ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(width,eps=1e-6)
        self.norm2 = nn.LayerNorm(width,eps=1e-6)
        self.attn = RotaryAttention(width,heads)
        self.mlp = nn.Sequential(nn.Linear(width,int(width*ratio)),nn.GELU(),
                                 nn.Linear(int(width*ratio),width))

    def forward(self,x,positions):
        x = x+self.attn(self.norm1(x),positions)
        return x+self.mlp(self.norm2(x))


class UnconditionalMD4DiT(nn.Module):
    def __init__(self,*,num_classes=304,voxel_shape=(256,16,16),patch_size=4,
                 hidden_dim=768,depth=25,num_heads=8,mlp_ratio=4.,
                 gradient_checkpointing=False,y_range=(-64.,192.)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.voxel_shape = tuple(voxel_shape)
        self.patch_size = patch_size
        self.hidden_dim,self.depth,self.num_heads = hidden_dim,depth,num_heads
        self.mlp_ratio = mlp_ratio
        self.gradient_checkpointing = gradient_checkpointing
        self.y_range = tuple(y_range)
        if (patch_size not in (2,4) or any(v<=0 or v%patch_size for v in self.voxel_shape)
                or hidden_dim%num_heads or (hidden_dim//num_heads)%6
                or depth<=0 or num_classes<=0 or y_range[1]<=y_range[0]):
            raise ValueError("Invalid patch rotary DiT configuration")
        self.patch_shape = tuple(v//patch_size for v in self.voxel_shape)
        self.target_patch = nn.Conv3d(num_classes+1,hidden_dim,patch_size,stride=patch_size)
        grid_yzx = torch.stack(torch.meshgrid(*(torch.arange(v).float() for v in self.patch_shape),
                                               indexing="ij"),-1).reshape(-1,3)
        self.register_buffer("patch_centers_xyz",(grid_yzx[:,[2,0,1]]*patch_size+patch_size/2))
        self.y_embed = nn.Sequential(nn.Linear(1,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        self.blocks = nn.ModuleList(UnconditionalBlock(hidden_dim,num_heads,mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden_dim,eps=1e-6)
        self.class_head = nn.Linear(hidden_dim,patch_size**3*num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.target_patch.weight.flatten(1))
        nn.init.zeros_(self.target_patch.bias)
        nn.init.zeros_(self.class_head.weight); nn.init.zeros_(self.class_head.bias)

    def config(self):
        return {k:getattr(self,k) for k in ("num_classes","voxel_shape","patch_size","hidden_dim",
            "depth","num_heads","mlp_ratio","gradient_checkpointing","y_range")}

    def patchify(self,tokens):
        # Direct scatter avoids F.one_hot's enormous intermediate int64 array.
        dtype = (torch.get_autocast_dtype("cuda") if tokens.is_cuda and torch.is_autocast_enabled("cuda")
                 else self.target_patch.weight.dtype)
        x = torch.zeros((len(tokens),self.num_classes+1,*tokens.shape[1:]),device=tokens.device,dtype=dtype)
        x.scatter_(1,tokens.long()[:,None],1.)
        return self.target_patch(x).flatten(2).transpose(1,2)

    def forward(self,tokens,*,timestep,origin_xyz=None):
        shape = tuple(tokens.shape[1:])
        if tokens.ndim!=4 or shape!=self.voxel_shape:
            raise ValueError("Invalid target shape")
        if timestep.shape!=(len(tokens),):
            raise ValueError("Invalid time shape")
        if origin_xyz is None:
            origin_xyz = tokens.new_tensor([0,-64,0],dtype=torch.float32)[None].expand(len(tokens),-1)
        if origin_xyz.shape!=(len(tokens),3):
            raise ValueError("Expected per-sample XYZ origin")
        local_centers = self.patch_centers_xyz
        centers = local_centers[None]+origin_xyz[:,None]
        # Common-origin shifts cancel in SA RoPE; local positions avoid large phases.
        positions = local_centers[None].expand(len(tokens),-1,-1)/self.patch_size
        y = 2*(centers[...,1:2]-self.y_range[0])/(self.y_range[1]-self.y_range[0])-1
        x = self.patchify(tokens)
        x = x+self.y_embed(y).to(x.dtype)
        for block in self.blocks:
            x = checkpoint(block,x,positions,use_reentrant=False) if self.training and self.gradient_checkpointing else block(x,positions)
        patches = self.class_head(self.final_norm(x))
        gy,gz,gx = (v//self.patch_size for v in shape)
        p = self.patch_size
        return patches.reshape(len(tokens),gy,gz,gx,p,p,p,self.num_classes).permute(
            0,7,1,4,2,5,3,6).reshape(len(tokens),self.num_classes,*shape)

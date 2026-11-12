"""Frozen patch4 MD4 backbone + full trainable copy with rotary cross attention."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from streamvoxel.models.unconditional_md4 import UnconditionalMD4DiT, AxialRoPE
from streamvoxel.models.shared_prefix_attention import shared_prefix_attention


def categorical_patch(conv, tokens, classes):
    dtype = torch.get_autocast_dtype("cuda") if tokens.is_cuda and torch.is_autocast_enabled("cuda") else conv.weight.dtype
    onehot = torch.zeros((len(tokens), classes + 1, *tokens.shape[1:]), device=tokens.device, dtype=dtype)
    onehot.scatter_(1, tokens.long()[:, None], 1.)
    return conv(onehot)


class RotaryCrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.width = heads, width
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)
        self.rope = AxialRoPE(width // heads)
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.backend = 'concat'

    def encode_kv(self, value, positions):
        # Unbatched shared bank: [heads, keys, head_dim].
        k, v = self.key_value(value).reshape(len(value), 2, self.heads, self.width // self.heads).unbind(1)
        k, v = k.transpose(0, 1), v.transpose(0, 1)
        return self.rope(k[None], positions[None])[0], v

    def forward(self, value, positions, prefix_kv, local, local_positions):
        b, n, d = value.shape
        q = self.query(self.norm(value)).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        q = self.rope(q, positions)
        k, v = self.key_value(local).reshape(b, local.shape[1], 2, self.heads, d // self.heads).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        k = self.rope(k, local_positions)
        pk, pv = prefix_kv
        if pk.ndim == 4:
            # Independent streaming samples: each has its own encoded L2 bank.
            # The original unbatched shared-prefix training path stays unchanged.
            if pk.shape[0] != b or pv.shape != pk.shape:
                raise ValueError('Invalid per-sample prefix K/V')
            k = torch.cat((pk, k), 2)
            v = torch.cat((pv, v), 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        elif self.backend == 'shared':
            out = shared_prefix_attention(q,pk,pv,k,v)
        elif pk.shape[1]:
            k = torch.cat((pk[None].expand(b, -1, -1, -1), k), 2)
            v = torch.cat((pv[None].expand(b, -1, -1, -1), v), 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class StreamingControlNet(nn.Module):
    def __init__(self, backbone, *, biome_classes=66, struct_classes=9, checkpointing=True, cell_chunk=2, biome_patch=1, biome_halo=16):
        super().__init__()
        if backbone.patch_size != 4:
            raise ValueError("ControlNet v1 requires the approved patch4 backbone")
        self.backbone = backbone.eval().requires_grad_(False)
        self.control = copy.deepcopy(backbone).requires_grad_(True)
        self.control.class_head = nn.Identity()
        self.control.final_norm = nn.Identity()
        if hasattr(self.control, "final_ada"):
            self.control.final_ada = nn.Identity()
        if getattr(self.control, "time_free", False):
            self.control._freeze_time_mlp()
        elif hasattr(self.control, "constant_time"):
            self.control.constant_time.requires_grad_(False)
        d, h = backbone.hidden_dim, backbone.num_heads
        self.cross = nn.ModuleList(RotaryCrossAttention(d, h) for _ in backbone.blocks)
        self.zero = nn.ModuleList(nn.Linear(d, d) for _ in backbone.blocks)
        for z in self.zero:
            nn.init.zeros_(z.weight); nn.init.zeros_(z.bias)
        self.l1_patch = copy.deepcopy(backbone.target_patch).requires_grad_(True)
        self.l2_patch = copy.deepcopy(backbone.target_patch).requires_grad_(True)
        self.l2_norm = nn.LayerNorm(d)
        self.l2_down = nn.Conv3d(d, d, 4, stride=4)
        nn.init.zeros_(self.l2_down.weight); nn.init.zeros_(self.l2_down.bias)
        with torch.no_grad():
            for i in range(d):
                self.l2_down.weight[i, i].fill_(1 / 64)
        self.biome_embed = nn.Embedding(biome_classes, d)
        if biome_patch not in (1,2,4):raise ValueError('Biome patch must be 1, 2 or 4 quart cells')
        self.biome_patch=biome_patch
        if biome_halo not in (4,16):raise ValueError('Biome halo must be 4 or 16 voxels')
        self.biome_halo=biome_halo
        if biome_patch>1:
            self.biome_down=nn.Conv3d(d,d,biome_patch,stride=biome_patch)
            nn.init.zeros_(self.biome_down.weight);nn.init.zeros_(self.biome_down.bias)
            with torch.no_grad():
                for i in range(d):self.biome_down.weight[i,i].fill_(1/biome_patch**3)
        if biome_patch==2:
            self.biome_size8=nn.Parameter(torch.empty(d))
            nn.init.normal_(self.biome_size8,std=.02)
        self.struct_classes = int(struct_classes)
        if self.struct_classes < 0:
            raise ValueError('struct_classes must be nonnegative')
        if self.struct_classes:
            self.struct_embed = nn.Embedding(self.struct_classes, d)
            self.struct_size = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.type_embed = nn.Embedding(4 if self.struct_classes else 3, d)
        self.size_embed = nn.Embedding(2, d)
        nn.init.normal_(self.type_embed.weight, std=.02)
        nn.init.normal_(self.size_embed.weight, std=.02)
        self.context_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.checkpointing, self.cell_chunk = checkpointing, int(cell_chunk)
        self.target_checkpointing = None
        self.attention_backend = 'concat'

    def set_runtime(self, *, target_checkpoint='all', attention_backend='concat', cell_chunk=None):
        if target_checkpoint not in ('all','none','blocks') or attention_backend not in ('concat','shared'):
            raise ValueError('Invalid checkpoint/attention runtime')
        self.target_checkpointing = target_checkpoint
        self.attention_backend = attention_backend
        for ca in self.cross:ca.backend=attention_backend
        if cell_chunk is not None:self.cell_chunk=int(cell_chunk)

    def config(self):
        return dict(backbone=self.backbone.config(), biome_classes=self.biome_embed.num_embeddings,
                    struct_classes=self.struct_classes, checkpointing=self.checkpointing,
                    cell_chunk=self.cell_chunk, condition_patch_trainable=True,biome_patch=self.biome_patch,
                    biome_halo=self.biome_halo)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def _encode_columns(self, tokens, level):
        conv = self.l1_patch if level == 0 else self.l2_patch
        def encode(part):
            x = categorical_patch(conv, part, self.backbone.num_classes)
            if level:
                x = F.gelu(self.l2_norm(x.movedim(1, -1))).movedim(-1, 1)
                x = self.l2_down(x)
            return x.flatten(2).transpose(1, 2)
        pieces = []
        for start in range(0, len(tokens), self.cell_chunk):
            part = tokens[start:start + self.cell_chunk]
            pieces.append(checkpoint(encode, part, use_reentrant=False) if self.training and self.checkpointing else encode(part))
        if not pieces:
            count = (tokens.shape[1] // (16 if level else 4)) * (1 if level else 16)
            return self.type_embed.weight.new_empty((0, count, self.backbone.hidden_dim))
        x = torch.cat(pieces)
        return x + self.type_embed.weight[level] + self.size_embed.weight[level]

    def encode_group(self, group):
        l2 = self._encode_columns(group["l2_blocks"], 1).flatten(0, 1)
        if self.struct_classes:
            struct = self.struct_embed(group["struct_labels"]) + self.struct_size(torch.log1p(group["struct_sizes"] / 4)) + self.type_embed.weight[3]
            l2 = torch.cat((l2, struct), 0)
        elif len(group.get('struct_labels', ())) or len(group['prefix_positions']) != group['l2_count']:
            raise ValueError('Struct-free model received struct tokens or positions')
        prefix = self.context_norm(l2)
        l1 = self.context_norm(self._encode_columns(group["l1_blocks"], 0))
        return dict(prefix=prefix, l1=l1)

    def prefix_kv(self, encoded, group, *, drop=None):
        value, positions = encoded["prefix"], group["prefix_positions"]
        if drop == "L2":
            count = group["l2_count"]
            value, positions = value[count:], positions[count:]
        return [ca.encode_kv(value, positions) for ca in self.cross]

    def local_condition(self, encoded, group, indices, *, drop=None):
        features, positions = [], []
        if drop != "L1":
            n = indices.condition_n if hasattr(indices,'condition_n') else int(group["n"][indices[0]])
            ids = group["l1_indices"][indices, :n]
            features.append(encoded["l1"][ids].flatten(1, 2))
            positions.append(group["l1_positions"][ids].flatten(1, 2))
        if drop != "biome":
            bio,coords=self._biome_condition(group,indices)
            features.append(bio)
            positions.append(coords)
        if not features:
            raise ValueError("No local keys")
        return torch.cat(features, 1), torch.cat(positions, 1)

    def biome_size_embedding(self):
        return self.biome_size8 if self.biome_patch==2 else self.size_embed.weight[int(self.biome_patch==4)]

    def _biome_condition(self, group, indices):
        if self.biome_halo==16 and self.biome_patch==1:
            values=self.biome_embed(group['biomes'][indices])
            positions=group['biome_positions'][indices]
        else:
            height=self.backbone.voxel_shape[0]//4
            labels=group['biomes'][indices].reshape(len(indices),height,12,12)
            positions=group['biome_positions'][indices].reshape(len(indices),height,12,12,3)
            if self.biome_halo==4:
                labels,positions=labels[:,:,3:9,3:9],positions[:,:,3:9,3:9]
            values=self.biome_embed(labels).permute(0,4,1,2,3)
            if self.biome_patch>1:
                p=self.biome_patch
                pad=(-labels.shape[-1])%p
                mask=values.new_ones((1,1,height,labels.shape[-2],labels.shape[-1]))
                fraction=F.avg_pool3d(F.pad(mask,(0,pad,0,pad)),p,p)
                values=self.biome_down(F.pad(values,(0,pad,0,pad)))/fraction
                coordinates=positions.permute(0,4,1,2,3)
                coordinates=F.avg_pool3d(F.pad(coordinates,(0,pad,0,pad)),p,p)/fraction
                positions=coordinates.flatten(2).transpose(1,2)
            else:positions=positions.reshape(len(indices),-1,3)
            values=values.flatten(2).transpose(1,2)
        values=values+self.type_embed.weight[2]+self.biome_size_embedding()
        return self.context_norm(values),positions

    def forward(self, tokens, *, timestep, origin_xyz, group_origin_xyz, prefix_kv, local, local_positions, scale=1.):
        if scale == 0:
            return self.backbone(tokens, timestep=timestep, origin_xyz=origin_xyz)
        bb, cc = self.backbone, self.control
        centers = bb.patch_centers_xyz[None] + origin_xyz[:, None]
        positions = (centers - group_origin_xyz[None, None]) / 4
        # Keep the original self-attention numerics; only CA needs a group reference.
        sa_positions = bb.patch_centers_xyz[None].expand(len(tokens), -1, -1) / 4
        y = 2 * (centers[..., 1:2] - bb.y_range[0]) / (bb.y_range[1] - bb.y_range[0]) - 1
        b = bb.patchify(tokens) + bb.y_embed(y)
        c = cc.patchify(tokens) + cc.y_embed(y)
        for i in range(bb.depth):
            def layer(b, c, positions, pk, pv, local, lp, index=i):
                c = cc.blocks[index](c, sa_positions)
                c = c + self.cross[index](c, positions, (pk, pv), local, lp)
                b = bb.blocks[index](b, sa_positions) + scale * self.zero[index](c)
                return b, c
            args = (b, c, positions, *prefix_kv[i], local, local_positions)
            policy=self.target_checkpointing or ('all' if self.checkpointing else 'none')
            if self.training and policy=='all':
                b,c=checkpoint(layer,*args,use_reentrant=False)
            elif self.training and policy=='blocks':
                c=checkpoint(cc.blocks[i],c,sa_positions,use_reentrant=False)
                c=c+self.cross[i](c,positions,prefix_kv[i],local,local_positions)
                b=checkpoint(bb.blocks[i],b,sa_positions,use_reentrant=False)+scale*self.zero[i](c)
            else:
                b,c=layer(*args)
        patches = bb.class_head(bb.final_norm(b))
        gy, gz, gx = bb.patch_shape
        return patches.reshape(len(tokens), gy, gz, gx, 4, 4, 4, bb.num_classes).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(len(tokens), bb.num_classes, *bb.voxel_shape)


def from_base_checkpoint(path, *, model_class=StreamingControlNet, **kwargs):
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    config = dict(saved["config"]["model"], gradient_checkpointing=False)
    state = saved["model"]
    # AdaLN checkpoints carry time_mlp. time_free ones also store constant_time; the 46k
    # time-conditioned checkpoint predates that parameter and leaves it unused.
    adaln = bool(config.get("time_free")) or any(str(key).startswith("time_mlp.") for key in state)
    if adaln:
        from streamvoxel.models.unconditional_md4_timefree import UnconditionalMD4DiT as AdaDiT
        config["time_free"] = bool(config.get("time_free"))
        fields = ("num_classes", "voxel_shape", "patch_size", "hidden_dim", "depth", "num_heads",
                  "mlp_ratio", "time_dim", "gradient_checkpointing", "y_range", "time_free")
        base = AdaDiT(**{key: config[key] for key in fields})
        missing, unexpected = base.load_state_dict(state, strict=False)
        if not config["time_free"]:
            missing = [key for key in missing if key != "constant_time"]
        if unexpected or missing:
            raise ValueError(f"Base checkpoint mismatch missing={missing} unexpected={unexpected}")
    else:
        config.pop("time_dim", None)
        config.pop("time_free", None)
        base = UnconditionalMD4DiT(**config)
        base.load_state_dict(state, strict=True)
    return model_class(base, **kwargs)

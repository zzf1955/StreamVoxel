"""Compact ControlNet with independent width and uniformly spaced backbone injections."""
from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from streamvoxel.models.streaming_controlnet import StreamingControlNet
from streamvoxel.models.unconditional_md4 import UnconditionalMD4DiT, UnconditionalBlock


class CompactStreamingControlNet(StreamingControlNet):
    def __init__(self, backbone, *, control_depth=13, control_hidden_dim=384, control_initialization='copied', **kwargs):
        depth = backbone.depth if control_depth is None else int(control_depth)
        width = backbone.hidden_dim if control_hidden_dim is None else int(control_hidden_dim)
        if not 1 <= depth <= backbone.depth or not 0 < width <= backbone.hidden_dim:
            raise ValueError("Control depth/width must not exceed frozen backbone")
        if width % backbone.num_heads or (width // backbone.num_heads) % 6:
            raise ValueError("Control width must support equal XYZ rotary head dimensions")
        if control_initialization not in ('copied','random_transformer'):
            raise ValueError('Unknown control initialization')
        super().__init__(backbone, **kwargs)
        self._resize_width(width)
        indices = ([0] if depth == 1 else
                   [round(i * (backbone.depth - 1) / (depth - 1)) for i in range(depth)])
        self._select_layers(indices)
        self.control_initialization=control_initialization
        if control_initialization=='random_transformer':
            self.control.blocks=nn.ModuleList(UnconditionalBlock(width,backbone.num_heads,backbone.mlp_ratio) for _ in indices)
            for block in self.control.blocks:
                for module in block.modules():
                    if isinstance(module,nn.Linear):
                        nn.init.xavier_uniform_(module.weight);nn.init.zeros_(module.bias)
            for projection in self.zero:
                nn.init.zeros_(projection.weight);nn.init.zeros_(projection.bias)

    def config(self):
        return dict(super().config(), control_depth=self.control.depth,
                    control_hidden_dim=self.control.hidden_dim,
                    control_mlp_ratio=self.control.mlp_ratio,
                    control_layer_indices=list(self.layer_indices),
                    control_initialization=("random_transformer_with_zero_output_projection" if self.control_initialization=='random_transformer'
                                            else "partitioned_leading_channels_with_recomputed_rope"))

    def _resize_width(self, width):
        old_width = self.control.hidden_dim
        if width == old_width:
            return
        if width > old_width:
            raise ValueError('Width probe supports reduction only')
        old = {k:v for k,v in self.state_dict().items() if not k.startswith('backbone.')}
        config = dict(self.backbone.config(), hidden_dim=width, gradient_checkpointing=False)
        template = StreamingControlNet(type(self.backbone)(**config),
            biome_classes=self.biome_embed.num_embeddings, struct_classes=self.struct_classes,
            checkpointing=self.checkpointing, cell_chunk=self.cell_chunk,
            biome_patch=self.biome_patch, biome_halo=self.biome_halo)
        for name, module in template._modules.items():
            if name != 'backbone':
                setattr(self, name, module)
        for name, value in template._parameters.items():
            setattr(self, name, value)
        self.zero = nn.ModuleList(nn.Linear(width, old_width) for _ in self.backbone.blocks)
        converted = {}
        for name, value in self.state_dict().items():
            if name.startswith('backbone.'):
                continue
            source = old[name]
            if name.endswith('rope.frequencies'):
                converted[name] = value  # Recompute XYZ frequencies for the new head width.
                continue
            partitions = (3 if name.startswith('control.blocks.') and '.attn.qkv.' in name else
                          2 if name.startswith('cross.') and '.key_value.' in name else 1)
            if partitions > 1:
                shaped = source.reshape(partitions, old_width, *source.shape[1:])
                converted[name] = shaped[(slice(None),slice(0,width)) +
                    tuple(slice(0,size) for size in value.shape[1:])].reshape(value.shape).contiguous()
            else:
                if source.ndim != value.ndim or any(a < b for a,b in zip(source.shape,value.shape)):
                    raise ValueError(f'Cannot narrow {name}: {source.shape} -> {value.shape}')
                converted[name] = source[tuple(slice(0,size) for size in value.shape)].contiguous()
        missing, unexpected = self.load_state_dict(converted, strict=False)
        if unexpected or any(not key.startswith('backbone.') for key in missing):
            raise ValueError('Incomplete narrowed ControlNet state')

    def _encode_columns(self, tokens, level):
        if not len(tokens):
            count = (tokens.shape[1] // (16 if level else 4)) * (1 if level else 16)
            return self.type_embed.weight.new_empty((0,count,self.control.hidden_dim))
        return super()._encode_columns(tokens,level)

    def _select_layers(self, indices):
        indices = tuple(indices)
        if not indices or sorted(set(indices)) != list(indices) or indices[-1] >= self.backbone.depth:
            raise ValueError('Expected unique increasing backbone layer indices')
        self.control.blocks = nn.ModuleList(self.control.blocks[i] for i in indices)
        self.cross = nn.ModuleList(self.cross[i] for i in indices)
        self.zero = nn.ModuleList(self.zero[i] for i in indices)
        self.control.depth = len(indices)
        self.layer_indices = indices
        self.layer_map = {i: j for j, i in enumerate(indices)}

    def forward(self, tokens, *, timestep, origin_xyz, group_origin_xyz,
                prefix_kv, local, local_positions, scale=1.):
        if scale == 0:
            return self.backbone(tokens, timestep=timestep, origin_xyz=origin_xyz)
        bb, cc = self.backbone, self.control
        centers = bb.patch_centers_xyz[None] + origin_xyz[:, None]
        reference = (group_origin_xyz[None, None] if group_origin_xyz.ndim == 1
                     else group_origin_xyz[:, None])
        positions = (centers - reference) / 4
        sa_positions = bb.patch_centers_xyz[None].expand(len(tokens), -1, -1) / 4
        y = 2 * (centers[..., 1:2] - bb.y_range[0]) / (bb.y_range[1] - bb.y_range[0]) - 1
        b = bb.patchify(tokens) + bb.y_embed(y)
        c = cc.patchify(tokens) + cc.y_embed(y)
        timed = hasattr(bb, "time_encoding")
        bt = bb.time_encoding(timestep) if timed else None
        ct = cc.time_encoding(timestep) if timed else None
        policy = self.target_checkpointing or ('all' if self.checkpointing else 'none')
        for i in range(bb.depth):
            if i not in self.layer_map:
                if timed:
                    b = (checkpoint(bb.blocks[i], b, bt, sa_positions, use_reentrant=False)
                         if self.training and policy != 'none' else bb.blocks[i](b, bt, sa_positions))
                else:
                    b = (checkpoint(bb.blocks[i], b, sa_positions, use_reentrant=False)
                         if self.training and policy != 'none' else bb.blocks[i](b, sa_positions))
                continue
            j = self.layer_map[i]
            def layer(b, c, positions, pk, pv, local, lp, index=i, control_index=j, bt=bt, ct=ct):
                if timed:
                    c = cc.blocks[control_index](c, ct, sa_positions)
                    c = c + self.cross[control_index](c, positions, (pk, pv), local, lp)
                    b = bb.blocks[index](b, bt, sa_positions) + scale * self.zero[control_index](c)
                else:
                    c = cc.blocks[control_index](c, sa_positions)
                    c = c + self.cross[control_index](c, positions, (pk, pv), local, lp)
                    b = bb.blocks[index](b, sa_positions) + scale * self.zero[control_index](c)
                return b, c
            args = (b, c, positions, *prefix_kv[j], local, local_positions)
            if self.training and policy == 'all':
                b, c = checkpoint(layer, *args, use_reentrant=False)
            elif self.training and policy == 'blocks':
                if timed:
                    c = checkpoint(cc.blocks[j], c, ct, sa_positions, use_reentrant=False)
                    c = c + self.cross[j](c, positions, prefix_kv[j], local, local_positions)
                    b = checkpoint(bb.blocks[i], b, bt, sa_positions, use_reentrant=False) + scale * self.zero[j](c)
                else:
                    c = checkpoint(cc.blocks[j], c, sa_positions, use_reentrant=False)
                    c = c + self.cross[j](c, positions, prefix_kv[j], local, local_positions)
                    b = checkpoint(bb.blocks[i], b, sa_positions, use_reentrant=False) + scale * self.zero[j](c)
            else:
                b, c = layer(*args)
        if timed:
            from streamvoxel.models.unconditional_md4_timefree import modulate
            shift, gain = bb.final_ada(bt).chunk(2, -1)
            patches = bb.class_head(modulate(bb.final_norm(b), shift, gain))
        else:
            patches = bb.class_head(bb.final_norm(b))
        gy, gz, gx = bb.patch_shape
        return patches.reshape(len(tokens), gy, gz, gx, 4, 4, 4, bb.num_classes).permute(
            0, 7, 1, 4, 2, 5, 3, 6).reshape(len(tokens), bb.num_classes, *bb.voxel_shape)

"""Compact production architecture, checkpoint, and measured-prototype contracts."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from types import SimpleNamespace
from streamvoxel.models.compact_controlnet import CompactStreamingControlNet
from streamvoxel.models.unconditional_md4 import UnconditionalMD4DiT
from streamvoxel.train.train_controlnet import control_state, load_control_state, train_step
from streamvoxel.train.validation_controlnet import sample_full_mask


def fixture():
    torch.manual_seed(54)
    base=UnconditionalMD4DiT(num_classes=3,voxel_shape=(16,16,16),hidden_dim=48,
                            depth=5,num_heads=4)
    with torch.no_grad():
        base.class_head.weight.normal_(std=.02)
    model=CompactStreamingControlNet(base,control_depth=3,control_hidden_dim=24,
                                    checkpointing=False,biome_halo=4,biome_patch=2)
    group=dict(l1_blocks=torch.randint(3,(1,16,16,16)),l2_blocks=torch.randint(3,(1,16,16,16)),
        struct_labels=torch.tensor([1]),struct_sizes=torch.ones(1,3),prefix_positions=torch.randn(2,3),l2_count=1,
        l1_indices=torch.zeros(2,1,dtype=torch.long),l1_positions=torch.randn(1,64,3),n=torch.ones(2,dtype=torch.long),
        biomes=torch.randint(66,(2,4*12*12)),biome_positions=torch.randn(2,4*12*12,3),
        targets=torch.randint(3,(2,16,16,16)),target_origins=torch.tensor([[0.,-64.,0.]]).expand(2,-1),
        group_origin=torch.tensor([0.,-64.,0.]))
    return model,group


def forward(model,group):
    e=model.encode_group(group);kv=model.prefix_kv(e,group)
    local,lp=model.local_condition(e,group,torch.arange(2))
    return model(group['targets'],timestep=torch.tensor([.3,.7]),origin_xyz=group['target_origins'],
                 group_origin_xyz=group['group_origin'],prefix_kv=kv,local=local,local_positions=lp)


def test_zero_residual_and_checkpoint_roundtrip():
    model,g=fixture()
    torch.testing.assert_close(forward(model,g),model.backbone(g['targets'],timestep=torch.tensor([.3,.7]),
                               origin_xyz=g['target_origins']),rtol=0,atol=0)
    cfg=model.config()
    assert cfg['control_layer_indices']==[0,2,4] and cfg['control_hidden_dim']==24
    assert cfg['backbone']['depth']==5 and cfg['backbone']['hidden_dim']==48
    assert model.zero[0].weight.shape==(48,24)
    with torch.no_grad():
        for z in model.zero:z.weight.normal_(std=.01)
    clone=CompactStreamingControlNet(copy.deepcopy(model.backbone),control_depth=cfg['control_depth'],
        control_hidden_dim=cfg['control_hidden_dim'],biome_halo=cfg['biome_halo'],
        biome_patch=cfg['biome_patch'],checkpointing=False)
    load_control_state(clone,control_state(model))
    assert clone.config()==cfg
    torch.testing.assert_close(forward(model,g),forward(clone,g),rtol=0,atol=0)
    assert not any(k.startswith('backbone.') for k in control_state(model))
    assert model.l1_patch.weight.data_ptr()!=model.l2_patch.weight.data_ptr()


def test_four_condition_variants_and_full_mask():
    model,g=fixture();model.eval();ids=torch.arange(2)
    e=model.encode_group(g)
    for drop in (None,'L1','L2','biome'):
        local,lp=model.local_condition(e,g,ids,drop=drop)
        out=model(g['targets'],timestep=torch.tensor([.3,.7]),origin_xyz=g['target_origins'],
            group_origin_xyz=g['group_origin'],prefix_kv=model.prefix_kv(e,g,drop=drop),local=local,local_positions=lp)
        assert out.shape==(2,3,16,16,16) and torch.isfinite(out).all()
    pred,ce,calls=sample_full_mask(model,g,e,ids,steps=8)
    assert calls==8 and (pred<3).all() and torch.isfinite(ce).all()


def test_random_transformer_zero_outputs_and_gradient_opening():
    copied,g=fixture()
    model=CompactStreamingControlNet(copy.deepcopy(copied.backbone),control_depth=3,control_hidden_dim=24,
        control_initialization='random_transformer',checkpointing=False,biome_halo=4,biome_patch=2)
    assert model.config()['control_initialization']=='random_transformer_with_zero_output_projection'
    assert not torch.equal(model.control.blocks[0].attn.qkv.weight,copied.control.blocks[0].attn.qkv.weight)
    torch.testing.assert_close(model.l1_patch.weight,copied.l1_patch.weight)
    for projection in model.zero:
        assert torch.count_nonzero(projection.weight)==0 and torch.count_nonzero(projection.bias)==0
    torch.testing.assert_close(forward(model,g),model.backbone(g['targets'],timestep=torch.tensor([.3,.7]),
                              origin_xyz=g['target_origins']),rtol=0,atol=0)
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-3)
    env=SimpleNamespace(device=torch.device('cpu'),enabled=False,world_size=1,rank=0,is_main=True)
    for i in range(3):
        torch.manual_seed(71)
        result=train_step(model,optimizer,g,env,microbatch=2)
        assert torch.isfinite(torch.tensor(result['train_loss']))
        if i==0:assert model.zero[0].weight.norm()>0
        if i==1:assert model.control.blocks[0].mlp[0].weight.grad.norm()>0
    assert model.control.blocks[0].attn.qkv.weight.grad.norm()>0
    assert model.l1_patch.weight.grad.norm()>0
    assert all(p.grad is None for p in model.backbone.parameters())


@pytest.mark.parametrize('depth,width',[(6,24),(0,24),(3,64),(3,32)])
def test_invalid_control_geometry(depth,width):
    base=UnconditionalMD4DiT(num_classes=3,voxel_shape=(16,16,16),hidden_dim=48,depth=5,num_heads=4)
    with pytest.raises(ValueError):
        CompactStreamingControlNet(base,control_depth=depth,control_hidden_dim=width)

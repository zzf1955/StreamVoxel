"""Critical spatial, frozen-base, gradient-sharing, and generation contracts."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from streamvoxel.common.controlnet_dataset import ring_targets, known_neighbors, validation_plans, full_mask_plans, training_validation_plans
from streamvoxel.models.unconditional_md4 import UnconditionalMD4DiT
from streamvoxel.models.streaming_controlnet import StreamingControlNet
from streamvoxel.train.train_controlnet import air_compact_ids, train_step, control_state, load_control_state, save_checkpoint
from streamvoxel.train.validation_controlnet import sample_full_mask, confusion_metrics, evaluate
from streamvoxel.objectives.md4 import strata_mask, weighted_masked_ce_sums
from streamvoxel.data.streaming_controlnet import ControlDataset
from streamvoxel.common.controlnet_dataset import context_plan
from streamvoxel.models.shared_prefix_attention import shared_prefix_attention
from streamvoxel.objectives.controlnet_compiled_ce import static_masked_ce_sums


def fixture():
    torch.manual_seed(18)
    base=UnconditionalMD4DiT(num_classes=5,voxel_shape=(16,16,16),hidden_dim=24,depth=2,
                            num_heads=4,mlp_ratio=2)
    with torch.no_grad():
        base.class_head.weight.normal_(std=.03)
    model=StreamingControlNet(base,biome_classes=4,struct_classes=2,checkpointing=False,cell_chunk=1)
    grid=torch.stack(torch.meshgrid(torch.arange(4),torch.arange(4),torch.arange(4),indexing='ij'),-1).reshape(-1,3)
    pos=(grid[:,[2,0,1]].float()+.5)
    group=dict(l2_blocks=torch.randint(5,(1,16,16,16)),l1_blocks=torch.randint(5,(1,16,16,16)),
        struct_labels=torch.tensor([1]),struct_sizes=torch.tensor([[16.,16.,16.]]),
        prefix_positions=torch.tensor([[2.,2.,2.],[2.,2.,2.]]),l2_count=1,
        l1_indices=torch.zeros((2,3),dtype=torch.long),l1_positions=pos[None],
        biomes=torch.randint(4,(2,16)),biome_positions=pos[None,:16].repeat(2,1,1),
        targets=torch.randint(5,(2,16,16,16)),n=torch.ones(2,dtype=torch.long),
        target_origins=torch.tensor([[16.,-64.,0.],[16.,-64.,16.]]),group_origin=torch.tensor([0.,-64.,0.]))
    return model,group


def test_ring_and_validation_panels():
    targets=ring_targets(32)
    assert len(targets)==132
    n=[len(known_neighbors(t,32)) for t in targets]
    assert [n.count(k) for k in (1,2,3)]==[4,8,120]
    assert all(not(0<=z<32 and 0<=x<32) for z,x in targets)
    plans=validation_plans([63,152]);full=full_mask_plans(plans)
    assert sum(len(p['targets']) for p in plans)==1536
    assert sum(len(p['targets']) for p in full)==48
    for k in (1,2,3):
        assert sum(p['n'].count(k) for p in plans)==512
        assert sum(p['n'].count(k) for p in full)==16
    assert validation_plans([63,152])==plans


def test_world_origins_remain_float32_for_bf16_forward():
    d=ControlDataset.__new__(ControlDataset)
    d.manifest={'worlds':{'0':{'origin_xyz':[160,-64,-32]}}}
    d.labels={'0':[]}
    biomes=np.zeros((64,512,512),dtype=np.uint16)
    d._world=lambda wi:(None,biomes)
    d.tiles=lambda wi,coords:np.zeros((len(coords),256,16,16),dtype=np.int16)
    g=d.gather(context_plan(0,[2,2],side=3,targets=3,balanced=True))
    assert g['group_origin'].dtype==torch.float32
    assert g['target_origins'].dtype==torch.float32
    assert g['prefix_positions'].dtype==torch.float32
    assert sorted(g['n'].tolist())==[1,2,3]


def test_zero_control_matches_frozen_base_and_branches_independent():
    model,g=fixture();model.eval()
    e=model.encode_group(g);kv=model.prefix_kv(e,g);ids=torch.arange(2)
    local,lp=model.local_condition(e,g,ids)
    t=torch.tensor([.4,.8])
    actual=model(g['targets'],timestep=t,origin_xyz=g['target_origins'],group_origin_xyz=g['group_origin'],
                 prefix_kv=kv,local=local,local_positions=lp)
    expected=model.backbone(g['targets'],timestep=t,origin_xyz=g['target_origins'])
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
    assert model.l1_patch.weight.data_ptr()!=model.l2_patch.weight.data_ptr()
    assert all(p.requires_grad for p in model.l1_patch.parameters())
    assert all(p.requires_grad for p in model.l2_patch.parameters())
    assert all(not p.requires_grad for p in model.backbone.parameters())


@pytest.mark.parametrize('checkpointing',[False,True])
def test_shared_vjp_matches_naive_microbatch_gradients(checkpointing):
    model,g=fixture();model.checkpointing=checkpointing
    with torch.no_grad():
        for z in model.zero:z.weight.normal_(std=.02)
    other=copy.deepcopy(model)
    opt=torch.optim.SGD(model.trainable_parameters(),lr=.001)
    opt2=torch.optim.SGD(other.trainable_parameters(),lr=.001)
    env=SimpleNamespace(device=torch.device('cpu'),enabled=False,world_size=1)
    torch.manual_seed(332);train_step(model,opt,g,env,microbatch=1)
    assert model.l1_patch.weight.grad.abs().sum()>0
    assert model.l2_patch.weight.grad.abs().sum()>0
    torch.manual_seed(332)
    x,mask,t,w=strata_mask(g['targets'],5,[0],band=8)
    encoded=other.encode_group(g)
    kv=other.prefix_kv(encoded,g)
    loss=0
    for i in range(2):
        ids=torch.tensor([i]);local,lp=other.local_condition(encoded,g,ids)
        logits=other(x[ids],timestep=t[ids],origin_xyz=g['target_origins'][ids],group_origin_xyz=g['group_origin'],
                    prefix_kv=kv,local=local,local_positions=lp)
        loss=loss+weighted_masked_ce_sums(logits,g['targets'][ids],mask[ids],w[ids])[0]/mask.sum()
    loss.backward();torch.nn.utils.clip_grad_norm_(list(other.trainable_parameters()),1.)
    opt2.step()
    for (n,a),(m,b) in zip(model.named_parameters(),other.named_parameters()):
        assert n==m
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6,msg=n)
    assert all(p.grad is None for p in model.backbone.parameters())


def test_budgeted_generation_ce_and_control_roundtrip():
    model,g=fixture();model.eval()
    state=control_state(model);clone=copy.deepcopy(model);load_control_state(clone,state)
    encoded=model.encode_group(g)
    a,ce,calls=sample_full_mask(model,g,encoded,torch.arange(2),steps=8,case_ids=[4,9])
    b,_,_=sample_full_mask(model,g,encoded,torch.arange(2),steps=8,case_ids=[4,9])
    assert calls==8 and a.max()<5 and a.min()>=0
    assert torch.isfinite(ce).all()
    assert torch.equal(a,b)
    metrics=confusion_metrics(torch.eye(5),[0])
    assert metrics['acc']==1 and metrics['macro_f1']==1


def test_removed_modalities_really_remove_keys():
    model,g=fixture();e=model.encode_group(g)
    full=model.prefix_kv(e,g)
    dropped=model.prefix_kv(e,g,drop='L2')
    assert full[0][0].shape[1]==2 and dropped[0][0].shape[1]==1
    ids=torch.arange(2)
    ordinary,_=model.local_condition(e,g,ids)
    no_l1,_=model.local_condition(e,g,ids,drop='L1')
    no_biome,_=model.local_condition(e,g,ids,drop='biome')
    assert ordinary.shape[1]==80 and no_l1.shape[1]==16 and no_biome.shape[1]==64


def test_validation_disables_autograd_and_reports_four_variants(tmp_path):
    _,g=fixture()
    base=UnconditionalMD4DiT(num_classes=304,voxel_shape=(16,16,16),hidden_dim=24,
                            depth=1,num_heads=4,mlp_ratio=2)
    model=StreamingControlNet(base,biome_classes=4,struct_classes=2,checkpointing=False)
    encode=model.encode_group
    def guarded(group):
        assert not torch.is_grad_enabled()
        return encode(group)
    model.encode_group=guarded
    mapping=list(range(304))
    (tmp_path/'vocab.json').write_text(json.dumps({'idToCleanName':['air']+['stone']*303}))
    data=SimpleNamespace(root=tmp_path,manifest={'splits':{'validation':[0]},'vocabulary_file':'vocab.json'},
                         mapping={'compact_to_global':mapping},gather=lambda plan:g)
    env=SimpleNamespace(device=torch.device('cpu'),enabled=False,world_size=1,rank=0,is_main=True)
    result=evaluate(model,data,{'ordinary':[{'world':0}],'full_mask':[]},env,microbatch=2)
    assert set(result['ordinary'])=={'full','no_L1','no_L2','no_biome'}
    assert all(result['ordinary'][k]['1']['samples']==2 for k in result['ordinary'])
    assert model.training and all(p.grad is None for p in model.parameters())


def test_shared_attention_matches_joint_softmax_and_all_gradients():
    torch.manual_seed(83)
    values=[torch.randn(*shape,dtype=torch.float64,requires_grad=True) for shape in
            [(3,2,7,6),(2,11,6),(2,11,6),(3,2,13,6),(3,2,13,6)]]
    q,pk,pv,lk,lv=values
    actual=shared_prefix_attention(*values)
    k=torch.cat((pk[None].expand(3,-1,-1,-1),lk),2)
    v=torch.cat((pv[None].expand(3,-1,-1,-1),lv),2)
    expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    grad=torch.randn_like(actual)
    a=torch.autograd.grad(actual,values,grad,retain_graph=True)
    b=torch.autograd.grad(expected,values,grad)
    torch.testing.assert_close(actual,expected)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)


@pytest.mark.parametrize('policy',['none','blocks'])
def test_target_checkpoint_policies_preserve_parameter_gradients(policy):
    model,g=fixture()
    with torch.no_grad():
        for z in model.zero:z.weight.normal_(std=.02)
    ref=copy.deepcopy(model)
    model.set_runtime(target_checkpoint=policy,attention_backend='shared')
    ref.set_runtime(target_checkpoint='all',attention_backend='concat')
    ids=torch.arange(2)
    def compute(m):
        encoded=m.encode_group(g);kv=m.prefix_kv(encoded,g);local,lp=m.local_condition(encoded,g,ids)
        out=m(g['targets'],timestep=torch.tensor([.4,.8]),origin_xyz=g['target_origins'],
              group_origin_xyz=g['group_origin'],prefix_kv=kv,local=local,local_positions=lp)
        out.square().mean().backward()
        return out
    a,b=compute(model),compute(ref)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    for (n,x),(k,y) in zip(model.named_parameters(),ref.named_parameters()):
        assert n==k
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=3e-4,atol=2e-6,msg=n)


def test_multiple_contexts_use_one_global_denominator_and_optimizer_step():
    model,g=fixture();other=copy.deepcopy(model)
    with torch.no_grad():
        for m in [model,other]:
            for z in m.zero:z.weight.fill_(.01)
    opt=torch.optim.SGD(model.trainable_parameters(),lr=.001)
    opt2=torch.optim.SGD(other.trainable_parameters(),lr=.001)
    env=SimpleNamespace(device=torch.device('cpu'),enabled=False,world_size=1)
    torch.manual_seed(991);train_step(model,opt,[g,g],env,microbatch=1)
    torch.manual_seed(991)
    batches=[strata_mask(g['targets'],5,[0],band=8) for _ in range(2)]
    denominator=sum(b[1].sum() for b in batches)
    loss=0
    for x,mask,t,w in batches:
        e=other.encode_group(g);kv=other.prefix_kv(e,g);ids=torch.arange(2)
        local,lp=other.local_condition(e,g,ids)
        logits=other(x,timestep=t,origin_xyz=g['target_origins'],group_origin_xyz=g['group_origin'],
                     prefix_kv=kv,local=local,local_positions=lp)
        loss+=weighted_masked_ce_sums(logits,g['targets'],mask,w)[0]/denominator
    loss.backward();torch.nn.utils.clip_grad_norm_(list(other.trainable_parameters()),1.);opt2.step()
    for (n,a),(k,b) in zip(model.named_parameters(),other.named_parameters()):
        assert n==k;torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6,msg=n)


def test_static_loss_preserves_masked_loss_and_gradients():
    torch.manual_seed(57)
    logits=torch.randn(3,5,8,4,4,requires_grad=True)
    targets=torch.randint(5,(3,8,4,4));mask=torch.rand(targets.shape)>.4
    weight=torch.tensor([.2,1.,7.])
    a=static_masked_ce_sums(logits,targets,mask,weight)
    b=weighted_masked_ce_sums(logits,targets,mask,weight)
    torch.testing.assert_close(a[0],b[0]);assert a[1]==b[1] and a[2]==b[2]
    ga=torch.autograd.grad(a[0],logits,retain_graph=True)[0]
    gb=torch.autograd.grad(b[0],logits)[0]
    torch.testing.assert_close(ga,gb)


def test_biome_patch4_uses_spatial_grid_and_is_trainable():
    model,g=fixture()
    compressed=StreamingControlNet(copy.deepcopy(model.backbone),biome_classes=4,struct_classes=2,
                                    checkpointing=False,biome_patch=4)
    load_control_state(compressed,control_state(model),allow_new_biome=True)
    g['biomes']=torch.randint(4,(2,4*12*12))
    grid=torch.stack(torch.meshgrid(torch.arange(4),torch.arange(12),torch.arange(12),indexing='ij'),-1)
    g['biome_positions']=(grid[..., [2,0,1]].float()+.5).reshape(1,-1,3).repeat(2,1,1)
    e=compressed.encode_group(g)
    bio,pos=compressed.local_condition(e,g,torch.arange(2),drop='L1')
    assert bio.shape==(2,9,24) and pos.shape==(2,9,3)
    torch.testing.assert_close(pos[0,0],torch.tensor([2.,2.,2.]))
    bio.square().sum().backward()
    assert compressed.biome_down.weight.grad is not None
    assert compressed.biome_embed.weight.grad is not None


def test_runtime_validation_uses_training_side_and_preserves_full_budget():
    manifest={'splits':{'validation':[63,152]}}
    original=validation_plans([63,152])
    packaged=dict(ordinary=original,full_mask=full_mask_plans(original))
    assert training_validation_plans(manifest,packaged,side=32) is packaged
    plans=training_validation_plans(manifest,packaged,side=8)
    assert all(p['side']==8 for kind in ('ordinary','full_mask') for p in plans[kind])
    assert sum(len(p['targets']) for p in plans['ordinary'])==1536
    assert sum(len(p['targets']) for p in plans['full_mask'])==48
    for n in (1,2,3):
        assert sum(p['n'].count(n) for p in plans['ordinary'])==512
        assert sum(p['n'].count(n) for p in plans['full_mask'])==16
    assert plans==training_validation_plans(manifest,packaged,side=8)
    assert all(p['side']==32 for p in packaged['ordinary'])


def test_voxel_weights_match_between_static_and_reference_loss():
    torch.manual_seed(57)
    logits=torch.randn(2,5,8,4,4,requires_grad=True)
    targets=torch.randint(5,(2,8,4,4));mask=torch.rand(targets.shape)>.4
    weight=torch.rand(targets.shape)
    a=static_masked_ce_sums(logits,targets,mask,weight)
    b=weighted_masked_ce_sums(logits,targets,mask,weight)
    torch.testing.assert_close(a[0],b[0]);assert a[1]==b[1] and a[2]==b[2]
    ga=torch.autograd.grad(a[0],logits,retain_graph=True)[0]
    gb=torch.autograd.grad(b[0],logits)[0]
    torch.testing.assert_close(ga,gb)


def test_latest_checkpoint_is_replaced(tmp_path):
    model,g=fixture()
    env=SimpleNamespace(device=torch.device('cpu'),enabled=False,world_size=1,rank=0,is_main=True)
    opt=torch.optim.SGD(model.trainable_parameters(),lr=0.)
    save_checkpoint(tmp_path,model,opt,{},dict(global_batch=4),2,env,numbered=False)
    save_checkpoint(tmp_path,model,opt,{},dict(global_batch=4),4,env,numbered=False)
    assert list(tmp_path.glob('checkpoint_step*.pt'))==[]
    assert torch.load(tmp_path/'checkpoint_latest.pt',weights_only=False)['step']==4
    names=['minecraft:air','minecraft:cave_air','stone','minecraft:void_air']
    assert air_compact_ids(dict(compact_to_clean_name=names))==[0,1,3]

import torch

torch.set_num_threads(2)

from streamvoxel.models.unconditional_md4 import AxialRoPE,UnconditionalMD4DiT
from streamvoxel.objectives.md4 import md4_mask,weighted_masked_ce_sums
from streamvoxel.train.train_base import RankBatches,sample_unconditional
from streamvoxel.data.batch_prefetch import BatchPrefetcher,prepare_batch


def tiny():
    return UnconditionalMD4DiT(num_classes=4,voxel_shape=(8,8,4),hidden_dim=48,
                               depth=2,num_heads=4)


def test_patch2_token_centers_output_voxel_order_backward_and_sampling():
    model=UnconditionalMD4DiT(num_classes=4,voxel_shape=(8,4,4),patch_size=2,
        hidden_dim=48,depth=2,num_heads=4)
    target=torch.randint(4,(2,8,4,4))
    assert model.patch_shape==(4,2,2)
    assert model.patchify(target).shape==(2,16,48)
    assert model.patch_centers_xyz[:4].tolist()==[[1.,1.,1.],[3.,1.,1.],[1.,1.,3.],[3.,1.,3.]]
    assert model.patch_centers_xyz[-1].tolist()==[3.,7.,3.]
    assert model.class_head.out_features==8*4
    with torch.no_grad():
        bias=torch.arange(8*4).reshape(8,4).float()
        model.class_head.bias.copy_(bias.flatten())
    out=model(target,timestep=torch.tensor([.2,.7]))
    assert out.shape==(2,4,8,4,4)
    for y in range(8):
        for z in range(4):
            for x in range(4):
                assert torch.equal(out[0,:,y,z,x],bias[(y%2)*4+(z%2)*2+x%2])
    x,mask,t,w=md4_mask(target,4,times=torch.tensor([.2,.8]))
    loss,_,count=weighted_masked_ce_sums(model(x,timestep=t),target,mask,w)
    (loss/count).backward()
    assert torch.isfinite(model.class_head.weight.grad).all()
    assert model.class_head.weight.grad.abs().sum()>0
    sample=sample_unconditional(model,steps=3,generator=torch.Generator().manual_seed(90))
    assert sample.shape==(1,8,4,4) and sample.max()<4


def test_doubled_mlp_preserves_backbone_and_has_finite_backward():
    model=UnconditionalMD4DiT(num_classes=4,voxel_shape=(8,8,4),hidden_dim=48,
        depth=2,num_heads=4,mlp_ratio=8.)
    assert model.config()['mlp_ratio']==8.
    for block in model.blocks:
        assert block.mlp[0].weight.shape==(384,48)
        assert block.mlp[2].weight.shape==(48,384)
    target=torch.randint(4,(2,8,8,4))
    x,mask,t,w=md4_mask(target,4,times=torch.tensor([.2,.8]))
    logits=model(x,timestep=t)
    loss,_,count=weighted_masked_ce_sums(logits,target,mask,w)
    (loss/count).backward()
    assert torch.isfinite(model.class_head.weight.grad).all()
    assert model.class_head.weight.grad.abs().sum()>0


def test_doubled_hidden_keeps_mlp_ratio_and_xyz_rope_partition():
    model=UnconditionalMD4DiT(num_classes=4,voxel_shape=(8,8,4),hidden_dim=96,
        depth=2,num_heads=4,mlp_ratio=4.)
    assert model.config()['hidden_dim']==96 and model.config()['mlp_ratio']==4.
    for block in model.blocks:
        assert block.mlp[0].weight.shape==(384,96)
        assert block.mlp[2].weight.shape==(96,384)
    target=torch.randint(4,(2,8,8,4));x,mask,t,w=md4_mask(target,4,times=torch.tensor([.3,.7]))
    logits=model(x,timestep=t)
    assert logits.shape==(2,4,8,8,4)
    loss,_,count=weighted_masked_ce_sums(logits,target,mask,w);(loss/count).backward()
    assert torch.isfinite(model.class_head.weight.grad).all()


def test_rope_relative_shift_norm_and_xyz_axis_partition():
    torch.manual_seed(3)
    rope=AxialRoPE(12)
    q,k=torch.randn(2,3,5,12),torch.randn(2,3,5,12)
    p=torch.randn(2,5,3)
    a,b=rope(q,p),rope(k,p)
    shifted=p+torch.tensor([23.,-9.,15.])
    assert torch.allclose(a.square().sum(-1),q.square().sum(-1),atol=3e-6)
    original=a@b.transpose(-1,-2)
    translated=rope(q,shifted)@rope(k,shifted).transpose(-1,-2)
    assert torch.allclose(original,translated,atol=2e-5)
    xonly=rope(q,p*0+torch.tensor([1.,0.,0.]))
    assert not torch.allclose(xonly[...,:4],q[...,:4])
    assert torch.equal(xonly[...,4:],q[...,4:])


def test_patch_input_equivalence_height_and_output_voxel_order():
    torch.manual_seed(4);model=tiny()
    target=torch.randint(5,(2,8,8,4))
    direct=model.patchify(target)
    onehot=torch.nn.functional.one_hot(target,5).permute(0,4,1,2,3).float()
    expected=model.target_patch(onehot).flatten(2).transpose(1,2)
    assert torch.equal(direct,expected)
    assert model.patch_centers_xyz.tolist()==[[2.,2.,2.],[2.,2.,6.],[2.,6.,2.],[2.,6.,6.]]
    with torch.no_grad():
        model.class_head.weight.zero_()
        bias=torch.arange(64*4).reshape(64,4).float()
        model.class_head.bias.copy_(bias.flatten())
    out=model(target,timestep=torch.tensor([.2,.7]))
    assert out.shape==(2,4,8,8,4)
    for y in range(8):
        for z in range(8):
            for x in range(4):
                assert torch.equal(out[0,:,y,z,x],bias[(y%4)*16+(z%4)*4+x])
    with torch.no_grad():model.class_head.weight.normal_(std=.02)
    t=torch.tensor([.4,.4]);origin=torch.tensor([[0.,-64.,0.],[0.,-64.,0.]])
    a=model(target,timestep=t,origin_xyz=origin)
    moved=origin+torch.tensor([800.,0.,-400.])
    assert torch.equal(a,model(target,timestep=t,origin_xyz=moved))
    moved[:,1]+=16
    assert not torch.allclose(a,model(target,timestep=t,origin_xyz=moved))


def test_zero_init_md4_gradients_and_full_mask_completion():
    torch.manual_seed(5);model=tiny()
    target=torch.randint(4,(2,8,8,4))
    x,mask,t,w=md4_mask(target,4,times=torch.tensor([.2,.8]))
    out=model(x,timestep=t)
    assert torch.count_nonzero(out)==0
    loss,_,count=weighted_masked_ce_sums(out,target,mask,w)
    (loss/count).backward()
    assert torch.isfinite(model.class_head.weight.grad).all()
    assert model.class_head.weight.grad.abs().sum()>0
    sample=sample_unconditional(model,steps=3,generator=torch.Generator().manual_seed(90))
    repeat=sample_unconditional(model,steps=3,generator=torch.Generator().manual_seed(90))
    assert sample.shape==(1,8,8,4) and sample.max()<4 and torch.equal(sample,repeat)


def test_ddp_rank_sampler_coverage_and_consumed_cursor_restart():
    streams=[iter(RankBatches(24,17,r,4,2,16)) for r in range(4)]
    first=[next(s) for s in streams]+[next(s) for s in streams]
    assert len(set(i for b in first for i in b))==16
    expected=[next(s) for s in streams]+[next(s) for s in streams]
    resumed=[iter(RankBatches(24,17,r,4,2,16,start_step=1)) for r in range(4)]
    actual=[next(s) for s in resumed]+[next(s) for s in resumed]
    assert actual==expected


def test_prefetch_preserves_masks_denominator_and_consumed_rng():
    class Loader:
        def __iter__(self):
            while True:yield dict(target=torch.arange(256).reshape(1,8,8,4)%4)
    torch.manual_seed(55);initial=torch.get_rng_state()
    expected=[prepare_batch(iter(Loader()),accumulation=2,num_classes=4) for _ in range(3)]
    prefetch=BatchPrefetcher(iter(Loader()),accumulation=2,num_classes=4,depth=3,rng=initial)
    try:
        for ref in expected:
            got=prefetch.take()
            assert got["count"]==ref["count"]
            assert torch.equal(torch.get_rng_state(),ref["rng_state"])
            for x,y in zip(got["prepared"],ref["prepared"]):
                for u,v in zip(x[1:],y[1:]):assert torch.equal(u,v)
    finally:prefetch.close()

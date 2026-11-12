import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from streamvoxel.common.distributed import DistributedEnv
from streamvoxel.train.validation_base import evaluate,collate_validation,validation_ids
from streamvoxel.train.train_base import validation_due,restore_optimizer
import pytest


def test_explicit_resume_lr_change_preserves_adam_moments_and_step():
    parameter=torch.nn.Parameter(torch.tensor([2.]))
    before=torch.optim.AdamW([parameter],lr=1e-4)
    parameter.square().sum().backward();before.step()
    saved=before.state_dict()
    replacement=torch.nn.Parameter(torch.tensor([2.]))
    resumed=torch.optim.AdamW([replacement],lr=1e-5)
    with pytest.raises(ValueError,match='explicit'):
        restore_optimizer(resumed,saved,lr=1e-5,previous_lr=1e-4)
    restore_optimizer(resumed,saved,lr=1e-5,previous_lr=1e-4,override_lr=True)
    assert resumed.param_groups[0]['lr']==1e-5
    for key in ('step','exp_avg','exp_avg_sq'):
        assert torch.equal(resumed.state[replacement][key],before.state[parameter][key])


def test_fresh_validation_runs_after_step1_then_every1000():
    assert not validation_due(0)
    assert [s for s in range(1,3001) if validation_due(s)]==[1,1000,2000,3000]
    assert not validation_due(1,first_step=-1)
    assert validation_due(2000,first_step=-1)


class Fixture(Dataset):
    num_classes=4
    def __init__(self):
        self.records=np.asarray([[0 if i<3 else 1,0,i] for i in range(7)])
        self.manifest={'worlds':[{'id':'a'},{'id':'b'}]}
    def __len__(self):return 7
    def __getitem__(self,i):
        return dict(target=(torch.arange(32).reshape(2,4,4)+i)%4,
            origin_xyz=torch.tensor([i,-64,0.],dtype=torch.float32),record_id=i,
            world_index=int(self.records[i,0]))


class Uniform(torch.nn.Module):
    def forward(self,x,*,timestep,origin_xyz):
        return torch.zeros(len(x),4,*x.shape[1:])


def cpu_env(rank=0,size=1):return DistributedEnv(rank,rank,size,torch.device('cpu'))


def test_full_validation_matches_analytic_totals_and_preserves_rng_and_mode():
    data=Fixture();model=Uniform();model.eval()
    initial=torch.get_rng_state().clone()
    expected_count=0;expected_weighted=0.;expected_correct=0
    for i in range(len(data)):
        b=collate_validation([data[i]],seed=71,num_classes=4)
        count=int(b['mask'].sum());expected_count+=count
        expected_weighted+=count*float(b['weight'][0])*math.log(4)
        expected_correct+=int(((b['target']==0)&b['mask']).sum())
    result=evaluate(model,data,cpu_env(),seed=71,batch_size=3,workers=0)
    assert result['cases']==7 and result['masked_voxels']==expected_count
    assert abs(result['validation_md4_loss']-expected_weighted/expected_count)<1e-6
    assert result['validation_masked_accuracy']==expected_correct/expected_count
    assert result['per_world']['a']['cases']==3 and result['per_world']['b']['cases']==4
    assert result['coverage']=='full' and result['weights_source']=='raw'
    assert torch.equal(initial,torch.get_rng_state()) and not model.training
    repeat=evaluate(model,data,cpu_env(),seed=71,batch_size=2,workers=2)
    assert repeat['masked_voxels']==expected_count
    assert abs(repeat['validation_md4_loss']-result['validation_md4_loss'])<1e-6
    assert repeat['validation_masked_accuracy']==result['validation_masked_accuracy']
    assert torch.equal(initial,torch.get_rng_state()) and not model.training


def distributed_worker(rank,rendezvous,directory):
    torch.set_num_threads(2)
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        result=evaluate(Uniform(),Fixture(),cpu_env(rank,2),seed=71,batch_size=2,workers=0)
        (Path(directory)/f'rank{rank}.json').write_text(json.dumps(result))
    finally:dist.destroy_process_group()


def test_distributed_validation_uneven_shards_have_no_duplicates_or_padding(tmp_path):
    ids=validation_ids(7,71)
    shards=[ids[r::2].tolist() for r in range(2)]
    assert sorted(shards[0]+shards[1])==list(range(7)) and not set(shards[0])&set(shards[1])
    mp.spawn(distributed_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
    expected=evaluate(Uniform(),Fixture(),cpu_env(),seed=71,batch_size=3,workers=0)
    for rank in range(2):
        result=json.loads((tmp_path/f'rank{rank}.json').read_text())
        for key in ('cases','masked_voxels','validation_masked_accuracy','per_world'):
            if key!='per_world':assert result[key]==expected[key]
        assert abs(result['validation_md4_loss']-expected['validation_md4_loss'])<1e-6
        for world in ('a','b'):
            assert result['per_world'][world]['cases']==expected['per_world'][world]['cases']
            assert result['per_world'][world]['masked_voxels']==expected['per_world'][world]['masked_voxels']

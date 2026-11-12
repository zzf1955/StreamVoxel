import pytest

from streamvoxel.train.train_controlnet import half_mask_role


def test_half_mask_roles_split_ranks():
    assert [half_mask_role(r,4,7) for r in range(4)]==['random','random','strata','strata']
    assert [half_mask_role(r,2,1) for r in range(2)]==['random','strata']


def test_single_rank_alternates_by_step():
    assert [half_mask_role(0,1,s) for s in (1,2,3,4)]==['random','strata','random','strata']


def test_odd_world_rejected():
    with pytest.raises(ValueError):
        half_mask_role(0,3,1)

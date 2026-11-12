import pytest
from streamvoxel.objectives.controlnet_schedule import learning_rate, validate_resume_schedule


def test_cosine_exact_duration_and_resume_values():
    args=dict(peak_lr=1e-4,schedule='cosine',cosine_start_step=9000,cosine_steps=10000)
    assert learning_rate(9000,**args)==1e-4
    assert learning_rate(9001,**args)==pytest.approx(1e-4,rel=1e-7)
    assert learning_rate(14000,**args)==pytest.approx(5e-5)
    assert learning_rate(19000,**args)==0
    assert learning_rate(20000,**args)==0
    assert learning_rate(100,peak_lr=1e-4)==5e-5
    assert learning_rate(9000,peak_lr=1e-4)==1e-4


def test_only_explicit_checkpoint_anchored_transition_allowed():
    old=dict(lr=1e-4,steps=10000)
    new=dict(lr=1e-4,steps=19000,lr_schedule='cosine',cosine_start_step=9000,cosine_steps=10000,min_lr=0.)
    with pytest.raises(ValueError):validate_resume_schedule(old,new,9000)
    validate_resume_schedule(old,new,9000,allow_change=True)
    validate_resume_schedule(new,new,12000)
    with pytest.raises(ValueError):validate_resume_schedule(old,new,10000,allow_change=True)
    with pytest.raises(ValueError):validate_resume_schedule(old,dict(new,steps=20000),9000,allow_change=True)
    with pytest.raises(ValueError):validate_resume_schedule(old,dict(new,min_lr=1e-3),9000,allow_change=True)

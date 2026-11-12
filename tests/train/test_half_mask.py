import pytest

from streamvoxel.train.train_base import half_mask_plan


def test_half_mask_splits_ranks_evenly():
    counts,roles=half_mask_plan(512,16,4)
    assert counts==[8,8,8,8] and roles==["random","random","strata","strata"]
    counts,roles=half_mask_plan(480,16,2)
    assert counts==[15,15] and roles==["random","strata"]


@pytest.mark.parametrize("batch,micro,world",[(512,16,3),(500,16,4)])
def test_half_mask_rejects_uneven_work(batch,micro,world):
    with pytest.raises(ValueError):
        half_mask_plan(batch,micro,world)


def test_half_mask_single_rank_alternates():
    assert half_mask_plan(512,8,1)==([64],["alternate"])


def test_config_file_sets_defaults_and_the_command_line_wins(tmp_path):
    from streamvoxel.train.train_base import parse_args
    config=tmp_path/"c.yaml"
    config.write_text("data_dir: d\nrun_dir: r\ntimefree_arch: true\nhalf_mask: true\nlr: 0.00001\n")
    args=parse_args(["--config",str(config)])
    assert args.half_mask and args.timefree_arch and args.lr==1e-5
    assert parse_args(["--config",str(config),"--lr","0.0002"]).lr==2e-4

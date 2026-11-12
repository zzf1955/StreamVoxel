import numpy as np
import pytest
from streamvoxel.common.colored_frontier import select_batch, validate_batch, color


def test_moving_player_reprioritizes_without_mutating_snapshot():
    known=np.zeros((16,16),bool);known[7:10,7:10]=True;before=known.copy()
    west=select_batch(known,[136,-100]);east=select_batch(known,[136,500])
    assert west!=east and sum(x for z,x in west)<sum(x for z,x in east)
    assert np.array_equal(known,before)
    for targets in (west,east):
        assert len(targets)==4 and len({color(c) for c in targets})==1
        validate_batch(known,targets)


def test_complete_finite_extent_and_tail_without_deadlock():
    known=np.zeros((16,16),bool);known[7:10,7:10]=True;seen=set();tails=0
    while not known.all():
        targets=select_batch(known,[137,500])
        assert targets and not set(targets)&seen
        validate_batch(known,targets);seen.update(targets)
        tails+=len(targets)<4
        for z,x in targets:known[z,x]=True
    assert len(seen)==247 and tails>0


def test_same_color_nearest_selection_and_readiness():
    from streamvoxel.common.sequential_rollout import neighbor_cells
    known=np.zeros((12,12),bool);known[4:7,4:7]=True;player=[80,140]
    chosen=select_batch(known,player);c=color(chosen[0]);queue=[]
    for z in range(12):
        for x in range(12):
            if color((z,x))==c and not known[z,x] and neighbor_cells(known,z,x):
                queue.append((((z+.5)*16-player[0])**2+((x+.5)*16-player[1])**2,z,x))
    assert chosen==[(z,x) for _,z,x in sorted(queue)[:4]]
    with pytest.raises(ValueError):validate_batch(known,[(3,4),(3,5)])
    assert select_batch(np.ones((2,2),bool),[0,0])==[]
    with pytest.raises(ValueError):select_batch(known,[float('nan'),0])


def test_variable_batch_is_same_color_nearest_and_short_batch_keeps_fullest_color():
    known=np.zeros((16,16),bool);known[7:10,7:10]=True
    for size in (1,4,8,16):
        targets=select_batch(known,[136,136],batch_size=size)
        assert len(targets)<=size and len({color(c) for c in targets})==1
        validate_batch(known,targets)
    assert len(select_batch(known,[136,136],batch_size=16))>4
    with pytest.raises(ValueError):select_batch(known,[0,0],batch_size=0)
    with pytest.raises(ValueError):select_batch(known,[0,0],batch_size=65)


def test_sparse_selection_matches_array_selection_and_grows_without_bound():
    from streamvoxel.common.colored_frontier import select_batch_cells, validate_cells
    known=np.zeros((16,16),bool);known[7:10,7:10]=True
    cells={(int(z),int(x)) for z,x in zip(*np.where(known))}
    for size in (1,4,8,16):
        for player in ([136,136],[0,500],[300,-40]):
            assert select_batch_cells(cells,player,batch_size=size)==select_batch(known,player,batch_size=size)
    for _ in range(60):  # far beyond the old 16x16 extent, including negative cells
        targets=select_batch_cells(cells,[-900,-900],batch_size=8)
        validate_cells(cells,targets);cells.update(targets)
    assert min(z for z,_ in cells)<-5 and len(cells)>200
    with pytest.raises(ValueError):validate_cells(cells,[next(iter(cells))])


def test_tiled_biomes_wrap_and_equal_original_crop_inside_the_world():
    from streamvoxel.stream.scheduler import tiled_biome_crop
    rng=np.random.default_rng(0);biomes=rng.integers(0,60,(64,66,66)).astype(np.uint16)
    for z,x in ((1,1),(8,3),(14,14)):
        assert np.array_equal(tiled_biome_crop(biomes,z,x),biomes[:,z*4:z*4+6,x*4:x*4+6])
    assert np.array_equal(tiled_biome_crop(biomes,16,5),tiled_biome_crop(biomes,0,5))
    assert np.array_equal(tiled_biome_crop(biomes,-1,-1),tiled_biome_crop(biomes,15,15))

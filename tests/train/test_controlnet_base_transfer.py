import copy
import json
import numpy as np
import pytest
import torch
from tests.train.test_compact_controlnet import fixture
from streamvoxel.models.compact_controlnet import CompactStreamingControlNet
from streamvoxel.train.train_controlnet import FORMAT,control_state,load_control_state,training_steps,rank_batch_plan
from streamvoxel.data.streaming_controlnet import ControlDataset


def test_unbounded_steps_preserve_numbering_and_finite_runs():
    import itertools
    assert list(itertools.islice(training_steps(9500,0),3))==[9501,9502,9503]
    assert list(training_steps(0,2))==[1,2]
    assert list(training_steps(2,2))==[]
    with pytest.raises(ValueError):training_steps(0,-1)


def test_three_rank_uneven_accumulation_preserves_global512():
    counts=[]
    for rank in range(3):
        accumulation,targets,groups=rank_batch_plan(512,8,3,rank,8)
        assert accumulation==[22,21,21] and targets==[176,168,168]
        assert sum(groups)==targets[rank] and max(groups)<=32
        counts.append(sum(groups))
    assert sum(counts)==512
    for rank in range(2):assert rank_batch_plan(512,8,2,rank,8)[2]==[32]*8


def test_control_state_roundtrips_and_rejects_backbone_weights():
    old,g=fixture()
    new=CompactStreamingControlNet(copy.deepcopy(old.backbone),control_depth=3,control_hidden_dim=24,
        checkpointing=False,biome_halo=4,biome_patch=2,struct_classes=0)
    with torch.no_grad():
        new.backbone.class_head.weight.add_(.1)
        for projection in old.zero:projection.weight.normal_(std=.02)
    before={k:v.clone() for k,v in new.backbone.state_dict().items()}
    load_control_state(new,control_state(old))
    assert all(torch.equal(before[k],v) for k,v in new.backbone.state_dict().items())
    for k,v in control_state(new).items():
        expected=control_state(old)[k]
        if k=='type_embed.weight':expected=expected[:3]
        assert torch.equal(v,expected),k
    with pytest.raises(ValueError,match='backbone'):
        load_control_state(new,dict(control_state(old),**{'backbone.class_head.weight':old.backbone.class_head.weight}))


def test_lossless_tile32_matches_global_crops_without_state_or_axis_change(tmp_path):
    import zstandard as zstd
    # A single native-height32² tile with distinct height/X/Z values.
    yy,zz,xx=np.indices((384,32,32));raw=((yy+zz*2+xx*3)%3).astype('<u2')
    packed=zstd.ZstdCompressor().compress(raw.tobytes());(tmp_path/'world.bin').write_bytes(packed)
    row=dict(entries=[[0,len(packed)]],shapeYZX=[384,32,32],tileShapeYZX=[384,32,32],codec='zstd')
    (tmp_path/'world.json').write_text(json.dumps(row));np.save(tmp_path/'bio.npy',np.zeros((64,512,512),np.uint16))
    (tmp_path/'mapping.json').write_text(json.dumps(dict(global_to_compact=[2,0,1],compact_to_global=[1,2,0])))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(num_classes=3,mapping_file='mapping.json',biome_names=[],struct_names=[],
        worlds={'0':dict(storage='lossless_tile32_zstd',blocks='world.bin',tile_index='world.json',biomes='bio.npy')})))
    data=ControlDataset(tmp_path,validate=False,use_struct=False)
    coords=np.array([[0,0],[0,1],[1,0],[1,1],[0,0]])
    values=data.tiles(0,coords);lut=np.array([2,0,1],np.int16)
    expected=np.stack([lut[raw[:256,z*16:(z+1)*16,x*16:(x+1)*16]] for z,x in coords])
    np.testing.assert_array_equal(values,expected)
    assert len(data.decoded_tiles)==1 and values.shape==(5,256,16,16)


def test_prefetch_preserves_old_hits_while_inserting_new_misses(tmp_path):
    import zstandard as zstd
    from collections import OrderedDict
    raw=np.zeros((384,32,32),dtype='<u2');packed=zstd.ZstdCompressor().compress(raw.tobytes())
    (tmp_path/'world.bin').write_bytes(packed)
    row=dict(entries=[[0,len(packed)]]*4096,shapeYZX=[384,2048,2048],tileShapeYZX=[384,32,32],codec='zstd')
    (tmp_path/'world.json').write_text(json.dumps(row));np.save(tmp_path/'bio.npy',np.zeros((64,512,512),np.uint16))
    (tmp_path/'mapping.json').write_text(json.dumps(dict(global_to_compact=[0,1,2],compact_to_global=[0,1,2])))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(num_classes=3,mapping_file='mapping.json',biome_names=[],struct_names=[],
        worlds={'0':dict(storage='lossless_tile32_zstd',blocks='world.bin',tile_index='world.json',biomes='bio.npy')})))
    data=ControlDataset(tmp_path,validate=False,use_struct=False)
    hit=np.ones((256,32,32),np.int16)
    # The required hit is oldest; an unprotected cache evicts it on the first miss.
    data.decoded_tiles=OrderedDict([((0,0),hit)]+[((1,i),hit) for i in range(255)])
    values=data.tiles(0,np.array([[0,0],[0,2]]))
    assert np.all(values[0]==1) and np.all(values[1]==0)
    assert (0,0) in data.decoded_tiles and len(data.decoded_tiles)==256

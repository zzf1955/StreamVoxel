"""Export the small initial World the live viewer starts from (CPU only; needs node + prismarine-nbt).

python -m viewer.world_export --panel <run>/panel --output <new dir> --template <void template world> --node <node> --nbt <prismarine-nbt dir>
Writes blocks.npy, world/ (with the streamvoxel_live datapack), initial_world.zip, export.json and COMPLETE, which viewer.start reads.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np
from streamvoxel.common.io import atomic_json, digest
from streamvoxel.common.preview_states import preview_lookup, load_registry, REGISTRY_PATH, DEFAULTS_PATH


def build(panel, output, template, node, nbt):
    output.mkdir(parents=True,exist_ok=False)
    pm=json.loads((panel/'manifest.json').read_text());case=pm['cases'][0]
    gt=np.load(panel/'case00/gt_global.npy',mmap_mode='r')
    z0,x0,z1,x1=case['initial_box_zx']
    lookup=preview_lookup('clean',json.loads((panel/'vocabulary.json').read_text()))
    states=load_registry()['idToState']; blocks=np.zeros((256,256,544),np.uint16)
    # Cropped panels store ground truth relative to gt_canvas_box_zx, not the canvas origin.
    canvas=case.get('gt_canvas_box_zx')
    if gt.shape[1] < z1 or gt.shape[2] < x1:
        if not canvas:
            raise ValueError(f'Ground truth {gt.shape} does not cover the initial box {case["initial_box_zx"]}')
        oz, ox = canvas[0], canvas[1]
        src = gt[:, z0-oz:z1-oz, x0-ox:x1-ox]
    else:
        src = gt[:, z0:z1, x0:x1]
    blocks[:,z0:z1,x0:x1]=lookup[src]
    bank={}
    for i,name in enumerate(('minecraft:cave_air','minecraft:void_air')):
        raw=states.index(name); x=512+2*i; blocks[64,0,x]=raw;bank[str(raw)]=[x,0,0]
    # An all-air section can be discarded by the native engine, losing the
    # cave/void distinction. Keep a solid voxel in the same palette section.
    blocks[64,0,516]=states.index('minecraft:stone')
    np.save(output/'blocks.npy',blocks,allow_pickle=False);shutil.copy2(REGISTRY_PATH,output/REGISTRY_PATH.name)
    shutil.copy2(DEFAULTS_PATH,output/'block_default_states.json')
    spawn=[(x0+x1)//2,140,(z0+z1)//2]
    atomic_json(output/'manifest.json',dict(format='minecraft_java_26_2_unified_preview_v1',game_version='26.2',data_version=4903,
        shape_yzx=list(blocks.shape),spawn_xyz=spawn,chunk_boxes_xz=[[0,0,16,16],[32,0,33,1]],biome='minecraft:plains'))
    subprocess.run([str(node),str(Path(__file__).with_name('build_world.js')),str(output),str(template),str(output/'world')],
                   env=dict(os.environ,PRISMARINE_NBT_PATH=str(nbt)),check=True)
    world=output/'world';pack=world/'datapacks/streamvoxel_live';functions=pack/'data/streamvoxel_live/function';functions.mkdir(parents=True)
    atomic_json(pack/'pack.mcmeta',dict(pack=dict(description='Live actual unmask deltas; bounded shared macros',min_format=[107,1],max_format=[107,1])))
    (functions/'fill_one.mcfunction').write_text('$fill $(x) $(y) $(z) $(ex) $(y) $(z) $(state) strict\n')
    (functions/'clone_one.mcfunction').write_text('$clone $(sx) $(sy) $(sz) $(sx) $(sy) $(sz) $(x) $(y) $(z) strict replace force\n')
    (functions/'loop.mcfunction').write_text('execute if data storage streamvoxel_live:frame runs[0].state run function streamvoxel_live:fill_one with storage streamvoxel_live:frame runs[0]\nexecute unless data storage streamvoxel_live:frame runs[0].state run function streamvoxel_live:clone_one with storage streamvoxel_live:frame runs[0]\ndata remove storage streamvoxel_live:frame runs[0]\nexecute if data storage streamvoxel_live:frame runs[0] run function streamvoxel_live:loop\n')
    archive=output/'initial_world.zip';files={str(p.relative_to(world)):digest(p) for p in world.rglob('*') if p.is_file()}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in files:z.write(world/name,name)
    report=dict(format='minecraft_live_frontier_initial_v1',world=case['world'],spawn_xyz=spawn,
        panel_manifest_sha256=digest(panel/'manifest.json'),initial_box_zx=case['initial_box_zx'],air_bank=bank,
        archive_sha256=digest(archive),world_files=files)
    atomic_json(output/'export.json',report);(output/'COMPLETE').write_text(digest(output/'export.json')+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('panel','output','template','node','nbt'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();build(a.panel,a.output,a.template,a.node,a.nbt)


if __name__=='__main__':main()

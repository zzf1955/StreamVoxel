"""Pure spatial contracts for immediate-commit paired rollout."""
import numpy as np


def clockwise_ring(lo,hi):
    """Start on an edge, then clockwise; every coordinate appears once."""
    if hi-lo<3:raise ValueError('Ring needs an interior')
    order=([(lo,x) for x in range(lo+1,hi)] + [(z,hi-1) for z in range(lo+1,hi)] +
           [(hi-1,x) for x in range(hi-2,lo-1,-1)] + [(z,lo) for z in range(hi-2,lo-1,-1)])
    start=order.index((lo,(lo+hi)//2))
    return order[start:]+order[:start]


def neighbor_cells(known,z,x):
    return [(a,b) for a in range(max(0,z-1),min(len(known),z+2))
            for b in range(max(0,x-1),min(known.shape[1],x+2)) if (a,b)!=(z,x) and known[a,b]]


def sliding_window(known,z,x,lo,hi):
    """Prefer three known8x8 cells; otherwise an adjacent known half-window."""
    candidates=[]
    for wz in (z-1,z):
        for wx in (x-1,x):
            if wz<lo or wx<lo or wz+2>hi or wx+2>hi:continue
            mask=known[wz:wz+2,wx:wx+2]
            n=int(mask.sum())
            adjacent=n==2 and (mask[0].all() or mask[1].all() or mask[:,0].all() or mask[:,1].all())
            if n==3 or adjacent:candidates.append((n,wz,wx))
    if not candidates:raise ValueError(f'No valid known half/L window for target {(z,x)}')
    _,wz,wx=max(candidates,key=lambda r:(r[0],-r[1],-r[2]))
    return wz,wx


def assert_known_contract(canvas,known,cell,classes):
    if known.ndim != 2 or canvas.shape[1:] != (known.shape[0]*cell,known.shape[1]*cell):
        raise ValueError('Canvas and known grid shapes differ')
    values=canvas.reshape(canvas.shape[0],known.shape[0],cell,known.shape[1],cell)
    complete=(values>=0).all(axis=(0,2,4));any_known=(values>=0).any(axis=(0,2,4))
    if not np.array_equal(known,complete) or not np.array_equal(known,any_known):
        raise ValueError('Partial known cells or GT leakage')
    if canvas[canvas>=0].size and canvas[canvas>=0].max()>=classes:raise ValueError('Invalid compact/MASK output')

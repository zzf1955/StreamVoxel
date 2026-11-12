"""Pure CPU metrics for E2/E3/E4: window-level array metrics with seam errors, ring-wise voxel scores, stale dependency edges."""
import numpy as np


def array_metrics(gt, pred, air, *, context=None, target=None, gt_context=None):
    """gt/pred [Y,16,16] compact ids, `air` the air class ids. With `context` (the canvas, -1 = unknown) and `target` (z, x tile),
    also the seam errors against the committed neighbours: occupancy disagreement and height-jump MAE relative to the GT seam."""
    from scipy import ndimage
    equal = gt == pred
    ga, pa = ~np.isin(gt, air), ~np.isin(pred, air)
    height = lambda occ: np.where(occ.any(0), len(occ) - 1 - np.argmax(occ[::-1], axis=0), -1)
    gh, ph = height(ga), height(pa)
    gt_top = np.take_along_axis(gt, np.maximum(gh, 0)[None], 0)[0]
    pr_top = np.take_along_axis(pred, np.maximum(ph, 0)[None], 0)[0]
    labels = np.union1d(np.unique(gt), np.unique(pred))
    f1 = []
    for k in labels:
        if k not in air:
            g, p = gt == k, pred == k
            f1.append(2 * int((g & p).sum()) / (int(g.sum()) + int(p.sum())))
    rates = lambda occ: [float((np.diff(occ.astype(np.int8), axis=i) != 0).mean()) for i in range(3)]
    gr, pr = rates(ga), rates(pa)
    ratios = [p / g if g else None for p, g in zip(pr, gr)]
    lab, n = ndimage.label(pa)
    sizes = np.bincount(lab.ravel())[1:]
    surface = np.abs(np.arange(len(gt))[:, None, None] - gh[None]) <= 8
    underground = np.arange(len(gt))[:, None, None] < gh[None] - 8
    r = dict(acc=float(equal.mean()), nonair_acc=float(equal[ga].mean()) if ga.any() else None,
             nonair_macro_f1=float(np.mean(f1)), height_mae=float(abs(ph - gh).mean()), surface_block_acc=float((gt_top == pr_top).mean()),
             surface_band_acc=float(equal[surface].mean()), underground_acc=float(equal[underground].mean()) if underground.any() else None,
             gt_interface_yzx=gr, pred_interface_yzx=pr, interface_ratio_yzx=ratios,
             interface_log_error=float(np.mean(np.abs(np.log((np.array(pr) + 1e-8) / (np.array(gr) + 1e-8))))),
             isolated_nonair_voxels=int((sizes == 1).sum()), small_nonair_voxels=int(sizes[(sizes >= 2) & (sizes <= 8)].sum()))
    if context is not None:
        z, x = target
        disagreements, hj = [], []
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = z + dz, x + dx
            if a < 0 or b < 0 or a * 16 >= context.shape[1] or b * 16 >= context.shape[2]:
                continue
            tile = context[:, a * 16:(a + 1) * 16, b * 16:(b + 1) * 16]
            if (tile < 0).any():
                continue
            gt_tile = tile if gt_context is None else gt_context[:, a * 16:(a + 1) * 16, b * 16:(b + 1) * 16]
            if dz:
                g, p, c = gt[:, 0 if dz < 0 else -1, :], pred[:, 0 if dz < 0 else -1, :], tile[:, -1 if dz < 0 else 0, :]
                cg = gt_tile[:, -1 if dz < 0 else 0, :]
            else:
                g, p, c = gt[:, :, 0 if dx < 0 else -1], pred[:, :, 0 if dx < 0 else -1], tile[:, :, -1 if dx < 0 else 0]
                cg = gt_tile[:, :, -1 if dx < 0 else 0]
            go, po, co, cgo = ~np.isin(g, air), ~np.isin(p, air), ~np.isin(c, air), ~np.isin(cg, air)
            disagreements.append(float(((po != co) != (go != cgo)).mean()))
            edgeh = lambda o: np.where(o.any(0), len(o) - 1 - np.argmax(o[::-1], axis=0), -1)
            hj.append(float(np.abs((edgeh(po) - edgeh(co)) - (edgeh(go) - edgeh(cgo))).mean()))
        r.update(seam_occupancy_error=float(np.mean(disagreements)), seam_height_jump_mae=float(np.mean(hj)))
    return r


def voxel_scores(gt, pred, order, air_ids=(0,), classes=981):
    """Class metrics over generated tiles only, pooled (`all`) and per ring. `order` lists `(ring, z, x)` of the generated 16x16 tiles;
    gt/pred are [Y,Z,X] ids in the same space (global 981 by default, so out-of-vocabulary GT counts as an error)."""
    rings = max(ring for ring, _, _ in order)
    matrices = [np.zeros((classes, classes), np.int64) for _ in range(rings + 1)]
    for ring, z, x in order:
        g = gt[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16].astype(np.int64)
        q = pred[:, z * 16:(z + 1) * 16, x * 16:(x + 1) * 16].astype(np.int64)
        m = np.bincount((g * classes + q).ravel(), minlength=classes ** 2).reshape(classes, classes)
        matrices[0] += m
        matrices[ring] += m

    def summarize(m):
        a, b = m.sum(1), m.sum(0)
        tp = m.diagonal()
        support = (a + b) > 0
        f1 = 2 * tp / np.maximum(a + b, 1)
        nonair = np.ones(classes, bool)
        nonair[list(air_ids)] = False
        return dict(acc=float(tp.sum() / max(int(a.sum()), 1)),
                    nonair_acc=float(tp[nonair].sum() / max(int(a[nonair].sum()), 1)),
                    macro_f1=float(f1[support].mean()), voxels=int(a.sum()), supported_classes=int(support.sum()))
    return {('all' if i == 0 else f'ring{i}'): summarize(m) for i, m in enumerate(matrices) if i == 0 or m.any()}


def stale_edges(batch):
    """Pairs of cells in one batch that share an edge or a corner. Two such windows would be generated from conditions that do not
    contain each other, so any positive count is a stale dependency edge; the four-colour frontier admits none."""
    cells = sorted(set(map(tuple, batch)))
    return sum(1 for i, a in enumerate(cells) for b in cells[i + 1:] if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1)

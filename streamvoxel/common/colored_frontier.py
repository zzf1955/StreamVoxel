"""Snapshot-ready, player-prioritized four-color scheduling for 16² windows."""
from itertools import combinations
import math


def color(cell):
    z, x = cell
    return 2 * (z % 2) + x % 2


MAX_BATCH = 64


def select_batch(known, player_zx, *, cell_size=16, batch_size=4):
    """Prefer a full same-color batch; otherwise take the color with most ready cells.

    Distance is squared horizontal distance to the window center. Readiness
    depends only on the supplied committed snapshot, never partial display.
    """
    from streamvoxel.common.sequential_rollout import neighbor_cells
    if not 1 <= batch_size <= MAX_BATCH or cell_size <= 0 or not all(map(math.isfinite, player_zx)):
        raise ValueError(f'Expected 1<=B<={MAX_BATCH} and finite player coordinates')
    pz, px = player_zx
    queues = [[] for _ in range(4)]
    for z in range(known.shape[0]):
        for x in range(known.shape[1]):
            if not known[z, x] and neighbor_cells(known, z, x):
                distance = ((z + .5) * cell_size - pz)**2 + ((x + .5) * cell_size - px)**2
                queues[color((z, x))].append((distance, z, x))
    for queue in queues:
        queue.sort()
    eligible = [q for q in queues if len(q) >= batch_size]
    if not eligible:
        # Short batch: keep the fullest color (nearest head breaks ties), so a large
        # requested B never degrades into one-window batches.
        fullest = max((len(q) for q in queues), default=0)
        if not fullest:
            return []
        eligible = [q for q in queues if len(q) == fullest]
    queue = min(eligible, key=lambda q: q[0])
    targets = [(z, x) for _, z, x in queue[:batch_size]]
    validate_batch(known, targets)
    return targets


def validate_batch(known, targets, max_batch=MAX_BATCH):
    from streamvoxel.common.sequential_rollout import neighbor_cells
    if not 1 <= len(targets) <= max_batch or len(set(targets)) != len(targets):
        raise ValueError('Invalid true batch size')
    if len({color(c) for c in targets}) != 1:
        raise ValueError('Mixed colors')
    for z, x in targets:
        if not (0 <= z < known.shape[0] and 0 <= x < known.shape[1]):
            raise ValueError('Outside target extent')
        if known[z, x] or not neighbor_cells(known, z, x):
            raise ValueError('Target not ready in committed snapshot')
    for a, b in combinations(targets, 2):
        if max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= 1:
            raise ValueError('Eight-neighbor conflict')


def ready_cells(known):
    """Unknown cells with at least one known 8-neighbor; `known` is a set of (z, x), unbounded."""
    ready = set()
    for z, x in known:
        for a in range(z - 1, z + 2):
            for b in range(x - 1, x + 2):
                if (a, b) not in known:
                    ready.add((a, b))
    return ready


def select_batch_cells(known, player_zx, *, cell_size=16, batch_size=4):
    """`select_batch` on an unbounded sparse extent: same colors, same nearest-first, same short-batch rule."""
    if not 1 <= batch_size <= MAX_BATCH or cell_size <= 0 or not all(map(math.isfinite, player_zx)):
        raise ValueError(f'Expected 1<=B<={MAX_BATCH} and finite player coordinates')
    pz, px = player_zx
    queues = [[] for _ in range(4)]
    for z, x in ready_cells(known):
        queues[color((z, x))].append((((z + .5) * cell_size - pz)**2 + ((x + .5) * cell_size - px)**2, z, x))
    for queue in queues:
        queue.sort()
    eligible = [q for q in queues if len(q) >= batch_size]
    if not eligible:
        fullest = max((len(q) for q in queues), default=0)
        eligible = [q for q in queues if len(q) == fullest]
    queue = min(eligible, key=lambda q: q[0])
    targets = [(z, x) for _, z, x in queue[:batch_size]]
    validate_cells(known, targets)
    return targets


def validate_cells(known, targets, max_batch=MAX_BATCH):
    if not 1 <= len(targets) <= max_batch or len(set(targets)) != len(targets):
        raise ValueError('Invalid true batch size')
    if len({color(c) for c in targets}) != 1:
        raise ValueError('Mixed colors')
    for z, x in targets:
        if (z, x) in known or not any((a, b) in known for a in range(z - 1, z + 2) for b in range(x - 1, x + 2)):
            raise ValueError('Target not ready in committed snapshot')
    for a, b in combinations(targets, 2):
        if max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= 1:
            raise ValueError('Eight-neighbor conflict')

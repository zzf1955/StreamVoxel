"""Per-stage timing for the viewer bridge: where the Mac spends its time between packet arrival and display.

The bridge writes `bridge_profile.jsonl` next to its other state: one `summary` line every few seconds (count, total,
mean and max per stage) and one `slow` line for each packet whose apply time exceeds the threshold. Stage keys:

* `wait:ssh_line`   bridge idle, nothing to apply (starved by the link or the generator)
* `decode`          json + unbatch + sha check of one transport line
* `apply:<kind>`    whole `apply_packet` per packet kind; `expand`/`write` are the row expansion and template file write
* `rcon:<a> <b>`    one main-connection RCON round trip, keyed by the command's first two words
* `poll:<a> <b>`    the same for the feedback thread's connection (shares the server thread with the main one)
* `batch:packets` / `batch:kbytes`  the mean_ms column is the mean packets / KB per transport line (how much got coalesced)
* `book`            journal/ack/active-cell bookkeeping after a packet
* `lag:emit_to_apply` / `lag:emit_to_shown`  seconds since the session stamped the packet (GPU-host clock minus viewer clock
  skew included, so compare differences, not absolutes)

Summarise a finished or running session with `python -m viewer.profile <bridge_profile.jsonl>`.
"""
from __future__ import annotations

import collections
import contextlib
import json
from pathlib import Path
import sys
import threading
import time


class Profile:
    def __init__(self, path, interval=5., slow=.2):
        self.handle = Path(path).open('a', buffering=1)
        self.interval, self.slow = interval, slow
        self.lock = threading.Lock()
        self.stats = collections.defaultdict(lambda: [0, 0., 0.])
        self.began = self.last = time.monotonic()

    def add(self, key, seconds):
        with self.lock:
            entry = self.stats[key]
            entry[0] += 1
            entry[1] += seconds
            entry[2] = max(entry[2], seconds)

    @contextlib.contextmanager
    def span(self, key):
        began = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - began)

    def event(self, kind, **row):
        self.handle.write(json.dumps(dict(kind=kind, unix=round(time.time(), 3), **row), separators=(',', ':')) + '\n')

    def tick(self):
        now = time.monotonic()
        if now - self.last < self.interval:
            return
        with self.lock:
            stats, self.stats = self.stats, collections.defaultdict(lambda: [0, 0., 0.])
        self.event('summary', wall=round(now - self.last, 3), stages={
            key: dict(n=n, total=round(total, 4), mean_ms=round(1000 * total / n, 2), max_ms=round(1000 * peak, 2))
            for key, (n, total, peak) in sorted(stats.items())})
        self.last = now


def report(path):
    total = collections.defaultdict(lambda: [0, 0., 0.])
    wall = 0.
    slow = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'summary':
            wall += row['wall']
            for key, stage in row['stages'].items():
                entry = total[key]
                entry[0] += stage['n']
                entry[1] += stage['total']
                entry[2] = max(entry[2], stage['max_ms'])
        else:
            slow.append(row)
    print(f'wall {wall:.1f} s over summaries; rows are totals (lag rows are seconds, not time spent)')
    print(f'{"stage":34s}{"n":>8s}{"total s":>10s}{"%wall":>8s}{"mean ms":>10s}{"max ms":>10s}')
    for key, (n, seconds, peak) in sorted(total.items(), key=lambda item: -item[1][1]):
        share = '' if key.startswith('lag:') else f'{100 * seconds / wall:7.1f}%' if wall else ''
        print(f'{key:34s}{n:8d}{seconds:10.2f}{share:>8s}{1000 * seconds / n:10.1f}{peak:10.1f}')
    print(f'\nslow packets (> threshold): {len(slow)}')
    for row in slow[-10:]:
        print(' ', json.dumps(row))


if __name__ == '__main__':
    report(sys.argv[1])

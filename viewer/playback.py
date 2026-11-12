"""Steady playback clock for the viewer: packets arrive in bursts, the display advances smoothly.

Each packet carries the session's emit time. The player starts once `delay` seconds of material are buffered (or the
oldest packet has waited that long), then advances a playhead through emit time at `speed` real seconds per second and
releases every packet the playhead has passed. A slow display therefore releases several packets in one tick (the
bridge merges their frame rows) instead of falling behind, and the delay is the cushion that rides out link jitter.
When the buffer runs dry the playhead stops and the buffer refills first. Idle stretches longer than `max_gap`
(a paused session, a dropped link) are skipped rather than waited out.
"""
from __future__ import annotations

import collections


class PlayBuffer:
    def __init__(self, delay=3., speed=1., max_gap=1.):
        self.delay, self.speed, self.max_gap = delay, speed, max_gap
        self.items = collections.deque()   # (emit seconds, arrival monotonic, packet), in sequence order
        self.head = self.clock = None
        self.playing = False
        self.last_emit = 0.
        self.stalls = 0

    def __len__(self):
        return len(self.items)

    def push(self, packet, now):
        stamp = packet.get('emitted_unix_ns')
        self.last_emit = max(self.last_emit, stamp / 1e9 if stamp else 0.)
        self.items.append((self.last_emit, now, packet))

    def buffered(self):
        """Seconds of emit time between the playhead and the newest packet."""
        if not self.items:
            return 0.
        return self.items[-1][0] - (self.head if self.playing else self.items[0][0])

    def due(self, now):
        if not self.items:
            if self.playing:
                self.playing = False
                self.stalls += 1
            return []
        first = self.items[0]
        if self.playing:
            self.head += (now - self.clock) * self.speed
            self.clock = now
        else:
            if self.items[-1][0] - first[0] < self.delay and now - first[1] < self.delay:
                return []
            self.playing, self.head, self.clock = True, first[0], now
        if first[0] - self.head > self.max_gap:
            self.head = first[0] - self.max_gap
        out = []
        while self.items and self.items[0][0] <= self.head:
            out.append(self.items.popleft()[2])
        return out

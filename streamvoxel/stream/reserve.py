"""GPU memory held back from other jobs, handed back when this process needs it.

`hold` fills the card down to `leave_bytes` free with fixed-size chunks. `ensure` gives chunks back whenever the
card's free memory falls under `leave_bytes`, so the process can grow without hitting OOM while the margin stays.
"""
from __future__ import annotations

import json

CHUNK = 256 * 1024 ** 2


class GpuReserve:
    def __init__(self, device, leave_bytes, *, rank=0, chunk=CHUNK):
        self.device = device
        self.leave = int(leave_bytes)
        self.rank = rank
        self.chunk = int(chunk)
        self.chunks = []

    @property
    def held(self):
        return len(self.chunks) * self.chunk

    def _free(self):
        import torch
        return int(torch.cuda.mem_get_info(self.device)[0])

    def hold(self):
        import torch
        torch.cuda.set_device(self.device)
        count = max(0, (self._free() - self.leave) // self.chunk)
        for _ in range(count):
            self.chunks.append(torch.empty(self.chunk // 2, dtype=torch.float16, device=self.device))
        torch.cuda.synchronize(self.device)
        free = self._free()
        print(json.dumps(dict(event='reserved', rank=self.rank, held_gb=round(self.held / 1024 ** 3, 2),
                              free_mb=int(free / 1024 ** 2), leave_gb=round(self.leave / 1024 ** 3, 2))), flush=True)

    def release(self, nbytes):
        """Give back at least `nbytes`. Returns the bytes freed."""
        import torch
        count = min(len(self.chunks), -(-int(nbytes) // self.chunk))
        for _ in range(count):
            self.chunks.pop()
        if count:
            torch.cuda.empty_cache()
            print(json.dumps(dict(event='reserve_release', rank=self.rank, released_mb=count * self.chunk // 1024 ** 2,
                                  held_gb=round(self.held / 1024 ** 3, 2))), flush=True)
        return count * self.chunk

    def ensure(self):
        """Release enough chunks that the card has `leave_bytes` free again."""
        short = self.leave - self._free()
        if short > 0 and self.chunks:
            self.release(short)

    def retry(self, call):
        """Run `call`; on CUDA OOM hand back the margin plus one chunk and run it once more."""
        import torch
        try:
            return call()
        except torch.cuda.OutOfMemoryError:
            if not self.chunks:
                raise
            self.release(self.leave + self.chunk)
            return call()

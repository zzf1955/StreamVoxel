"""Per-rank GPU utilization/power sampler for the live session (nvidia-smi, no pynvml)."""
from __future__ import annotations
from pathlib import Path
import subprocess
import threading
import time
from streamvoxel.common.io import atomic_json

QUERY = 'index,name,utilization.gpu,power.draw,power.limit,memory.used,memory.total'


def parse_line(line):
    index, name, util, power, limit, used, total = [v.strip() for v in line.split(',')]
    return dict(index=int(index), name=name, util=float(util), power_w=float(power), power_limit_w=float(limit),
                mem_used_mb=float(used), mem_total_mb=float(total))


class Telemetry:
    """Streams `nvidia-smi -lms` into `telemetry_r<rank>.json` (latest) and `gpu_telemetry_r<rank>.csv` (history)."""
    def __init__(self, run, rank, interval_ms=500, gpu_index=None):
        self.latest = Path(run) / f'telemetry_r{rank}.json'
        self.history = Path(run) / f'gpu_telemetry_r{rank}.csv'
        self.rank = rank
        self.gpu_index = None if gpu_index is None else int(gpu_index)
        self.proc = subprocess.Popen(['nvidia-smi', f'--query-gpu={QUERY}', '--format=csv,noheader,nounits', '-lms', str(interval_ms)],
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self.pump, daemon=True)
        self.thread.start()

    def pump(self):
        with self.history.open('a') as csv:
            for line in self.proc.stdout:
                try:
                    row = parse_line(line.split('\n')[0])
                except ValueError:
                    continue
                if self.gpu_index is not None and row['index'] != self.gpu_index:
                    continue
                row.update(rank=self.rank, unix_ns=time.time_ns())
                atomic_json(self.latest, row)
                csv.write(f"{row['unix_ns']},{row['util']},{row['power_w']},{row['mem_used_mb']}\n")
                csv.flush()

    def close(self):
        self.proc.terminate()

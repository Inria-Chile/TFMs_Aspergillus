"""Lightweight per-seed resource accounting."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from pathlib import Path


def gpu_snapshot() -> list[dict[str, str]]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            rows.append(dict(zip(("index", "name", "memory_used_mib", "memory_total_mib", "utilization_percent"), fields)))
    return rows


class ResourceMonitor:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.started = time.time()
        self.peak_gpu_memory_mib = {}

    def observe(self):
        for row in gpu_snapshot():
            index = row["index"]
            used = int(float(row["memory_used_mib"]))
            self.peak_gpu_memory_mib[index] = max(self.peak_gpu_memory_mib.get(index, 0), used)

    def stop(self, *, model: str, task: str, scenario: str, seed: int, fold_count: int, workers: int = 1, device: str = "cpu") -> dict:
        self.observe()
        finished = time.time()
        record = {
            "model": model,
            "task": task,
            "scenario": scenario,
            "seed": int(seed),
            "fold_count": int(fold_count),
            "workers": int(workers),
            "device": device,
            "duration_seconds": round(finished - self.started, 3),
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
            "gpu_snapshot_at_finish": gpu_snapshot(),
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib,
            "pid": os.getpid(),
            "finished_epoch": finished,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record

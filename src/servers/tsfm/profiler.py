"""profiler.py — Stopwatch + WandB logger for TSFM tool calls."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import Optional

import torch
import wandb


class TSFMProfiler:
    def __init__(self, run_name: str, config: dict = None):
        self._stages = {}
        self._call_count = 0
        self.run_name = run_name
        self._wandb_enabled = os.environ.get("DISABLE_WANDB_INIT") != "true"

        if self._wandb_enabled:
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_CONSOLE"] = "off"
            wandb.init(
                project="hpml-tsfm-optimization",
                name=run_name,
                config=config or {},
                settings=wandb.Settings(silent=True, console="off"),
            )

    @contextlib.contextmanager
    def measure(self, stage_name: str):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        yield
        wall_ms = (time.perf_counter() - t0) * 1000
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        self._stages[stage_name] = {"ms": round(wall_ms, 2), "vram_mb": round(vram_mb, 2)}

    def _build_log(self, extra: dict = None) -> dict:
        log = {"call": self._call_count}
        total_ms = 0.0
        for name, vals in self._stages.items():
            log[f"latency/{name}_ms"] = vals["ms"]
            log[f"vram/{name}_mb"] = vals["vram_mb"]
            total_ms += vals["ms"]
        log["latency/total_ms"] = round(total_ms, 2)
        if extra:
            log.update(extra)
        return log

    def log_and_reset(self, extra: dict = None):
        self._call_count += 1
        log = self._build_log(extra)

        if self._wandb_enabled:
            wandb.log(log)
            print(f"\n[Profiler] Call #{self._call_count} | {self.run_name}", file=sys.stderr)
            for name, vals in self._stages.items():
                print(f"  {name:<20} {vals['ms']:>8.1f} ms   {vals['vram_mb']:>6.0f} MB", file=sys.stderr)
            print(f"  {'TOTAL':<20} {log['latency/total_ms']:>8.1f} ms", file=sys.stderr)
        else:
            with open("/tmp/tsfm_profiler_last.json", "w") as f:
                json.dump(log, f)

        self._stages.clear()


_profiler: Optional[TSFMProfiler] = None

def init_global_profiler(run_name: str, config: dict = None) -> TSFMProfiler:
    global _profiler
    _profiler = TSFMProfiler(run_name, config)
    return _profiler

def get_profiler() -> Optional[TSFMProfiler]:
    return _profiler
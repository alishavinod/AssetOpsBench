"""
baseline_sweep_v2.py — Final baseline sweep via plan-execute.

Run from repo root:
    python baseline_sweep_v2.py
"""

import os
import statistics
import subprocess
import json

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

import wandb

DATASET_PATH = "src/tmp/assetopsbench/sample_data/chiller6_june2020_sensordata_couchdb.csv"
REPO_ROOT = os.path.expanduser("~/AssetOpsBench")

SENSOR_COUNTS = {
    1: ["Chiller 6 Tonnage"],
    2: ["Chiller 6 Tonnage", "Chiller 6 Power Input"],
    5: ["Chiller 6 Tonnage", "Chiller 6 Power Input", "Chiller 6 Supply Temperature",
        "Chiller 6 Return Temperature", "Chiller 6 Condenser Water Flow"],
    9: ["Chiller 6 Tonnage", "Chiller 6 Power Input", "Chiller 6 Supply Temperature",
        "Chiller 6 Return Temperature", "Chiller 6 Condenser Water Flow",
        "Chiller 6 Chiller Efficiency", "Chiller 6 Chiller % Loaded",
        "Chiller 6 Setpoint Temperature",
        "Chiller 6 Condenser Water Return To Tower Temperature"],
}

def build_query(sensors: list) -> str:
    target = sensors[0]
    conditionals = sensors[1:]
    cond_str = ", ".join(f"'{s}'" for s in conditionals) if conditionals else ""
    query = f"Forecast '{target}' using data in '{DATASET_PATH}'. Use parameter 'timestamp' as a timestamp."
    if cond_str:
        query += f" Use the following parameters as inputs {cond_str}"
    return query


def run_once(sensors: list) -> dict | None:
    result = subprocess.run(
        ["uv", "run", "plan-execute", build_query(sensors)],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={**os.environ, "DISABLE_WANDB_INIT": "true"},
    )
    success = any(x in result.stdout.lower() for x in [
        "successful",
        "forecasting complete",
        "forecast results are saved",
        "success message",
        "can be found in",
        "saved to /tmp",
        "the task to forecast",
        "forecast for",
    ])
    
    if not success:
        return None
    
    # Parse profiler output from stderr
    import re
    metrics = {}
    patterns = {
        "latency/data_loading_ms": r"data_loading\s+([\d.]+) ms",
        "latency/preprocessing_ms": r"preprocessing\s+([\d.]+) ms",
        "latency/model_loading_ms": r"model_loading\s+([\d.]+) ms",
        "latency/ttm_forward_ms": r"ttm_forward\s+([\d.]+) ms",
        "latency/postprocessing_ms": r"postprocessing\s+([\d.]+) ms",
        "latency/total_ms": r"TOTAL\s+([\d.]+) ms",
        "vram/ttm_forward_mb": r"ttm_forward\s+[\d.]+ ms\s+([\d.]+) MB",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, result.stderr)
        if match:
            metrics[key] = float(match.group(1))
    
    return metrics if metrics else None


def avg(sensors: list, n: int) -> dict | None:
    # warmup
    print("    warmup...", end=" ", flush=True)
    run_once(sensors)
    print("done")
    # measured runs
    results = []
    for i in range(n):
        print(f"    run {i+1}/{n}...", end=" ", flush=True)
        r = run_once(sensors)
        if r:
            results.append(r)
            print(f"{r['latency/ttm_forward_ms']:.0f}ms")
        else:
            print("failed")
    if not results:
        return None
    keys = results[0].keys()
    return {k: round(statistics.mean(r[k] for r in results if k in r), 2) for k in keys}


if __name__ == "__main__":
    wandb.init(
        project="hpml-tsfm-optimization",
        name="baseline_fp32_final",
        config={"opt": "baseline", "precision": "fp32", "model": "ttm_96_28"},
        settings=wandb.Settings(silent=True, console="off"),
    )

    print("\n=== n_sensors scaling (warm, 3 runs each) ===")
    for n, sensors in SENSOR_COUNTS.items():
        print(f"\n  n_sensors={n}:")
        r = avg(sensors, n=3)
        if not r:
            continue
        print(f"  → ttm_forward: {r['latency/ttm_forward_ms']:.0f}ms | "
              f"model_loading: {r['latency/model_loading_ms']:.0f}ms | "
              f"vram: {r.get('vram/ttm_forward_mb', 0):.0f}MB")
        wandb.log({
            "n_sensors/n_sensors":        n,
            "n_sensors/ttm_forward_ms":   r["latency/ttm_forward_ms"],
            "n_sensors/model_loading_ms": r["latency/model_loading_ms"],
            "n_sensors/data_loading_ms":  r["latency/data_loading_ms"],
            "n_sensors/total_ms":         r["latency/total_ms"],
            "n_sensors/vram_mb":          r.get("vram/ttm_forward_mb", 0),
        })

    wandb.finish()
    print("\n=== Done ===")
    print("https://wandb.ai/av3311-columbia-university/hpml-tsfm-optimization")
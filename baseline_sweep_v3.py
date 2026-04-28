"""
baseline_sweep_v3.py — Baseline sweep mirroring scenario 216 structure.
Run from repo root: conda run -n assetops python baseline_sweep_v3.py
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
TIMESTAMP_COL = "timestamp"
TARGET_COL = "Chiller 6 Tonnage"

ALL_CONDITIONALS = [
    "Chiller 6 Power Input",
    "Chiller 6 Supply Temperature",
    "Chiller 6 Return Temperature",
    "Chiller 6 Condenser Water Flow",
    "Chiller 6 Chiller Efficiency",
    "Chiller 6 Chiller % Loaded",
    "Chiller 6 Condenser Water Return To Tower Temperature",
    "Chiller 6 Liquid Refrigerant Evaporator Temperature",
]

SENSOR_COUNTS = {
    1: [],
    2: ALL_CONDITIONALS[:1],
    5: ALL_CONDITIONALS[:4],
    9: ALL_CONDITIONALS[:8],
}


def build_query(conditional_columns: list) -> str:
    all_cols = [TARGET_COL] + conditional_columns
    sensor_str = " and ".join(f"'{s}'" for s in all_cols)
    return (
        f"Run tsfm forecasting on '{DATASET_PATH}' "
        f"using model_checkpoint 'ttm_96_28', "
        f"timestamp_column '{TIMESTAMP_COL}', "
        f"target_columns {sensor_str}, "
        f"forecast_horizon 96, "
        f"frequency_sampling '15_minutes'"
    )


def run_once(conditional_columns: list) -> dict | None:
    try:
        os.remove("/tmp/tsfm_profiler_last.json")
    except FileNotFoundError:
        pass

    result = subprocess.run(
        ["uv", "run", "plan-execute", build_query(conditional_columns)],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={**os.environ, "DISABLE_WANDB_INIT": "true"},
    )
    success = (
        "Forecasting complete" in result.stdout
        or "success" in result.stdout.lower()
    )
    if not success:
        return None
    try:
        with open("/tmp/tsfm_profiler_last.json") as f:
            data = json.load(f)
        return {k: v for k, v in data.items()
                if k.startswith("latency/") or k.startswith("vram/")}
    except Exception:
        return None


def avg(conditional_columns: list, n: int) -> dict | None:
    print("    warmup...", end=" ", flush=True)
    run_once(conditional_columns)
    print("done")
    results = []
    for i in range(n):
        print(f"    run {i+1}/{n}...", end=" ", flush=True)
        r = run_once(conditional_columns)
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

    print("\n=== Baseline sweep (warm, 3 runs each) ===")

    for n, conditionals in SENSOR_COUNTS.items():
        print(f"\n  n_sensors={n}:")
        r = avg(conditionals, n=3)
        if not r:
            print("  -> all runs failed")
            continue
        print(f"  -> ttm_forward: {r['latency/ttm_forward_ms']:.0f}ms | "
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
import sys, os, time, json
import numpy as np
import pandas as pd
import torch
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from servers.tsfm.forecasting import (
    _get_ttm_hf_inference,
    _tsfm_data_quality_filter,
)
from servers.tsfm.io import _read_ts_data, _get_model_checkpoint_path, _get_dataset_path

# ── Config ────────────────────────────────────────────────────────────────────

CHILLER_CSV = "src/tmp/assetopsbench/sample_data/chiller6_june2020_sensordata_couchdb.csv"
MODELS = ["ttm_96_28", "ttm_512_96"]
NUM_RUNS = 3

# ── Load data and get sensor columns ─────────────────────────────────────────

df_full = pd.read_csv(CHILLER_CSV)
ALL_SENSORS = [c for c in df_full.columns if c not in ["asset_id", "timestamp"]]
# exclude Run Status — binary column that fails data quality filter
SENSORS = [c for c in ALL_SENSORS if "Run Status" not in c]
print(f"Using {len(SENSORS)} sensors: {SENSORS}")

# ── WandB ─────────────────────────────────────────────────────────────────────

wandb.init(
    project="hpml-team27",
    name="baseline-full-profiling",
    tags=["baseline", "fp32", "serial", "no-optimization"],
    config={
        "num_sensors": len(SENSORS),
        "num_runs": NUM_RUNS,
        "optimization": "none",
        "precision": "fp32",
        "dataset": "chiller6_june2020",
    }
)

# ── Helper: build dataset config ──────────────────────────────────────────────

def make_config(target_col):
    return {
        "column_specifiers": {
            "autoregressive_modeling": True,
            "timestamp_column": "timestamp",
            "conditional_columns": [],
            "target_columns": [target_col],
        },
        "id_columns": [],
        "frequency_sampling": "15_minutes",
    }

# ── Per-stage timing wrapper ───────────────────────────────────────────────────

def profile_single_sensor(sensor_col, model_checkpoint, model_config, data_df):
    """
    Runs one sensor through the full pipeline and returns
    a dict of per-stage latencies in ms.
    """
    stages = {}
    dataset_config = make_config(sensor_col)

    # Stage 1: Data quality filter
    t0 = time.perf_counter()
    try:
        dq_output = _tsfm_data_quality_filter(
            data_df[["timestamp", sensor_col]].copy(),
            dataset_config,
            model_config,
            task="inference"
        )
        stages["data_quality_ms"] = (time.perf_counter() - t0) * 1000
        filtered_df = dq_output["data"]
        filtered_config = dq_output["dataset_config_dictionary"]
    except Exception as e:
        return {"error": str(e)}

    if len(filtered_df) == 0:
        return {"error": "empty after data quality filter"}

    # Stage 2: TSP fit + dataset build + model load + forward pass
    # These are all inside _get_ttm_hf_inference so we time it as one block
    # and then subtract data quality time to get inference time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**2

    t0 = time.perf_counter()
    try:
        output = _get_ttm_hf_inference(
            filtered_df,
            filtered_config,
            model_config,
            model_checkpoint,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stages["inference_ms"] = (time.perf_counter() - t0) * 1000
    except Exception as e:
        stages["inference_ms"] = -1
        stages["error"] = str(e)

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**2
        stages["gpu_memory_delta_mb"] = mem_after - mem_before

    stages["total_ms"] = stages.get("data_quality_ms", 0) + stages.get("inference_ms", 0)
    stages["sensor"] = sensor_col
    stages["model"] = os.path.basename(model_checkpoint)
    return stages


# ── Model load time isolation ─────────────────────────────────────────────────

print("\n" + "="*60)
print("MODEL LOAD TIME ISOLATION")
print("="*60)

from tsfm_public import TinyTimeMixerForPrediction

model_load_times = {}
for model_name in MODELS:
    checkpoint = _get_model_checkpoint_path(model_name)
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        m = TinyTimeMixerForPrediction.from_pretrained(checkpoint)
        times.append((time.perf_counter() - t0) * 1000)
        del m
    model_load_times[model_name] = float(np.mean(times))
    print(f"  {model_name}: {model_load_times[model_name]:.1f}ms avg load time")

wandb.log({f"model_load/{k}": v for k, v in model_load_times.items()})


# ── Main profiling loop ───────────────────────────────────────────────────────

for model_name in MODELS:
    checkpoint = _get_model_checkpoint_path(model_name)
    with open(checkpoint + "/config.json") as f:
        model_config = json.load(f)

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    all_run_results = []

    for run in range(NUM_RUNS):
        run_label = "cold" if run == 0 else f"warm_{run}"
        print(f"\nRun {run+1}/{NUM_RUNS} ({run_label})")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        run_start = time.perf_counter()
        run_results = []

        for sensor in SENSORS:
            result = profile_single_sensor(
                sensor, checkpoint, model_config, df_full
            )
            result["run"] = run
            result["run_label"] = run_label
            run_results.append(result)

            if "error" in result:
                print(f"  {sensor[:45]}: ERROR — {result['error']}")
            else:
                print(
                    f"  {sensor[:45]}: "
                    f"dq={result.get('data_quality_ms', 0):.0f}ms "
                    f"inf={result.get('inference_ms', 0):.0f}ms "
                    f"total={result.get('total_ms', 0):.0f}ms"
                )

        run_total_ms = (time.perf_counter() - run_start) * 1000
        peak_gpu_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0
        )

        # Log per-run summary
        valid = [r for r in run_results if "error" not in r]
        if valid:
            avg_dq = float(np.mean([r["data_quality_ms"] for r in valid]))
            avg_inf = float(np.mean([r["inference_ms"] for r in valid]))
            avg_total = float(np.mean([r["total_ms"] for r in valid]))

            wandb.log({
                f"{model_name}/{run_label}/serial_total_ms": run_total_ms,
                f"{model_name}/{run_label}/avg_dq_filter_ms": avg_dq,
                f"{model_name}/{run_label}/avg_inference_ms": avg_inf,
                f"{model_name}/{run_label}/avg_per_sensor_ms": avg_total,
                f"{model_name}/{run_label}/peak_gpu_mb": peak_gpu_mb,
                f"{model_name}/{run_label}/successful_sensors": len(valid),
            })

            print(f"\n  Run {run+1} summary:")
            print(f"    Serial total:        {run_total_ms:.1f}ms")
            print(f"    Avg data quality:    {avg_dq:.1f}ms")
            print(f"    Avg inference:       {avg_inf:.1f}ms")
            print(f"    Peak GPU memory:     {peak_gpu_mb:.1f}MB")

        all_run_results.extend(run_results)

    # Per-sensor breakdown table for WandB
    valid_all = [r for r in all_run_results if "error" not in r]
    if valid_all:
        sensor_table = wandb.Table(
            columns=["sensor", "run", "run_label", "data_quality_ms",
                     "inference_ms", "total_ms", "model"],
            data=[
                [
                    r["sensor"], r["run"], r["run_label"],
                    r.get("data_quality_ms", 0),
                    r.get("inference_ms", 0),
                    r.get("total_ms", 0),
                    r["model"]
                ]
                for r in valid_all
            ]
        )
        wandb.log({f"{model_name}/per_sensor_breakdown": sensor_table})

        # Stacked bar: avg time split between data quality vs inference
        warm_results = [r for r in valid_all if r["run_label"] != "cold"]
        if warm_results:
            by_sensor = {}
            for r in warm_results:
                s = r["sensor"][:35]
                if s not in by_sensor:
                    by_sensor[s] = {"dq": [], "inf": []}
                by_sensor[s]["dq"].append(r.get("data_quality_ms", 0))
                by_sensor[s]["inf"].append(r.get("inference_ms", 0))

            stage_table = wandb.Table(
                columns=["sensor", "stage", "latency_ms"],
                data=[
                    [s, "data_quality_filter", float(np.mean(v["dq"]))]
                    for s, v in by_sensor.items()
                ] + [
                    [s, "inference", float(np.mean(v["inf"]))]
                    for s, v in by_sensor.items()
                ]
            )
            wandb.log({
                f"{model_name}/stage_breakdown_chart": wandb.plot.bar(
                    stage_table, "sensor", "latency_ms",
                    title=f"{model_name}: per-sensor stage breakdown (warm runs)"
                )
            })


# ── CPU vs GPU comparison (one sensor, both devices) ─────────────────────────

print("\n" + "="*60)
print("CPU vs GPU COMPARISON")
print("="*60)

test_sensor = SENSORS[0]
checkpoint = _get_model_checkpoint_path("ttm_96_28")
with open(checkpoint + "/config.json") as f:
    model_config = json.load(f)

dataset_config = make_config(test_sensor)
dq_out = _tsfm_data_quality_filter(
    df_full[["timestamp", test_sensor]].copy(),
    dataset_config, model_config, task="inference"
)
filtered_df = dq_out["data"]
filtered_config = dq_out["dataset_config_dictionary"]

device_times = {}
for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _get_ttm_hf_inference(
            filtered_df.copy(), filtered_config,
            model_config, checkpoint
        )
        times.append((time.perf_counter() - t0) * 1000)
    device_times[device] = float(np.mean(times))
    print(f"  {device}: {device_times[device]:.1f}ms")

wandb.log({f"cpu_gpu_comparison/{k}": v for k, v in device_times.items()})

device_table = wandb.Table(
    columns=["device", "latency_ms"],
    data=[[k, v] for k, v in device_times.items()]
)
wandb.log({
    "cpu_gpu_comparison/chart": wandb.plot.bar(
        device_table, "device", "latency_ms",
        title="CPU vs GPU: single sensor inference (ttm_96_28)"
    )
})


# ── Final summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("BASELINE PROFILING COMPLETE")
print("="*60)
print(f"WandB run: {wandb.run.get_url()}")

wandb.finish()
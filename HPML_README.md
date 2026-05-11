# AssetOpsBench TSFM Performance Optimization

**Team 27 - Columbia University HPML Spring 2026**  
Alisha Vinod, Thomas Ajai, Jonathan Ang, Sanjaii Vijayakumar

[![WandB Dashboard](https://img.shields.io/badge/WandB-Dashboard-orange)](https://wandb.ai/av3311-columbia-university/hpml-tsfm-optimization)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/alishavinod/AssetOpsBench)

---

## Overview

This project optimizes the TSFM (TinyTimeMixer) MCP agent within IBM's AssetOpsBench industrial AI benchmark. We target inference-time workflow completion latency through model pre-loading, `torch.compile` kernel fusion, HuggingFace Trainer replacement, and GPU placement with mixed-precision evaluation.

**Headline result: 3.3× reduction in workflow completion latency (43,021ms → 13,798ms)**

---

## Quick Start

### 1. Install dependencies

Run from the **repo root**:

```bash
uv sync
```

`uv sync` creates a virtual environment at `.venv/`, installs all dependencies, and registers the CLI entry points (`plan-execute`, `*-mcp-server`). You can either prefix commands with `uv run` (no activation needed) or activate the venv once for your shell session:

```bash
source .venv/bin/activate   # macOS / Linux
```

### 2. Configure environment

Copy `.env.public` to `.env` and fill in the required values (see [Environment Variables](#environment-variables)):

```bash
cp .env.public .env
# Then edit .env and set WATSONX_APIKEY, WATSONX_PROJECT_ID
# CouchDB defaults work out of the box with the Docker setup
```

### 3. Start CouchDB

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
```

Verify CouchDB is running:

```bash
curl -X GET http://localhost:5984/
```


### 4. Run optimized sweep
```
uv run -n assetops python src/servers/tsfm/baseline_sweep.py
```

---

## Repository Structure
```
AssetOpsBench/
├── src/
│   ├── servers/tsfm/
│   │   ├── main.py              # TSFM MCP server with pre-loading
│   │   ├── forecasting.py       # Inference pipeline with profiler hooks
│   │   ├── model_cache.py       # Model singleton cache + torch.compile
│   │   ├── profiler.py          # Custom TSFMProfiler class
│   │   └── baseline_sweep.py    # Sweep script (n_sensors scaling)
│   └── agent/
│       └── plan_execute/
│           └── executor.py      # MCP orchestrator
├── locustfile.py                # Concurrent load testing
├── deliverables/
│   ├── HPML_Team27_Final_Project_Presentation.pptx
│   └── Team27_HPML_Final_Report.pdf
├── environment.yml      # Pinned working environment
├── LICENSE
└── README.md
```
---

## Environment Setup

**Requirements:**
- GCP VM with NVIDIA T4 GPU (or equivalent CUDA 12.4 device)
- uv
- IBM watsonx.ai API key and project ID

**Create environment:**
```bash
  uv sync
```

**Key package versions:**
- Python 3.12.13
- PyTorch 2.6.0+cu124
- granite-tsfm 0.3.6
- transformers 4.57.6
- huggingface-hub 0.36.2
- wandb 0.25.1
- mcp 1.27.0

## Install TTM model
```bash
MODEL_DIR="src/servers/tsfm/artifacts/output/tuned_models/ttm_96_28"

echo "Creating model directory..."
mkdir -p $MODEL_DIR

echo "Downloading IBM Granite TTM weights from Hugging Face..."
uv run huggingface-cli download ibm-granite/granite-timeseries-ttm-r2 --local-dir $MODEL_DIR
```


## Environment Variables

Create a `.env` file at the repo root:

```bash
WATSONX_APIKEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WANDB_API_KEY=your_wandb_key_here
LOG_LEVEL=WARNING
WANDB_SILENT=true
WANDB_CONSOLE=off
```

---

## Reproducing Results

**Run baseline sweep (CPU, no optimizations):**
```bash
# Revert to original IBM code
git checkout ab8c333 -- src/servers/tsfm/main.py
git checkout ab8c333 -- src/servers/tsfm/forecasting.py

conda run -n assetops python src/servers/tsfm/baseline_sweep.py
```

**Run optimized sweep (pre-loading + torch.compile + GPU):**
```bash
# Restore optimized code
git checkout HEAD -- src/servers/tsfm/main.py
git checkout HEAD -- src/servers/tsfm/forecasting.py

conda run -n assetops python src/servers/tsfm/baseline_sweep.py
```

**Run single query with profiler output:**
```bash
uv run plan-execute \
  "Forecast 'Chiller 6 Tonnage' using data in \
  'src/tmp/assetopsbench/sample_data/chiller6_june2020_sensordata_couchdb.csv'. \
  Use parameter 'timestamp' as a timestamp. \
  Use the following parameters as inputs \
  'Chiller 6 Power Input,Chiller 6 Supply Temperature,Chiller 6 Return Temperature,\
  Chiller 6 Condenser Water Flow,Chiller 6 Chiller Efficiency,Chiller 6 Chiller % Loaded,\
  Chiller 6 Condenser Water Return To Tower Temperature,\
  Chiller 6 Liquid Refrigerant Evaporator Temperature,Chiller 6 Setpoint Temperature'" \
  2>&1 | grep -E "Profiler|ttm_forward|TOTAL|MB"
```



---

## Results Summary

| Configuration | ttm\_forward (ms) | Total (ms) | Speedup |
|---|---|---|---|
| Baseline CPU | 40,161 | 43,021 | 1.0× |
| + Pre-loading + torch.compile + no Trainer | 16,000 | 17,000 | 2.5× |
| + GPU placement + TF32 | 12,293 | 13,798 | **3.3×** |
| + bf16 autocast | 15,272 | 16,414 | 2.6× (regression) |

**WandB Dashboard:** https://wandb.ai/av3311-columbia-university/hpml-tsfm-optimization

---

## Dataset

We use IBM chiller6 sensor data (2,896 rows, 12 sensor columns, 15-minute frequency, June 2020) as a proxy for the official AssetOpsBench chiller9 dataset. The data is located at:
```
src/tmp/assetopsbench/sample_data/chiller6_june2020_sensordata_couchdb.csv
```

The official AssetOpsBench benchmark dataset is available at:
https://huggingface.co/datasets/ibm-research/AssetOpsBench

---

## Hardware

- **Instance:** GCP n1-standard-4
- **GPU:** NVIDIA L4
- **CUDA:** 12.4
- **CPU:** Intel Skylake, 4 vCPUs
- **RAM:** 15GB

---

## Team Contributions

| Member | Contributions |
|---|---|
| Alisha Vinod | Baseline sweep, Wandb logging, bf16 evaluation |
| Thomas Ajai | GPU optimization, resolving dependency/integration conflicts |
| Jonathan Ang | torch.compile integration, model_cache.py, Trainer replacement |
| Sanjaii Vijayakumar | Initial baseline setup, building configurations, testing out the plan-execute pipeline|
---

## AI Tool Use

This project used Claude as a debugging and implementation aid for environment setup, code instrumentation, and prose polishing. All profiling interpretations, performance reasoning, experimental design decisions, and written analysis are our own. Claude was not used to generate profiling conclusions or performance reasoning. This disclosure is included per the HPML AI Use Policy.

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

```bibtex
@misc{assetopsbench,
  title  = {AssetOpsBench: Benchmarking AI Agents for Task Automation
            in Industrial Asset Operations and Maintenance},
  author = {Patel, Dhaval and Lin, Shuxin and Rayfield, James and
            Zhou, Nianjun and Vaculin, Roman and Martinez, Natalia and
            O'Donncha, Fearghal and Kalagnanam, Jayant},
  year   = {2025},
  url    = {https://arxiv.org/abs/2506.03828}
}
```
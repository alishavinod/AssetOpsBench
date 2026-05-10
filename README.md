```                                          
                                          ,--,    
        ,--,,-.----.             ____  ,---.'|    
      ,--.'|\    /  \          ,'  , `.|   | :    
   ,--,  | :|   :    \      ,-+-,.' _ |:   : |    
,---.'|  : '|   |  .\ :  ,-+-. ;   , |||   ' :    
|   | : _' |.   :  |: | ,--.'|'   |  ;|;   ; '    
:   : |.'  ||   |   \ :|   |  ,', |  ':'   | |__  
|   ' '  ; :|   : .   /|   | /  | |  |||   | :.'| 
'   |  .'. |;   | |`-' '   | :  | :  |,'   :    ; 
|   | :  | '|   | ;    ;   . |  ; |--' |   |  ./  
'   : |  : ;:   ' |    |   : |  | ,    ;   : ;    
|   | '  ,/ :   : :    |   : '  |/     |   ,/     
;   : ;--'  |   | :    ;   | |`-'      '---'      
|   ,/      `---'.|    |   ;/                     
'---'         `---`    '---'                      
```                                                  

# Final Project: [Performance Optimization of the TSFM Agent in an Industrial Agentic Benchmark]

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Team 27
- **Members:**
  - Jonathan Ang ma4624 — model caching + bf16
  - Alisha Vinod av3311 — bf16 + baseline results
  - Thomas Ajai · tl3444 — gpu integration
  - Sanjaii Vijayakumar · sv2851 — 

## Submission

- **GitHub repository:** [https://github.com/alishavinod/AssetOpsBench/tree/baseline](https://github.com/alishavinod/AssetOpsBench/tree/baseline)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](/deliverables/HPML_Team27_Final_Project_Presentation.pdf)
- **Experiment-tracking dashboard:** [https://wandb.ai/av3311-columbia-university/hpml-tsfm-optimization/workspace?nw=nwuserav3311]

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

Industrial maintenance workflows depend on fast, reliable TSFM forecasting.
Current MCP tool calls add latency from model loading, data filtering, and inference.
A naive baseline makes workflows slower and harder to scale across many sensors.
We profile and optimize the TSFM MCP agent in IBM’s AssetOpsBench benchmark, targeting latency bottlenecks across data quality filtering, model loading, and forecasting inference.
Goal:
Reduce per-tool-call and end-to-end workflow latency through three targeted optimizations while preserving forecast quality.

---

## 2. Model/Application Description

Briefly describe the model(s) and stack you used:

- **Model architecture:** TinyTimeMixer (TTM)
- **Dataset:** IBM AssetOpsBench - Chiller 6 Sensor Data -> chiller6_june2020_sensordata_couchdb
- **Hardware:** NVIDIA T4 (GCP n1-standard-4 instance)

---

## 3. Final Results Summary

Replace the numbers below with your measured values. Add or remove rows to fit your study.

+------------------------------+---------------+---------------+--------------------+
| Metric                       | Baseline      | Optimized     | Δ (Improvement)    |
+------------------------------+---------------+---------------+--------------------+
| Workflow Latency (End-to-End)| 43,021 ms     | 13,798 ms     | 68% reduction      |
| Inference Latency (ttm_fwd)  | ~38,000 ms    | 12,293 ms     | 3.1x faster        |
| Model Loading Overhead       | 233 ms        | 26 ms         | 9x faster          |
| Inference Precision          | FP32          | FP32+BF16     | No change          |
| Peak GPU Memory (9 sensors)  | ~150 MB       | ~300 MB       | 100% more (BF16)   |
| Model Size (Parameters)      | ~1-5M         | ~1-5M         | No change          |
+------------------------------+---------------+---------------+--------------------+


**Headline result (one sentence):** *Optimizing the TSFM MCP agent through model pre-loading, torch.compile kernel fusion, and GPU placement achieved a 3.3x reduction in total workflow latency (from 43,021 ms to 13,798 ms) while identifying that model caching was the dominant factor in performance gains.*

---

## 4. Repository Structure
```
.
├── HPML_README.md
├── LICENSE
├── requirements.txt
├── configs/                # YAML / JSON configs for every reported experiment
├── deliverables/           # Final report (PDF) and final presentation (PPT/PDF) — same files uploaded to CourseWorks
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
├── src/
│   ├── servers/                  # Server definitions & request handling
│   │   ├── tsfm/                 # Code that uses the tsfm model
│   │   │   ├── forecasting.py    # Calls the TTM model and uses it to make predictions
│   │   │   ├── model_cache.py    # Caches the model to be used by forecasting.py
│   │   │   └── main.py           # tsfm initialization
```
---

## 5. Reproducibility Instructions

### 1. Environment Setup

```bash
# Clone
git clone https://github.com/alishavinod/AssetOpsBench.git
cd <repo>
```

## 2. Environment Variables

**WatsonX** — plan-execute runner (when `--model-id` starts with `watsonx/`) in .env.public

| Variable             | Default                             | Description                 |
| -------------------- | ----------------------------------- | --------------------------- |
| `WATSONX_APIKEY`     | _(required)_                        | IBM WatsonX API key         |
| `WATSONX_PROJECT_ID` | _(required)_                        | IBM WatsonX project ID      |

### 3. Install dependencies

Run from the **repo root**:

```bash

uv sync
```

`uv sync` creates a virtual environment at `.venv/`, installs all dependencies, and registers the CLI entry points (`plan-execute`, `*-mcp-server`). You can either prefix commands with `uv run` (no activation needed) or activate the venv once for your shell session:

```bash
source .venv/bin/activate   # macOS / Linux
```

### 4. Configure environment

Copy `.env.public` to `.env` and fill in the required values (see [Environment Variables](#environment-variables)):

```bash
cp .env.public .env
# Then edit .env and set WATSONX_APIKEY, WATSONX_PROJECT_ID
# CouchDB defaults work out of the box with the Docker setup
```

### 5. Start CouchDB

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
```

Verify CouchDB is running:

```bash
curl -X GET http://localhost:5984/
```

### 6. Run servers

> **Note:** MCP servers use stdio transport — they are spawned on-demand by clients (Claude Desktop, `plan-execute`) and exit when the client disconnects. They are not long-running daemons.

To start a server manually for testing:

```bash
uv run plan-execute "Run TSFMAgent forecasting on 'src/tmp/assetopsbench/sample_data/chiller6_june2020_sensordata_couchdb.csv' using model_checkpoint 'ttm_96_28', timestamp_column 'timestamp', target_columns 'Chiller 6 Tonnage', forecast_horizon 96, frequency_sampling '15_minutes'"
```

### C. Dataset

The dataset is committed to the repository. The script fetches it from [source] (license: [license]) and stores it under `data/`.

### D. Training

To reproduce the baseline:

```bash
python src/train.py --config configs/baseline.yaml
```

To reproduce the optimized run:

```bash
python src/train.py --config configs/optimized.yaml
```

### E. Evaluation

```bash
python src/eval.py --weights checkpoints/best_model.pth --config configs/optimized.yaml
```

### F. Profiling

To regenerate the profiler traces referenced in the report:

```bash
python src/profile.py --config configs/optimized.yaml --output results/trace.json
# View in chrome://tracing or perfetto.dev
```

### G. Quickstart: Reproduce the Headline Result

The following sequence reproduces the headline number in Section 3 end-to-end (≈ XX minutes on a single A100):

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Download dataset
bash scripts/download_dataset.sh

# 3. Run optimized training (or skip if checkpoint provided in releases)
bash scripts/run_optimized.sh

# 4. Evaluate
python src/eval.py --weights checkpoints/best_model.pth
```

---

## 6. Results and Observations

- *Optimization 1 Model pre-loading + caching: Reduced model loading time from 233ms to 26ms per tool call (9× speedup), by eliminating repeated from_pretrained() disk reads on every MCP invocation.
- *Optimization 2 torch.compile + Trainer removal: Fusing 19,032 eager-mode CUDA kernel launches and removing the HuggingFace Trainer loop cut ttm_forward from 40,161ms to 16,000ms (2.5× total speedup).
- *Optimization 3 GPU placement (TF32): Moving the model to CUDA with TF32 precision further reduced ttm_forward to 12,293ms, achieving a 3.3× total speedup over baseline.
- *What did not work:* bf16 mixed precision: Regressed latency to 15,272ms; TTM's small MLP matrix dimensions don't saturate Tensor Cores, so dtype conversion overhead outweighed any memory bandwidth savings.

![Baseline vs Optimized latency](results/figures/latency_comparison.png)

---

## 7. Notes

- All secrets (API keys, Wandb tokens) are loaded from environment variables. See `.env.public`.

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [X] Yes, we used AI assistance as described below.

**Tool(s) used:** *Claude, Copilot*

**Specific purpose:** *polished prose in the report/presentation, research, dependency debugging*

**Sections affected:** *e.g., src/profile.py setup, README §6 results narrative, report §V Discussion*

**How we verified correctness:** *re-ran every reported experiment ourselves*

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{teamname2026hpml,
  title  = {[Project Title]},
  author = {Last1, First1 and Last2, First2 and Last3, First3},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/<org>/<repo>}
}
```

### Contact

Open a GitHub Issue or email *ma4624@columbia.edu]*.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*

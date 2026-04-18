# Airline Graph Neural Network for Flight Delay Prediction

Graph-temporal flight delay forecasting over a large U.S. airline network using heterogeneous graphs, recurrent airport/tail state, and multi-horizon evaluation.

This repository is an end-to-end ML systems project: data validation, graph construction, temporal training, cluster evaluation, and demo-facing inference scripts. It started as an arrival-delay project and was later migrated to a departure-delay formulation for cleaner benchmarking and more realistic pre-departure use.

## What This Project Does

The model predicts flight delay across multiple pre-departure horizons:

- `0h`: `<1h before departure`
- `1h`: `1–3h before departure`
- `3h`: `3–6h before departure`
- `6h`: `>6h before departure`

The graph combines:

- airport nodes
- flight nodes
- airport-to-airport network edges
- congestion edges
- airport-to-flight assignment edges
- flight-to-flight rotation edges
- per-airport recurrent state
- per-tail recurrent state

The current primary target is **departure delay**.

## Why This Is Interesting

Flight delay prediction is a hard systems problem:

- delays propagate through aircraft rotations
- airport congestion shifts quickly
- weather matters at both origin and destination
- severe disruptions are rare but operationally important
- evaluation can be misleading if the pipeline leaks future information

This project focuses not just on model accuracy, but on **causal-ish feature design, leakage control, and honest evaluation**.

## Current Status

The repository contains both stable results and active experiments.

### Current primary model

The current primary checkpoint is:

- checkpoint: `best_model_dep_12k_ordinal_ep25.pt`

This is the strongest corrected overall departure-delay model so far after the ordinal-tail reformulation.

Corrected departure-delay validation performance:

| Metric | Value |
| --- | ---: |
| Overall MAE | `13.244` min |
| Overall RMSE | `40.485` min |
| `0h` MAE | `12.25` |
| `1h` MAE | `12.38` |
| `3h` MAE | `13.49` |
| `6h` MAE | `14.23` |

Corrected departure-delay test performance on the rebuilt 2022 holdout:

| Metric | Value |
| --- | ---: |
| Overall MAE | `16.572` min |
| Overall RMSE | `47.448` min |
| `0h` MAE | `15.17` |
| `1h` MAE | `15.42` |
| `3h` MAE | `16.97` |
| `6h` MAE | `17.83` |

Severe-delay detection using a simple `pred >= 120` rule:

- all bands precision: `0.611`
- all bands recall: `0.370`
- 6h precision: `0.543`
- 6h recall: `0.299`

The ordinal classifier head is more conservative than the regression threshold:

- classifier all-bands precision: `0.673`
- classifier all-bands recall: `0.163`
- classifier 6h precision: `0.622`
- classifier 6h recall: `0.147`

### Strong baseline

The simpler departure-delay baseline remains an important comparison point:

- checkpoint: `best_model_before_severe_patch.pt`

Corrected departure-delay test performance:

| Metric | Value |
| --- | ---: |
| Overall MAE | `16.656` min |
| Overall RMSE | `49.040` min |
| `0h` MAE | `15.46` |
| `1h` MAE | `15.64` |
| `3h` MAE | `17.00` |
| `6h` MAE | `17.76` |

This baseline is slightly simpler and still competitive, but the ordinal checkpoint edges it out overall and gives much better tail interpretability.

### Active experiments

Recent and ongoing experiments include:

- coarse severe-tail model (`best_model_dep_12k_severe_ep6.pt`)
  - useful for understanding recall / calibration tradeoffs
  - not better than the base or ordinal model overall
- signed-`log1p` regression target
  - improved stability but became too conservative in the tail
  - not adopted as the main checkpoint
- tail uplift head for `>=240` and `>=720`
  - current active experiment
  - intended to improve extreme-tail minute prediction without sacrificing overall calibration

### Important evaluation note

The 2022 test split is relatively sparse compared with train/val and behaves partly like an out-of-distribution stress test. Earlier test runs in this repo were invalid until `snapshots_test.pt` was refreshed after the departure-delay migration. The numbers above are from the corrected test setup.

## Repository Layout

### Core training pipeline

- [00_validate_dataset.py](00_validate_dataset.py): basic dataset validation and sanity checks
- [02_build_rotation_edges.py](02_build_rotation_edges.py): build aircraft rotation edges
- [03_build_weather_edge.py](03_build_weather_edge.py): weather feature/edge construction
- [04_build_congestion_edges.py](04_build_congestion_edges.py): congestion edge construction
- [05_build_graph_snapshots.py](05_build_graph_snapshots.py): build graph snapshots and labels
- [06_train_gnn.py](06_train_gnn.py): train the graph-temporal GNN
- [07_dashboard.py](07_dashboard.py): offline evaluation and model comparison

### Demo / inference scripts

- [08_realtime_connector.py](08_realtime_connector.py): live API connector for real-time data refresh
- [09_day_simulation.py](09_day_simulation.py): track predictions for flights across a day
- [10_flight_finder.py](10_flight_finder.py): lightweight web app for on-demand predictions

### Data / outputs

- [`graph_data/`](graph_data): built graph artifacts, lookups, static edges, snapshot tensors
- [`checkpoints/`](checkpoints): saved model checkpoints
- [`evaluation/`](evaluation): evaluation CSVs and comparison tables
- [`outputs/`](outputs): simulation and demo outputs
- [`data/`](data): data cleaning, retrieval, and audit utilities

## Data

The project uses a large U.S. flight dataset from 2018–2022 plus airport weather data and graph-derived operational features.

Main flight parquet:

- [`flights_2018_2022.parquet`](flights_2018_2022.parquet)

Graph data artifacts generated by step 5 include:

- [`graph_data/snapshots_train.pt`](graph_data/snapshots_train.pt)
- [`graph_data/snapshots_val.pt`](graph_data/snapshots_val.pt)
- [`graph_data/snapshots_test.pt`](graph_data/snapshots_test.pt)
- [`graph_data/static_edges.pt`](graph_data/static_edges.pt)
- [`graph_data/flight_lookup.parquet`](graph_data/flight_lookup.parquet)
- [`graph_data/tail2idx.json`](graph_data/tail2idx.json)

## Model Summary

The main model in [06_train_gnn.py](06_train_gnn.py) is a heterogeneous graph neural network with recurrent state:

- HGT-style message passing over airport/flight nodes
- airport GRU to maintain temporal airport state
- per-tail GRUCell to model aircraft state propagation
- dynamic rotation gate
- multi-horizon supervision

Current default hidden sizes:

- hidden dim: `512`
- heads: `8`
- GNN layers: `2`
- airport GRU: `2 x 512`
- tail hidden dim: `128`

## Feature Design

Examples of signals used:

- route-history priors
- airport dynamic delay / taxi aggregates
- scheduled traffic load
- forecast weather
- inbound tail state
- actual previous-leg arrival delay, but only after the previous leg has landed
- aircraft rotation structure

Examples of signals intentionally removed or zeroed to avoid direct leakage for departure prediction:

- current flight realized `DepDelay`
- current flight realized `TaxiOut`
- current flight `CarrierDelay`

## Leakage / Causality Notes

This project has gone through multiple leakage audits and bug fixes.

### Fixed

- removed direct current-flight departure-delay leakage
- removed direct current-flight taxi-out leakage
- fixed a rotation self-loop bug by building true `leg1 -> leg2` flight rotation edges
- corrected stale test snapshots after the departure-delay migration

### Still important to know

The project still uses **same-hour airport dynamic aggregates** (average departure delay, taxi, etc.). These are operationally plausible in a near-real-time system, but they are somewhat optimistic for strict benchmark-style forecasting because they may include information from later in the same hour.

So this repo is designed to be **causal-aware and much cleaner than a naive pipeline**, but not a final publication-grade benchmark yet.

## Key Experiments So Far

### 1. Arrival-delay baseline

Earlier versions predicted arrival delay and achieved decent holdout performance, but the target was harder to compare to public departure-delay baselines.

### 2. Departure-delay migration

The full pipeline was migrated from arrival delay to departure delay:

- route priors switched to `DepDelay`
- airport labels switched to departure-delay averages
- flight labels switched to `DepDelay`
- direct leakage features removed

### 3. Severe-tail experiments

The project explored models focused on rare extreme delays:

- coarse `>=120 min` severe classifier
- severity-aware window sampling
- ordinal-tail reformulation with thresholds:
  - `>=0`
  - `>=15`
  - `>=60`
  - `>=120`
  - `>=240`
  - `>=720`
- signed-`log1p` regression target
- tail uplift head for:
  - `>=240`
  - `>=720`

The strongest result from this line so far is the ordinal checkpoint `best_model_dep_12k_ordinal_ep25.pt`. Tail-specific refinement beyond that is still active work.

## Running the Pipeline

### 1. Validate / inspect data

```bash
python 00_validate_dataset.py
```

### 2. Build graph components

```bash
python 02_build_rotation_edges.py
python 03_build_weather_edge.py
python 04_build_congestion_edges.py
python 05_build_graph_snapshots.py
```

### 3. Train

```bash
python 06_train_gnn.py
```

### 4. Evaluate

```bash
python 07_dashboard.py --checkpoint best_model.pt --split val
python 07_dashboard.py --checkpoint best_model.pt --split test
```

## Running on Rice / Slurm

This project was trained and evaluated on Rice NOTS using Slurm.

Typical pattern:

1. build snapshots locally or on cluster
2. copy `graph_data/` artifacts if needed
3. upload updated scripts
4. train with `sbatch`
5. evaluate with `sbatch`

The current scripts support overriding the project root with:

```bash
python 07_dashboard.py --checkpoint best_model.pt --split test --drive_base /scratch/jb310/airline_project
```

## Dependencies

There is no final pinned `requirements.txt` yet, but the core stack is:

- Python `3.11+`
- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `pyarrow`
- `requests`
- `python-dotenv`
- `dash`
- `plotly`
- `dash-bootstrap-components`

Example:

```bash
pip install torch torch-geometric numpy pandas pyarrow requests python-dotenv dash plotly dash-bootstrap-components
```

## Notable Findings

- A graph-temporal model can learn useful multi-horizon delay structure over a large airline network.
- Contiguous training windows improve temporal realism, but broader regime coverage is still important for robustness.
- A model can look excellent on validation while still failing under a shifted holdout regime.
- Severe-delay detection benefits from explicit tail-aware objectives, but overly coarse severe buckets can hurt calibration.
- The ordinal-tail reformulation improved overall corrected test performance while giving a more interpretable breakdown of `120-240`, `240-720`, and `720+` disruption tiers.
- Simply compressing the regression target with `log1p` was not enough to fix the far tail by itself.
- Honest evaluation and pipeline debugging mattered as much as the raw architecture.

## Limitations

- same-hour airport aggregates are still somewhat optimistic for strict benchmarking
- no final public benchmark protocol yet matching an exact `2h before departure, one prediction per flight` paper-style setup
- severe/extreme tail minute prediction is still under active iteration, especially for the `720+` terminal tier
- demo scripts ([09_day_simulation.py](09_day_simulation.py), [10_flight_finder.py](10_flight_finder.py)) were originally written around earlier arrival-delay checkpoints and may need light refresh depending on which checkpoint you want to showcase

## Next Steps

- evaluate the tail-uplift architecture against the ordinal baseline
- strengthen minute prediction for `240-720` and especially `720+`
- keep the ordinal tier outputs as the main risk signal for demos
- evaluate exact-horizon per-flight benchmarks
- tighten same-hour feature realism
- refresh demo-facing scripts around the final departure-delay checkpoint

## Why This Repo Exists

This project is meant to show:

- graph ML engineering
- large-scale temporal data handling
- debugging under real-world data messiness
- model evaluation discipline
- practical deployment thinking

It is both a forecasting project and a systems/debugging project.

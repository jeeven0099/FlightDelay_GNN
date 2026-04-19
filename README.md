# Airline Graph Neural Network for Flight Disruption Forecasting

Graph-temporal forecasting of U.S. flight disruptions using heterogeneous graphs, recurrent airport and aircraft state, multi-horizon supervision, and precision-first severe-delay alerting.

This repository is an end-to-end ML systems project, not just a modeling notebook. It includes data validation, graph construction, temporal training, cluster-scale evaluation, leakage audits, ablation experiments, and an interactive replay dashboard for demoing model behavior throughout a day of airline operations.

The project started as an arrival-delay model and was later migrated to a departure-delay formulation for cleaner benchmarking, better causal alignment, and more realistic pre-departure use.

## Highlights

- Forecasts departure delay across four pre-departure horizons: `0h`, `1h`, `3h`, and `6h`
- Models propagation through airports, aircraft rotations, congestion structure, and weather
- Uses a heterogeneous graph with recurrent airport state and per-tail recurrent memory
- Produces both minute predictions and interpretable ordinal disruption tiers
- Supports high-precision severe-delay alerting for personal-use and demo scenarios
- Includes a replay website that visualizes predictions on a U.S. map over the course of a validation day

## Why This Problem Is Hard

Flight delay prediction is a real systems problem, not a clean tabular benchmark.

- Delays propagate through aircraft rotations and airport congestion
- Weather affects both the origin and the downstream network
- Extreme disruptions are rare, heterogeneous, and operationally important
- Evaluation is easy to get wrong if the pipeline leaks future information
- A model can look excellent on validation while failing on a shifted holdout split

This project emphasizes model quality, but just as importantly, it emphasizes evaluation integrity, leakage control, and product-minded interpretation of outputs.

## Final Model

The current primary checkpoint is:

- `best_model_dep_12k_ordinal_ep25.pt`

This is the strongest corrected overall departure-delay model in the repository. It combines:

- multi-horizon departure-delay regression
- ordinal disruption tiers with thresholds at `0`, `15`, `60`, `120`, `240`, and `720` minutes
- a conservative classifier signal that is useful for high-precision severe alerts

### Corrected Validation Performance

| Metric | Value |
| --- | ---: |
| Overall MAE | `13.244` min |
| Overall RMSE | `40.485` min |
| `0h` MAE | `12.25` |
| `1h` MAE | `12.38` |
| `3h` MAE | `13.49` |
| `6h` MAE | `14.23` |

### Corrected Test Performance

These results are from the corrected 2022 departure-delay holdout after refreshing the stale test snapshot used in earlier runs.

| Metric | Value |
| --- | ---: |
| Overall MAE | `16.572` min |
| Overall RMSE | `47.448` min |
| `0h` MAE | `15.17` |
| `1h` MAE | `15.42` |
| `3h` MAE | `16.97` |
| `6h` MAE | `17.83` |

### Severe Delay Performance

Using a regression rule of `pred >= 120` minutes:

- all bands precision: `0.611`
- all bands recall: `0.370`
- `6h` precision: `0.543`
- `6h` recall: `0.299`

Using the ordinal classifier head as a severe alert signal:

- classifier all-bands precision: `0.673`
- classifier all-bands recall: `0.163`
- classifier `6h` precision: `0.622`
- classifier `6h` recall: `0.147`

The regression head is the broader detector. The classifier head is more conservative and works better as a high-confidence alert signal.

## Precision-First Severe Alerting

For personal-use alerting, the most useful output is not raw MAE. It is a trustworthy severe-alert rule.

This project uses:

- default severe alert: `severe_prob >= 0.60`
- strict severe alert: `severe_prob >= 0.70`

Threshold sweep on corrected test:

### All Horizons

| Threshold | Precision | Recall |
| --- | ---: | ---: |
| `0.50` | `0.673` | `0.163` |
| `0.60` | `0.752` | `0.091` |
| `0.70` | `0.832` | `0.036` |

### `6h` Horizon Only

| Threshold | Precision | Recall |
| --- | ---: | ---: |
| `0.50` | `0.622` | `0.147` |
| `0.60` | `0.721` | `0.089` |
| `0.70` | `0.839` | `0.038` |

The repository's final demo and personal-use workflow are built around the `0.60` setting because it gives a better precision-first tradeoff than the default `0.50` threshold.

## Strong Baseline

The main baseline is:

- `best_model_before_severe_patch.pt`

Corrected departure-delay test performance:

| Metric | Value |
| --- | ---: |
| Overall MAE | `16.656` min |
| Overall RMSE | `49.040` min |
| `0h` MAE | `15.46` |
| `1h` MAE | `15.64` |
| `3h` MAE | `17.00` |
| `6h` MAE | `17.76` |

This simpler model remains strong, but the ordinal checkpoint edges it out overall, provides better tail interpretability, and aligns better with precision-first severe alerting.

## Demo

The main demo surface is:

- [11_val_day_replay_dashboard.py](11_val_day_replay_dashboard.py)

It replays one day from the validation set and shows:

- a timeline-driven U.S. map visualization
- predicted disruption severity by flight
- a full flight table with model outputs and flight metadata
- correctness views that distinguish predicted severity from high-confidence severe alerts

This dashboard exists to make the model interpretable, demoable, and portfolio-ready. It is the recommended way to showcase the project on a personal website or in a short video walkthrough.

The primary demo configuration uses:

- checkpoint: `best_model_dep_12k_ordinal_ep25.pt`
- date replay from validation data
- severe threshold: `0.60`

## Repository Layout

### Core Pipeline

- [00_validate_dataset.py](00_validate_dataset.py): dataset validation and sanity checks
- [02_build_rotation_edges.py](02_build_rotation_edges.py): aircraft rotation edges
- [03_build_weather_edge.py](03_build_weather_edge.py): weather feature and edge construction
- [04_build_congestion_edges.py](04_build_congestion_edges.py): congestion edge construction
- [05_build_graph_snapshots.py](05_build_graph_snapshots.py): graph snapshots and labels
- [06_train_gnn.py](06_train_gnn.py): graph-temporal GNN training
- [07_dashboard.py](07_dashboard.py): offline evaluation, CSV export, and comparison metrics

### Demo and Inference

- [08_realtime_connector.py](08_realtime_connector.py): live data connector utilities
- [09_day_simulation.py](09_day_simulation.py): earlier day-based simulation flow
- [10_flight_finder.py](10_flight_finder.py): lightweight web app for on-demand lookup
- [11_val_day_replay_dashboard.py](11_val_day_replay_dashboard.py): current replay dashboard for portfolio demo

### Outputs and Artifacts

- [`graph_data/`](graph_data): snapshots, static graph artifacts, and lookups
- [`checkpoints/`](checkpoints): trained checkpoints
- [`evaluation/`](evaluation): evaluation CSVs, threshold analysis, and comparison tables
- [`outputs/`](outputs): simulation and demo outputs
- [`data/`](data): retrieval, cleaning, and audit utilities

## Data

The project uses a large U.S. flight dataset spanning 2018-2022, enriched with weather features and graph-derived operational signals.

Main flight parquet:

- `flights_2018_2022.parquet`

Important step-5 graph artifacts:

- [`graph_data/snapshots_train.pt`](graph_data/snapshots_train.pt)
- [`graph_data/snapshots_val.pt`](graph_data/snapshots_val.pt)
- [`graph_data/snapshots_test.pt`](graph_data/snapshots_test.pt)
- [`graph_data/static_edges.pt`](graph_data/static_edges.pt)
- [`graph_data/flight_lookup.parquet`](graph_data/flight_lookup.parquet)
- [`graph_data/tail2idx.json`](graph_data/tail2idx.json)

## Model Architecture

The main model in [06_train_gnn.py](06_train_gnn.py) is a heterogeneous graph neural network with recurrent state:

- airport nodes and flight nodes
- airport-to-airport network edges
- congestion edges
- airport-to-flight assignment edges
- flight-to-flight rotation edges
- HGT-style message passing
- airport GRU for temporal airport state
- per-tail GRUCell for aircraft propagation state
- multi-horizon supervision
- ordinal disruption-tier head plus minute regression head

Current default sizes:

- hidden dim: `512`
- attention heads: `8`
- GNN layers: `2`
- airport GRU: `2 x 512`
- tail hidden dim: `128`

## Prediction Targets

The model predicts departure delay over four pre-departure horizons:

- `0h`: `<1h before departure`
- `1h`: `1-3h before departure`
- `3h`: `3-6h before departure`
- `6h`: `>6h before departure`

It also produces ordinal disruption tiers:

- `<0`
- `0-15`
- `15-60`
- `60-120`
- `120-240`
- `240-720`
- `720+`

In practice, `720+` is treated as a terminal risk tier rather than a precise minute-estimation regime.

## Feature Design

Examples of signals used:

- route-history priors
- airport dynamic delay and taxi aggregates
- scheduled traffic load
- forecast weather
- inbound aircraft state
- actual previous-leg arrival delay, but only after the previous leg has landed
- aircraft rotation structure

Signals intentionally removed or zeroed to reduce direct leakage for departure prediction:

- current flight realized `DepDelay`
- current flight realized `TaxiOut`
- current flight `CarrierDelay`

## Leakage and Evaluation Integrity

This repo went through multiple debugging and evaluation corrections. That work is one of the main reasons the project is strong.

### Fixed

- removed direct current-flight departure-delay leakage
- removed direct current-flight taxi-out leakage
- fixed a rotation self-loop bug by building true `leg1 -> leg2` flight rotation edges
- corrected stale test snapshots after the departure-delay migration
- re-ran holdout evaluation on the corrected departure-delay test artifacts

### Important Caveat

The project still uses same-hour airport dynamic aggregates such as average departure delay and taxi behavior. These are plausible in a near-real-time operational setting, but they are still somewhat optimistic for strict publication-style benchmarking because they may include information from later within the same hour.

So this repository is best described as:

- much cleaner and more causal-aware than a naive delay pipeline
- honest about limitations
- not yet a fully publication-grade causal benchmark

## Major Experiments

### 1. Arrival to Departure Migration

The original version predicted arrival delay. The project was later migrated to departure delay so that:

- benchmarking was cleaner
- pre-departure inference was more realistic
- direct leakage risks were easier to reason about

This required updating route priors, airport labels, flight labels, and evaluation scripts.

### 2. Coarse Severe-Tail Modeling

The project explored a binary `>=120` severe-delay formulation:

- useful for understanding recall and calibration tradeoffs
- not strong enough as a final formulation
- too coarse to distinguish `120` minutes from multi-hour or next-day disruptions

### 3. Ordinal Tail Reformulation

The strongest final result came from reformulating the tail as ordered thresholds:

- `>=0`
- `>=15`
- `>=60`
- `>=120`
- `>=240`
- `>=720`

This improved interpretability, stabilized evaluation, and produced the current best checkpoint.

### 4. Signed-`log1p` Regression Target

This experiment compressed the regression target to help the far tail. It improved numerical stability, but it became too conservative and did not outperform the standard ordinal setup.

### 5. Tail Uplift Head

A later experiment added explicit uplift channels for `>=240` and `>=720` cases. It improved tail movement mechanically, but it did not beat the precision-first ordinal model as a final choice.

## Running the Pipeline

### 1. Validate Data

```bash
python 00_validate_dataset.py
```

### 2. Build Graph Components

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

## Running the Demo

Install the UI stack if needed:

```bash
pip install dash dash-bootstrap-components plotly
```

Then launch the replay dashboard:

```bash
python 11_val_day_replay_dashboard.py --date 2021-11-28 --threshold 0.60
```

If you want a more visually active demo mode:

```bash
python 11_val_day_replay_dashboard.py --date 2021-11-28 --threshold 0.50
```

The `0.50` setting shows more alerts. The `0.60` setting is the recommended precision-first configuration.

## Running on Rice / Slurm

This project was trained and evaluated on Rice NOTS using Slurm.

Typical workflow:

1. build or copy graph artifacts
2. upload updated scripts
3. train with `sbatch`
4. evaluate with `sbatch`
5. pull checkpoints and evaluation CSVs back locally

The evaluation scripts support overriding the project root with:

```bash
python 07_dashboard.py --checkpoint best_model.pt --split test --drive_base /scratch/jb310/airline_project
```

## Dependencies

Core stack:

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

Example install:

```bash
pip install torch torch-geometric numpy pandas pyarrow requests python-dotenv dash plotly dash-bootstrap-components
```

## What This Project Demonstrates

This repository is meant to showcase:

- graph ML engineering
- temporal sequence modeling on large operational data
- debugging under real-world data messiness
- leakage-aware evaluation
- model selection under competing business goals
- productization of ML outputs into an interactive demo

## Limitations

- same-hour airport aggregates are still somewhat optimistic for strict benchmarking
- the 2022 test set is sparse and partly out-of-distribution
- minute prediction remains hard in the extreme tail, especially for `720+`
- internal airline recovery factors such as crew, maintenance, and operational interventions are not explicitly modeled
- the strongest user-facing signal is the severity tier and alert probability, not exact-minute accuracy in the far tail

## Current Recommendation

For the current repository state:

- use `best_model_dep_12k_ordinal_ep25.pt` as the primary checkpoint
- use `severe_prob >= 0.60` as the default severe alert rule
- use `severe_prob >= 0.70` for a stricter high-confidence mode
- treat the replay dashboard as the main demo surface

For portfolio and recruiting use, this is the version of the project to present.

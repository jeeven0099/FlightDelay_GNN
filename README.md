# Flight Delay GNN — Two-Level Heterogeneous Graph Neural Network

**Predicts individual US domestic flight arrival delays up to 6 hours before departure** using a novel two-level heterogeneous graph that jointly models airport congestion networks and individual flight tail rotation chains — without requiring any real-time gate data at long horizons.

> **Live demo:** [jeevenbalasubramaniam.github.io](https://jeevenbalasubramaniam.github.io) — Flight predictor tab

---

## Results

### Honest two-number summary

| Mode | MAE | What it knows |
|------|-----|---------------|
| **6h ahead — no gate data** | **19.6 min** | Weather, network state, tail history, route patterns, cascade |
| **Near departure — with gate data** | **9.68 min** | All of above + actual dep delay and taxi time |

The 6h-ahead number is the **novel result**. No published model evaluates honest individual flight prediction at 6h ahead without gate data.

### Comparison with published baselines

| Method | Year | Horizon | MAE | Gate data? |
|--------|------|---------|-----|-----------|
| Dynamic RF/CatBoost | 2025 | 90 min | 8.46 min | ✅ Private airline data |
| UC Berkeley XGBoost | 2025 | ~0h | 12.79 min | ✅ Yes |
| Saudi Airlines CatBoost | 2024 | ~0h | 12.19 min | ✅ Yes |
| FDPP-ML (Springer) | 2023 | 2h | 16.70 min | ❌ No |
| **This work — with gate data** | **2026** | **1h** | **9.68 min** | **✅ Yes** |
| **This work — no gate data** | **2026** | **6h** | **19.6 min** | **❌ No** |

The honest comparison is against **FDPP-ML** — same public BTS data, no gate information. This model predicts at 6h (vs FDPP-ML's 2h) with comparable accuracy.

### Day simulation — 2022-11-04

Evaluated on 7,254 flights with GRU warmed up from the previous evening:

| Metric | All flights | Operational (≤120 min actual) |
|--------|-------------|-------------------------------|
| MAE 6h ahead | 25.43 min | **20.3 min** |
| MAE 3h ahead | 25.45 min | — |
| MAE 1h ahead | 25.28 min | — |
| RMSE | 57.1 min | — |
| Delay classification accuracy | 73.5% | — |

The gap between all flights and operational flights is entirely from cancellations coded as 900–1300 min in BTS data — no public signal predicts those at 6h ahead.

**On-time flight accuracy:** MAE ~13 min for the 5,461 on-time flights — competitive with at-gate baselines.

**Calibration:** Predicted delayed >15 min: **15.9%** vs actual **17.1%** — 1.2 pp off.

**GRU accumulation confirmed:**

| Time | Avg predicted delay |
|------|-------------------|
| 06:00 | −3.0 min |
| 12:00 | −0.7 min |
| 18:00 | +2.4 min |
| 22:00 | +10.9 min |
| 23:00 | +12.1 min |

**Airline ordering:** Frontier/Spirit predicted worst, Delta/SkyWest predicted best — matches known operational reality.

---

## What makes this different

### Eight novel contributions

**1. Honest horizon-aware feature masking**
Gate features (dep_delay, taxi_out, turnaround, carrier_delay, immed_inbound) are deterministically zeroed for flights more than 2 hours from departure — during both training and inference. No published flight delay paper applies this protocol.

**2. Two-level heterogeneous graph**
Airport nodes and individual flight nodes coexist in a unified graph. Most GNN papers model airports only. Each flight has its own node with tail history, route statistics, and causal inbound signal.

**3. Causal rotation edges — fired only on observed arrivals**
A self-edge on each downstream flight fires only when the upstream aircraft has actually landed — carrying the observed arrival delay as edge attribute. No edge fires when leg1 is still airborne, eliminating gradient noise from unobserved signals. Cascade delay at 6h ahead is carried through cumul_delay node features (always available).

**4. GRU temporal memory per airport**
Each of the 36 airports maintains a 256-dimensional hidden state across the day. Validated: −3.0 min at 6am → +12.1 min at 11pm, matching real-world cascade patterns.

**5. Dynamic congestion edge weights**
Weights computed per snapshot from actual taxi-out anomalies (z-score vs airport baseline), not static 5-year averages.

**6. Per-date actual weather features**
Airport dynamic features use actual hourly METAR observations keyed by (date, hour, airport) — not 5-year climatological averages.

**7. Historical route statistics — training years only**
hist_route_delay_avg and hist_route_delay_std per route × hour × day-of-week from 2018–2020 only. No leakage. Available at all horizons.

**8. Joint regression + classification**
Classification head predicts P(ArrDelay > 15 min) jointly with regression, improving calibration around the DOT delay threshold.

---

## Architecture

### Graph structure (one hourly snapshot)

```
Two node types:
  airport (36 nodes)         — 30 features, persistent GRU state
  flight  (~2,272/snapshot)  — 19 features, causal masking at >2h

Six edge types:
  (airport, rotation,     airport)  dynamic      — tail delay at turnaround hour
  (airport, congestion,   airport)  dynamic      — taxi z-score per snapshot
  (airport, network,      airport)  static       — 5yr correlation-weighted mesh
  (flight,  rotation,     flight)   causal       — fires only when leg1 has landed
  (flight,  departs_from, airport)  unconditional — every flight to origin
  (flight,  arrives_at,   airport)  causal       — immed_inbound > 15 min only
```

**Verified edge counts (avg per snapshot):**

| Edge type | Avg/snapshot |
|-----------|-------------|
| departs_from | 2,272 |
| arrives_at | 183 |
| flight rotation | 99 |
| congestion (dynamic) | 5,074 |
| network (static) | 1,260 |

### Airport node features (30 dims)

| Group | Dims | Description |
|-------|------|-------------|
| Static | 5 | is_hub, hist avg dep/arr delay, hist avg taxi, total departures |
| Current dynamic | 9 | avg dep/arr delay, taxi times, counts, wind, visibility, precip (per-date actual) |
| Traffic load | 6 | scheduled dep/arr next 1h, 3h, 6h |
| Forecast weather | 6 | avg wind/precip/visibility at +3h and +6h |
| Time embeddings | 4 | hour sin/cos, month sin/cos |

### Flight node features (19 dims)

| Group | Dims | Description |
|-------|------|-------------|
| Gate state | 5 | dep_delay, taxi_out, turnaround, carrier_delay, immed_inbound *(zeroed >2h out)* |
| Tail history | 3 | tail_cumul_delay, tail_legs_today, is_first_flight |
| Schedule | 6 | dep/arr hour sin/cos, day_of_week sin/cos |
| Context | 3 | distance, is_hub_origin, time_to_dep |
| Route stats | 2 | hist_route_avg, hist_route_std *(never masked)* |

### Model

```
Input projection  (30d airport / 19d flight → 256d hidden)
      ↓
HGTConv × 2  (4 attention heads, residual + LayerNorm)
      ↓
GRUCell per airport  (256d, persistent across snapshots)
      ↓
Gated MLP per flight  (128d)
      ↓
Regression head  → predicted ArrDelay (minutes)
Classifier head  → P(ArrDelay > 15 min)
```

**Parameters:** ~2,005,175 | **Hardware:** Rice University NOTS — Tesla V100-PCIE-32GB

### Loss

```
Total = airport×0.25 + flight×0.75 + classification×0.20
Flight = Huber(δ=20) + delay weighting (1.5× for ≥60 min actual)
       + horizon weights: 1h×0.20 · 3h×0.35 · 6h×0.45
```

---

## Training

| Setting | Value |
|---------|-------|
| Features | 30 airport / 19 flight |
| Hidden / GRU / MLP | 256 / 256 / 128 |
| HGT heads / layers | 4 / 2 |
| Parameters | ~2,005,175 |
| Batch size | 16 consecutive snapshots |
| LR schedule | 1e-3 → 5e-4 → 2.5e-4 (ReduceLROnPlateau, patience=4) |
| Best epoch | 23 |
| Train / Val / Test | 2018–2020 / 2021 / 2022 |
| Dataset | 9,179,915 flights · 36 hub airports · 43,177 hourly snapshots |
| Rotation pairs | 17,850,533 |

---

## Data sources

| Source | Used for |
|--------|---------|
| BTS On-Time Performance (2018–2022) | Training, validation, test |
| NWS METAR | Per-date hourly weather features |
| AviationStack API | Live flight schedules + tail numbers |
| NWS Weather API | Current + 6h forecast |
| FAA NASSTATUS | Ground delay programs |

Key stats: mean ArrDelay +3.0 min · std 48.3 min · 17.1% flights delayed >15 min

---

## Web apps

### Production flight finder (step 10)
```bash
python 10_flight_finder.py   # http://127.0.0.1:8051
```
3 API calls per request. Returns 6h/3h/1h predictions with weather context.

### Live real-time dashboard (step 8 + step 7)
```bash
python 08_realtime_connector.py --mode test   # test API connections first
python 08_realtime_connector.py               # fetch live data
python 07_dashboard.py --mode realtime        # launch dashboard
```
Requires `AVIATIONSTACK_KEY` in `.env`. Refreshes every 3 hours.

### Historical dashboard (step 7)
```bash
python 07_dashboard.py --mode dash --date 2022-11-04
```

---

## Setup

```bash
git clone https://github.com/yourusername/flight-delay-gnn
cd flight-delay-gnn
python -m venv gnn_env
source gnn_env/bin/activate        # Mac/Linux
# gnn_env\Scripts\activate         # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pandas numpy pyarrow tqdm dash plotly requests python-dotenv
```

Create `.env`:
```
AVIATIONSTACK_KEY=your_key_here
```

---

## Pipeline

```bash
python 02_build_rotation_edges.py   # 17.85M tail rotation pairs (~20 min)
python 03_build_weather_edges.py    # METAR weather enrichment
python 04_build_congestion_edges.py # dynamic congestion topology
python 05_build_graph_snapshots.py  # 43,177 snapshots (~45 min)
python 06_train_gnn.py              # train (~10h on V100)
python 10_flight_finder.py          # production app
python 09_day_simulation.py --date 2022-11-04 --save_plots
```

---

## File structure

```
flight-delay-gnn/
├── 02_build_rotation_edges.py     17.85M tail rotation pairs
├── 03_build_weather_edges.py      METAR weather enrichment
├── 04_build_congestion_edges.py   dynamic congestion topology
├── 05_build_graph_snapshots.py    43,177 hourly graph snapshots
├── 06_train_gnn.py                HGT + GRU training, per-horizon MAE logging
├── 07_dashboard.py                historical + live dashboard
├── 08_realtime_connector.py       live API connector (NWS, FAA, AviationStack)
├── 09_day_simulation.py           full-day evaluation with GRU warmup
├── 10_flight_finder.py            on-demand flight finder (3 API calls)
├── graph_data/
│   ├── snapshots_train.pt         7.0 GB  — not in git (see .gitignore)
│   ├── snapshots_val.pt           2.3 GB  — not in git
│   ├── snapshots_test.pt          0.2 GB  — not in git
│   ├── static_edges.pt
│   ├── flight_lookup.parquet
│   ├── route_stats.parquet        route × hour × dow (training only)
│   ├── rotation_edges.parquet     — not in git (large)
│   ├── congestion_edges.parquet
│   ├── network_edges.parquet
│   └── airport_index.parquet
└── checkpoints/
    ├── best_model.pt              epoch 23, val MAE 19.579 min
    └── training_history.csv       per-epoch 1h/3h/6h MAE breakdown
```

---

## Known limitations

**36 hub airports only.** Smaller regional airports not modelled.

**Cancellations not predicted.** No public signal distinguishes a cancellation from a normal flight 6 hours ahead. This is the primary driver of RMSE.

**~19 min floor at 6h ahead.** ~90% of delay variance is irreducible from public data — mechanical failures, crew positioning, ATC decisions, last-minute weather. Breaking below 15 min requires private operational data.

**RMSE ~57 min.** Driven by cancellations/diversions coded as extreme delays in BTS data.

---

## Citation

```bibtex
@misc{balasubramaniam2026flightgnn,
  title  = {Two-Level Heterogeneous GNN for Honest Multi-Horizon
             Individual Flight Delay Prediction},
  author = {Balasubramaniam, Jeeven},
  year   = {2026},
  url    = {https://github.com/jeeven0099/FlightDelay_GNN}
}
```

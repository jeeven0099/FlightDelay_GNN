"""
STEP 6 — TWO-LEVEL FLIGHT DELAY GNN (CLEAN 6-EDGE)
===================================================
Colab version — reads/writes Google Drive.

6 edge types — each non-redundant:
  airport→airport : rotation (dynamic), congestion/taxi (dynamic),
                    network (static, full connectivity)
  flight→flight   : rotation (tail chain)
  flight→airport  : departs_from (outgoing manifest)
                    arrives_at   (incoming manifest)

NO affects edge — 2-round message passing makes it redundant:
  Round 1: flights → airports via departs_from + arrives_at
           airports → airports via rotation + congestion + network
  Round 2: airports updated state flows back to influence flights
           in next snapshot via departs_from/arrives_at aggregation

Airport features: 28 dims (static + current + traffic + forecast + time)
Flight features:  15 dims (including time_to_dep for multi-horizon)
Labels:           airport 1h ahead + flight 1/3/6/12h ahead

Training: sequential mini-batching (BATCH_SIZE consecutive snapshots)
  — preserves GRU temporal order
  — improves GPU utilization ~10x over single-snapshot processing
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

# ── CONFIG ──────────────────────────────────────────────────────────────────
DRIVE_BASE     = "/content/drive/MyDrive/Airline_Graphs_Project"
GRAPH_DATA_DIR = f"{DRIVE_BASE}/graph_data"
CHECKPOINT_DIR = f"{DRIVE_BASE}/checkpoints"

# Model
NODE_FEAT_AP        = 30    # +2: dep_1h, arr_1h traffic features
NODE_FEAT_FL        = 19    # +2: hist_route_delay_avg, hist_route_delay_std
HIDDEN_DIM          = 256   # was 128 — larger model
NUM_HEADS           = 4
NUM_GNN_LAYERS      = 2
GRU_HIDDEN_DIM      = 256   # was 128
MLP_HIDDEN_DIM      = 128   # was 64
DROPOUT             = 0.1

# Loss
AIRPORT_LOSS_WEIGHT = 0.25
FLIGHT_LOSS_WEIGHT  = 0.75
HORIZON_WEIGHTS     = {1: 0.20, 3: 0.35, 6: 0.45}  # 6h is max, weighted highest

# Training
LEARNING_RATE          = 1e-3
WEIGHT_DECAY           = 1e-4
NUM_EPOCHS             = 75
CLIP_GRAD_NORM         = 1.0
EARLY_STOP_PATIENCE    = 10
LR_SCHEDULER_PATIENCE  = 4
LR_SCHEDULER_FACTOR    = 0.5
RESUME_FROM_CHECKPOINT = False   # retrain with horizon masking
BATCH_SIZE             = 16   # consecutive snapshots per optimizer step

LABEL_HORIZONS_FL = [1, 3, 6]

# Constants shared with step 5 — must match exactly
INBOUND_DELAY_THRESH = 15.0   # minutes — soft mask centre
MAX_DELAY_MIN        = 300.0  # normalisation denominator

# 6 edge types — matches step 5 exactly
NODE_TYPES = ["airport", "flight"]
EDGE_TYPES = [
    ("airport", "rotation",    "airport"),   # dynamic tail delay
    ("airport", "congestion",  "airport"),   # dynamic taxi anomaly
    ("airport", "network",     "airport"),   # static full connectivity
    ("flight",  "rotation",    "flight"),    # tail chain
    ("flight",  "departs_from","airport"),   # outgoing manifest
    ("flight",  "arrives_at",  "airport"),   # incoming manifest
]
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════

class FlightDelayGNN(nn.Module):
    """
    Two-level HGT with 6 non-redundant edge types.

    Current improvements over baseline:
      - Unconditional departs_from edges (all flights connect to origin airport)
      - Horizon-aware feature masking (gate features zeroed for far-out flights)
      - Route stats features (hist_avg, hist_std — available at all horizons)
      - Larger model: 256 hidden, 256 GRU, 128 MLP
      - Classification head: P(ArrDelay > 15min) jointly trained
      - Huber loss + delay sample weighting (3× penalty for severe delays)
      - 1h scheduled traffic window as airport feature
    """

    def __init__(self, ap_in_dim, fl_in_dim, hidden_dim,
                 num_heads, num_layers, gru_hidden, mlp_hidden,
                 num_airports, dropout=0.1):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.gru_hidden   = gru_hidden
        self.num_airports = num_airports
        self.num_layers   = num_layers

        # Input projections
        self.ap_proj = nn.Sequential(
            Linear(ap_in_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.fl_proj = nn.Sequential(
            Linear(fl_in_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        # HGT layers — 6 edge types
        metadata = (NODE_TYPES, EDGE_TYPES)
        self.convs    = nn.ModuleList([
            HGTConv(in_channels=hidden_dim, out_channels=hidden_dim,
                    metadata=metadata, heads=num_heads)
            for _ in range(num_layers)])
        self.ap_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.fl_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drops    = nn.ModuleList(
            [nn.Dropout(dropout)     for _ in range(num_layers)])

        # Airport GRU — persistent temporal state across snapshots
        self.ap_gru = nn.GRUCell(hidden_dim, gru_hidden)

        # Flight gate
        self.fl_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        # Regression head
        self.ap_head = nn.Sequential(
            nn.Linear(gru_hidden, mlp_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))
        self.fl_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))

        # Classification head — P(ArrDelay > 15min)
        self.fl_classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden // 2, 1))

    def forward(self, data: HeteroData, ap_h: torch.Tensor):
        n_fl = data["flight"].num_nodes
        dev  = ap_h.device

        x_dict = {"airport": self.ap_proj(data["airport"].x.float())}
        x_dict["flight"] = (self.fl_proj(data["flight"].x.float())
                             if n_fl > 0
                             else torch.zeros(0, self.hidden_dim, device=dev))

        # Build edge_index dict for HGT — edge_index only, no edge_attr.
        # HGTConv uses node representations for attention, not edge features.
        # Edge attrs (congestion weights, delay norms) are used in the loss
        # and masking logic, not in the message passing itself.
        # Stripping them here prevents HGT's internal bipartite builder from
        # encountering inconsistent attr shapes across edge types.
        eid = {}
        for et in EDGE_TYPES:
            if (hasattr(data[et], "edge_index") and
                    data[et].edge_index.shape[1] > 0):
                eid[et] = data[et].edge_index

        for i, conv in enumerate(self.convs):
            x_new = conv(x_dict, eid)
            if "airport" in x_new:
                x_dict["airport"] = self.ap_norms[i](
                    self.drops[i](x_new["airport"]) + x_dict["airport"])
            if "flight" in x_new and n_fl > 0:
                x_dict["flight"] = self.fl_norms[i](
                    self.drops[i](x_new["flight"]) + x_dict["flight"])

        ap_h_new = self.ap_gru(x_dict["airport"], ap_h)
        ap_pred  = self.ap_head(ap_h_new).squeeze(-1)

        if n_fl > 0:
            fl_out    = x_dict["flight"]
            fl_gated  = fl_out * self.fl_gate(fl_out)
            fl_pred   = self.fl_head(fl_gated).squeeze(-1)
            fl_logits = self.fl_classifier(fl_gated).squeeze(-1)
        else:
            fl_pred   = torch.zeros(0, device=dev)
            fl_logits = torch.zeros(0, device=dev)

        return ap_pred, fl_pred, fl_logits, ap_h_new

    def init_hidden(self, device):
        return torch.zeros(self.num_airports, self.gru_hidden, device=device)


# ════════════════════════════════════════════════════════════════════════════
# LOSS AND METRICS
# ════════════════════════════════════════════════════════════════════════════

def masked_mae(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.l1_loss(pred[mask], target[mask])


def masked_rmse(pred, target, mask):
    if mask.sum() == 0:
        return 0.0
    return float(((pred[mask] - target[mask])**2).mean().sqrt())


DELAY_THRESHOLD   = 15.0    # DOT definition of "delayed"
CLASS_LOSS_WEIGHT = 0.20    # classification loss weight
HUBER_DELTA       = 20.0    # Huber transition point in minutes
DELAY_WEIGHT_MAX  = 1.5     # max sample weight for severe delays
                             # Kept modest — 1.5× not 3× — so early training
                             # converges on mean before focusing on tail risk.


def masked_huber_weighted(pred, target, mask, delta=HUBER_DELTA,
                           w_max=DELAY_WEIGHT_MAX):
    """
    Huber loss with sample weighting by actual delay magnitude.

    Weight schedule:
      actual < 0    → 1.0  (early arrivals, no special treatment)
      actual 0-15   → 1.0  (on-time)
      actual 15-30  → 1.5  (minor delay — worth catching)
      actual 30-60  → 2.0  (moderate — operationally significant)
      actual > 60   → w_max (severe — most important to get right)

    Huber vs MAE: quadratic below delta=20min (sensitive to small errors),
    linear above (robust to outlier cancellations/diversions coded as 1000min).
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    p = pred[mask]
    t = target[mask]

    # Sample weights by actual delay
    w = torch.ones_like(t)
    w = torch.where(t >= 15,  torch.full_like(t, 1.5), w)
    w = torch.where(t >= 30,  torch.full_like(t, 2.0), w)
    w = torch.where(t >= 60,  torch.full_like(t, w_max), w)

    # Huber loss per sample
    err     = torch.abs(p - t)
    huber   = torch.where(err <= delta,
                          0.5 * err ** 2 / delta,
                          err - 0.5 * delta)

    return (w * huber).mean()


def compute_loss(ap_pred, ap_y, ap_mask, fl_pred, fl_logits, fl_y, snap):
    """
    Combined loss:
      - Airport: Huber (smooth, handles outliers better than MAE)
      - Flight regression: Huber + delay weighting per horizon
      - Flight classification: BCE for P(delay > 15min)

    Loss breakdown:
      airport  × 0.25 — airport-level delay prediction
      flight   × 0.75 — individual flight delay prediction
      classify × 0.20 — binary delay/not-delayed classification
    """
    # Airport loss — Huber without sample weighting
    if ap_mask.sum() > 0:
        ap_err   = torch.abs(ap_pred[ap_mask] - ap_y[ap_mask])
        ap_loss  = torch.where(ap_err <= HUBER_DELTA,
                               0.5 * ap_err**2 / HUBER_DELTA,
                               ap_err - 0.5 * HUBER_DELTA).mean()
    else:
        ap_loss = torch.tensor(0.0, device=ap_pred.device, requires_grad=True)

    fl_losses  = []
    cls_losses = []

    for h in LABEL_HORIZONS_FL:
        attr = f"y_mask_{h}h"
        if hasattr(snap["flight"], attr):
            mask = getattr(snap["flight"], attr)
            if mask.sum() > 0:
                # Weighted Huber regression loss
                fl_losses.append(
                    HORIZON_WEIGHTS[h] *
                    masked_huber_weighted(fl_pred, fl_y, mask))

                # Classification loss
                if fl_logits.shape[0] > 0:
                    labels = (fl_y[mask] > DELAY_THRESHOLD).float()
                    cls_losses.append(
                        HORIZON_WEIGHTS[h] *
                        F.binary_cross_entropy_with_logits(
                            fl_logits[mask], labels))

    fl_loss  = (sum(fl_losses)  if fl_losses
                else torch.tensor(0.0, device=ap_pred.device,
                                   requires_grad=True))
    cls_loss = (sum(cls_losses) if cls_losses
                else torch.tensor(0.0, device=ap_pred.device,
                                   requires_grad=True))

    total = (AIRPORT_LOSS_WEIGHT * ap_loss +
             FLIGHT_LOSS_WEIGHT  * fl_loss +
             CLASS_LOSS_WEIGHT   * cls_loss)

    return total, ap_loss, fl_loss


# ════════════════════════════════════════════════════════════════════════════
# TRAINING — sequential mini-batching
# ════════════════════════════════════════════════════════════════════════════


# ── FEATURE MASKING CONFIG ──────────────────────────────────────────────────
# Real-time gate features that are UNKNOWN before departure.
# Deterministically zeroed based on time_to_dep — no randomness.
#
# Feature indices (must match step 5 build_flight_features_fast):
#   0: dep_delay      (actual DepDelay)       — unknown until pushback
#   3: turnaround     (actual turnaround)      — unknown until plane arrives
#   4: immed_inbound  (actual inbound delay)   — unknown until plane lands
#   7: taxi_out       (actual TaxiOut)         — unknown until wheels up
#  11: carrier_delay  (actual carrier delay)   — unknown until pushback
#  17: hist_route_avg (precomputed from train) — NEVER masked, always available
#  18: hist_route_std (precomputed from train) — NEVER masked, always available
GATE_FEATURE_INDICES = [0, 3, 4, 7, 11]   # indices 17,18 excluded — historical

# Masking thresholds (time_to_dep is normalized 0-1, where 1.0 = 24h)
MASK_FULL_THRESHOLD    = 2.0 / 24   # > 2h  → zero ALL gate features
MASK_PARTIAL_THRESHOLD = 1.0 / 24   # 1-2h  → zero dep_delay + taxi_out only
#                                   # < 1h  → no masking (gate data available)

# Partially masked features (1-2h window — plane approaching gate)
PARTIAL_GATE_FEATURES = [0, 7]      # dep_delay + taxi_out only


def apply_horizon_masking(x_fl, training=True):
    """
    Deterministically zero out gate features based on time_to_dep.

    This mirrors what a real production system knows at each horizon:
      < 1h  : plane is at gate, all features available
      1-2h  : plane arriving, dep_delay + taxi still unknown
      > 2h  : no gate data at all — use network/weather/schedule only

    Applied identically during training AND inference, so the model
    learns exactly what it will see in production.

    x_fl    : float16 tensor [n_flights, 17]
    training: included for API compatibility (masking always applied)
    """
    if x_fl.shape[0] == 0:
        return x_fl

    x             = x_fl.float().clone()
    time_to_dep   = x[:, 14]            # normalized 0-1

    # > 2h before departure — zero all gate features
    full_mask = time_to_dep >= MASK_FULL_THRESHOLD
    if full_mask.any():
        idx = full_mask.nonzero(as_tuple=True)[0]
        x[idx[:, None],
          torch.tensor(GATE_FEATURE_INDICES, device=x.device)] = 0.0

    # 1-2h before departure — zero dep_delay + taxi only
    partial_mask = (time_to_dep >= MASK_PARTIAL_THRESHOLD) & ~full_mask
    if partial_mask.any():
        idx = partial_mask.nonzero(as_tuple=True)[0]
        x[idx[:, None],
          torch.tensor(PARTIAL_GATE_FEATURES, device=x.device)] = 0.0

    return x.to(x_fl.dtype)



def run_epoch(model, snapshots, optimizer, device,
              static_edges=None, is_train=True):
    """
    Sequential mini-batch processing with horizon-aware feature masking.

    During training only: gate features (dep_delay, taxi_out, inbound_delay,
    turnaround, carrier_delay) are stochastically zeroed out for far-out
    flights. This forces the model to rely on network/weather at long horizons.
    """
    model.train(is_train)
    tot = {"loss":0.0,"ap":0.0,"fl":0.0,"ap_r":0.0,"fl_r":0.0,
           "ap_mae":0.0,"fl_mae":0.0,
           "fl_1h":0.0,"fl_3h":0.0,"fl_6h":0.0,
           "n_1h":0,   "n_3h":0,   "n_6h":0}
    ap_h = model.init_hidden(device)
    n    = 0

    if static_edges is not None:
        nw_ei = static_edges["network_ei"].to(device)
        nw_ea = static_edges["network_ea"].to(device)

    for batch_start in range(0, len(snapshots), BATCH_SIZE):
        batch = snapshots[batch_start: batch_start + BATCH_SIZE]

        if is_train:
            optimizer.zero_grad()

        accumulated_loss = None

        for snap in batch:
            snap  = snap.to(device)

            if static_edges is not None:
                snap["airport","network","airport"].edge_index = nw_ei
                snap["airport","network","airport"].edge_attr  = nw_ea

            # ── HORIZON-AWARE FEATURE MASKING ────────────────────────────────
            if snap["flight"].num_nodes > 0:
                snap["flight"].x = apply_horizon_masking(snap["flight"].x)

            ap_y  = snap["airport"].y
            ap_m  = snap["airport"].y_mask
            fl_y  = snap["flight"].y
            fl_m1 = getattr(snap["flight"], "y_mask_1h",
                            torch.zeros(snap["flight"].num_nodes,
                                        dtype=torch.bool, device=device))

            with torch.set_grad_enabled(is_train):
                ap_pred, fl_pred, fl_logits, ap_h = model(snap, ap_h)
                ap_h = ap_h.detach()

                loss, ap_l, fl_l = compute_loss(
                    ap_pred, ap_y, ap_m, fl_pred, fl_logits, fl_y, snap)

                if is_train:
                    scaled = loss / len(batch)
                    accumulated_loss = (scaled if accumulated_loss is None
                                        else accumulated_loss + scaled)

            tot["loss"]   += loss.item()
            tot["ap"]     += ap_l.item()
            tot["fl"]     += fl_l.item()
            tot["ap_r"]   += masked_rmse(ap_pred.detach(), ap_y, ap_m)
            tot["fl_r"]   += masked_rmse(fl_pred.detach(), fl_y, fl_m1)
            tot["ap_mae"] += masked_mae(ap_pred.detach(), ap_y, ap_m).item()
            tot["fl_mae"] += masked_mae(fl_pred.detach(), fl_y, fl_m1).item()

            # Per-horizon MAE reporting using existing masks
            # y_mask_1h: flights ≥1h from dep  → "at least 1h ahead" cohort
            # y_mask_3h: flights ≥3h from dep  → "at least 3h ahead" cohort
            # y_mask_6h: flights ≥6h from dep  → "at least 6h ahead" (hardest)
            # These are the natural evaluation cohorts — each mask represents
            # the set of flights where that horizon's prediction applies.
            with torch.no_grad():
                m1 = getattr(snap["flight"], "y_mask_1h",
                             torch.zeros(snap["flight"].num_nodes,
                                         dtype=torch.bool, device=device))
                m3 = getattr(snap["flight"], "y_mask_3h",
                             torch.zeros(snap["flight"].num_nodes,
                                         dtype=torch.bool, device=device))
                m6 = getattr(snap["flight"], "y_mask_6h",
                             torch.zeros(snap["flight"].num_nodes,
                                         dtype=torch.bool, device=device))
                fl_y_d = snap["flight"].y
                if m1.sum() > 0:
                    tot["fl_1h"] += masked_mae(
                        fl_pred.detach(), fl_y_d, m1).item()
                    tot["n_1h"]  += 1
                if m3.sum() > 0:
                    tot["fl_3h"] += masked_mae(
                        fl_pred.detach(), fl_y_d, m3).item()
                    tot["n_3h"]  += 1
                if m6.sum() > 0:
                    tot["fl_6h"] += masked_mae(
                        fl_pred.detach(), fl_y_d, m6).item()
                    tot["n_6h"]  += 1
            n += 1

        if is_train and accumulated_loss is not None:
            accumulated_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

    d = max(n, 1)
    mae_1h = tot["fl_1h"] / max(tot["n_1h"], 1)
    mae_3h = tot["fl_3h"] / max(tot["n_3h"], 1)
    mae_6h = tot["fl_6h"] / max(tot["n_6h"], 1)
    return (tot["loss"]/d, tot["ap_mae"]/d, tot["fl_mae"]/d,
            tot["ap_r"]/d, tot["fl_r"]/d,
            mae_1h, mae_3h, mae_6h)


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ════════════════════════════════════════════════════════════════════════════

def save_ckpt(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch"        : epoch,
        "model_state"  : model.state_dict(),
        "optim_state"  : optimizer.state_dict(),
        "metrics"      : metrics,
        "num_airports" : model.num_airports,
        "hidden_dim"   : model.hidden_dim,
        "gru_hidden"   : model.gru_hidden,
        "node_feat_ap" : NODE_FEAT_AP,
        "node_feat_fl" : NODE_FEAT_FL,
        "num_heads"    : NUM_HEADS,
        "num_layers"   : NUM_GNN_LAYERS,
        "mlp_hidden"   : MLP_HIDDEN_DIM,
    }, path)


def load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    start = ckpt["epoch"] + 1
    best  = ckpt["metrics"].get("val_ap_mae", float("inf"))
    print(f"  Resumed from epoch {ckpt['epoch']} | "
          f"best val airport MAE = {best:.3f}")
    return start, best


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("STEP 6 — FLIGHT DELAY GNN (CLEAN 6-EDGE + MINI-BATCH)")
    print("=" * 65)
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Batch size : {BATCH_SIZE} consecutive snapshots/step")

    print(f"\nLoading snapshots ...")
    def load(name):
        path  = os.path.join(GRAPH_DATA_DIR, f"snapshots_{name}.pt")
        snaps = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  {name:>5}: {len(snaps):,} snapshots")
        return snaps

    train_snaps = load("train")
    val_snaps   = load("val")

    # Load static edges (saved once, shared across all snapshots)
    print(f"\nLoading static edges ...")
    static = torch.load(
        os.path.join(GRAPH_DATA_DIR, "static_edges.pt"),
        map_location="cpu", weights_only=False)

    # Handle both old format (has congestion_ea) and new format (dynamic weights)
    cg_ei = static["congestion_ei"]
    nw_ei = static["network_ei"]
    nw_ea = static["network_ea"]

    # Old static_edges.pt has congestion_ea — new one doesn't (weights are in snapshots)
    # If running with old static_edges.pt, congestion edges are still in snapshots
    # so we just need the topology (ei) which both formats have
    if "congestion_ea" in static:
        print(f"  ⚠  Old static_edges format detected — congestion weights from snapshot")

    print(f"  Congestion edges : {cg_ei.shape[1]} (dynamic weights per snapshot)")
    print(f"  Network edges    : {nw_ei.shape[1]}")

    num_airports = train_snaps[0]["airport"].num_nodes
    ap_in_dim    = train_snaps[0]["airport"].x.shape[1]
    fl_in_dim    = (train_snaps[0]["flight"].x.shape[1]
                    if train_snaps[0]["flight"].num_nodes > 0
                    else NODE_FEAT_FL)

    print(f"\n  Airports      : {num_airports}")
    print(f"  Airport feats : {ap_in_dim}  (expected {NODE_FEAT_AP})")
    print(f"  Flight feats  : {fl_in_dim}  (expected {NODE_FEAT_FL})")

    fl_counts = [s["flight"].num_nodes for s in train_snaps[:200]]
    print(f"  Avg flights   : {np.mean(fl_counts):.0f}/snapshot (24h window)")

    print(f"\n  Edge types in first non-empty snapshot:")
    for snap in train_snaps:
        if snap["flight"].num_nodes > 0:
            for et in EDGE_TYPES:
                try:
                    n_e = snap[et].edge_index.shape[1]
                    print(f"    {str(et):<50} {n_e:>6} edges (in snapshot)")
                except AttributeError:
                    print(f"    {str(et):<50}  static (loaded separately)")
            print(f"\n  Multi-horizon label coverage (sample):")
            for h in LABEL_HORIZONS_FL:
                attr = f"y_mask_{h}h"
                if hasattr(snap["flight"], attr):
                    nv  = getattr(snap["flight"], attr).sum().item()
                    pct = 100 * nv / max(snap["flight"].num_nodes, 1)
                    print(f"    {h:>2}h ahead: {nv:>5} flights ({pct:.0f}%)")
            break

    print(f"\nBuilding model ...")
    model = FlightDelayGNN(
        ap_in_dim=ap_in_dim, fl_in_dim=fl_in_dim,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_GNN_LAYERS, gru_hidden=GRU_HIDDEN_DIM,
        mlp_hidden=MLP_HIDDEN_DIM, num_airports=num_airports,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_batches = (len(train_snaps) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Parameters     : {n_params:,}")
    print(f"  Hidden dim     : {HIDDEN_DIM} | GRU: {GRU_HIDDEN_DIM}")
    print(f"  Edge types     : {len(EDGE_TYPES)} (all non-redundant)")
    print(f"  Loss weights   : airport={AIRPORT_LOSS_WEIGHT} "
          f"flight={FLIGHT_LOSS_WEIGHT}")
    print(f"  Horizon weights: {HORIZON_WEIGHTS}")
    print(f"  Train batches  : {n_batches:,}/epoch")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR)

    start_epoch  = 1
    best_val_mae = float("inf")
    patience_ctr = 0
    history      = []
    ckpt_path    = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if RESUME_FROM_CHECKPOINT and os.path.exists(ckpt_path):
        print(f"\nResuming from checkpoint ...")
        try:
            start_epoch, best_val_mae = load_ckpt(
                ckpt_path, model, optimizer, device)
        except Exception as e:
            print(f"  ⚠ Incompatible checkpoint ({e}) — starting fresh")
            start_epoch  = 1
            best_val_mae = float("inf")
    else:
        print(f"\nStarting fresh training ...")

    print(f"\n{'─'*100}")
    print(f"{'Epoch':>6}  {'Loss':>8}  "
          f"{'Fl MAE':>8}  "
          f"{'vFl MAE':>8}  {'v1h MAE':>8}  {'v3h MAE':>8}  {'v6h MAE':>8}  "
          f"{'vAp MAE':>8}  {'LR':>8}")
    print(f"  (all MAE = true L1 in minutes — loss uses Huber internally)")
    print(f"{'─'*100}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        tr_loss, tr_ap, tr_fl, tr_ap_r, tr_fl_r, \
            tr_1h, tr_3h, tr_6h = run_epoch(
            model, train_snaps, optimizer, device,
            static_edges=static, is_train=True)
        vl_loss, vl_ap, vl_fl, vl_ap_r, vl_fl_r, \
            vl_1h, vl_3h, vl_6h = run_epoch(
            model, val_snaps, optimizer, device,
            static_edges=static, is_train=False)

        lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch"      : epoch,
            "tr_fl_mae"  : tr_fl,
            "vl_ap_mae"  : vl_ap,   "vl_fl_mae"  : vl_fl,
            "vl_1h_mae"  : vl_1h,   "vl_3h_mae"  : vl_3h,   "vl_6h_mae": vl_6h,
            "vl_ap_rmse" : vl_ap_r, "vl_fl_rmse" : vl_fl_r,
            "lr"         : lr,
        })

        print(f"{epoch:>6}  {tr_loss:>8.3f}  "
              f"{tr_fl:>8.3f}  "
              f"{vl_fl:>8.3f}  {vl_1h:>8.3f}  {vl_3h:>8.3f}  {vl_6h:>8.3f}  "
              f"{vl_ap:>8.3f}  {lr:>8.2e}")

        # Use flight MAE as primary metric
        scheduler.step(vl_fl)

        if vl_fl < best_val_mae:
            best_val_mae = vl_fl
            patience_ctr = 0
            save_ckpt(model, optimizer, epoch,
                      {"val_ap_mae":vl_ap, "val_fl_mae":vl_fl,
                       "val_ap_rmse":vl_ap_r,"val_fl_rmse":vl_fl_r},
                      ckpt_path)
            print(f"  ✅  Flight MAE={vl_fl:.3f} | Airport MAE={vl_ap:.3f} → saved")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping after {epoch} epochs "
                      f"(no flight MAE improvement for "
                      f"{EARLY_STOP_PATIENCE} epochs)")
                break

        # Save history every epoch — survives Colab disconnects
        pd.DataFrame(history).to_csv(
            os.path.join(CHECKPOINT_DIR, "training_history.csv"),
            index=False)

    print(f"\n{'='*65}")
    print(f"TRAINING COMPLETE")
    print(f"  Best flight val MAE  : {best_val_mae:.3f} min")
    print(f"  Epochs               : {len(history)}")
    print(f"  Checkpoint           : {ckpt_path}")
    print(f"\nNext: run 07_evaluate_and_infer.py")


if __name__ == "__main__":
    main()
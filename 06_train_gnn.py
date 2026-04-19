"""
STEP 6 — DEPARTURE DELAY GNN v6  (clean snapshot training)
=========================================================
v6 model architecture with proven sequential snapshot training.
Snapshot mutation fixed: clone() per snapshot, SNAPS_PER_EPOCH limits cost.

Architecture:
  - 512 hidden, 8 heads, 2-layer airport GRU
  - Per-tail GRU (vectorised — one batched GRUCell call per snapshot)
  - ap_context_proj: airport GRU output injected into flight embeddings
  - Dynamic rotation gate (turnaround-weighted)
  - 8 edge types including reverse airport→flight edges
  - Severe-tail aware Huber loss + >120 min classifier
  - Exclusive horizon bands {0,1,3,6}h
  - Flight-weighted MAE via BandMAE
"""

import os, time, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

# ── PATHS ────────────────────────────────────────────────────────────────────
DRIVE_BASE     = "/content/drive/MyDrive/Airline_Graphs_Project"
GRAPH_DATA_DIR = f"{DRIVE_BASE}/graph_data"
CHECKPOINT_DIR = f"{DRIVE_BASE}/checkpoints"

# ── MODEL CONFIG ─────────────────────────────────────────────────────────────
NODE_FEAT_AP    = 30
NODE_FEAT_FL    = 19
HIDDEN_DIM      = 512
NUM_HEADS       = 8
NUM_GNN_LAYERS  = 2
GRU_HIDDEN_DIM  = 512
GRU_NUM_LAYERS  = 2
MLP_HIDDEN_DIM  = 256
TAIL_HIDDEN_DIM = 128
NUM_TAILS       = 12000
DROPOUT         = 0.10

# ── LOSS CONFIG ──────────────────────────────────────────────────────────────
HUBER_DELTA             = 20.0
DELAY_WEIGHT_MAX        = 5.0
ASYM_ALPHA              = 0.50
ORDINAL_THRESHOLDS      = [0.0, 15.0, 60.0, 120.0, 240.0, 720.0]
SEVERE_DELAY_THRESHOLD  = 120.0
SEVERE_PROB_THRESHOLD   = 0.50
SEVERE_ORDINAL_INDEX    = ORDINAL_THRESHOLDS.index(SEVERE_DELAY_THRESHOLD)
REGRESSION_TARGET_TRANSFORM = "signed_log1p"
USE_TAIL_UPLIFT          = True
TAIL_UPLIFT_THRESHOLDS   = [240.0, 720.0]
TAIL_UPLIFT_DETACH_GATES = True
AIRPORT_LOSS_WEIGHT = 0.25
FLIGHT_LOSS_WEIGHT  = 0.75
CLASS_LOSS_WEIGHT   = 0.15
TRAJ_LAMBDA         = 0.20
TRAJ_IMPROVE_W      = 0.60
TRAJ_CONSIST_W      = 0.40
PRED_BUFFER_HOURS   = 8
NS_PER_HOUR         = 3_600_000_000_000

HORIZON_WEIGHTS   = {0: 0.10, 1: 0.20, 3: 0.30, 6: 0.40}
LABEL_HORIZONS_FL = [0, 1, 3, 6]
SEVERE_WINDOW_WEIGHT_FLOOR = 1.0
SEVERE_WINDOW_WEIGHT_POWER = 1.0

# ── TRAINING CONFIG ──────────────────────────────────────────────────────────
LEARNING_RATE         = 1e-3
WEIGHT_DECAY          = 1e-4
NUM_EPOCHS            = 75
CLIP_GRAD_NORM        = 1.0
EARLY_STOP_PATIENCE   = 10
LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_FACTOR   = 0.5
BATCH_SIZE            = 16
SNAPS_PER_EPOCH       = 500    # train snapshots per epoch (~70 sec/epoch)
SNAPS_VAL             = 500    # val snapshots per epoch
RESUME_FROM_CHECKPOINT = False

# ── GRAPH STRUCTURE ──────────────────────────────────────────────────────────
NODE_TYPES = ["airport", "flight"]
EDGE_TYPES = [
    ("airport", "rotation",    "airport"),
    ("airport", "congestion",  "airport"),
    ("airport", "network",     "airport"),
    ("airport", "departs_to",  "flight"),
    ("airport", "arrives_from","flight"),
    ("flight",  "rotation",    "flight"),
    ("flight",  "departs_from","airport"),
    ("flight",  "arrives_at",  "airport"),
]

# ── MASKING ──────────────────────────────────────────────────────────────────
GATE_FEATURE_INDICES  = [0, 3, 4, 7, 11]
PARTIAL_GATE_FEATURES = [0, 7]
MASK_FULL_THRESHOLD    = 2.0 / 24
MASK_PARTIAL_THRESHOLD = 1.0 / 24


def apply_masking(x_fl):
    if x_fl.shape[0] == 0:
        return x_fl
    x   = x_fl.float().clone()
    t2d = x[:, 14]
    fm  = t2d > MASK_FULL_THRESHOLD
    pm  = (t2d > MASK_PARTIAL_THRESHOLD) & ~fm
    for i in GATE_FEATURE_INDICES:
        x[fm, i] = 0.0
    for i in PARTIAL_GATE_FEATURES:
        x[pm, i] = 0.0
    return x.half()


def delay_to_model_target(delay_min):
    if REGRESSION_TARGET_TRANSFORM == "signed_log1p":
        return torch.sign(delay_min) * torch.log1p(torch.abs(delay_min))
    return delay_min


def model_target_to_delay(model_target):
    if REGRESSION_TARGET_TRANSFORM == "signed_log1p":
        return torch.sign(model_target) * torch.expm1(torch.abs(model_target))
    return model_target


def regression_huber_delta():
    if REGRESSION_TARGET_TRANSFORM == "signed_log1p":
        return float(np.log1p(HUBER_DELTA))
    return HUBER_DELTA


# ════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════

class FlightDelayGNN(nn.Module):
    def __init__(self, ap_in, fl_in, hidden, heads, layers,
                 gru_h, gru_layers, mlp_h, tail_h,
                 n_airports, n_tails, dropout=0.1,
                 use_tail_uplift=USE_TAIL_UPLIFT,
                 tail_uplift_thresholds=TAIL_UPLIFT_THRESHOLDS,
                 tail_uplift_detach_gates=TAIL_UPLIFT_DETACH_GATES):
        super().__init__()
        self.hidden_dim     = hidden
        self.gru_hidden     = gru_h
        self.gru_num_layers = gru_layers
        self.num_airports   = n_airports
        self.num_tails      = n_tails
        self.tail_hidden    = tail_h
        self.num_layers     = layers
        self.use_tail_uplift = use_tail_uplift
        self.tail_uplift_thresholds = [float(x) for x in tail_uplift_thresholds]
        self.tail_uplift_detach_gates = tail_uplift_detach_gates
        self.tail_uplift_indices = [
            ORDINAL_THRESHOLDS.index(float(thr))
            for thr in self.tail_uplift_thresholds
            if float(thr) in ORDINAL_THRESHOLDS
        ]

        self.ap_proj = nn.Sequential(Linear(ap_in, hidden), nn.LayerNorm(hidden))
        self.fl_proj = nn.Sequential(Linear(fl_in, hidden), nn.LayerNorm(hidden))
        self.rotation_gate = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        meta = (NODE_TYPES, EDGE_TYPES)
        self.convs    = nn.ModuleList([HGTConv(hidden, hidden, meta, heads=heads)
                                       for _ in range(layers)])
        self.ap_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.fl_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drops    = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers)])

        self.ap_gru = nn.GRU(hidden, gru_h, num_layers=gru_layers, batch_first=False)
        self.ap_context_proj = nn.Linear(gru_h, hidden)
        self.tail_gru  = nn.GRUCell(hidden, tail_h)
        self.tail_proj = nn.Linear(tail_h, hidden)
        self.fl_fuse = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.fl_gate = nn.Sequential(nn.Linear(hidden*3, hidden), nn.Sigmoid())
        self.ap_head = nn.Sequential(
            nn.Linear(gru_h, mlp_h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_h, mlp_h//2), nn.ReLU(), nn.Linear(mlp_h//2, 1))
        self.fl_head = nn.Sequential(
            nn.Linear(hidden, mlp_h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_h, mlp_h//2), nn.ReLU(), nn.Linear(mlp_h//2, 1))
        self.fl_cls = nn.Sequential(
            nn.Linear(hidden, mlp_h//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_h//2, len(ORDINAL_THRESHOLDS)))
        if self.use_tail_uplift and self.tail_uplift_indices:
            self.fl_tail_uplift = nn.Sequential(
                nn.Linear(hidden, mlp_h//2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(mlp_h//2, len(self.tail_uplift_indices)))
        else:
            self.fl_tail_uplift = None

    def forward(self, data: HeteroData, ap_h, tail_h=None):
        n_fl = data["flight"].num_nodes
        dev  = ap_h.device
        if tail_h is None:
            tail_h = {}

        x_ap = self.ap_proj(data["airport"].x.float())
        x_fl = (self.fl_proj(data["flight"].x.float()) if n_fl > 0
                else torch.zeros(0, self.hidden_dim, device=dev))
        x = {"airport": x_ap, "flight": x_fl}

        rot_et = ("flight","rotation","flight")
        if (n_fl > 0 and hasattr(data[rot_et], "edge_index")
                and data[rot_et].edge_index.shape[1] > 0):
            rei = data[rot_et].edge_index
            rea = data[rot_et].edge_attr.float()
            if rea.shape[1] >= 2:
                g = self.rotation_gate(rea[:,:2])
                m = g * x["flight"][rei[0]]
                x["flight"] = x["flight"].clone()
                x["flight"].scatter_add_(
                    0, rei[1].unsqueeze(-1).expand_as(m), m)

        eid = {et: data[et].edge_index
               for et in EDGE_TYPES
               if et != rot_et
               and hasattr(data[et], "edge_index")
               and data[et].edge_index.shape[1] > 0}

        for i, conv in enumerate(self.convs):
            xn = conv(x, eid)
            if "airport" in xn:
                x["airport"] = self.ap_norms[i](
                    self.drops[i](xn["airport"]) + x["airport"])
            if "flight" in xn and n_fl > 0:
                x["flight"] = self.fl_norms[i](
                    self.drops[i](xn["flight"]) + x["flight"])

        _, ap_h_new = self.ap_gru(x["airport"].unsqueeze(0), ap_h)
        ap_pred = self.ap_head(ap_h_new[-1]).squeeze(-1)

        new_tail = dict(tail_h)
        if n_fl > 0:
            emb = x["flight"]
            ap_ctx_all = self.ap_context_proj(ap_h_new[-1])
            ap_context  = torch.zeros(n_fl, self.hidden_dim, device=dev)
            dep_et = ("flight","departs_from","airport")
            if hasattr(data[dep_et], "edge_index") and data[dep_et].edge_index.shape[1] > 0:
                fl_idx = data[dep_et].edge_index[0]
                ap_idx = data[dep_et].edge_index[1]
                ap_context[fl_idx] = ap_ctx_all[ap_idx]

            tail_context = torch.zeros(n_fl, self.hidden_dim, device=dev)
            if hasattr(data["flight"], "tail_id") and data["flight"].tail_id is not None:
                tids = data["flight"].tail_id.tolist()
                prev_h = torch.stack([
                    tail_h.get(int(tid), torch.zeros(self.tail_hidden, device=dev))
                    for tid in tids])
                new_h = self.tail_gru(emb, prev_h)
                tail_context = self.tail_proj(new_h)
                with torch.no_grad():
                    nh_det = new_h.detach()
                    for j, tid in enumerate(tids):
                        new_tail[int(tid)] = nh_det[j]

            cat    = torch.cat([emb, ap_context, tail_context], dim=-1)
            fl_out = self.fl_gate(cat) * self.fl_fuse(cat) + emb
            fl_pred_z = self.fl_head(fl_out).squeeze(-1)
            fl_logits = self.fl_cls(fl_out)
            if self.fl_tail_uplift is not None:
                uplift_mag = F.softplus(self.fl_tail_uplift(fl_out))
                tail_probs = torch.sigmoid(fl_logits[:, self.tail_uplift_indices])
                if self.tail_uplift_detach_gates:
                    tail_probs = tail_probs.detach()
                fl_pred_z = fl_pred_z + (tail_probs * uplift_mag).sum(dim=-1)
            fl_pred   = model_target_to_delay(fl_pred_z)
        else:
            fl_pred = torch.zeros(0, device=dev)
            fl_pred_z = torch.zeros(0, device=dev)
            fl_logits = torch.zeros(0, len(ORDINAL_THRESHOLDS), device=dev)

        return ap_pred, fl_pred, fl_pred_z, fl_logits, ap_h_new, new_tail

    def init_hidden(self, device):
        return (torch.zeros(self.gru_num_layers, self.num_airports,
                            self.gru_hidden, device=device), {})


# ════════════════════════════════════════════════════════════════════════════
# LOSS + METRICS
# ════════════════════════════════════════════════════════════════════════════

def asym_huber(pred, target, raw_target, mask,
               delta=None, alpha=ASYM_ALPHA, wmax=DELAY_WEIGHT_MAX):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    if delta is None:
        delta = regression_huber_delta()
    p = pred[mask]; t = target[mask]; raw_t = raw_target[mask]
    w = torch.ones_like(t)
    w = torch.where(raw_t >= 15, torch.full_like(t, 1.5), w)
    w = torch.where(raw_t >= 30, torch.full_like(t, 2.0), w)
    w = torch.where(raw_t >= 60, torch.full_like(t, 3.5), w)
    w = torch.where(raw_t >= 120, torch.full_like(t, wmax), w)
    e = p - t; ae = e.abs()
    h = torch.where(ae <= delta, 0.5*ae**2/delta, ae-0.5*delta)
    a = torch.where(e < 0, torch.full_like(e, alpha),
                            torch.full_like(e, 1-alpha))
    return (w*a*h).mean()


def compute_loss(ap_pred, ap_y, ap_m, fl_pred_z, fl_logits, fl_y, snap, device):
    if ap_m.sum() > 0:
        ae = (ap_pred[ap_m] - ap_y[ap_m]).abs()
        ap_loss = torch.where(ae <= HUBER_DELTA,
                              0.5*ae**2/HUBER_DELTA, ae-0.5*HUBER_DELTA).mean()
    else:
        ap_loss = torch.tensor(0.0, device=device, requires_grad=True)

    fl_y_z = delay_to_model_target(fl_y)
    fl_losses = []; cls_losses = []
    for h in LABEL_HORIZONS_FL:
        m = getattr(snap["flight"], f"y_mask_{h}h", None)
        if m is None or m.sum() == 0:
            continue
        fl_losses.append(HORIZON_WEIGHTS[h] * asym_huber(fl_pred_z, fl_y_z, fl_y, m))
        if fl_logits.shape[0] > 0:
            ordinal_target = torch.stack(
                [(fl_y[m] >= thr).float() for thr in ORDINAL_THRESHOLDS], dim=-1)
            cls_losses.append(HORIZON_WEIGHTS[h] *
                F.binary_cross_entropy_with_logits(
                    fl_logits[m], ordinal_target))

    fl_loss  = sum(fl_losses)  if fl_losses  else torch.tensor(0.0, device=device, requires_grad=True)
    cls_loss = sum(cls_losses) if cls_losses else torch.tensor(0.0, device=device, requires_grad=True)
    return (AIRPORT_LOSS_WEIGHT*ap_loss + FLIGHT_LOSS_WEIGHT*fl_loss +
            CLASS_LOSS_WEIGHT*cls_loss, ap_loss, fl_loss)


class BandMAE:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sums = {h: 0.0 for h in LABEL_HORIZONS_FL}
        self.cnts = {h: 0   for h in LABEL_HORIZONS_FL}
    def update(self, pred, target, snap):
        with torch.no_grad():
            for h in LABEL_HORIZONS_FL:
                m = getattr(snap["flight"], f"y_mask_{h}h", None)
                if m is None or m.sum() == 0: continue
                self.sums[h] += (pred[m]-target[m]).abs().sum().item()
                self.cnts[h] += m.sum().item()
    def mae(self, h): return self.sums[h] / max(self.cnts[h], 1)
    def overall(self): return sum(self.sums.values()) / max(sum(self.cnts.values()), 1)
    def ckpt(self):
        return (0.50*self.mae(6) + 0.30*self.mae(3) +
                0.15*self.mae(1) + 0.05*self.mae(0))


class SevereMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.err_sum = 0.0
        self.err_cnt = 0

    def update(self, logits, pred, target, snap):
        with torch.no_grad():
            valid = torch.zeros_like(target, dtype=torch.bool)
            for h in LABEL_HORIZONS_FL:
                m = getattr(snap["flight"], f"y_mask_{h}h", None)
                if m is not None:
                    valid = valid | m
            if valid.sum() == 0:
                return

            actual = target[valid] >= SEVERE_DELAY_THRESHOLD
            severe_prob = torch.sigmoid(logits[valid, SEVERE_ORDINAL_INDEX])
            pred_pos = severe_prob >= SEVERE_PROB_THRESHOLD

            self.tp += (pred_pos & actual).sum().item()
            self.fp += (pred_pos & ~actual).sum().item()
            self.fn += (~pred_pos & actual).sum().item()

            if actual.any():
                severe_err = (pred[valid][actual] - target[valid][actual]).abs()
                self.err_sum += severe_err.sum().item()
                self.err_cnt += severe_err.numel()

    def precision(self):
        return self.tp / max(self.tp + self.fp, 1)

    def recall(self):
        return self.tp / max(self.tp + self.fn, 1)

    def mae(self):
        return self.err_sum / max(self.err_cnt, 1)


def score_snapshot_tail(snap):
    if snap["flight"].num_nodes == 0:
        return 0.0
    target = snap["flight"].y
    valid = torch.zeros_like(target, dtype=torch.bool)
    for h in LABEL_HORIZONS_FL:
        m = getattr(snap["flight"], f"y_mask_{h}h", None)
        if m is not None:
            valid = valid | m
    tail_score = (
        0.5 * (((target >= 60) & (target < 120) & valid).sum().item()) +
        1.0 * (((target >= 120) & (target < 240) & valid).sum().item()) +
        2.0 * (((target >= 240) & (target < 720) & valid).sum().item()) +
        3.0 * (((target >= 720) & valid).sum().item())
    )
    return float(tail_score)


def build_window_scores(snapshot_counts, window_size):
    if len(snapshot_counts) == 0:
        return np.zeros(0, dtype=np.float64)
    if window_size >= len(snapshot_counts):
        return np.array([float(np.sum(snapshot_counts))], dtype=np.float64)

    csum = np.concatenate([[0.0], np.cumsum(snapshot_counts, dtype=np.float64)])
    max_start = len(snapshot_counts) - window_size
    return np.array([
        csum[start + window_size] - csum[start]
        for start in range(max_start + 1)
    ], dtype=np.float64)


def sample_window_start(window_scores):
    if len(window_scores) <= 1:
        return 0
    weights = np.power(window_scores + SEVERE_WINDOW_WEIGHT_FLOOR,
                       SEVERE_WINDOW_WEIGHT_POWER)
    return random.choices(range(len(window_scores)), weights=weights.tolist(), k=1)[0]


def compute_traj_loss(pred_buffer, snap, fl_pred, fl_y, snap_ns, device):
    if not hasattr(snap["flight"], "flight_id"):
        return torch.tensor(0.0, device=device, requires_grad=True)
    n_fl = snap["flight"].num_nodes
    valid = torch.zeros(n_fl, dtype=torch.bool, device=device)
    for h in [0,1,3,6]:
        m = getattr(snap["flight"], f"y_mask_{h}h", None)
        if m is not None: valid = valid | m
    fids = snap["flight"].flight_id.tolist()
    prev_preds, curr_preds, actuals = [], [], []
    for j, fid in enumerate(fids):
        fid = int(fid)
        if fid not in pred_buffer or not valid[j].item(): continue
        prev_preds.append(pred_buffer[fid]["pred"])
        curr_preds.append(fl_pred[j]); actuals.append(fl_y[j])
    if len(prev_preds) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    prev_t = torch.tensor(prev_preds, device=device)
    curr_t = torch.stack(curr_preds); act_t = torch.stack(actuals)
    ec = (curr_t-act_t).abs(); ep = (prev_t-act_t).abs()
    improve = torch.clamp(ec-ep, min=0.0).mean()
    jump    = (curr_t-prev_t).abs()
    consist = torch.where(jump<=HUBER_DELTA,
                          0.5*jump**2/HUBER_DELTA, jump-0.5*HUBER_DELTA).mean()
    return TRAJ_IMPROVE_W*improve + TRAJ_CONSIST_W*consist


# ════════════════════════════════════════════════════════════════════════════
# TRAINING EPOCH
# ════════════════════════════════════════════════════════════════════════════

def run_epoch(model, snapshots, optimizer, device,
              static_edges=None, is_train=True,
              epoch_idx=None):
    """
    Sequential snapshot training. clone() used per snapshot to prevent
    in-place mutation of the CPU snapshot list.
    epoch_idx: pre-determined snapshot indices for this epoch.
               Train indices resampled each epoch in main().
               Val indices fixed at startup (seeded) for stable metrics.
    """
    model.train(is_train)
    nw_ei = static_edges["network_ei"].to(device) if static_edges else None
    nw_ea = static_edges["network_ea"].to(device) if static_edges else None

    if epoch_idx is None:
        epoch_idx = list(range(len(snapshots)))

    ap_h, tail_h = model.init_hidden(device)
    metrics  = BandMAE()
    severe_m = SevereMetrics()
    tot_loss = tot_fl = 0.0
    n = 0

    for batch_start in range(0, len(epoch_idx), BATCH_SIZE):
        batch_si = epoch_idx[batch_start: batch_start + BATCH_SIZE]
        if is_train:
            optimizer.zero_grad()

        for si in batch_si:
            # clone() creates independent copy — no mutation of source list
            snap = snapshots[si].clone().to(device)
            if static_edges is not None:
                snap["airport","network","airport"].edge_index = nw_ei
                snap["airport","network","airport"].edge_attr  = nw_ea
            if snap["flight"].num_nodes > 0:
                snap["flight"].x = apply_masking(snap["flight"].x)

            ap_y = snap["airport"].y
            ap_m = snap["airport"].y_mask
            fl_y = snap["flight"].y

            with torch.set_grad_enabled(is_train):
                ap_pred, fl_pred, fl_pred_z, fl_logits, ap_h, tail_h = \
                    model(snap, ap_h, tail_h)
                ap_h   = ap_h.detach()
                tail_h = {k: v.detach() for k, v in tail_h.items()}

                loss, ap_l, fl_l = compute_loss(
                    ap_pred, ap_y, ap_m, fl_pred_z, fl_logits, fl_y, snap, device)

                if is_train:
                    (loss / len(batch_si)).backward()

            metrics.update(fl_pred.detach(), fl_y, snap)
            if fl_logits.shape[0] > 0:
                severe_m.update(fl_logits.detach(), fl_pred.detach(), fl_y, snap)
            tot_loss += loss.item(); tot_fl += fl_l.item()
            n += 1

        if is_train:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

    d = max(n, 1)
    return tot_loss/d, tot_fl/d, metrics, severe_m


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ════════════════════════════════════════════════════════════════════════════

def save_ckpt(model, optimizer, epoch, mdict, path):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(), "metrics": mdict,
        "num_airports": model.num_airports, "hidden_dim": model.hidden_dim,
        "gru_hidden": model.gru_hidden, "gru_num_layers": model.gru_num_layers,
        "tail_hidden": model.tail_hidden, "num_tails": model.num_tails,
        "cls_out_dim": len(ORDINAL_THRESHOLDS),
        "class_target_threshold": SEVERE_DELAY_THRESHOLD,
        "ordinal_thresholds": ORDINAL_THRESHOLDS,
        "regression_target_transform": REGRESSION_TARGET_TRANSFORM,
        "use_tail_uplift": model.use_tail_uplift,
        "tail_uplift_thresholds": model.tail_uplift_thresholds,
        "tail_uplift_detach_gates": model.tail_uplift_detach_gates,
    }, path)

def load_ckpt(path, model, optimizer, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    try: optimizer.load_state_dict(ck["optim_state"])
    except Exception: pass
    m = ck.get("metrics", {})
    print(f"  Loaded epoch {ck['epoch']} | "
          f"val_fl_mae={m.get('val_fl_mae', float('nan')):.3f}")
    return ck["epoch"] + 1, m.get("val_ckpt", float("inf"))


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("STEP 6 — DEPARTURE DELAY GNN v6  (clean snapshot training)")
    print("=" * 70)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("\nLoading snapshots ...")
    def load(name):
        s = torch.load(os.path.join(GRAPH_DATA_DIR, f"snapshots_{name}.pt"),
                       map_location="cpu", weights_only=False)
        print(f"  {name:>5}: {len(s):,}")
        return s
    train_snaps = load("train")
    val_snaps   = load("val")

    print("\nLoading static edges ...")
    static = torch.load(os.path.join(GRAPH_DATA_DIR, "static_edges.pt"),
                        map_location="cpu", weights_only=False)

    t2i_path = os.path.join(GRAPH_DATA_DIR, "tail2idx.json")
    n_tails = NUM_TAILS
    if os.path.exists(t2i_path):
        with open(t2i_path) as f:
            tail2idx = json.load(f)
        n_tails = max(tail2idx.values()) + 1
        print(f"  Tail numbers : {len(tail2idx):,} → {n_tails} indices")

    ap_in = train_snaps[0]["airport"].x.shape[1]
    fl_in = (train_snaps[0]["flight"].x.shape[1]
             if train_snaps[0]["flight"].num_nodes > 0 else NODE_FEAT_FL)
    n_ap  = train_snaps[0]["airport"].num_nodes

    print(f"\n  Airport feats : {ap_in}  Flight feats : {fl_in}")
    print(f"  Train : {len(train_snaps):,}  Val : {len(val_snaps):,}")
    print(f"  Snaps/epoch (train): {SNAPS_PER_EPOCH}  (val): {SNAPS_VAL}")

    print("\nIndexing weighted tail delays (60+ min, ordinal buckets) ...")
    train_tail_scores = np.array([score_snapshot_tail(s) for s in train_snaps],
                                 dtype=np.float64)
    val_tail_scores = np.array([score_snapshot_tail(s) for s in val_snaps],
                               dtype=np.float64)
    train_window_scores = build_window_scores(train_tail_scores, SNAPS_PER_EPOCH)
    val_window_scores = build_window_scores(val_tail_scores, SNAPS_VAL)
    print(f"  Train tail score : {train_tail_scores.sum():,.1f}")
    print(f"  Val tail score   : {val_tail_scores.sum():,.1f}")

    print("\nBuilding model ...")
    model = FlightDelayGNN(
        ap_in=ap_in, fl_in=fl_in,
        hidden=HIDDEN_DIM, heads=NUM_HEADS, layers=NUM_GNN_LAYERS,
        gru_h=GRU_HIDDEN_DIM, gru_layers=GRU_NUM_LAYERS,
        mlp_h=MLP_HIDDEN_DIM, tail_h=TAIL_HIDDEN_DIM,
        n_airports=n_ap, n_tails=n_tails, dropout=DROPOUT,
        use_tail_uplift=USE_TAIL_UPLIFT,
        tail_uplift_thresholds=TAIL_UPLIFT_THRESHOLDS,
        tail_uplift_detach_gates=TAIL_UPLIFT_DETACH_GATES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")
    print(f"  Hidden     : {HIDDEN_DIM} | GRU {GRU_NUM_LAYERS}×{GRU_HIDDEN_DIM}")
    print(f"  Horizon W  : {HORIZON_WEIGHTS}")
    if USE_TAIL_UPLIFT:
        print(f"  Tail uplift: >= {TAIL_UPLIFT_THRESHOLDS} "
              f"(detach_gates={TAIL_UPLIFT_DETACH_GATES})")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR)

    start_epoch = 1; best_ckpt = float("inf"); patience_ctr = 0
    history = []; ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if RESUME_FROM_CHECKPOINT and os.path.exists(ckpt_path):
        print("\nResuming ...")
        try:
            start_epoch, best_ckpt = load_ckpt(
                ckpt_path, model, optimizer, device)
        except Exception as e:
            print(f"  ⚠ {e} — starting fresh")
            start_epoch = 1; best_ckpt = float("inf")
    else:
        print("\nStarting fresh ...")

    hdr = (f"{'Ep':>4}  {'Loss':>8}  {'FlMAE':>7}  "
           f"{'vMAE':>7}  {'v6h':>6}  {'v3h':>6}  {'v1h':>6}  "
           f"{'vCkpt':>7}  {'LR':>8}")
    sep = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}")
    print(f"  v6h=[>=6h] v3h=[3-6h] v1h=[1-3h] | "
          f"vMAE=flight-weighted | vCkpt=checkpoint metric (includes 0h)")
    print(sep)

    # Val: fixed contiguous window from a stable start point (seed=42)
    # Same window every epoch → stable vl_ckpt, scheduler, early stopping
    rng = random.Random(42)
    val_start = rng.randint(0, max(0, len(val_snaps) - SNAPS_VAL))
    val_idx   = list(range(val_start, min(val_start + SNAPS_VAL, len(val_snaps))))
    print(f"  Val window   : snaps {val_idx[0]}–{val_idx[-1]} "
          f"({len(val_idx)} contiguous, fixed | "
          f"{val_window_scores[val_start]:.1f} tail score)")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Train: severity-focused contiguous window each epoch.
        # GRU state still sees a real uninterrupted timeline, but start
        # points are biased toward windows that contain more >120 min flights.
        tr_start = sample_window_start(train_window_scores)
        tr_idx    = list(range(tr_start,
                               min(tr_start + SNAPS_PER_EPOCH, len(train_snaps))))

        tr_loss, tr_fl, tr_m, tr_s = run_epoch(
            model, train_snaps, optimizer, device,
            static_edges=static, is_train=True,
            epoch_idx=tr_idx)

        vl_loss, vl_fl, vl_m, vl_s = run_epoch(
            model, val_snaps, optimizer, device,
            static_edges=static, is_train=False,
            epoch_idx=val_idx)

        lr      = optimizer.param_groups[0]["lr"]
        vl_ckpt = vl_m.ckpt()

        history.append({
            "epoch": epoch, "tr_loss": tr_loss,
            "tr_mae": tr_m.overall(), "vl_mae": vl_m.overall(),
            "vl_ckpt": vl_ckpt, "vl_6h": vl_m.mae(6),
            "vl_3h": vl_m.mae(3), "vl_1h": vl_m.mae(1), "lr": lr,
        })

        print(f"{epoch:>4}  {tr_loss:>8.3f}  {tr_m.overall():>7.3f}  "
              f"{vl_m.overall():>7.3f}  {vl_m.mae(6):>6.2f}  {vl_m.mae(3):>6.2f}  "
              f"{vl_m.mae(1):>6.2f}  {vl_ckpt:>7.3f}  {lr:>8.2e}")
        print(f"      severe>120  vRec={vl_s.recall():.3f}  "
              f"vPrec={vl_s.precision():.3f}  vMAE={vl_s.mae():.2f}")

        scheduler.step(vl_ckpt)

        if vl_ckpt < best_ckpt:
            best_ckpt = vl_ckpt; patience_ctr = 0
            save_ckpt(model, optimizer, epoch,
                      {"val_fl_mae": vl_m.overall(), "val_ckpt": vl_ckpt,
                       "val_6h": vl_m.mae(6), "val_3h": vl_m.mae(3),
                       "val_1h": vl_m.mae(1),
                       "val_severe_recall": vl_s.recall(),
                       "val_severe_precision": vl_s.precision(),
                       "val_severe_mae": vl_s.mae()},
                      ckpt_path)
            print(f"  ✅ ckpt={vl_ckpt:.3f} "
                  f"[6h={vl_m.mae(6):.2f} 3h={vl_m.mae(3):.2f} "
                  f"1h={vl_m.mae(1):.2f} sevR={vl_s.recall():.3f}] → saved")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(
        os.path.join(CHECKPOINT_DIR, "training_history.csv"), index=False)
    print(f"\n✅ Best ckpt metric: {best_ckpt:.3f}")


if __name__ == "__main__":
    main()

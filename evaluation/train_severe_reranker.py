import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ORDINAL_THRESHOLDS = [0.0, 15.0, 60.0, 120.0, 240.0, 720.0]


def binary_stats(actual, pred):
    actual = np.asarray(actual, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((actual & pred).sum())
    fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "pred_pos": int(pred.sum()),
        "actual_pos": int(actual.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def threshold_sweep(actual, probs, thresholds):
    rows = []
    for thr in thresholds:
        pred = probs >= thr
        s = binary_stats(actual, pred)
        rows.append(
            {
                "threshold": float(thr),
                "precision": s["precision"],
                "recall": s["recall"],
                "pred_pos": s["pred_pos"],
                "actual_pos": s["actual_pos"],
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
            }
        )
    return pd.DataFrame(rows)


def choose_best_from_sweep(sweep, min_recall=0.08, min_pred_pos=100):
    valid = sweep[(sweep["recall"] >= min_recall) & (sweep["pred_pos"] >= min_pred_pos)]
    if len(valid) == 0:
        best = sweep.sort_values(
            ["precision", "recall", "pred_pos", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    else:
        best = valid.sort_values(
            ["precision", "recall", "pred_pos", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    return best


def choose_precision_first_threshold(
    actual,
    probs,
    min_recall=0.08,
    min_pred_pos=100,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    sweep = threshold_sweep(actual, probs, thresholds)
    best = choose_best_from_sweep(sweep, min_recall=min_recall, min_pred_pos=min_pred_pos)
    return float(best["threshold"]), sweep


def pred_bucket_idx(pred):
    vals = np.asarray(pred, dtype=np.float32)
    idx = np.zeros(len(vals), dtype=np.int8)
    idx[vals >= 0] = 1
    idx[vals >= 15] = 2
    idx[vals >= 60] = 3
    idx[vals >= 120] = 4
    idx[vals >= 240] = 5
    idx[vals >= 720] = 6
    return idx


def load_lookup(path):
    lookup = pd.read_parquet(path)
    keep = ["flight_id", "ORIGIN", "DEST", "Operating_Airline", "dep_datetime"]
    keep = [c for c in keep if c in lookup.columns]
    return lookup[keep].copy()


def sample_training_rows(
    csv_path,
    severe_minutes,
    neg_keep_prob,
    random_state,
    chunksize,
    max_rows,
):
    rng = np.random.RandomState(random_state)
    usecols = ["flight_id", "snap_idx", "horizon_h", "pred", "severe_prob", "actual"]
    parts = []

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        target = chunk["actual"].to_numpy() >= severe_minutes
        keep = target | (rng.random(len(chunk)) < neg_keep_prob)
        sub = chunk.loc[keep].copy()
        sub["target_severe"] = target[keep].astype(np.int8)
        parts.append(sub)

    df = pd.concat(parts, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        pos = df[df["target_severe"] == 1]
        neg = df[df["target_severe"] == 0]
        max_neg = max(max_rows - len(pos), 0)
        if len(neg) > max_neg:
            neg = neg.sample(n=max_neg, random_state=random_state)
        df = pd.concat([pos, neg], ignore_index=True)
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def attach_metadata(df, lookup):
    out = df.merge(lookup, on="flight_id", how="left")

    dep_dt = pd.to_datetime(out.get("dep_datetime"), errors="coerce")
    out["dep_hour"] = dep_dt.dt.hour.fillna(-1).astype(int)
    out["dep_dow"] = dep_dt.dt.dayofweek.fillna(-1).astype(int)
    out["dep_month"] = dep_dt.dt.month.fillna(-1).astype(int)
    out["route"] = (
        out.get("ORIGIN", pd.Series("", index=out.index)).fillna("UNK")
        + "_"
        + out.get("DEST", pd.Series("", index=out.index)).fillna("UNK")
    )

    out["pred_bucket_idx"] = pred_bucket_idx(out["pred"])
    out["pred_ge_120"] = (out["pred"] >= 120).astype(int)
    out["pred_ge_240"] = (out["pred"] >= 240).astype(int)
    out["pred_ge_720"] = (out["pred"] >= 720).astype(int)
    out["pred_minus_120"] = np.maximum(out["pred"] - 120.0, 0.0)
    out["pred_minus_240"] = np.maximum(out["pred"] - 240.0, 0.0)
    out["pred_minus_720"] = np.maximum(out["pred"] - 720.0, 0.0)
    out["sev_x_pred"] = out["severe_prob"] * out["pred"]
    out["sev_x_pred120"] = out["severe_prob"] * out["pred_ge_120"]
    out["sev_x_h6"] = out["severe_prob"] * (out["horizon_h"] == 6).astype(float)
    out["pred_abs"] = out["pred"].abs()
    out["pred_clip"] = out["pred"].clip(-60, 720)

    return out


def make_pipeline():
    numeric_features = [
        "pred",
        "pred_abs",
        "pred_clip",
        "severe_prob",
        "pred_minus_120",
        "pred_minus_240",
        "pred_minus_720",
        "sev_x_pred",
        "sev_x_pred120",
        "sev_x_h6",
        "pred_ge_120",
        "pred_ge_240",
        "pred_ge_720",
        "dep_hour",
        "dep_dow",
        "dep_month",
    ]
    categorical_features = [
        "horizon_h",
        "pred_bucket_idx",
        "Operating_Airline",
        "ORIGIN",
        "DEST",
        "route",
    ]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=200,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline([("pre", pre), ("clf", clf)]), numeric_features + categorical_features


def evaluate_split(name, actual, probs, threshold):
    pred = probs >= threshold
    s = binary_stats(actual, pred)
    print(
        f"{name:>8s} | thr={threshold:.3f} | "
        f"precision={s['precision']:.3f} recall={s['recall']:.3f} "
        f"pred+={s['pred_pos']:,} actual+={s['actual_pos']:,}"
    )
    return s


def score_csv_sweeps(
    csv_path,
    lookup,
    pipe,
    feature_cols,
    severe_minutes,
    thresholds,
    chunksize,
    base_thresholds=(0.50, 0.60, 0.70),
):
    usecols = ["flight_id", "snap_idx", "horizon_h", "pred", "severe_prob", "actual"]
    thr_arr = np.asarray(list(thresholds), dtype=np.float32)
    tp = np.zeros(len(thr_arr), dtype=np.int64)
    fp = np.zeros(len(thr_arr), dtype=np.int64)
    fn = np.zeros(len(thr_arr), dtype=np.int64)

    base_stats = {
        float(thr): {"tp": 0, "fp": 0, "fn": 0, "pred_pos": 0, "actual_pos": 0}
        for thr in base_thresholds
    }

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["target_severe"] = (chunk["actual"] >= severe_minutes).astype(np.int8)
        feat_df = attach_metadata(chunk, lookup)
        probs = pipe.predict_proba(feat_df[feature_cols])[:, 1]
        actual = feat_df["target_severe"].to_numpy(dtype=bool)

        pred_matrix = probs[:, None] >= thr_arr[None, :]
        actual_col = actual[:, None]
        tp += (pred_matrix & actual_col).sum(axis=0)
        fp += (pred_matrix & ~actual_col).sum(axis=0)
        fn += (~pred_matrix & actual_col).sum(axis=0)

        for thr in base_thresholds:
            pred = feat_df["severe_prob"].to_numpy() >= thr
            s = binary_stats(actual, pred)
            base_stats[float(thr)]["tp"] += s["tp"]
            base_stats[float(thr)]["fp"] += s["fp"]
            base_stats[float(thr)]["fn"] += s["fn"]
            base_stats[float(thr)]["pred_pos"] += s["pred_pos"]
            base_stats[float(thr)]["actual_pos"] += s["actual_pos"]

    rows = []
    for i, thr in enumerate(thr_arr):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(p),
                "recall": float(r),
                "pred_pos": int(tp[i] + fp[i]),
                "actual_pos": int(tp[i] + fn[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame.from_dict(base_stats, orient="index").reset_index().rename(columns={"index": "threshold"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        default="evaluation/eval_best_model_dep_12k_ordinal_ep25_val.csv",
    )
    parser.add_argument(
        "--test_csv",
        default="evaluation/eval_best_model_dep_12k_ordinal_ep25_test.csv",
    )
    parser.add_argument(
        "--lookup",
        default="graph_data/flight_lookup.parquet",
    )
    parser.add_argument(
        "--severe_minutes",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--neg_keep_prob",
        type=float,
        default=0.03,
        help="Keep probability for non-severe rows when sampling the train CSV.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=1_000_000,
        help="Maximum sampled rows used from the train CSV after keeping all positives.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--min_recall",
        type=float,
        default=0.08,
        help="Minimum recall constraint when choosing a precision-first threshold.",
    )
    parser.add_argument(
        "--min_pred_pos",
        type=int,
        default=100,
        help="Minimum positive predictions required when choosing the threshold.",
    )
    parser.add_argument(
        "--output_prefix",
        default="evaluation/severe_reranker_ord_ep25",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(f"{out_prefix}.pkl")

    print("Loading lookup ...")
    lookup = load_lookup(args.lookup)
    print(f"  lookup rows: {len(lookup):,}")

    print("Sampling training rows from validation CSV ...")
    train_sample = sample_training_rows(
        args.train_csv,
        severe_minutes=args.severe_minutes,
        neg_keep_prob=args.neg_keep_prob,
        random_state=args.random_state,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
    print(f"  sampled rows: {len(train_sample):,}")
    print(f"  severe rate : {train_sample['target_severe'].mean():.3%}")

    print("Attaching metadata and engineering features ...")
    train_sample = attach_metadata(train_sample, lookup)

    # Hold out entire snapshots for threshold tuning.
    tune_mask = (train_sample["snap_idx"] % 5) == 0
    fit_df = train_sample.loc[~tune_mask].copy()
    tune_df = train_sample.loc[tune_mask].copy()
    if len(tune_df) == 0:
        raise RuntimeError("Tune split is empty. Adjust sampling or split logic.")

    pipe, feature_cols = make_pipeline()
    X_fit = fit_df[feature_cols]
    y_fit = fit_df["target_severe"].to_numpy()
    X_tune = tune_df[feature_cols]
    y_tune = tune_df["target_severe"].to_numpy()

    print("Training reranker ...")
    pipe.fit(X_fit, y_fit)
    tune_probs = pipe.predict_proba(X_tune)[:, 1]
    threshold, tune_sweep = choose_precision_first_threshold(
        y_tune,
        tune_probs,
        min_recall=args.min_recall,
        min_pred_pos=args.min_pred_pos,
    )

    base_thresholds = [0.50, 0.60, 0.70]
    print("\nSampled tune split comparison:")
    for thr in base_thresholds:
        s = binary_stats(y_tune, tune_df["severe_prob"].to_numpy() >= thr)
        print(
            f"base@{thr:.2f} | precision={s['precision']:.3f} "
            f"recall={s['recall']:.3f} pred+={s['pred_pos']:,}"
        )
    evaluate_split("rerank", y_tune, tune_probs, threshold)

    print("\nRefitting reranker on full sampled validation set ...")
    pipe.fit(train_sample[feature_cols], train_sample["target_severe"].to_numpy())

    sweep_thresholds = np.linspace(0.05, 0.95, 181)
    print("Sweeping threshold against full validation distribution ...")
    val_sweep, val_base = score_csv_sweeps(
        args.train_csv,
        lookup=lookup,
        pipe=pipe,
        feature_cols=feature_cols,
        severe_minutes=args.severe_minutes,
        thresholds=sweep_thresholds,
        chunksize=args.chunksize,
        base_thresholds=base_thresholds,
    )
    threshold = float(
        choose_best_from_sweep(
            val_sweep,
            min_recall=args.min_recall,
            min_pred_pos=args.min_pred_pos,
        )["threshold"]
    )

    print("\nFull validation comparison:")
    for thr in base_thresholds:
        row = val_base[val_base["threshold"] == float(thr)].iloc[0]
        p = row["tp"] / max(row["tp"] + row["fp"], 1)
        r = row["tp"] / max(row["tp"] + row["fn"], 1)
        print(
            f"base@{thr:.2f} | precision={p:.3f} "
            f"recall={r:.3f} pred+={int(row['pred_pos']):,}"
        )
    val_best = val_sweep[val_sweep["threshold"] == threshold].iloc[0]
    print(
        f" rerank | thr={threshold:.3f} | precision={val_best['precision']:.3f} "
        f"recall={val_best['recall']:.3f} pred+={int(val_best['pred_pos']):,}"
    )

    print("Scoring corrected test CSV ...")
    test_df = pd.read_csv(args.test_csv)
    test_df["target_severe"] = (test_df["actual"] >= args.severe_minutes).astype(np.int8)
    test_df = attach_metadata(test_df, lookup)
    test_probs = pipe.predict_proba(test_df[feature_cols])[:, 1]

    print("\nTest comparison:")
    for thr in base_thresholds:
        s = binary_stats(test_df["target_severe"].to_numpy(), test_df["severe_prob"].to_numpy() >= thr)
        print(
            f"base@{thr:.2f} | precision={s['precision']:.3f} "
            f"recall={s['recall']:.3f} pred+={s['pred_pos']:,}"
        )
    test_stats = evaluate_split("rerank", test_df["target_severe"].to_numpy(), test_probs, threshold)

    tune_sweep.to_csv(f"{out_prefix}_sampled_tune_sweep.csv", index=False)
    val_sweep.to_csv(f"{out_prefix}_val_sweep.csv", index=False)
    val_base.to_csv(f"{out_prefix}_val_base_compare.csv", index=False)
    test_sweep = threshold_sweep(test_df["target_severe"].to_numpy(), test_probs, np.linspace(0.05, 0.95, 181))
    test_sweep.to_csv(f"{out_prefix}_test_sweep.csv", index=False)

    scored_cols = [
        "flight_id",
        "snap_idx",
        "horizon_h",
        "pred",
        "severe_prob",
        "actual",
        "target_severe",
    ]
    scored = test_df[scored_cols].copy()
    scored["rerank_prob"] = test_probs
    scored["rerank_alert"] = scored["rerank_prob"] >= threshold
    scored.to_csv(f"{out_prefix}_test_scores.csv", index=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": pipe,
                "threshold": threshold,
                "feature_cols": feature_cols,
                "severe_minutes": args.severe_minutes,
                "min_recall": args.min_recall,
                "min_pred_pos": args.min_pred_pos,
            },
            f,
        )

    report = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "sampled_train_rows": int(len(train_sample)),
        "sampled_train_severe_rate": float(train_sample["target_severe"].mean()),
        "tuned_threshold": float(threshold),
        "test_precision": float(test_stats["precision"]),
        "test_recall": float(test_stats["recall"]),
        "test_pred_pos": int(test_stats["pred_pos"]),
        "test_actual_pos": int(test_stats["actual_pos"]),
        "model_path": str(model_path),
    }
    Path(f"{out_prefix}_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved model      -> {model_path}")
    print(f"Saved report     -> {out_prefix}_report.json")
    print(f"Saved tune sweep -> {out_prefix}_sampled_tune_sweep.csv")
    print(f"Saved val sweep  -> {out_prefix}_val_sweep.csv")
    print(f"Saved test sweep -> {out_prefix}_test_sweep.csv")
    print(f"Saved test score -> {out_prefix}_test_scores.csv")


if __name__ == "__main__":
    main()

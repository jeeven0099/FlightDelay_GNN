import argparse
import pandas as pd


def binary_stats(actual, pred):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="evaluation/eval_best_model_dep_12k_ordinal_ep25_test.csv",
        help="Path to evaluation CSV with severe_prob and actual columns.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Probability thresholds to sweep.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional horizon filter, e.g. 6 for 6h only.",
    )
    parser.add_argument(
        "--severe_minutes",
        type=float,
        default=120.0,
        help="Actual delay cutoff for severe events.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.horizon is not None:
        df = df[df["horizon_h"] == args.horizon].copy()

    actual = df["actual"] >= args.severe_minutes

    print(f"CSV           : {args.csv}")
    print(f"Rows          : {len(df):,}")
    print(f"Horizon       : {args.horizon if args.horizon is not None else 'all'}")
    print(f"Severe cutoff : >= {args.severe_minutes:.0f} min")
    print()
    print(f"{'thr':>5} {'precision':>10} {'recall':>10} {'pred+':>10} {'actual+':>10} {'tp':>8} {'fp':>8} {'fn':>8}")
    print("-" * 75)

    for thr in args.thresholds:
        pred = df["severe_prob"] >= thr
        s = binary_stats(actual, pred)
        print(
            f"{thr:>5.2f} "
            f"{s['precision']:>10.3f} "
            f"{s['recall']:>10.3f} "
            f"{s['pred_pos']:>10,} "
            f"{s['actual_pos']:>10,} "
            f"{s['tp']:>8,} "
            f"{s['fp']:>8,} "
            f"{s['fn']:>8,}"
        )


if __name__ == "__main__":
    main()

"""Step 2: Evaluation harness for DTB challenge.
Uses exact leaderboard MAPE formula. All experiments log to outputs/experiments_dtb.csv.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import GroupKFold

OUT_DIR = "outputs"
DAILY = f"{OUT_DIR}/dtb_daily.parquet"
EXPERIMENTS = f"{OUT_DIR}/experiments_dtb.csv"
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]

# --- MAPE: exact leaderboard formula ---
def mape_per_target(y_true, y_pred):
    """Per-target MAPE = mean(|true - pred| / |true|) * 100.
    Uses tiny guard on denominator to survive true==0 edge cases (not present in data)."""
    denom = np.maximum(np.abs(y_true), 1e-12)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def mean_mape(y_true, y_pred):
    """Mean of the four per-target MAPEs (leaderboard metric)."""
    return float(np.mean([mape_per_target(y_true[:, j], y_pred[:, j]) for j in range(len(TARGETS))]))

def evaluate(predict_fn, daily_df, cv_folds=5, random_state=42):
    """GroupKFold CV. predict_fn(train_df, test_df) -> (n_test, 4) numpy array.
    Returns dict with per-target MAPE, mean MAPE, per-fold breakdown, and worst batches.
    """
    train_df = daily_df[~daily_df.is_test].copy()
    Y = train_df[TARGETS].to_numpy()
    gkf = GroupKFold(n_splits=cv_folds)
    folds = []
    all_preds = np.zeros_like(Y)

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_df, groups=train_df.batch_id)):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        pred = predict_fn(tr, va)
        all_preds[va_idx] = pred
        per_target = {c: mape_per_target(va[c].to_numpy(), pred[:, j]) for j, c in enumerate(TARGETS)}
        folds.append({
            "fold": fold_i,
            **per_target,
            "mean_mape": float(np.mean(list(per_target.values()))),
            "n_batches": va.batch_id.nunique(),
        })

    per_target = {c: mape_per_target(Y[:, j], all_preds[:, j]) for j, c in enumerate(TARGETS)}
    overall_mean = float(np.mean(list(per_target.values())))

    # Per-batch MAPE for worst-batch diagnostic
    batch_mapes = []
    for bid in train_df.batch_id.unique():
        mask = train_df.batch_id == bid
        if mask.sum() == 0:
            continue
        bm = mean_mape(Y[mask], all_preds[mask])
        batch_mapes.append((bid, bm, mask.sum()))
    batch_mapes.sort(key=lambda x: -x[1])  # worst first

    return {
        **{f"{c}_mape": v for c, v in per_target.items()},
        "mean_mape": overall_mean,
        "per_fold": folds,
        "per_fold_mapes": [f["mean_mape"] for f in folds],
        "fold_std": float(np.std([f["mean_mape"] for f in folds])),
        "worst_batches": [(b, round(m, 2)) for b, m, _ in batch_mapes[:5]],
    }


def log_experiment(name, description, results, notes=""):
    """Append one row to experiments CSV."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": name,
        "short_description": description,
        "vcd_mape": round(results.get("vcd_mape", np.nan), 4),
        "tcd_mape": round(results.get("tcd_mape", np.nan), 4),
        "substrate_mape": round(results.get("substrate_mape", np.nan), 4),
        "byproduct_mape": round(results.get("byproduct_mape", np.nan), 4),
        "mean_mape": round(results["mean_mape"], 4),
        "per_fold_mapes": json.dumps([round(m, 4) for m in results.get("per_fold_mapes", [])]),
        "fold_std": round(results.get("fold_std", np.nan), 4),
        "notes": notes,
    }
    df = pd.DataFrame([row])
    if os.path.exists(EXPERIMENTS):
        df.to_csv(EXPERIMENTS, mode="a", header=False, index=False)
    else:
        df.to_csv(EXPERIMENTS, index=False)
    print(f"  -> logged to {EXPERIMENTS}")


def print_results(name, results):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  mean MAPE: {results['mean_mape']:.4f}  (fold std: {results.get('fold_std', np.nan):.4f})")
    for c in TARGETS:
        print(f"    {c}: {results[f'{c}_mape']:.4f}")
    per_fold = results.get("per_fold_mapes", [])
    if per_fold:
        print(f"  per-fold: {[f'{m:.4f}' for m in per_fold]}")
    worst = results.get("worst_batches", [])
    if worst:
        print(f"  worst batches: {worst}")


def predict_test(predict_fn, daily_df):
    """Generate predictions for test batches using model trained on all train data."""
    train_df = daily_df[~daily_df.is_test].copy()
    test_df = daily_df[daily_df.is_test].copy()
    pred = predict_fn(train_df, test_df)
    # clip and enforce physical constraint
    pred = np.clip(pred, 0, None)
    pred[:, 1] = np.maximum(pred[:, 1], pred[:, 0])  # tcd >= vcd
    return test_df[["batch_id", "elapsed_time", "day"]].copy(), pred


# ============================================================
# Model-building utilities shared across experiments
# ============================================================

def get_feature_cols(daily_df, extra_cols=None):
    """Columns usable as features: all numeric except targets, batch_id, and metadata."""
    exclude = set(TARGETS + ["batch_id", "elapsed_time", "is_test", "dcd", "viability"])
    cols = [c for c in daily_df.columns if c not in exclude and daily_df[c].dtype in ('float64', 'int64', 'bool', 'float32', 'int32')]
    if extra_cols:
        cols = [c for c in cols if c not in set(extra_cols)] + extra_cols
    return cols


def prepare_xy(df, feature_cols):
    """Return X (features, with NaN→0) and y (targets, only rows with targets)."""
    mask = df[TARGETS].notna().all(axis=1)
    X = df.loc[mask, feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y = df.loc[mask, TARGETS].to_numpy(dtype=np.float32)
    return X, y


if __name__ == "__main__":
    # Quick smoke test
    daily = pd.read_parquet(DAILY)
    print(f"Loaded {len(daily)} rows from {DAILY}")

    # Verify MAPE on a trivial baseline: predict day-0 value for all days
    def trivial_baseline(train_df, test_df):
        pred = np.zeros((len(test_df), 4))
        for j, c in enumerate(TARGETS):
            pred[:, j] = test_df[f"init_{c}"].to_numpy()
        return pred

    res = evaluate(trivial_baseline, daily)
    print_results("Trivial baseline (predict day-0 everywhere)", res)

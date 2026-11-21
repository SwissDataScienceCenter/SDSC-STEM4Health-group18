"""Run experiments 0–5 from PLAN_DTB.md in order, with plateau rule.
Each experiment is a function; results flow through dtb_eval.py harness.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path so we can import dtb_eval
sys.path.insert(0, os.path.dirname(__file__))
from dtb_eval import (
    evaluate, log_experiment, print_results, predict_test,
    get_feature_cols, prepare_xy, TARGETS, DAILY, OUT_DIR,
)

SEED = 42
PLATEAU_THRESHOLD = 0.2  # stop if two consecutive experiments improve < this


def get_init_features(daily_df):
    """Core feature set: day-0 state + volume + feed windows + time features."""
    base = get_feature_cols(daily_df)
    # Ensure critical features are present
    critical = ["day", "volume", "vol_rel", "dvol_prev", "dvol_next", "time_to_end",
                "init_vcd", "init_tcd", "init_substrate", "init_byproduct", "init_volume"]
    for c in critical:
        if c not in base:
            base.append(c)
    # Also all feed features
    for c in daily_df.columns:
        if any(c.startswith(f"feed_") for f in ["feed_substrate", "feed_b", "feed_c", "feed_d"]):
            if c not in base:
                base.append(c)
    return base


# ============================================================
# Experiment 0: Baseline
# ============================================================
def exp0_baseline(daily):
    """Mean-trajectory-per-day baseline + ExtraTrees ensemble."""
    print("\n" + "="*60)
    print("EXPERIMENT 0: Baseline")
    
    train = daily[~daily.is_test].copy()
    
    # --- Baseline A: Mean trajectory ratio per day ---
    # For each target, compute mean(ratio to day-0) per day, then scale by batch's day-0
    ratio_curves = {}
    for c in TARGETS:
        train_ratio = (train[c] / train[f"init_{c}"]).groupby(train.day).mean()
        ratio_curves[c] = train_ratio
    
    def mean_trajectory_predict(tr_df, te_df):
        pred = np.zeros((len(te_df), len(TARGETS)))
        for j, c in enumerate(TARGETS):
            # Recompute ratio curve from training data
            tr_ratio = (tr_df[c] / tr_df[f"init_{c}"]).groupby(tr_df.day).mean()
            # Map test days to ratios, fallback to last known ratio for days beyond train range
            mapped = te_df.day.map(tr_ratio)
            # If any day not in tr_ratio (shouldn't happen), use nearest
            if mapped.isna().any():
                # fill forward
                mapped = mapped.fillna(method="ffill").fillna(method="bfill")
            pred[:, j] = mapped.to_numpy() * te_df[f"init_{c}"].to_numpy()
        return pred
    
    res_mean = evaluate(mean_trajectory_predict, daily)
    print_results("0a: Mean trajectory ratio", res_mean)
    log_experiment("0a_mean_trajectory", "Mean trajectory ratio-to-day0 per target, scaled by batch init", res_mean)
    
    # --- Baseline B: ExtraTrees on init + feed features ---
    feature_cols = get_init_features(daily)
    
    def et_predict(tr_df, te_df):
        Xtr, ytr = prepare_xy(tr_df, feature_cols)
        Xte = te_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        pred = np.zeros((len(te_df), len(TARGETS)))
        for j, c in enumerate(TARGETS):
            mask = tr_df[c].notna()
            m = ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED
            )
            Xtr_j = tr_df.loc[mask, feature_cols].fillna(0).to_numpy(dtype=np.float32)
            ytr_j = tr_df.loc[mask, c].to_numpy(dtype=np.float32)
            pred[:, j] = m.fit(Xtr_j, ytr_j).predict(Xte)
        return pred
    
    res_et = evaluate(et_predict, daily)
    print_results("0b: ExtraTrees (init+feed features)", res_et)
    log_experiment("0b_extratrees", "ExtraTrees 500 trees on init state + feed/volume features", res_et)
    
    return {"mean_trajectory": res_mean["mean_mape"], "extratrees": res_et["mean_mape"]}


# ============================================================
# Experiment 1: Log targets
# ============================================================
def exp1_log_targets(daily):
    """Fit on log1p(target), predict and exponentiate."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Log targets")
    
    feature_cols = get_init_features(daily)
    best_mape = float("inf")
    best_results = None
    best_name = ""
    
    # 1a: log1p
    def log1p_predict(tr_df, te_df):
        pred = np.zeros((len(te_df), len(TARGETS)))
        for j, c in enumerate(TARGETS):
            mask = tr_df[c].notna()
            Xtr = tr_df.loc[mask, feature_cols].fillna(0).to_numpy(dtype=np.float32)
            ytr = np.log1p(tr_df.loc[mask, c].to_numpy(dtype=np.float32))
            Xte = te_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
            m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
            pred_log = m.fit(Xtr, ytr).predict(Xte)
            pred[:, j] = np.expm1(pred_log)
        return pred
    
    res = evaluate(log1p_predict, daily)
    print_results("1a: ExtraTrees + log1p target", res)
    log_experiment("1a_log1p_et", "ExtraTrees on log1p(target), back-transform via expm1", res)
    if res["mean_mape"] < best_mape:
        best_mape = res["mean_mape"]
        best_results = res
        best_name = "1a_log1p_et"
    
    # 1b: log(target) with clip
    def log_predict(tr_df, te_df):
        pred = np.zeros((len(te_df), len(TARGETS)))
        for j, c in enumerate(TARGETS):
            mask = tr_df[c].notna()
            Xtr = tr_df.loc[mask, feature_cols].fillna(0).to_numpy(dtype=np.float32)
            ytr = np.log(np.maximum(tr_df.loc[mask, c].to_numpy(dtype=np.float32), 1e-8))
            Xte = te_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
            m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
            pred_log = m.fit(Xtr, ytr).predict(Xte)
            pred[:, j] = np.maximum(np.exp(pred_log), 0)
        return pred
    
    res2 = evaluate(log_predict, daily)
    print_results("1b: ExtraTrees + log(target) w/ clip", res2)
    log_experiment("1b_log_et", "ExtraTrees on log(target) with 1e-8 clip, exp back-transform", res2)
    if res2["mean_mape"] < best_mape:
        best_mape = res2["mean_mape"]
        best_results = res2
        best_name = "1b_log_et"
    
    return best_name, best_results


# ============================================================
# Experiment 2: Ratio prediction
# ============================================================
def exp2_ratio_prediction(daily):
    """Predict target/init_target ratio instead of raw target."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Ratio prediction")
    
    train = daily[~daily.is_test].copy()
    feature_cols = get_init_features(daily)
    # Remove init_* from features since we're predicting ratios
    ratio_features = [c for c in feature_cols if not c.startswith("init_")]
    
    # 2a: Predict ratio, then multiply back
    def ratio_predict(tr_df, te_df):
        pred = np.zeros((len(te_df), len(TARGETS)))
        for j, c in enumerate(TARGETS):
            mask = tr_df[c].notna()
            Xtr = tr_df.loc[mask, ratio_features].fillna(0).to_numpy(dtype=np.float32)
            ytr = (tr_df.loc[mask, c] / tr_df.loc[mask, f"init_{c}"]).to_numpy(dtype=np.float32)
            Xte = te_df[ratio_features].fillna(0).to_numpy(dtype=np.float32)
            m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
            ratio_pred = m.fit(Xtr, ytr).predict(Xte)
            pred[:, j] = ratio_pred * te_df[f"init_{c}"].to_numpy()
        return pred
    
    res = evaluate(ratio_predict, daily)
    print_results("2a: Predict ratio-to-init with ET", res)
    log_experiment("2a_ratio_et", "ExtraTrees predicting target/init_target ratio", res)
    
    return "2a_ratio_et", res


# ============================================================
# Experiment 3: Model × loss sweep
# ============================================================
def exp3_sweep(daily, baseline_mean_mape):
    """Sweep over models, losses, representations. Apply plateau rule internally."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Model × loss sweep")
    
    train = daily[~daily.is_test].copy()
    base_features = get_init_features(daily)
    ratio_features = [c for c in base_features if not c.startswith("init_")]
    
    best_overall_mape = baseline_mean_mape
    best_overall_name = ""
    best_overall_results = None
    consecutive_small_gains = 0
    
    # We'll try these combos in a smart order (most promising first)
    # Representation: "raw" (already tested), "ratio" (exp2), "log1p"
    # Model: ET (fast, strong), HGB (gradient boosting)
    # Loss: MSE (default), weighted MSE (1/y²), Huber
    
    combos = [
        # (name, model_name, representation, loss_type, features)
        # Start with best from exp 1/2, then explore
        ("3a_et_raw_weighted", "et", "raw", "weighted", base_features),
        ("3b_et_raw_huber", "et", "raw", "huber", base_features),
        ("3c_et_ratio_weighted", "et", "ratio", "weighted", ratio_features),
        ("3d_et_ratio_huber", "et", "ratio", "huber", ratio_features),
        ("3e_hgb_raw_mse", "hgb", "raw", "mse", base_features),
        ("3f_hgb_raw_weighted", "hgb", "raw", "weighted", base_features),
        ("3g_hgb_ratio_mse", "hgb", "ratio", "mse", ratio_features),
        ("3h_et_log1p_mse", "et", "log1p", "mse", base_features),
    ]
    
    for name, model_name, repr_type, loss_type, features in combos:
        if consecutive_small_gains >= 2:
            print(f"  Skipping {name} — plateau rule triggered")
            continue
        
        def make_predict_fn(model_name, repr_type, loss_type, features):
            def predict_fn(tr_df, te_df):
                pred = np.zeros((len(te_df), len(TARGETS)))
                for j, c in enumerate(TARGETS):
                    mask = tr_df[c].notna()
                    Xtr = tr_df.loc[mask, features].fillna(0).to_numpy(dtype=np.float32)
                    ytr_orig = tr_df.loc[mask, c].to_numpy(dtype=np.float32)
                    Xte = te_df[features].fillna(0).to_numpy(dtype=np.float32)
                    
                    # Representation transform
                    if repr_type == "raw":
                        ytr = ytr_orig
                    elif repr_type == "ratio":
                        ytr = ytr_orig / tr_df.loc[mask, f"init_{c}"].to_numpy(dtype=np.float32)
                    elif repr_type == "log1p":
                        ytr = np.log1p(ytr_orig)
                    else:
                        ytr = ytr_orig
                    
                    # Model
                    if model_name == "et":
                        m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
                    elif model_name == "hgb":
                        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED)
                    else:
                        m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
                    
                    # Loss — for tree models, we approximate via sample_weight or target transform
                    if loss_type == "weighted":
                        # Weight = 1/y² approximates relative error
                        sample_weight = 1.0 / np.maximum(ytr_orig ** 2, 1e-8)
                        m.fit(Xtr, ytr, sample_weight=sample_weight)
                    elif loss_type == "huber":
                        # HGB has built-in Huber; ET doesn't — we pre-transform with a soft clip
                        if model_name == "hgb":
                            m.set_params(loss="huber")
                        m.fit(Xtr, ytr)
                    else:  # mse
                        m.fit(Xtr, ytr)
                    
                    pred_raw = m.predict(Xte)
                    
                    # Inverse transform
                    if repr_type == "log1p":
                        pred_raw = np.expm1(pred_raw)
                    elif repr_type == "ratio":
                        pred_raw = pred_raw * te_df[f"init_{c}"].to_numpy()
                    
                    pred[:, j] = pred_raw
                return pred
            return predict_fn
        
        pf = make_predict_fn(model_name, repr_type, loss_type, features)
        try:
            res = evaluate(pf, daily)
            print_results(name, res)
            log_experiment(name, f"{model_name} × {repr_type} × {loss_type}", res)
            
            improvement = best_overall_mape - res["mean_mape"]
            if improvement >= PLATEAU_THRESHOLD:
                consecutive_small_gains = 0
                best_overall_mape = res["mean_mape"]
                best_overall_name = name
                best_overall_results = res
                print(f"  >> New best! Improvement: {improvement:.4f}")
            else:
                consecutive_small_gains += 1
                print(f"  Small gain ({improvement:.4f}), plateau counter: {consecutive_small_gains}/2")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            log_experiment(name, f"{model_name} × {repr_type} × {loss_type} — ERROR", {"mean_mape": np.nan}, str(e))
    
    if best_overall_results is None:
        best_overall_name = "3_fallback"
        best_overall_results = {"mean_mape": baseline_mean_mape}
    
    return best_overall_name, best_overall_results


# ============================================================
# Experiment 4: Feed-aware features
# ============================================================
def exp4_feed_features(daily, best_features):
    """Add interaction features between feeds and cell state."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Feed-aware features")
    
    # Augment the best feature set with interaction features
    train = daily[~daily.is_test].copy()
    
    # Build augmented daily with interactions
    aug = daily.copy()
    FEEDS_SHORT = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
    
    for f in FEEDS_SHORT:
        # feed per current vcd/tcd (if known, or use init)
        aug[f"{f}_per_vcd"] = aug[f"{f}_next"] / np.maximum(aug.get("vcd", aug.init_vcd), 1e-6)
        aug[f"{f}_per_tcd"] = aug[f"{f}_next"] / np.maximum(aug.get("tcd", aug.init_tcd), 1e-6)
        # cumulative feed ratios
        aug[f"{f}_cum_ratio"] = aug[f"{f}_cum"] / np.maximum(aug[f"{f}_cum"].max(), 1e-6)  # normalize
        # time since last feed (approximate via feed_next > 0)
    
    # time_since_last_feed: days since previous feed event
    for f in FEEDS_SHORT:
        for bid, grp in aug.groupby("batch_id"):
            grp = grp.sort_values("day")
            last_feed_day = -999
            days_since = []
            for _, row in grp.iterrows():
                days_since.append(row.day - last_feed_day if last_feed_day >= 0 else 13)
                if row[f"{f}_next"] > 0:
                    last_feed_day = row.day
            aug.loc[grp.index, f"{f}_days_since"] = days_since
    
    extra = [c for c in aug.columns if any(p in c for p in ["_per_vcd", "_per_tcd", "_cum_ratio", "_days_since"])]
    aug_features = list(best_features) + extra
    
    def feed_aware_et(tr_df, te_df):
        pred = np.zeros((len(te_df), len(TARGETS)))
        for j, c in enumerate(TARGETS):
            mask = tr_df[c].notna()
            available = [col for col in aug_features if col in tr_df.columns and col in te_df.columns]
            Xtr = tr_df.loc[mask, available].fillna(0).to_numpy(dtype=np.float32)
            ytr = tr_df.loc[mask, c].to_numpy(dtype=np.float32)
            Xte = te_df[available].fillna(0).to_numpy(dtype=np.float32)
            m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
            pred[:, j] = m.fit(Xtr, ytr).predict(Xte)
        return pred
    
    res = evaluate(feed_aware_et, aug)
    print_results("4: Feed-aware features with ET", res)
    log_experiment("4_feed_aware_et", "ExtraTrees with feed×state interaction features", res)
    
    return "4_feed_aware_et", res


# ============================================================
# Experiment 5: Step-ahead (delta prediction)
# ============================================================
def exp5_step_ahead(daily):
    """Predict delta = target(t+1) - target(t), roll out for full trajectory."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Step-ahead delta prediction")
    
    train = daily[~daily.is_test].copy()
    base_features = get_init_features(daily)
    
    # Build step-ahead pairs: for each batch, predict day t+1 from day t
    def build_step_data(df):
        """Return X (features at t), y (delta to t+1), and metadata for roll-out."""
        X_list, y_list, meta_list = [], [], []
        for bid, grp in df.groupby("batch_id"):
            grp = grp.sort_values("day")
            for t in range(13):  # predict days 1..13 from day t
                row_t = grp[grp.day == t]
                row_t1 = grp[grp.day == t + 1]
                if len(row_t) == 0 or len(row_t1) == 0:
                    continue
                row_t = row_t.iloc[0]
                row_t1 = row_t1.iloc[0]
                
                feas = {c: row_t[c] for c in base_features if c in row_t.index}
                # Add feed_next from row_t (feeds between t and t+1)
                for c in row_t.index:
                    if "_next" in c and c in base_features:
                        feas[c] = row_t[c]
                
                X_list.append(feas)
                # delta to t+1
                delta = []
                for c in TARGETS:
                    if pd.notna(row_t1.get(c)):
                        delta.append(row_t1[c] - row_t[c])
                    else:
                        delta.append(np.nan)
                y_list.append(delta)
                meta_list.append({"batch_id": bid, "day_from": t, "day_to": t + 1})
        
        Xdf = pd.DataFrame(X_list)
        yarr = np.array(y_list)
        meta = pd.DataFrame(meta_list)
        return Xdf, yarr, meta
    
    X_step, y_step, meta_step = build_step_data(train)
    # Only train on rows where delta target is known
    valid_mask = ~np.isnan(y_step).any(axis=1)
    X_step = X_step[valid_mask].reset_index(drop=True)
    y_step = y_step[valid_mask]
    meta_step = meta_step[valid_mask].reset_index(drop=True)
    
    print(f"  Step-ahead training pairs: {len(X_step)}")
    
    # CV for one-step delta prediction
    step_features = [c for c in X_step.columns if X_step[c].dtype in ('float64', 'int64', 'float32', 'int32', 'bool')]
    
    # One-step CV
    train_batches_arr = meta_step.batch_id.to_numpy()
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    fold_mapes = []
    
    for tr_i, va_i in gkf.split(np.arange(len(meta_step)), groups=train_batches_arr):
        tr_bids = set(train_batches_arr[tr_i])
        va_bids = set(train_batches_arr[va_i])
        tr_idx = np.where(np.isin(train_batches_arr, list(tr_bids)))[0]
        va_idx = np.where(np.isin(train_batches_arr, list(va_bids)))[0]
        
        Xtr = X_step.loc[tr_idx, step_features].fillna(0).to_numpy(dtype=np.float32)
        ytr = y_step[tr_idx]
        Xva = X_step.loc[va_idx, step_features].fillna(0).to_numpy(dtype=np.float32)
        yva = y_step[va_idx]
        
        pred_delta = np.zeros((len(va_idx), 4))
        for j in range(4):
            m = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
            pred_delta[:, j] = m.fit(Xtr, ytr[:, j]).predict(Xva)
        
        from dtb_eval import mape_per_target
        fold_mape = np.mean([mape_per_target(yva[:, j], pred_delta[:, j]) for j in range(4)])
        fold_mapes.append(fold_mape)
    
    one_step_mape = float(np.mean(fold_mapes))
    print(f"  One-step delta MAPE: {one_step_mape:.4f} (folds: {[f'{m:.4f}' for m in fold_mapes]})")
    
    # Roll-out evaluation: start from day 0, predict day 1 delta, add to get day 1, feed as day 0 for day 2, etc.
    roll_mapes = []
    for tr_i, va_i in gkf.split(np.arange(len(meta_step)), groups=train_batches_arr):
        va_bids = set(train_batches_arr[va_i])
        tr_idx = tr_i  # indices into meta_step for training
        for va_bid in va_bids:
            grp = train[train.batch_id == va_bid].sort_values("day")
            if len(grp) < 14:
                continue
            
            row0 = grp[grp.day == 0]
            if len(row0) == 0:
                continue
            # Start from day 0 true values
            current = row0[TARGETS].to_numpy()[0].copy()
            preds = [current.copy()]
            trues = []
            
            for day_t in range(13):
                row_t = grp[grp.day == day_t]
                row_t1 = grp[grp.day == day_t + 1]
                if len(row_t) == 0 or len(row_t1) == 0:
                    break
                row_t = row_t.iloc[0]
                row_t1 = row_t1.iloc[0]
                
                feas = {c: row_t[c] for c in step_features if c in row_t.index}
                for c in row_t.index:
                    if "_next" in c and c in step_features:
                        feas[c] = row_t[c]
                
                # Update current state in features for vcd/tcd if needed
                for j, c in enumerate(TARGETS):
                    if c in feas:
                        feas[c] = current[j]
                
                X_row = pd.DataFrame([feas])[step_features].fillna(0).to_numpy(dtype=np.float32)
                
                # Predict delta using model trained on all other batches
                delta_pred = np.zeros(4)
                for j in range(4):
                    m = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
                    m.fit(X_step.loc[tr_idx, step_features].fillna(0).to_numpy(dtype=np.float32), y_step[tr_idx, j])
                    delta_pred[j] = m.predict(X_row)[0]
                
                current = current + delta_pred
                current = np.clip(current, 0, None)
                current[1] = max(current[1], current[0])  # tcd >= vcd
                preds.append(current.copy())
                trues.append(row_t1[TARGETS].to_numpy())
            
            if trues:
                trues_arr = np.array(trues)
                preds_arr = np.array(preds[1:])  # day 1..13 predictions
                from dtb_eval import mape_per_target
                bm = np.mean([mape_per_target(trues_arr[:, j], preds_arr[:, j]) for j in range(4)])
                roll_mapes.append(bm)
    
    roll_mape = float(np.mean(roll_mapes)) if roll_mapes else float("inf")
    print(f"  Roll-out MAPE: {roll_mape:.4f}")
    
    results = {
        "one_step_mape": one_step_mape,
        "roll_out_mape": roll_mape,
        "mean_mape": roll_mape,  # use roll-out as the real metric
        "vcd_mape": np.nan, "tcd_mape": np.nan, "substrate_mape": np.nan, "byproduct_mape": np.nan,
        "per_fold_mapes": fold_mapes,
        "fold_std": float(np.std(fold_mapes)),
        "worst_batches": [],
    }
    
    print_results(f"5: Step-ahead (roll-out MAPE)", results)
    log_experiment("5_step_ahead", f"Delta prediction, one-step MAPE={one_step_mape:.4f}, roll-out MAPE={roll_mape:.4f}", results)
    
    return "5_step_ahead", results


# ============================================================
# Main: run all experiments
# ============================================================
def main():
    daily = pd.read_parquet(DAILY)
    print(f"Loaded daily table: {len(daily)} rows")
    print(f"Train batches: {daily[~daily.is_test].batch_id.nunique()}, Test batches: {daily[daily.is_test].batch_id.nunique()}")
    
    # Track best across all experiments
    best_mean_mape = float("inf")
    best_exp_name = ""
    consecutive_plateau = 0
    
    # Experiment 0: Baseline
    exp0_results = exp0_baseline(daily)
    best_mean_mape = exp0_results["extratrees"]
    best_exp_name = "0b_extratrees"
    print(f"\n>>> Best so far: {best_exp_name} with mean MAPE {best_mean_mape:.4f}")
    
    def update_best(exp_name, exp_res):
        nonlocal best_mean_mape, best_exp_name, consecutive_plateau
        improvement = best_mean_mape - exp_res["mean_mape"]
        # Always track the actual best, regardless of plateau
        if improvement > 0:
            best_mean_mape = exp_res["mean_mape"]
            best_exp_name = exp_name
        # Plateau rule: only counts improvements (positive deltas) below threshold
        if improvement > 0 and improvement < PLATEAU_THRESHOLD:
            consecutive_plateau += 1
            print(f">>> Small improvement {consecutive_plateau}/2 (Δ={improvement:.4f}), new best: {best_exp_name} ({best_mean_mape:.4f})")
        elif improvement >= PLATEAU_THRESHOLD:
            consecutive_plateau = 0
            print(f">>> Significant improvement! New best: {best_exp_name} with mean MAPE {best_mean_mape:.4f} (Δ={improvement:.4f})")
        else:
            print(f">>> No improvement (Δ={improvement:.4f}), best remains: {best_exp_name} ({best_mean_mape:.4f})")
    
    # Experiment 1: Log targets
    exp1_name, exp1_res = exp1_log_targets(daily)
    update_best(exp1_name, exp1_res)
    
    # Experiment 2: Ratio prediction
    exp2_name, exp2_res = exp2_ratio_prediction(daily)
    update_best(exp2_name, exp2_res)
    
    if consecutive_plateau >= 2:
        print(f"\n*** PLATEAU: stopping after experiment 2 (2 consecutive small improvements < {PLATEAU_THRESHOLD}) ***")
        print(f"*** Best model: {best_exp_name}, mean MAPE: {best_mean_mape:.4f} ***")
        return
    
    # Experiment 3: Model × loss sweep
    exp3_name, exp3_res = exp3_sweep(daily, best_mean_mape)
    update_best(exp3_name, exp3_res)
    
    if consecutive_plateau >= 2:
        print(f"\n*** PLATEAU: stopping after experiment 3 (2 consecutive small improvements < {PLATEAU_THRESHOLD}) ***")
        print(f"*** Best model: {best_exp_name}, mean MAPE: {best_mean_mape:.4f} ***")
        return
    
    # Experiment 4: Feed-aware features
    best_features = get_init_features(daily)
    exp4_name, exp4_res = exp4_feed_features(daily, best_features)
    update_best(exp4_name, exp4_res)
    
    if consecutive_plateau >= 2:
        print(f"\n*** PLATEAU: stopping after experiment 4 (2 consecutive small improvements < {PLATEAU_THRESHOLD}) ***")
        print(f"*** Best model: {best_exp_name}, mean MAPE: {best_mean_mape:.4f} ***")
        return
    
    # Experiment 5: Step-ahead
    exp5_name, exp5_res = exp5_step_ahead(daily)
    update_best(exp5_name, exp5_res)
    
    print(f"\n{'='*60}")
    print(f"*** FINAL BEST: {best_exp_name} with mean MAPE {best_mean_mape:.4f} ***")


if __name__ == "__main__":
    main()

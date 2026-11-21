# Challenge 1 (DTB) — Plan

Goal: predict `vcd`, `tcd`, `substrate`, `byproduct` for 12 test batches (days 1–13) from their
day-0 state and known feed/volume schedule. Leaderboard metric: **mean MAPE** over the 4 targets.

> Rule: never submit to the leaderboard unless a team member explicitly says so.

## Where we are

- `src/dtb_checks.py` — data checks. Data is clean: 61 batches × 14 daily samples, no duplicates,
  no negatives, `tcd ≥ vcd` everywhere. NaNs are structural (rows are either measurement,
  to-predict, or a single feed event) — reshape, don't impute.
- `src/dtb_solve.py` — first model (ExtraTrees on day-0 state + feed/volume features).
  CV normalized RMSE 0.259 vs baseline 0.302. One leaderboard attempt: mean MAPE 8.03.

**Leaderboard MAPE formula** (confirmed from API): per-target `mean(|true - pred| / |true|) * 100`,
then average the four per-target MAPEs. No epsilon trick — denominators are raw |true|.
All true values in the data are > 0 for measurement rows, but predictions can get arbitrarily
close to zero so we must guard against division by zero in our own eval.

## Steps

### 1. Daily table (`src/dtb_data.py`)
One row per batch × sampling day (14 rows per batch). Columns:
- `batch_id`, `elapsed_time`, `day` (integer, `round(elapsed_time)`)
- `volume`, `dvol_prev`, `dvol_next`, `vol_rel_to_t0` (= `volume / day0_volume`)
- Targets: `vcd`, `tcd`, `substrate`, `byproduct` (NaN for test days to predict)
- Day-0 state: `init_vcd`, `init_tcd`, `init_substrate`, `init_byproduct`, `init_volume`
- Per feed: `{feed}_prev` (dosed between previous sample and this one), `{feed}_next` (dosed
  between this sample and next), `{feed}_cum` (total dosed up to this sample time),
  `{feed}_total` (total per batch). Also `{feed}_next_per_vol` (= `{feed}_next / volume` at
  this sample time) for all four feeds — feeds scale with current reactor state and `dtb_checks`
  confirmed strong correlation of `feed_b` with `vcd * volume`.
- `time_to_end` (= total batch duration - elapsed_time), for models that need to know how
  much time is left.

Saved to `outputs/dtb_daily.parquet`. Every model reads this; no one re-parses the raw CSV.

**Specification to verify before proceeding:**
- Every batch has exactly 14 rows (days 0–13).
- Train batches have targets for all 14 days, test batches only for day 0.
- `day` is integer 0–13 captured via `round(elapsed_time)`; assert `max(|elapsed_time - day|) < 0.01`.
- Per-feed window logic handles edge cases: day 0 has no `_prev` (→ 0), day 13 has `_next` = NaN
  (→ 0 in features). Document these choices.

### 2. Evaluation harness (`src/dtb_eval.py`) — the "checker"

#### 2a. MAPE computation
```python
def mape(y_true, y_pred, eps=1e-8):
    """Per-target MAPE = mean(|true - pred| / max(|true|, eps)) * 100."""
    return np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100

def evaluate(model_predict_fn) -> dict:
    """Returns dict with per-target MAPE + mean_mape, plus per-fold breakdown."""
```

Use `max(|true|, 1e-8)` in the denominator so our CV eval never divides by zero.
The leaderboard doesn't have this issue (true values are non-zero) but our CV
folds might see near-zero predictions hitting near-zero true values.

#### 2b. Cross-validation
- **GroupKFold, 5 folds, split by batch, fixed `random_state=42`**.
- For each fold, train on 4/5 of batches, predict on held-out 1/5.
- Report: per-target MAPE and mean MAPE (average of the four).
- Also report **per-fold** mean MAPE so we can see variance.

#### 2c. Experiment log
- Append one row to `outputs/experiments_dtb.csv`:
  `timestamp, experiment_name, short_description, vcd_mape, tcd_mape, substrate_mape, byproduct_mape, mean_mape, per_fold_mapes, notes`
- `per_fold_mapes` is a list of the 5 fold-level mean MAPEs so we can check stability.
- Every idea goes through this harness. **The number decides between ideas, not opinions.**

#### 2d. Roll-out evaluation (for step-ahead models)
In addition to the standard one-step CV above, models that predict day t+1 from day t
must also pass a **full roll-out test**: start from day-0 state, predict day 1, use that
prediction as input for day 2, etc. through day 13. Report roll-out MAPE separately.
A step-ahead model that looks great in one-step CV but terrible on roll-out is not usable.

### 3. Experiments, in order

| # | Name | What | Success criterion |
|---|------|------|-------------------|
| 0 | **Baseline** | Mean-trajectory per day (ratio-to-day0 mean curve, scaled per batch) + current ExtraTrees ensemble from `dtb_solve.py`, both re-scored in MAPE via the new harness | Honest starting point; expect ET near 8.0 mean MAPE |
| 1 | **Log targets** | Fit on `log1p(target)`, predict and exponentiate. Also try `log(target)` with clip — MAPE penalizes relative error so log-space MSE aligns better | Improvement over row 0 on at least 3 of 4 targets |
| 2 | **Per-target models** | Auto-select best model per target: try Ridge, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting. Use Optuna with 50 trials per model type, GroupKFold inner CV | Meaningful gain on the worst-performing target from row 1 |
| 3 | **Ratio prediction** | Predict `target / init_target` (ratio to day-0 value). This removes batch-to-batch scale differences — a batch that starts with 2× cells will have ~2× trajectory. Also try ratio to population median curve ("detrended" prediction) | Lower per-fold variance than row 2 |
| 4 | **Feed-aware features** | Add interaction features: `feed_{f}_per_vcd`, `feed_{f}_per_tcd`, cumulative feed ratios, time-since-last-feed. Feed timing matters — a dose at day 2 hits differently than at day 10 | Improvement on substrate (the target most affected by feeds) |
| 5 | **Step-ahead** | Model `target(t+1) - target(t)` (delta) from state at t + feeds between t and t+1. Roll out for test batches. Compare one-step vs roll-out MAPE | Roll-out MAPE within 10% of one-step; if error explodes, abandon |
| 6 | **Hybrid / ODE** | Simple mass-balance ODE: `dVCD/dt = μ*VCD - kd*VCD`, `dSubstrate/dt = -Y*μ*VCD`, etc. Fit parameters (μ, kd, Y) per batch or per phase via least squares on train. Use ODE solution as features alongside ML, or as a prior mean for a GP | Only if rows 1–5 plateau (gain < 0.3 MAPE for 2 consecutive experiments) |

**Constraints applied to every prediction:** clip ≥ 0, enforce `tcd ≥ vcd`.

**Plateau rule:** if two consecutive experiments improve mean MAPE by less than 0.2, stop
and move to review + submission. Don't chase hundredths — the leaderboard variance likely
exceeds that.

### 4. Review — the "critic"
Before trusting a result that beats the best so far:

1. **Leakage check:** verify no test batches or future-information features appear in training.
   Run `git diff` and manually inspect the feature columns for anything that couldn't be
   known at prediction time.
2. **Per-fold check:** if one fold is much worse than others, investigate which batch(es)
   drive that error.
3. **Outlier sensitivity:** flagged points from `dtb_checks.py` (Batch14 d1 vcd, Batch29/34
   d13 byproduct, Batch42 substrate) — if a single batch dominates error, test dropping it
   as sensitivity, but default to keeping it (they look like real process behaviour).
4. **Prediction distribution check:** compare predicted vs true distributions (mean, std per day).
   Should roughly align. Systematic bias (always predicting too low on substrate, for example)
   is fixable with a bias-correction layer.

### 5. Submission
- Write `outputs/submission_dtb.csv` from the best CV model, in the exact format of
  `bioreaction_template.csv` (columns: `batch_id, elapsed_time, vcd, tcd, substrate, byproduct`).
- Verify: row count matches template, `batch_id` and `elapsed_time` match exactly, no NaNs,
  all values ≥ 0, `tcd ≥ vcd` everywhere.
- A team member decides whether to submit.

## Not doing (unless evidence changes)

- **Multi-agent / parallel infrastructure:** overhead > benefit for one small dataset.
  If two team members need to run independent experiments simultaneously, they can just
  run separate scripts against the shared `dtb_daily.parquet` and `dtb_eval.py`.
- **Blind outlier removal:** flagged points look like real process behaviour.
  Keep them unless sensitivity analysis (step 4.3) proves otherwise.
- **Deep learning / LSTM / transformers:** 61 batches × 14 timepoints is too little
  data for sequence models to beat tree-based methods. Keep the option open only if
  the ODE row (6) motivates a neural hybrid, but even then a small MLP with ODE features
  is more appropriate than a sequence model.
- **Uncertainty quantification / conformal prediction:** not asked for by the challenge.
  Only add if we need tie-breaking between two models with identical MAPE.

# Challenge 1 (DTB) — Plan

Goal: predict `vcd`, `tcd`, `substrate`, `byproduct` for 12 test batches (days 1–13) from their
day-0 state and known feed/volume schedule. Leaderboard metric: **mean MAPE** over the 4 targets.

**From the challenge brief** ("Decode the Bioreactor"): fed-batch cell culture data with explicit
**nonlinear growth dynamics, coupled metabolism, and feeding effects**; the brief encourages
literature approaches (growth kinetics, mass-balance) and says **interpretability/insight into the
process is valued**, not just the leaderboard score — a pure black-box winner is an incomplete
answer. The brief's "dead cell density (DCD)" maps onto our columns as `dcd = tcd - vcd`.

> Rule: never submit to the leaderboard unless a team member explicitly says so.

## Where we are

- `src/dtb_checks.py` — data checks. Data is clean: 61 batches × 14 daily samples, no duplicates,
  no negatives, `tcd ≥ vcd` everywhere. NaNs are structural (rows are either measurement,
  to-predict, or a single feed event) — reshape, don't impute.
- `src/dtb_solve.py` — first model (ExtraTrees on day-0 state + feed/volume features).
  CV normalized RMSE 0.259 vs baseline 0.302. One leaderboard attempt: mean MAPE 8.03.
- `src/dtb_viz.py` — exploratory plots (one PNG per view, not a notebook, so they're easy to
  regenerate and diff): `outputs/dtb_viz_trajectories.png` (all 61 batches, one line each, per
  target), `outputs/dtb_viz_volume.png`, `outputs/dtb_viz_feeds.png` (feed timing/dose). See
  "What the data looks like" below for what they show.

## What the data looks like

Each batch is a **14-point daily time series** (day 0–13) for `vcd`, `tcd`, `substrate`,
`byproduct`, plus a known `volume` schedule and four feed streams. From `dtb_viz.py`:

- **vcd / tcd** (cell density): classic batch-culture shape — exponential-ish growth to a peak
  around day 7–8 (~25–35), then a decline/plateau phase with batch-to-batch noise. `tcd` tracks
  `vcd` closely (viability stays high) but the two diverge slightly in the decline phase.
- **substrate**: steadily consumed (drops ~4.5 → ~2 by day 4), but **not monotonic** — a visible
  bump back up around day 5–7 in most batches (re-feed effect), then a sharp drop around day
  10–11. A single growth/decay curve won't fit this; it's feed-driven and has at least 3 regimes.
- **byproduct**: rises during growth (day 0–6, peak ~2.5), then falls as cells die off/dilute —
  mirrors the growth/death phases of vcd/tcd rather than accumulating monotonically.
- **volume**: nearly identical ramp across *all* batches, train and test alike — it's close to a
  fixed schedule, so by itself it carries little signal, but it's needed to convert feed doses
  into concentrations.
- **feeds**: `feed_c` is dosed almost daily throughout the run; `feed_b` roughly daily with
  growing dose size over time; `feed_substrate` fires only a handful of times (later, larger
  doses); `feed_d` is rare — about 3 events total, clustered around days 5, 8, 10.
- **train vs test day 0**: test batches' day-0 values sit inside the same range as train, but
  trajectories fan out a lot by day 7–13 — day-0 state alone is a weak predictor; the feed
  schedule (which differs per batch) is likely doing real work.

**Implication for the experiment plan:** the non-monotonic, multi-phase substrate/byproduct
curves are why the mechanistic/phase-aware ideas in §6 exist — a single smooth regression target
may not capture the day-5-7 substrate bump. Re-run `dtb_viz.py` (or add a per-model prediction
overlay to it) whenever a model's residuals look systematically off on a particular phase.

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
- Derived, interpretable quantities (matches the brief's "DCD" language): `dcd = tcd - vcd`
  (dead cell density) and `viability = vcd / tcd`. Not new information, but worth reporting
  explicitly since the brief frames the process in these terms.
- Per feed: `{feed}_prev` (dosed between previous sample and this one), `{feed}_next` (dosed
  between this sample and next), `{feed}_cum` (total dosed up to this sample time).
  Also `{feed}_next_per_vol` (= `{feed}_next / volume` at this sample time) for all four
  feeds — feeds scale with current reactor state and `dtb_checks` confirmed strong
  correlation of `feed_b` with `vcd * volume`.
  **No `{feed}_total`:** the total feed for an entire batch is future information at
  prediction time (you don't know on day 5 how much feed will be added on day 12).
  `{feed}_cum` captures everything known up to the current time without leakage.
- `time_to_end` (= total batch duration - elapsed_time), for models that need to know how
  much time is left.

Saved to `outputs/dtb_daily.parquet`. Every model reads this; no one re-parses the raw CSV.

**On `dvol_next`:** this column encodes the volume change between the current sample and the
next one. This is safe to include for test batches because the volume schedule is nearly
identical across *all* batches, train and test alike (confirmed in `dtb_viz.py` — "nearly
identical ramp across all batches, train and test alike"). The volume at day t+1 is effectively
known from the fixed schedule. If this assumption ever breaks (e.g., a future data release
introduces batch-specific volume deviations), drop `dvol_next` for test rows.

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

Use the **exact leaderboard formula** — `|true - pred| / |true|` with **no epsilon** —
so our CV numbers are directly comparable to the leaderboard. Run a pre-check: assert
all training true values are > 0.001 (they are — `dtb_checks.py` confirms zero true values
are absent from measurement rows). If any true value is exactly zero, fall back to
the epsilon-guarded version and document the discrepancy, but as of the current data
this is a non-issue. Since predictions can be arbitrarily close to zero, guard against
`pred / 0` (division by zero in the *prediction* side never happens; it's `true` in
the denominator). A `max(|true|, 1e-12)` guard costs nothing and handles edge cases
if the data changes in future releases.

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
| 2 | **Ratio prediction** | Predict `target / init_target` (ratio to day-0 value). This removes batch-to-batch scale differences — a batch that starts with 2× cells will have ~2× trajectory. Also try ratio to population median curve ("detrended" prediction) | Lower per-fold variance than row 1; ratio representation is now a known quantity before the sweep in row 3 |
| 3 | **Model × loss sweep** | Run the queued combinations in §3a: GBM / ExtraTrees × {log1p, weighted-MSE, Huber} loss × {raw, ratio-to-day0, log} representation (bounded — see §3a for why it stays small) | Meaningful gain on the worst-performing target from rows 1–2; pick winner by CV mean MAPE |
| 4 | **Feed-aware features** | Add interaction features: `feed_{f}_per_vcd`, `feed_{f}_per_tcd`, cumulative feed ratios, time-since-last-feed. Feed timing matters — a dose at day 2 hits differently than at day 10 | Improvement on substrate (the target most affected by feeds) |
| 5 | **Step-ahead** | Model `target(t+1) - target(t)` (delta) from state at t + feeds between t and t+1. Roll out for test batches. Compare one-step vs roll-out MAPE | Roll-out MAPE within 10% of one-step; if error explodes, abandon |

**Constraints applied to every prediction:** clip ≥ 0, enforce `tcd ≥ vcd`.

**Plateau rule:** if two consecutive experiments improve mean MAPE by less than 0.2, stop
and move to review + submission. Don't chase hundredths — the leaderboard variance likely
exceeds that. This rule also applies *inside* §3a's sweep — see its last line.

### 3a. Models and losses queued for experiment 3

Deliberately kept small and low-risk (see §6 for the bigger, riskier candidate list that's
parked rather than queued):

- **Models:** gradient-boosted trees (HistGradientBoostingRegressor / LightGBM / XGBoost, with
  monotonic constraints on physically-monotonic features such as cumulative feed — verify the
  library actually supports the constraint direction you want before relying on it) and
  ExtraTrees/RandomForest (already validated in `dtb_solve.py`).
- **Losses:** `log1p(target)` + MSE (experiment 1), weighted MSE with weight `= 1/true²`
  (approximates squared relative error with no back-transform bias), and Huber loss (robust to
  the flagged outlier batches: Batch14 d1 vcd, Batch29/34 d13 byproduct).
- **Representations:** raw, ratio-to-day0, log. Ratio is already validated by
  experiment 2 before this sweep starts, so it's a known-good baseline here rather
  than an unknown.

That's 2 models × 3 losses × 3 representations = 18 runs maximum, each logged to
`outputs/experiments_dtb.csv`. **Apply the plateau rule inside this sweep too:** once a loss or
representation choice stops moving the number (< 0.2 MAPE for 2 consecutive runs), stop varying
it and fix it for the rest of the sweep — don't run all 18 blind.

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
- **Uncertainty quantification / conformal prediction:** not asked for by the challenge.
  Only add if we need tie-breaking between two models with identical MAPE.

## 6. High-risk / high-effort track (parked — revisit only after §3a plateaus)

These are real, worthwhile ideas — several directly match what the brief asks for — but each
has an open question or a cost that makes it the wrong place to start. Keep them here rather
than in the main flow so §3 stays a clean, executable checklist. Escalate into one of these
**one at a time**, re-applying the plateau rule after each, instead of committing to all of them
or to a single giant grid search.

- **Mechanistic ODE** (Monod growth `dVCD/dt = μ(S)·VCD − kd·VCD` with `μ(S) = μmax·S/(Ks+S)`,
  substrate consumption `dS/dt = −(1/Yxs)·μ·VCD` + maintenance, byproduct via Luedeking-Piret
  `dP/dt = α·dVCD/dt + β·VCD`), parameters fit per batch by least squares. Gives the kinetic
  parameters (`μmax`, `Ks`, `Yxs`, `α`, `β`) as the interpretability deliverable the brief asks
  for. **Unresolved:** test batches have no trajectory to fit against (day 0 only) — this plan
  doesn't yet specify how test-batch parameters get produced. Needs a second-stage regression
  (day-0 state → fitted kinetic params, learned on train) before it can generate a submission at
  all, not just train-batch insight. Effort: medium-high.
- **Hybrid: ODE trajectory as a GBM feature/prior, GBM learns the residual.** Best shot at the
  non-monotonic substrate bump (day 5–7). Must fit the ODE on train-only data per CV fold to
  avoid leakage — and the same test-batch-parameter gap above applies at inference time.
  Effort: medium-high.
- **Gaussian Process with a composite/changepoint kernel**, to handle the growth→bump→crash
  regimes (a plain stationary RBF/Matern kernel will underfit the bump). This isn't a library
  default — it's custom kernel code, so treat it as medium-high effort, not medium.
- **Piecewise / phase-aware regressors** (separate regressor per growth/decline phase).
  **Leakage risk:** the phase must be classified from `day` and/or already-predicted history
  only — never from the target's own slope on days not yet predicted, or it's circular. Also,
  predicting the phase boundary itself for test batches is a sub-problem that needs its own
  validation.
- **Custom approximate-MAPE training objective** for LightGBM/XGBoost. Theoretically the
  best-aligned loss with the leaderboard metric, but `|y−ŷ|/y` is non-smooth at 0 and its second
  derivative is 0 — a naive gradient/hessian breaks the leaf-weight update. Needs a smoothed
  surrogate (e.g. pseudo-Huber in relative space) to be usable at all; this is real
  implementation work, not a one-line `objective=`. Try §3a's log1p/weighted-MSE losses first —
  they get most of the benefit for a fraction of the cost.
- **Sequence models (LSTM/GRU/transformer):** 61 sequences × 14 steps is too little data for
  these to generalize. Kept here only as a note in case the ODE track above motivates a small
  neural hybrid later — not a plan item on its own.

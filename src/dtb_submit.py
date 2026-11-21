"""Generate submission CSV using the best model (4h: feed-aware + extended timing + per-target weighted ET)."""
import sys, os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, os.path.dirname(__file__))
from dtb_eval import TARGETS, DAILY, OUT_DIR

SEED = 42
DAILY_PATH = DAILY
TEMPLATE = os.environ.get("DTB_DATA", "/home/renku/work/input-data/Project 1 - DTB/") + "bioreaction_template.csv"

daily = pd.read_parquet(DAILY_PATH)
tpl = pd.read_csv(TEMPLATE, dtype={"batch_id": str})

# Build augmented features (same as 4h)
FEEDS_SHORT = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']
aug = daily.copy()

for f in FEEDS_SHORT:
    aug[f'{f}_per_vcd'] = aug[f'{f}_next'] / np.maximum(aug.get('vcd', aug.init_vcd), 1e-6)
    aug[f'{f}_per_tcd'] = aug[f'{f}_next'] / np.maximum(aug.get('tcd', aug.init_tcd), 1e-6)
    aug[f'{f}_cum_ratio'] = aug[f'{f}_cum'] / np.maximum(aug[f'{f}_cum'].max(), 1e-6)

for f in FEEDS_SHORT:
    for bid, grp in aug.groupby('batch_id'):
        grp = grp.sort_values('day')
        last_feed_day = -999
        days_since = []
        just_fed = []
        for _, row in grp.iterrows():
            ds = row.day - last_feed_day if last_feed_day >= 0 else 13
            days_since.append(ds)
            just_fed.append(1 if ds <= 1 else 0)
            if row[f'{f}_next'] > 0:
                last_feed_day = row.day
        aug.loc[grp.index, f'{f}_days_since'] = days_since
        aug.loc[grp.index, f'{f}_just_fed'] = just_fed

for f in FEEDS_SHORT:
    aug[f'{f}_cum_frac'] = aug[f'{f}_cum'] / np.maximum(aug.groupby('batch_id')[f'{f}_cum'].transform('max'), 1e-6)

# Feature selection
extra = [c for c in aug.columns if any(p in c for p in ['_per_vcd', '_per_tcd', '_cum_ratio', '_cum_frac', '_days_since', '_just_fed'])]
exclude = set(TARGETS + ['batch_id', 'elapsed_time', 'is_test', 'dcd', 'viability'])
base_features = [c for c in aug.columns if c not in exclude and aug[c].dtype in ('float64', 'int64', 'bool', 'float32', 'int32')]
all_features = base_features + [e for e in extra if e not in base_features]

# Train/test split
train_df = aug[~aug.is_test].copy()
test_df = aug[aug.is_test].copy()

# Predict on all test rows (days 0-13)
pred = np.zeros((len(test_df), len(TARGETS)))
for j, c in enumerate(TARGETS):
    mask = train_df[c].notna()
    Xtr = train_df.loc[mask, all_features].fillna(0).to_numpy(dtype=np.float32)
    ytr = train_df.loc[mask, c].to_numpy(dtype=np.float32)
    Xte = test_df[all_features].fillna(0).to_numpy(dtype=np.float32)
    
    if c in ['vcd', 'tcd']:
        m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
        m.fit(Xtr, ytr)
    else:
        w = 1.0 / np.maximum(ytr, 1e-6)
        m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
        m.fit(Xtr, ytr, sample_weight=w)
    pred[:, j] = m.predict(Xte)

# Store predictions
test_df = test_df.copy()
for j, c in enumerate(TARGETS):
    test_df[c] = pred[:, j]

# Physical constraints
test_df['vcd'] = np.clip(test_df['vcd'], 0, None)
test_df['tcd'] = np.clip(test_df['tcd'], 0, None)
test_df['substrate'] = np.clip(test_df['substrate'], 0, None)
test_df['byproduct'] = np.clip(test_df['byproduct'], 0, None)
test_df['tcd'] = np.maximum(test_df['tcd'], test_df['vcd'])

# Map onto template: test days 1-13 only, day 0 is already known
sub = tpl[['batch_id', 'elapsed_time']].copy()
for c in TARGETS:
    sub[c] = np.nan

for i, r in sub.iterrows():
    cand = test_df[test_df.batch_id == r.batch_id]
    k = (cand.elapsed_time - r.elapsed_time).abs().idxmin()
    time_diff = abs(cand.loc[k, 'elapsed_time'] - r.elapsed_time)
    assert time_diff < 1e-4, f"Time mismatch for {r.batch_id} at {r.elapsed_time}: nearest is {cand.loc[k, 'elapsed_time']} (diff={time_diff})"
    sub.loc[i, TARGETS] = cand.loc[k, TARGETS].to_numpy(dtype=float)

# Final verification
assert sub[TARGETS].notna().all().all(), "NaNs in submission"
assert len(sub) == len(tpl), f"Row count mismatch: {len(sub)} vs {len(tpl)}"
assert (sub[TARGETS] >= 0).all().all(), "Negative values in submission"
assert (sub['tcd'] >= sub['vcd']).all(), "tcd < vcd violation"

out_path = f"{OUT_DIR}/submission_dtb.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
print(f"Rows: {len(sub)}, columns: {list(sub.columns)}")
print(sub.head())
print(f"\nPrediction ranges:")
print(sub[TARGETS].describe().round(3).to_string())

"""Step 1: build the daily table (one row per batch×day) from raw CSV.
Saves outputs/dtb_daily.parquet — every downstream script reads this.
"""
import os
import numpy as np
import pandas as pd

DATA = os.environ.get("DTB_DATA", "/home/renku/work/input-data/Project 1 - DTB/")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]

df = pd.read_csv(DATA + "bioreaction_remaining.csv")
tpl = pd.read_csv(DATA + "bioreaction_template.csv", dtype={"batch_id": str})
test_ids = set(tpl.batch_id)


def build_rows(raw_df):
    rows = []
    for bid, b in raw_df.groupby("batch_id", sort=False):
        b = b.sort_values("elapsed_time").reset_index(drop=True)
        t0 = b.iloc[0]
        # Sampling rows: either measurement or to_predict (both carry volume + targets info)
        samp = b[b.volume.notna() & (b.elapsed_time >= 0)]
        feeds = b[b[FEEDS].notna().any(axis=1)]
        times = samp.elapsed_time.to_numpy()
        vols = samp.volume.to_numpy()
        batch_duration = times[-1]  # last sample time ≈ day 13

        for k, (_, s) in enumerate(samp.iterrows()):
            t = s.elapsed_time
            r = {
                "batch_id": bid,
                "elapsed_time": t,
                "day": k,  # 0-indexed sample position within batch (0–13)
                "volume": s.volume,
                # volume deltas
                "dvol_prev": s.volume - (vols[k - 1] if k > 0 else t0.volume),
                "dvol_next": (vols[k + 1] - s.volume) if k + 1 < len(vols) else 0.0,
                "vol_rel": s.volume / t0.volume,
            }
            # day-0 init state (same for all rows in this batch)
            for c in TARGETS:
                r[f"init_{c}"] = t0[c]
                r[c] = s[c]  # NaN for test rows beyond day 0
            r["init_volume"] = t0.volume

            # Derived / interpretable
            if pd.notna(s.get("vcd")) and pd.notna(s.get("tcd")) and s["tcd"] > 0:
                r["dcd"] = s["tcd"] - s["vcd"]
                r["viability"] = s["vcd"] / s["tcd"]
            else:
                r["dcd"] = np.nan
                r["viability"] = np.nan

            # Feed windows
            nxt = times[k + 1] if k + 1 < len(times) else t + 1.0
            prv = times[k - 1] if k > 0 else 0.0
            for f in FEEDS:
                fe = feeds[[f, "elapsed_time"]].dropna(subset=[f])
                # _prev: feeds between previous sample and current (inclusive upper bound)
                r[f"{f}_prev"] = fe[(fe.elapsed_time > prv) & (fe.elapsed_time <= t)][f].sum()
                # _next: feeds between current sample and next (exclusive upper bound)
                r[f"{f}_next"] = fe[(fe.elapsed_time > t) & (fe.elapsed_time < nxt)][f].sum()
                # _cum: all feed up to and including current time
                r[f"{f}_cum"] = fe[fe.elapsed_time <= t][f].sum()
                # per-volume version (feeds scale with reactor state)
                r[f"{f}_next_per_vol"] = r[f"{f}_next"] / s.volume if s.volume > 0 else 0.0

            r["time_to_end"] = batch_duration - t
            r["is_test"] = bid in test_ids
            rows.append(r)

    return pd.DataFrame(rows)


# Build and verify
daily = build_rows(df)

# --- Verification assertions ---
# Every batch has exactly 14 rows
batch_counts = daily.groupby("batch_id").size()
assert (batch_counts == 14).all(), f"Not all batches have 14 rows: {batch_counts[batch_counts != 14].to_dict()}"

# Train batches have targets for all 14 days, test batches only for day 0
train_batches = daily[~daily.is_test]
test_batches = daily[daily.is_test]
assert train_batches[TARGETS].notna().all().all(), "Some train rows missing targets"
test_target_na = test_batches[TARGETS].isna()
assert test_batches[test_batches.day == 0][TARGETS].notna().all().all(), "Test day-0 missing targets"
# Days 1-13 for test: all targets should be NaN
test_d1_plus = test_batches[test_batches.day > 0]
assert test_d1_plus[TARGETS].isna().all().all(), "Test rows beyond day 0 should have NaN targets"

# day is the sample index (0–13) within each batch. Verify by checking:
# within each batch, day increases by 1 each row and there are exactly 14 unique day values
for bid, grp in daily.groupby("batch_id"):
    assert list(grp.day) == list(range(14)), f"{bid}: day not 0–13: {list(grp.day)}"

# Per-feed edge cases: day 0 _prev should be 0, day 13 _next can be small/0
d0 = daily[daily.day == 0]
for f in FEEDS:
    assert (d0[f"{f}_prev"] == 0).all(), f"{f}_prev non-zero on day 0"

# No {feed}_total column (leakage prevention)
for f in FEEDS:
    assert f"{f}_total" not in daily.columns, f"{f}_total column found — remove it (leakage)"

print(f"Built daily table: {len(daily)} rows × {len(daily.columns)} cols")
print(f"  Train batches: {daily[~daily.is_test].batch_id.nunique()}")
print(f"  Test batches:  {daily[daily.is_test].batch_id.nunique()}")
print(f"  Columns: {list(daily.columns)}")

daily.to_parquet(f"{OUT_DIR}/dtb_daily.parquet", index=False)
print(f"Saved to {OUT_DIR}/dtb_daily.parquet")

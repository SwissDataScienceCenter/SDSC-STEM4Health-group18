"""Data-quality checks for Challenge 1 (DTB): bioreaction_remaining.csv."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
DATA = "/home/renku/work/input-data/Project 1 - DTB/"
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]

df = pd.read_csv(DATA + "bioreaction_remaining.csv")
tpl = pd.read_csv(DATA + "bioreaction_template.csv")
test_ids = set(tpl.batch_id)
df["is_test"] = df.batch_id.isin(test_ids)


def h(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


h("1. Schema / basic")
print(df.dtypes.to_string())
print("rows", len(df), "batches", df.batch_id.nunique(), "test batches", len(test_ids))
print("exact duplicate rows:", df.duplicated().sum())
print("duplicate (batch,time):", df.duplicated(["batch_id", "elapsed_time"]).sum())
print("time sorted within batch:", df.groupby("batch_id").elapsed_time.apply(lambda s: s.is_monotonic_increasing).all())

h("2. Row types (which columns are filled together)")
pattern = df[TARGETS + FEEDS + ["volume"]].notna().astype(int).astype(str).agg("".join, axis=1)
df["rowtype"] = np.select(
    [df[TARGETS].notna().all(axis=1), df.to_predict, df[FEEDS].notna().any(axis=1)],
    ["measurement", "to_predict", "feed_event"], "other")
print(pd.crosstab(pattern, df.rowtype).rename_axis("pattern: " + "".join(c[0] for c in TARGETS + FEEDS + ["volume"])))
print("feed rows with >1 feed:", (df[FEEDS].notna().sum(axis=1) > 1).sum())
print("measurement rows with partial targets:", (df[TARGETS].notna().any(axis=1) & ~df[TARGETS].notna().all(axis=1)).sum())
print("rows with volume but no targets and not to_predict:",
      (df.volume.notna() & df[TARGETS].isna().all(axis=1) & ~df.to_predict).sum())

h("3. Sampling grid per batch")
samp = df[df.rowtype.isin(["measurement", "to_predict"])].copy()
samp["day"] = samp.elapsed_time.round().astype(int)
g = samp.groupby("batch_id")
print("samples/batch:", g.size().value_counts().to_dict())
print("day-0 present in every batch:", (df[df.elapsed_time == 0].groupby("batch_id").size() == 1).all()
      and df.batch_id.nunique() == (df.elapsed_time == 0).sum())
print("duplicate days in a batch:", samp.duplicated(["batch_id", "day"]).sum())
print("max |time - round(time)|:", (samp.elapsed_time - samp.day).abs().max().round(3))
missing_days = g.day.apply(lambda d: sorted(set(range(0, 14)) - set(d)))
print("batches missing a day 0..13:", missing_days[missing_days.str.len() > 0].to_dict())
print("test: day-0 measured, all later rows to_predict:",
      all((b.iloc[0][TARGETS].notna().all() and b.iloc[1:].to_predict.all())
          for _, b in samp[samp.is_test].groupby("batch_id")))
print("template rows match to_predict rows:",
      np.allclose(np.sort(tpl.elapsed_time), np.sort(df[df.to_predict].elapsed_time)))

h("4. Ranges / physical sanity (measurements)")
m = df[df.rowtype == "measurement"]
print(m[TARGETS + ["volume"]].describe().T.round(3))
print("negative values:", (df[TARGETS + FEEDS + ["volume"]] < 0).sum().to_dict())
print("zero values:", (df[TARGETS + FEEDS + ["volume"]] == 0).sum().to_dict())
bad = m[m.tcd < m.vcd]
print(f"tcd < vcd: {len(bad)} rows; max deficit {(bad.vcd - bad.tcd).max() if len(bad) else 0:.4f}")
print("viability (vcd/tcd) range:", (m.vcd / m.tcd).agg(["min", "median", "max"]).round(3).to_dict())

h("5. Outliers: per-day robust z-score > 4 (measurements)")
md = m.assign(day=m.elapsed_time.round().astype(int))
for c in TARGETS:
    med = md.groupby("day")[c].transform("median")
    mad = md.groupby("day")[c].transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    z = (md[c] - med) / mad.replace(0, np.nan)
    o = md[z.abs() > 4]
    print(f"{c}: {len(o)} outliers", o[["batch_id", "day", c]].assign(z=z[o.index].round(1)).to_string(index=False) if len(o) else "")

h("6. Jumps within a batch (day-to-day change > 4 robust sd)")
md = md.sort_values(["batch_id", "elapsed_time"])
for c in TARGETS:
    d = md.groupby("batch_id")[c].diff()
    zd = (d - d.median()) / ((d - d.median()).abs().median() * 1.4826)
    o = md[zd.abs() > 5]
    print(f"{c}: {len(o)} large jumps", o[["batch_id", "day", c]].to_string(index=False) if len(o) else "")

h("7. Volume consistency: does volume rise by the feeds added?")
rows = []
for bid, b in df.groupby("batch_id"):
    s = b[b.volume.notna()]
    for (t0, v0), (t1, v1) in zip(s[["elapsed_time", "volume"]].values[:-1], s[["elapsed_time", "volume"]].values[1:]):
        f = b[(b.elapsed_time > t0) & (b.elapsed_time <= t1)][FEEDS].sum()
        rows.append({"batch_id": bid, "dvol": v1 - v0, **f.to_dict()})
vc = pd.DataFrame(rows)
print(vc[["dvol"] + FEEDS].corr().round(2).loc["dvol"].to_string())
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(vc[FEEDS], vc.dvol)
print("dvol ~ feeds coef:", dict(zip(FEEDS, lr.coef_.round(2))), "intercept", round(lr.intercept_, 3),
      "R2", round(lr.score(vc[FEEDS], vc.dvol), 3))
print("volume decreases between samples:", (vc.dvol < 0).sum(), "min", vc.dvol.min().round(3))

h("8. Feed schedule per batch")
fs = df.groupby("batch_id")[FEEDS].agg(["count", "sum"]).round(3)
print(fs.describe().round(3).loc[["min", "50%", "max"]].to_string())
print("batches with zero events of a feed:", (df.groupby("batch_id")[FEEDS].count() == 0).sum().to_dict())
first = {f: df[df[f].notna()].groupby("batch_id").elapsed_time.min().describe()[["min", "50%", "max"]].round(2).to_dict()
         for f in FEEDS}
print("first feed time:", first)

h("9. feed_b vs state (is the feed rate proportional to cells?)")
mm = df[df.rowtype == "measurement"][["batch_id", "elapsed_time", "vcd", "volume"]]
fb = df[df.feed_b.notna()][["batch_id", "elapsed_time", "feed_b"]]
j = pd.merge_asof(fb.sort_values("elapsed_time"), mm.sort_values("elapsed_time"), on="elapsed_time",
                  by="batch_id", direction="backward", tolerance=0.2).dropna()
print("corr(feed_b, vcd*volume):", np.corrcoef(j.feed_b, j.vcd * j.volume)[0, 1].round(3),
      " corr(feed_b, vcd):", np.corrcoef(j.feed_b, j.vcd)[0, 1].round(3))

h("10. Train vs test batches: day-0 state & feed totals")
d0 = df[df.elapsed_time == 0].set_index("batch_id")[TARGETS + ["volume", "is_test"]]
d0 = d0.join(df.groupby("batch_id")[FEEDS].sum())
print(d0.groupby("is_test").agg(["mean", "std"]).round(3).T.to_string())

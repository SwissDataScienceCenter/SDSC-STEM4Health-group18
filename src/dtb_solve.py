"""Challenge 1 (DTB): predict vcd/tcd/substrate/byproduct trajectories for test batches.

Approach: one row per sampling time; features = day-0 state + volume history + feed events
around that time (feeds are dosed shortly after each sample and scale with the current state).
Models are validated with GroupKFold over batches and compared with a mean-trajectory baseline.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "/home/renku/work/input-data/Project 1 - DTB/"
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "submission_dtb.csv"

raw = pd.read_csv(DATA + "bioreaction_remaining.csv")
template = pd.read_csv(DATA + "bioreaction_template.csv", dtype={"batch_id": str})


def build_rows(df):
    rows = []
    for bid, b in df.groupby("batch_id", sort=False):
        b = b.sort_values("elapsed_time")
        t0 = b.iloc[0]
        # sampling rows: either measured or to_predict (both carry volume)
        samp = b[b.volume.notna() & (b.elapsed_time > 0)]
        feeds = b[b[FEEDS].notna().any(axis=1)]
        times = samp.elapsed_time.to_numpy()
        vols = samp.volume.to_numpy()
        for k, (_, s) in enumerate(samp.iterrows()):
            t = s.elapsed_time
            r = {"batch_id": bid, "elapsed_time": t, "day": round(t), "volume": s.volume,
                 "dvol_prev": s.volume - (vols[k - 1] if k else t0.volume),
                 "dvol_next": (vols[k + 1] - s.volume) if k + 1 < len(vols) else np.nan,
                 "vol_rel": s.volume / t0.volume}
            for c in TARGETS:
                r[f"init_{c}"] = t0[c]
                r[c] = s[c]
            nxt = times[k + 1] if k + 1 < len(times) else t + 1.0
            prv = times[k - 1] if k else 0.0
            for f in FEEDS:
                fe = feeds[[f, "elapsed_time"]].dropna()
                r[f"{f}_next"] = fe[(fe.elapsed_time > t) & (fe.elapsed_time < nxt)][f].sum()
                r[f"{f}_prev"] = fe[(fe.elapsed_time > prv) & (fe.elapsed_time <= t)][f].sum()
                r[f"{f}_cum"] = fe[fe.elapsed_time <= t][f].sum()
                r[f"{f}_total"] = fe[f].sum()
            r["fb_next_per_vol"] = r["feed_b_next"] / s.volume
            rows.append(r)
    return pd.DataFrame(rows)


test_ids = set(template.batch_id)
data = build_rows(raw)
train = data[~data.batch_id.isin(test_ids)].reset_index(drop=True)
test = data[data.batch_id.isin(test_ids)].reset_index(drop=True)
FEATS = [c for c in data.columns if c not in TARGETS + ["batch_id"]]
print(f"train rows {len(train)} ({train.batch_id.nunique()} batches), test rows {len(test)}")

models = {
    "extratrees": lambda: ExtraTreesRegressor(500, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "hgb": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, random_state=0),
    "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 20))),
}


def fit_predict(name, Xtr, ytr, Xte):
    m = models[name]()
    Xtr, Xte = Xtr.fillna(0), Xte.fillna(0)
    return m.fit(Xtr, ytr).predict(Xte)


def baseline(tr, te):  # mean trajectory per day, scaled to batch's day-0 value
    out = np.zeros((len(te), len(TARGETS)))
    for j, c in enumerate(TARGETS):
        ratio = (tr[c] / tr[f"init_{c}"]).groupby(tr.day).mean()
        out[:, j] = te.day.map(ratio).fillna(ratio.iloc[-1]).to_numpy() * te[f"init_{c}"].to_numpy()
    return out


# ---- cross-validation over batches
gkf = GroupKFold(n_splits=5)
oof = {n: np.zeros((len(train), len(TARGETS))) for n in ["baseline", *models, "ensemble"]}
for tr_i, va_i in gkf.split(train, groups=train.batch_id):
    tr, va = train.iloc[tr_i], train.iloc[va_i]
    oof["baseline"][va_i] = baseline(tr, va)
    for n in models:
        for j, c in enumerate(TARGETS):
            oof[n][va_i, j] = fit_predict(n, tr[FEATS], tr[c], va[FEATS])
oof["ensemble"] = (oof["extratrees"] + oof["hgb"]) / 2

Y = train[TARGETS].to_numpy()
rows = []
for n, p in oof.items():
    rmse = np.sqrt(((p - Y) ** 2).mean(0))
    nrmse = rmse / Y.std(0)
    rows.append({"model": n, **{f"rmse_{c}": v for c, v in zip(TARGETS, rmse)}, "mean_nrmse": nrmse.mean()})
cv = pd.DataFrame(rows).set_index("model").round(3)
print(cv.to_string())

best = cv.mean_nrmse.idxmin()
print("best:", best)

# ---- final fit on all training batches
pred = np.zeros((len(test), len(TARGETS)))
if best == "baseline":
    pred = baseline(train, test)
else:
    names = ["extratrees", "hgb"] if best == "ensemble" else [best]
    for j, c in enumerate(TARGETS):
        pred[:, j] = np.mean([fit_predict(n, train[FEATS], train[c], test[FEATS]) for n in names], axis=0)
pred = np.clip(pred, 0, None)
test[TARGETS] = pred
test["tcd"] = np.maximum(test.tcd, test.vcd)  # physical constraint

# map onto template rows by (batch, nearest time)
sub = template[["batch_id", "elapsed_time"]].copy()
for c in TARGETS:
    sub[c] = np.nan
for i, r in sub.iterrows():
    cand = test[test.batch_id == r.batch_id]
    k = (cand.elapsed_time - r.elapsed_time).abs().idxmin()
    assert abs(cand.elapsed_time[k] - r.elapsed_time) < 1e-6
    sub.loc[i, TARGETS] = cand.loc[k, TARGETS].to_numpy(dtype=float)
assert sub[TARGETS].notna().all().all() and len(sub) == len(template)
sub.to_csv(OUT, index=False)
print(f"wrote {OUT}\n", sub.head())

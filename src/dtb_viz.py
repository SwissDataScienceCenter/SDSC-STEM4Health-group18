"""Exploratory visualization for Challenge 1 (DTB): what the target trajectories,
volume, and feed events actually look like. Run after dtb_checks.py; writes PNGs
to outputs/ instead of a notebook so results are easy to diff/regenerate in CI-less runs.
"""
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt

DATA = "/home/renku/work/input-data/Project 1 - DTB/"
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
OUT = "outputs/"
TRAIN_COLOR, TEST_COLOR = "#4477AA", "#CC6677"  # fixed categorical assignment, reused across all plots

df = pd.read_csv(DATA + "bioreaction_remaining.csv")
tpl = pd.read_csv(DATA + "bioreaction_template.csv")
test_ids = set(tpl.batch_id)
df["is_test"] = df.batch_id.isin(test_ids)


def spaghetti(ax, data, y, title):
    for bid, b in data.groupby("batch_id"):
        b = b.sort_values("elapsed_time")
        is_test = b.is_test.iloc[0]
        # test batches get higher alpha: there are few of them (12) vs 61 train,
        # so equal alpha would make them invisible under the train batches
        # marker=o: test batches often have a single point (day 0 only), which a
        # bare line wouldn't render at all
        ax.plot(b.elapsed_time, b[y], color=TEST_COLOR if is_test else TRAIN_COLOR,
                 alpha=0.6 if is_test else 0.25, lw=1, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("elapsed_time (days)")


legend_handles = [Line2D([0], [0], color=TRAIN_COLOR, lw=2), Line2D([0], [0], color=TEST_COLOR, lw=2)]

# 1. target trajectories, one line per batch. Test batches only have a day-0
# point measured (rest is to_predict / NaN), so their lines are necessarily short.
measured = df[df[TARGETS].notna().all(axis=1)]
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, target in zip(axes.flat, TARGETS):
    spaghetti(ax, measured, target, target)
fig.legend(legend_handles, ["train batch", "test batch (day 0 only)"], loc="upper right", ncol=2)
fig.suptitle("Target trajectories per batch")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT + "dtb_viz_trajectories.png", dpi=130)

# 2. volume trajectories
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
spaghetti(ax2, df[df.volume.notna()], "volume", "Volume per batch")
fig2.legend(legend_handles, ["train", "test"], loc="upper left")
fig2.tight_layout()
fig2.savefig(OUT + "dtb_viz_volume.png", dpi=130)

# 3. feed timing: when does each feed fire, and how large is the dose (marker size)
fig3, ax3 = plt.subplots(figsize=(8, 4.5))
palette = {"feed_substrate": "#4477AA", "feed_b": "#CC6677", "feed_c": "#228833", "feed_d": "#CCBB44"}
for f in FEEDS:
    fe = df[df[f].notna()]
    ax3.scatter(fe.elapsed_time, [f] * len(fe), s=fe[f] / fe[f].max() * 120 + 8,
                color=palette[f], alpha=0.5, label=f)
ax3.set_xlabel("elapsed_time (days)")
ax3.set_title("Feed events over time (marker size ~ dose)")
fig3.tight_layout()
fig3.savefig(OUT + "dtb_viz_feeds.png", dpi=130)

print(f"wrote {OUT}dtb_viz_trajectories.png, {OUT}dtb_viz_volume.png, {OUT}dtb_viz_feeds.png")

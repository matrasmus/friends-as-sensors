"""Reproduce all figures used in the paper, without titles.

Loads the Sensor and Random parquets, runs every analysis from scratch, and
writes 11 SVGs to OUT_DIR. Figures match the paper's content but drop chart
titles (information goes into captions); side-panel "Distribution" sub-label
is removed; Figure 10 carries a soft, hedged label on the September
marker pointing to the §4.4 monetization-rule co-occurrence.

Output filenames are kept identical to the originals shipped in
misinformation_detection_prodeminfo.zip, with one exception: Figure 4 uses
the _neutral_ filename because it now plots three orientation series.

Usage:
    python reproduce_paper_figures.py
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterMathtext
from scipy import stats
from scipy.stats import gaussian_kde
import pymannkendall as mk

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "/home/mangermaier/cs2/twitter/user"
SENSOR_PARQUET = f"{DATA_DIR}/result_sensor_with_users_unraveled_v5.parquet"
RANDOM_PARQUET = f"{DATA_DIR}/result_random_with_users_unraveled_v5.parquet"

OUT_DIR = Path("/home/mangermaier/friendship_paradox/paper_drafts/figs_remake")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_COLOR    = "#1f77b4"
RIGHT_COLOR   = "#d32f2f"
NEUTRAL_COLOR = "#2ca02c"
NG_COLOR      = "#6A1B9A"
DECAY_COLOR   = "#c0392b"
GRAY_COLOR    = "#9aa0a6"
TEXT_GRAY     = "#555555"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_parquet(path, name):
    t0 = time.time()
    table = pq.read_table(path)
    table = table.replace_schema_metadata(None)  # drop pandas-extension metadata
    df = table.to_pandas()
    print(f"[{name}] loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Shared utilities (consolidated from notebook cells 124, 161, 170)
# ---------------------------------------------------------------------------
def to_naive_utc(s):
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_localize(None)


def _to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if pd.isna(x): return []
    return [x]


def _has_left(lst):
    if not isinstance(lst, list): return False
    for v in lst:
        if str(v).strip().lower().startswith("left"): return True
    return False


def _has_right(lst):
    if not isinstance(lst, list): return False
    for v in lst:
        if str(v).strip().lower().startswith("right"): return True
    return False


def _is_retweet(x):
    if isinstance(x, list):
        return any(str(t).lower() == "retweeted" for t in x)
    if pd.isna(x): return False
    return str(x).lower() == "retweeted"


def _tweet_mean_newsguard(df):
    scores = df[["TWEET_id", "newsguard_scores_expanded"]].explode(
        "newsguard_scores_expanded", ignore_index=False)
    scores["score"] = pd.to_numeric(scores["newsguard_scores_expanded"], errors="coerce")
    scores = scores.dropna(subset=["score"])
    return scores.groupby("TWEET_id", observed=True)["score"].mean().rename("tweet_ng")


def _series_by_kind(df, bucket_col, value_col, mask_kind, mask_side):
    sub = df.loc[mask_kind & mask_side, [bucket_col, value_col]].rename(columns={value_col: "val"})
    if sub.empty:
        return pd.DataFrame(columns=[bucket_col, "mean", "ci_low", "ci_high", "n"])
    g = sub.groupby(bucket_col, observed=True)["val"]
    s = g.agg(mean="mean", sd=lambda s: s.std(ddof=1), n="count").reset_index()
    s["se"] = s["sd"] / np.sqrt(s["n"].replace(0, np.nan))
    z = 1.96
    s["ci_low"] = s["mean"] - z * s["se"]
    s["ci_high"] = s["mean"] + z * s["se"]
    return s[[bucket_col, "mean", "ci_low", "ci_high", "n"]]


def _norm_lr(x):
    t = str(x).strip().lower()
    if t.startswith("right"): return "Right"
    if t.startswith("left"):  return "Left"
    return "Center"


def build_domain_orientation_map(sensor_df):
    """Derive a domain -> dominant orientation map from sensor's NG-orientation column.
    The Random parquet does not carry a `newsguard_orientation` field, so we
    rebuild it for random tweets via this map (paper convention)."""
    sm = sensor_df[["domain_from_urls_expanded", "newsguard_orientation"]].copy()
    sm["d"] = sm["domain_from_urls_expanded"].map(_to_list)
    sm["o"] = sm["newsguard_orientation"].map(_to_list)
    sm = sm[["d", "o"]].explode(["d", "o"], ignore_index=True)
    sm = sm.dropna(subset=["d", "o"])
    sm["d"] = sm["d"].astype(str).str.lower().str.strip()
    sm["o"] = sm["o"].astype(str).str.strip()
    sm = sm[(sm["d"] != "") & (sm["o"] != "") & (sm["o"].str.lower() != "nan")]
    return sm.groupby("d")["o"].agg(lambda s: s.value_counts().idxmax()).to_dict()


def inject_orientation_from_domains(df, dom_map):
    """Add a `newsguard_orientation` column derived from `domain_from_urls_expanded`."""
    out = df.copy()
    def _doms_to_orient(dlist):
        if not isinstance(dlist, list): return []
        return [dom_map.get(str(x).lower().strip()) for x in dlist]
    out["newsguard_orientation"] = out["domain_from_urls_expanded"].map(_to_list).map(_doms_to_orient)
    return out


def _prepare_scores_frame(df, start="2024-06-01", end="2024-11-01"):
    cols = ["TWEET_id", "TWEET_created_at", "referenced_tweet_type",
            "newsguard_orientation", "newsguard_scores_expanded"]
    use_cols = [c for c in cols if c in df.columns]
    x = df.loc[:, use_cols].copy()
    x["TWEET_created_at"] = to_naive_utc(x["TWEET_created_at"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    x = x[(x["TWEET_created_at"] >= start_ts) & (x["TWEET_created_at"] < end_ts)]
    if x.empty:
        return x.assign(tweet_ng=np.nan, is_ret=False, is_left=False, is_right=False)
    tweet_ng = _tweet_mean_newsguard(x)
    x = x.merge(tweet_ng, on="TWEET_id", how="left")
    ori_list = x["newsguard_orientation"].map(_to_list)
    x["is_left"]  = ori_list.map(_has_left)
    x["is_right"] = ori_list.map(_has_right)
    x["is_ret"]   = x["referenced_tweet_type"].map(_is_retweet)
    return x


# ---------------------------------------------------------------------------
# Figure 4: impressions by orientation (with neutral)
# ---------------------------------------------------------------------------
def analyze_impressions_by_orientation(df, start="2024-05-01", end="2024-11-01",
                                       freq="3D", mode="any", tweet_kind="original",
                                       zero_values=True):
    cols = ["TWEET_id", "TWEET_created_at", "newsguard_orientation",
            "impression_count", "referenced_tweet_type", "newsguard_scores_expanded"]
    df = df[cols].copy()
    df["TWEET_created_at"] = to_naive_utc(df["TWEET_created_at"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["TWEET_created_at"] >= start_ts) & (df["TWEET_created_at"] < end_ts)]

    mask_ret = df["referenced_tweet_type"].map(_is_retweet)
    if tweet_kind == "retweet": df = df[mask_ret]
    elif tweet_kind == "original": df = df[~mask_ret]

    df["impression_count"] = pd.to_numeric(df["impression_count"], errors="coerce")
    df = df[df["impression_count"].notna()]
    if not zero_values:
        df = df[df["impression_count"] > 0]
    if df.empty:
        return {"empty": True}

    df["impression_count"] = df["impression_count"].clip(lower=0)
    df["log_impr"] = np.log1p(df["impression_count"])

    ori_list = df["newsguard_orientation"].map(_to_list)
    left_any  = ori_list.map(_has_left)
    right_any = ori_list.map(_has_right)
    if mode == "only":
        mask_left  = left_any & ~right_any
        mask_right = right_any & ~left_any
    else:
        mask_left, mask_right = left_any, right_any

    scores_list = df["newsguard_scores_expanded"].map(_to_list)
    has_any_score = scores_list.map(
        lambda x: any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in x))
    mask_neutral = ~mask_left & ~mask_right & has_any_score

    left_label, right_label, neutral_label = "Left media", "Right media", "Neutral media"
    df["bucket"] = df["TWEET_created_at"].dt.floor(freq)

    impr_parts = [
        df.loc[mask_left,    ["bucket", "log_impr"]].assign(group=left_label),
        df.loc[mask_right,   ["bucket", "log_impr"]].assign(group=right_label),
        df.loc[mask_neutral, ["bucket", "log_impr"]].assign(group=neutral_label),
    ]
    lr_impr = pd.concat(impr_parts, ignore_index=True)
    weekly_impr = (lr_impr.groupby(["bucket", "group"], observed=True)["log_impr"]
                   .agg(mean_log_impr="mean",
                        sd_log_impr=lambda s: s.std(ddof=1),
                        n_impr="count")
                   .reset_index())
    weekly_impr["se_log_impr"] = weekly_impr["sd_log_impr"] / np.sqrt(weekly_impr["n_impr"].replace(0, np.nan))
    z = 1.96
    weekly_impr["ci_low_log_impr"]  = weekly_impr["mean_log_impr"] - z * weekly_impr["se_log_impr"]
    weekly_impr["ci_high_log_impr"] = weekly_impr["mean_log_impr"] + z * weekly_impr["se_log_impr"]
    for col in ["mean_log_impr", "ci_low_log_impr", "ci_high_log_impr"]:
        weekly_impr[col.replace("_log", "").replace("mean_impr", "mean_impr").replace("ci_low_impr", "ci_low_impr").replace("ci_high_impr", "ci_high_impr")] = np.expm1(weekly_impr[col]).clip(lower=1.0)
    # be explicit about names
    weekly_impr["mean_impr"]    = np.expm1(weekly_impr["mean_log_impr"]).clip(lower=1.0)
    weekly_impr["ci_low_impr"]  = np.expm1(weekly_impr["ci_low_log_impr"]).clip(lower=1.0)
    weekly_impr["ci_high_impr"] = np.expm1(weekly_impr["ci_high_log_impr"]).clip(lower=1.0)

    # NG tweet-level mean per bucket
    scores = df[["TWEET_id", "newsguard_scores_expanded"]].explode("newsguard_scores_expanded", ignore_index=False)
    scores["score"] = pd.to_numeric(scores["newsguard_scores_expanded"], errors="coerce")
    scores = scores.dropna(subset=["score"])
    tweet_ng = scores.groupby("TWEET_id", observed=True)["score"].mean().rename("tweet_ng")

    tweet_frame = df[["TWEET_id", "bucket"]].copy()
    tweet_frame["mask_left"]    = mask_left.values
    tweet_frame["mask_right"]   = mask_right.values
    tweet_frame["mask_neutral"] = mask_neutral.values
    tweet_frame = tweet_frame.merge(tweet_ng, on="TWEET_id", how="left").dropna(subset=["tweet_ng"])

    ng_parts = [
        tweet_frame.loc[tweet_frame["mask_left"],    ["bucket", "tweet_ng"]].assign(group=left_label),
        tweet_frame.loc[tweet_frame["mask_right"],   ["bucket", "tweet_ng"]].assign(group=right_label),
        tweet_frame.loc[tweet_frame["mask_neutral"], ["bucket", "tweet_ng"]].assign(group=neutral_label),
    ]
    lr_ng = pd.concat(ng_parts, ignore_index=True)
    weekly_ng = (lr_ng.groupby(["bucket", "group"], observed=True)["tweet_ng"]
                 .agg(mean_ng="mean",
                      sd_ng=lambda s: s.std(ddof=1),
                      n_ng="count")
                 .reset_index())
    weekly_ng["se_ng"] = weekly_ng["sd_ng"] / np.sqrt(weekly_ng["n_ng"].replace(0, np.nan))
    weekly_ng["ci_low_ng"]  = weekly_ng["mean_ng"] - z * weekly_ng["se_ng"]
    weekly_ng["ci_high_ng"] = weekly_ng["mean_ng"] + z * weekly_ng["se_ng"]
    weekly = weekly_impr.merge(weekly_ng, on=["bucket", "group"], how="outer")

    # trend tests
    trend_results = {}
    for group_label in [left_label, right_label, neutral_label]:
        sub = weekly_impr[weekly_impr["group"] == group_label].sort_values("bucket").dropna(subset=["mean_log_impr"])
        if len(sub) < 3:
            trend_results[group_label] = {"ols_slope": np.nan, "ols_r2": np.nan, "ols_p": np.nan,
                                          "mk_tau": np.nan, "mk_p": np.nan, "mk_slope": np.nan,
                                          "n_buckets": len(sub)}
            continue
        bucket_delta = (sub["bucket"].iloc[1] - sub["bucket"].iloc[0]).total_seconds()
        t = (sub["bucket"] - sub["bucket"].min()).dt.total_seconds() / bucket_delta
        y = sub["mean_log_impr"].values
        slope, intercept, r, p_ols, se = stats.linregress(t, y)
        mkr = mk.original_test(y)
        trend_results[group_label] = {
            "ols_slope": float(slope), "ols_r2": float(r ** 2), "ols_p": float(p_ols),
            "mk_tau": float(mkr.Tau), "mk_p": float(mkr.p), "mk_slope": float(mkr.slope),
            "n_buckets": int(len(sub))}

    return {"empty": False, "weekly": weekly, "freq": freq, "tweet_kind": tweet_kind,
            "left_label": left_label, "right_label": right_label, "neutral_label": neutral_label,
            "trend_results": trend_results}


def plot_impressions_by_orientation(analysis, include_neutral=True, savepath=None):
    """No title; no 'Distribution' subtitle; same layout otherwise."""
    if analysis.get("empty", True): return None, None
    weekly = analysis["weekly"]
    left_label    = analysis["left_label"]
    right_label   = analysis["right_label"]
    neutral_label = analysis["neutral_label"]

    colors = {left_label: LEFT_COLOR, right_label: RIGHT_COLOR, neutral_label: NEUTRAL_COLOR}
    all_labels = [left_label, right_label] + ([neutral_label] if include_neutral else [])

    fig = plt.figure(figsize=(13, 4.5))
    gs = GridSpec(1, 2, width_ratios=[10, 1], wspace=0.30, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])

    for group in all_labels:
        sub = weekly[weekly["group"] == group].sort_values("bucket")
        if len(sub) == 0: continue
        ax.plot(sub["bucket"], sub["mean_impr"], "-o", ms=3, lw=1.8, color=colors[group],
                label=f"{group}: avg. impressions per tweet")
        ax.fill_between(sub["bucket"], sub["ci_low_impr"], sub["ci_high_impr"],
                        color=colors[group], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("Avg. impressions per tweet (95% CI) — log scale")
    ax.set_xlabel("Time")
    ax.grid(True, axis="y", alpha=0.2)

    ax2 = ax.twinx()
    for group in all_labels:
        sub = weekly[weekly["group"] == group].sort_values("bucket")
        if len(sub) == 0 or sub["mean_ng"].isna().all(): continue
        ax2.plot(sub["bucket"], sub["mean_ng"], "--", lw=1.8, color=colors[group],
                 label=f"{group}: mean NG-score")
        ax2.fill_between(sub["bucket"], sub["ci_low_ng"], sub["ci_high_ng"],
                         color=colors[group], alpha=0.10, lw=0)
    ax2.set_ylabel("NewsGuard (mean, 95% CI)")
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True, fontsize=8, ncol=2)
    # NO TITLE

    # Side panel: half-violin distribution (NO "Distribution" label)
    for group in all_labels:
        sub = weekly[weekly["group"] == group].dropna(subset=["mean_impr"])
        vals = sub["mean_impr"].values
        vals = vals[vals > 0]
        if len(vals) < 3: continue
        log_vals = np.log(vals)
        kde = gaussian_kde(log_vals, bw_method=0.3)
        y_grid = np.linspace(log_vals.min(), log_vals.max(), 300)
        density = kde(y_grid); density = density / density.max()
        ax_dist.fill_betweenx(np.exp(y_grid), 0, density, color=colors[group], alpha=0.35, linewidth=0)
        ax_dist.plot(density, np.exp(y_grid), color=colors[group], lw=1.0, alpha=0.7)

    ax_dist.set_yscale("log")
    ax_dist.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_dist.set_ylim(ax.get_ylim())
    ax_dist.set_xlim(0, None)
    ax_dist.grid(True, axis="y", alpha=0.2)
    ax_dist.tick_params(axis="y", labelleft=True)
    ax_dist.tick_params(axis="x", labelbottom=False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.spines["bottom"].set_visible(False)
    # NO "Distribution" subtitle

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2: 4-panel sensor/random NG over 3-day intervals
# ---------------------------------------------------------------------------
def plot_left_right_3day_newsguard_4panel(datasets, start="2024-06-01", end="2024-11-01",
                                          savepath=None):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)
    cols = ["TWEET_id", "TWEET_created_at", "referenced_tweet_type",
            "newsguard_orientation", "newsguard_scores_expanded"]

    for r, (result_df, dataset) in enumerate(datasets[:2]):
        df = result_df.loc[:, [c for c in cols if c in result_df.columns]].copy()
        df["TWEET_created_at"] = to_naive_utc(df["TWEET_created_at"])
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        df = df[(df["TWEET_created_at"] >= start_ts) & (df["TWEET_created_at"] <= end_ts)]
        if df.empty: continue
        df["bucket"] = df["TWEET_created_at"].dt.floor("3D")

        ori_list = df["newsguard_orientation"].map(_to_list)
        mask_left  = ori_list.map(_has_left)
        mask_right = ori_list.map(_has_right)
        mask_ret   = df["referenced_tweet_type"].map(_is_retweet)

        tweet_ng = _tweet_mean_newsguard(df)
        tframe = df[["TWEET_id", "bucket"]].copy()
        tframe["is_left"]  = mask_left.values
        tframe["is_right"] = mask_right.values
        tframe["is_ret"]   = mask_ret.values
        tframe = tframe.merge(tweet_ng, on="TWEET_id", how="left").dropna(subset=["tweet_ng"])

        L_org_ng = _series_by_kind(tframe.rename(columns={"tweet_ng": "val"}), "bucket", "val",
                                   (~tframe["is_ret"]), tframe["is_left"]).sort_values("bucket")
        L_ret_ng = _series_by_kind(tframe.rename(columns={"tweet_ng": "val"}), "bucket", "val",
                                   ( tframe["is_ret"]), tframe["is_left"]).sort_values("bucket")
        R_org_ng = _series_by_kind(tframe.rename(columns={"tweet_ng": "val"}), "bucket", "val",
                                   (~tframe["is_ret"]), tframe["is_right"]).sort_values("bucket")
        R_ret_ng = _series_by_kind(tframe.rename(columns={"tweet_ng": "val"}), "bucket", "val",
                                   ( tframe["is_ret"]), tframe["is_right"]).sort_values("bucket")

        axL, axR = axes[r, 0], axes[r, 1]
        if len(L_org_ng):
            axL.plot(L_org_ng["bucket"], L_org_ng["mean"], color=LEFT_COLOR, lw=2.0, label="Original")
            axL.fill_between(L_org_ng["bucket"], L_org_ng["ci_low"], L_org_ng["ci_high"],
                             color=LEFT_COLOR, alpha=0.15, lw=0)
        if len(L_ret_ng):
            axL.plot(L_ret_ng["bucket"], L_ret_ng["mean"], color=LEFT_COLOR, lw=2.0, ls="--", label="Retweet")
            axL.fill_between(L_ret_ng["bucket"], L_ret_ng["ci_low"], L_ret_ng["ci_high"],
                             color=LEFT_COLOR, alpha=0.10, lw=0)
        axL.set_ylabel("Mean NewsGuard")
        axL.set_ylim(0, 100)
        axL.grid(True, axis="y", alpha=0.25)
        axL.legend()
        for lbl in axL.get_xticklabels(): lbl.set_rotation(45)

        if len(R_org_ng):
            axR.plot(R_org_ng["bucket"], R_org_ng["mean"], color=RIGHT_COLOR, lw=2.0, label="Original")
            axR.fill_between(R_org_ng["bucket"], R_org_ng["ci_low"], R_org_ng["ci_high"],
                             color=RIGHT_COLOR, alpha=0.15, lw=0)
        if len(R_ret_ng):
            axR.plot(R_ret_ng["bucket"], R_ret_ng["mean"], color=RIGHT_COLOR, lw=2.0, ls="--", label="Retweet")
            axR.fill_between(R_ret_ng["bucket"], R_ret_ng["ci_low"], R_ret_ng["ci_high"],
                             color=RIGHT_COLOR, alpha=0.10, lw=0)
        axR.set_ylim(0, 100)
        axR.grid(True, axis="y", alpha=0.25)
        axR.legend()
        for lbl in axR.get_xticklabels(): lbl.set_rotation(45)

        # Row labels (subtle, only y-axis side, since we're losing titles)
        axL.text(-0.12, 0.5, dataset.capitalize(), transform=axL.transAxes,
                 fontsize=11, ha="center", va="center", rotation=90, color="#444")

        # Panel labels (A/B/C/D in reading order: TL=A, TR=B, BL=C, BR=D)
        # Placed in lower-right corner to avoid overlapping with the lines
        panel_labels = [["A", "B"], ["C", "D"]][r]
        axL.text(0.97, 0.04, panel_labels[0], transform=axL.transAxes,
                 fontsize=18, fontweight="bold", va="bottom", ha="right", color="#222",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#222", lw=0.8, alpha=0.85))
        axR.text(0.97, 0.04, panel_labels[1], transform=axR.transAxes,
                 fontsize=18, fontweight="bold", va="bottom", ha="right", color="#222",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#222", lw=0.8, alpha=0.85))
    # NO suptitle
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# Figures 8 & 9: random_left_newsguard_3day, random_right_newsguard_3day
# (single-panel NG-over-time plots for the random sample)
# ---------------------------------------------------------------------------
def plot_single_panel_newsguard_3day(result_df, side, start="2024-06-01", end="2024-11-01",
                                     savepath=None):
    """side ∈ {'left', 'right'}; saves a 1-panel NG-over-time figure."""
    cols = ["TWEET_id", "TWEET_created_at", "referenced_tweet_type",
            "newsguard_orientation", "newsguard_scores_expanded"]
    df = result_df.loc[:, [c for c in cols if c in result_df.columns]].copy()
    df["TWEET_created_at"] = to_naive_utc(df["TWEET_created_at"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["TWEET_created_at"] >= start_ts) & (df["TWEET_created_at"] <= end_ts)]
    df["bucket"] = df["TWEET_created_at"].dt.floor("3D")

    ori_list = df["newsguard_orientation"].map(_to_list)
    if side == "left":
        mask_side = ori_list.map(_has_left); color = LEFT_COLOR
    else:
        mask_side = ori_list.map(_has_right); color = RIGHT_COLOR
    mask_ret = df["referenced_tweet_type"].map(_is_retweet)

    tweet_ng = _tweet_mean_newsguard(df)
    tframe = df[["TWEET_id", "bucket"]].copy()
    tframe["is_side"] = mask_side.values
    tframe["is_ret"]  = mask_ret.values
    tframe = tframe.merge(tweet_ng, on="TWEET_id", how="left").dropna(subset=["tweet_ng"])

    org_ng = _series_by_kind(tframe.rename(columns={"tweet_ng": "val"}), "bucket", "val",
                              (~tframe["is_ret"]), tframe["is_side"]).sort_values("bucket")
    ret_ng = _series_by_kind(tframe.rename(columns={"tweet_ng": "val"}), "bucket", "val",
                              ( tframe["is_ret"]), tframe["is_side"]).sort_values("bucket")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    if len(org_ng):
        ax.plot(org_ng["bucket"], org_ng["mean"], color=color, lw=2.0, label="Original")
        ax.fill_between(org_ng["bucket"], org_ng["ci_low"], org_ng["ci_high"], color=color, alpha=0.15, lw=0)
    if len(ret_ng):
        ax.plot(ret_ng["bucket"], ret_ng["mean"], color=color, lw=2.0, ls="--", label="Retweet")
        ax.fill_between(ret_ng["bucket"], ret_ng["ci_low"], ret_ng["ci_high"], color=color, alpha=0.10, lw=0)
    ax.set_ylabel("Mean NewsGuard")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    for lbl in ax.get_xticklabels(): lbl.set_rotation(45)
    # NO TITLE
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Figures 6 & 7: violin plots
# ---------------------------------------------------------------------------
def plot_ng_overall_violin(sensor_df, random_df, start="2024-06-01", end="2024-11-01",
                           savepath=None):
    sen = _prepare_scores_frame(sensor_df, start, end)
    rnd = _prepare_scores_frame(random_df, start, end)
    s_arr = pd.to_numeric(sen["tweet_ng"], errors="coerce").dropna().to_numpy(dtype="float64")
    r_arr = pd.to_numeric(rnd["tweet_ng"], errors="coerce").dropna().to_numpy(dtype="float64")

    overall_df = pd.concat([
        pd.DataFrame({"tweet_ng": s_arr, "dataset": "Sensor"}),
        pd.DataFrame({"tweet_ng": r_arr, "dataset": "Random"}),
    ], ignore_index=True)
    palette = {"Sensor": "#666666", "Random": "#BBBBBB"}

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=overall_df, x="dataset", y="tweet_ng", hue="dataset",
                   dodge=False, palette=palette, cut=0, inner=None, linewidth=0, width=1.0, ax=ax)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for coll in ax.collections:
        paths = getattr(coll, "get_paths", None)
        if not callable(paths): continue
        paths = coll.get_paths()
        if not paths: continue
        bbox = paths[0].get_extents()
        x0, y0, w, h = bbox.bounds
        coll.set_clip_path(plt.Rectangle((x0, y0), w / 2.0, h, transform=ax.transData))
    sns.boxplot(data=overall_df, x="dataset", y="tweet_ng", showfliers=False, width=0.3,
                boxprops={"zorder": 3, "facecolor": "none", "edgecolor": "#333", "linewidth": 1.0},
                whiskerprops={"color": "#333", "linewidth": 1.0},
                capprops={"color": "#333", "linewidth": 1.0},
                medianprops={"color": "#333", "linewidth": 1.2}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("NewsGuard score")
    ax.set_ylim(-10, 110)
    ax.set_xticklabels(["All Sensor", "All Random"], rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    if ax.legend_ is not None: ax.legend_.remove()
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    # NO TITLE
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_ng_left_right_grouped_violin(sensor_df, random_df, start="2024-06-01", end="2024-11-01",
                                      savepath=None):
    sen = _prepare_scores_frame(sensor_df, start, end)
    rnd = _prepare_scores_frame(random_df, start, end)
    s_left  = sen.loc[sen["is_left"],  "tweet_ng"].dropna().values
    r_left  = rnd.loc[rnd["is_left"],  "tweet_ng"].dropna().values
    s_right = sen.loc[sen["is_right"], "tweet_ng"].dropna().values
    r_right = rnd.loc[rnd["is_right"], "tweet_ng"].dropna().values

    desired = [
        ("Sensor Left",  s_left,  1, LEFT_COLOR),
        ("Random Left",  r_left,  2, LEFT_COLOR),
        ("Sensor Right", s_right, 5, RIGHT_COLOR),
        ("Random Right", r_right, 6, RIGHT_COLOR),
    ]
    data, positions, colors, labels = [], [], [], []
    for name, arr, pos, color in desired:
        if len(arr) > 0:
            data.append(arr); positions.append(pos); colors.append(color); labels.append(name)

    fig, ax = plt.subplots(figsize=(9, 5))
    if data:
        vp = ax.violinplot(data, positions=positions, widths=0.9,
                           showmeans=False, showmedians=False, showextrema=False)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col); body.set_edgecolor("#333"); body.set_alpha(0.9)
            paths = getattr(body, "get_paths", lambda: [])()
            if paths:
                bbox = paths[0].get_extents()
                x0, y0, w, h = bbox.bounds
                body.set_clip_path(plt.Rectangle((x0, y0), w / 2.0, h, transform=ax.transData))
        bp = ax.boxplot(data, positions=positions, widths=0.35, patch_artist=True, showfliers=False)
        for box in bp["boxes"]: box.set(facecolor="none", edgecolor="#333", linewidth=1.0)
        for w_ in bp["whiskers"]: w_.set(color="#333", linewidth=1.0)
        for c in bp["caps"]: c.set(color="#333", linewidth=1.0)
        for m_ in bp["medians"]: m_.set(color="#333", linewidth=1.2)
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("NewsGuard score")
    ax.set_ylim(-10, 110)
    ax.grid(True, axis="y", alpha=0.25)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True); ax.spines[side].set_color("black")
    # NO TITLE
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 1: combined overall + left/right violins
# ---------------------------------------------------------------------------
def plot_ng_overall_and_left_right(sensor_df, random_df, start="2024-06-01", end="2024-11-01",
                                   savepath=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # Top panel: overall violin (call helper directly into provided ax)
    sen = _prepare_scores_frame(sensor_df, start, end)
    rnd = _prepare_scores_frame(random_df, start, end)
    s_arr = pd.to_numeric(sen["tweet_ng"], errors="coerce").dropna().to_numpy(dtype="float64")
    r_arr = pd.to_numeric(rnd["tweet_ng"], errors="coerce").dropna().to_numpy(dtype="float64")
    overall_df = pd.concat([
        pd.DataFrame({"tweet_ng": s_arr, "dataset": "Sensor"}),
        pd.DataFrame({"tweet_ng": r_arr, "dataset": "Random"}),
    ], ignore_index=True)
    palette = {"Sensor": "#666666", "Random": "#BBBBBB"}
    ax0 = axes[0]
    sns.violinplot(data=overall_df, x="dataset", y="tweet_ng", hue="dataset",
                   dodge=False, palette=palette, cut=0, inner=None, linewidth=0, width=1.0, ax=ax0)
    xlim0, ylim0 = ax0.get_xlim(), ax0.get_ylim()
    for coll in ax0.collections:
        paths = getattr(coll, "get_paths", None)
        if not callable(paths): continue
        paths = coll.get_paths()
        if not paths: continue
        bbox = paths[0].get_extents()
        x0_, y0_, w_, h_ = bbox.bounds
        coll.set_clip_path(plt.Rectangle((x0_, y0_), w_ / 2.0, h_, transform=ax0.transData))
    sns.boxplot(data=overall_df, x="dataset", y="tweet_ng", showfliers=False, width=0.3,
                boxprops={"zorder": 3, "facecolor": "none", "edgecolor": "#333", "linewidth": 1.0},
                whiskerprops={"color": "#333", "linewidth": 1.0},
                capprops={"color": "#333", "linewidth": 1.0},
                medianprops={"color": "#333", "linewidth": 1.2}, ax=ax0)
    ax0.set_xlabel("")
    ax0.set_ylabel("NewsGuard score")
    ax0.set_ylim(-10, 110)
    ax0.set_xticklabels(["All Sensor", "All Random"], rotation=45)
    ax0.grid(True, axis="y", alpha=0.25)
    if ax0.legend_ is not None: ax0.legend_.remove()
    ax0.set_xlim(xlim0); ax0.set_ylim(ylim0)

    # Bottom panel: left/right grouped
    ax1 = axes[1]
    s_left  = sen.loc[sen["is_left"],  "tweet_ng"].dropna().values
    r_left  = rnd.loc[rnd["is_left"],  "tweet_ng"].dropna().values
    s_right = sen.loc[sen["is_right"], "tweet_ng"].dropna().values
    r_right = rnd.loc[rnd["is_right"], "tweet_ng"].dropna().values

    desired = [("Sensor Left", s_left, 1, LEFT_COLOR),
               ("Random Left", r_left, 2, LEFT_COLOR),
               ("Sensor Right", s_right, 5, RIGHT_COLOR),
               ("Random Right", r_right, 6, RIGHT_COLOR)]
    data, positions, colors_v, labels = [], [], [], []
    for name, arr, pos, color in desired:
        if len(arr) > 0:
            data.append(arr); positions.append(pos); colors_v.append(color); labels.append(name)

    if data:
        vp = ax1.violinplot(data, positions=positions, widths=0.9,
                            showmeans=False, showmedians=False, showextrema=False)
        for body, col in zip(vp["bodies"], colors_v):
            body.set_facecolor(col); body.set_edgecolor("#333"); body.set_alpha(0.9)
            paths = getattr(body, "get_paths", lambda: [])()
            if paths:
                bbox = paths[0].get_extents()
                x0_, y0_, w_, h_ = bbox.bounds
                body.set_clip_path(plt.Rectangle((x0_, y0_), w_ / 2.0, h_, transform=ax1.transData))
        bp = ax1.boxplot(data, positions=positions, widths=0.35, patch_artist=True, showfliers=False)
        for box in bp["boxes"]: box.set(facecolor="none", edgecolor="#333", linewidth=1.0)
        for w_ in bp["whiskers"]: w_.set(color="#333", linewidth=1.0)
        for c in bp["caps"]: c.set(color="#333", linewidth=1.0)
        for m_ in bp["medians"]: m_.set(color="#333", linewidth=1.2)
        ax1.set_xticks(positions); ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel("NewsGuard score")
    ax1.set_ylim(-10, 110)
    ax1.grid(True, axis="y", alpha=0.25)
    for side in ["top", "right", "bottom", "left"]:
        ax1.spines[side].set_visible(True); ax1.spines[side].set_color("black")
    # NO TITLES
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 5 / 11: user-share-weighted impressions
# (re-uses the analysis already in tweet_exposure/fp_weighted_trend*.py)
# ---------------------------------------------------------------------------
SHARES = ["left_share", "right_share", "neutral_share"]
SHARE_LABELS = {
    "left_share":    "Left-leaning",
    "right_share":   "Right-leaning",
    "neutral_share": "Neutral",
}
SHARE_COLORS = {
    "left_share":    LEFT_COLOR,
    "right_share":   RIGHT_COLOR,
    "neutral_share": NEUTRAL_COLOR,
}


def _build_base_frame(result_df, start="2024-05-01", end="2024-11-01", freq="3D"):
    """Build pre-filtered base frame (originals only, with has_ng flag, log_impr,
    bucket).  Mirrors `fp_threshold_sensitivity.py`."""
    cols = ["USER_id", "TWEET_id", "TWEET_created_at", "impression_count",
            "referenced_tweet_type", "newsguard_scores_expanded", "newsguard_orientation"]
    cols = [c for c in cols if c in result_df.columns]
    d = result_df[cols].copy()
    d["TWEET_created_at"] = to_naive_utc(d["TWEET_created_at"])
    d = d[(d["TWEET_created_at"] >= pd.Timestamp(start)) &
          (d["TWEET_created_at"] <  pd.Timestamp(end))]
    mask_ret = d["referenced_tweet_type"].map(_is_retweet)
    d = d[~mask_ret]
    d["impression_count"] = pd.to_numeric(d["impression_count"], errors="coerce")
    d = d[d["impression_count"].notna()]
    d["log_impr"] = np.log1p(d["impression_count"].clip(lower=0))
    d["bucket"]   = d["TWEET_created_at"].dt.floor(freq)
    scores_list   = d["newsguard_scores_expanded"].map(_to_list)
    d["has_ng"]   = scores_list.map(
        lambda xs: any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in xs))
    return d


def _classify_users_for_share_weighted(d, min_ng=5):
    """Build user_lean (n_left/n_right/n_neutral/total + shares) using the
    paired-explode logic from fp_threshold_sensitivity.py.  Pairs each NG
    score with its corresponding orientation per tweet, drops rows with no
    score, then normalises orientation labels."""
    def _norm(x):
        t = str(x).strip().lower()
        if t.startswith("right"): return "right"
        if t.startswith("left"):  return "left"
        return "neutral"

    ng = d.loc[d["has_ng"], ["USER_id", "newsguard_scores_expanded", "newsguard_orientation"]].copy()
    ng["newsguard_scores_expanded"] = ng["newsguard_scores_expanded"].map(_to_list)
    ng["newsguard_orientation"]     = ng["newsguard_orientation"].map(_to_list)
    # Drop rows where the two lists have mismatched lengths (rare data issue)
    len_ok = ng.apply(lambda r: len(r["newsguard_scores_expanded"]) == len(r["newsguard_orientation"]), axis=1)
    ng = ng[len_ok]
    ex = ng.explode(["newsguard_scores_expanded", "newsguard_orientation"], ignore_index=False)
    ex["score"] = pd.to_numeric(ex["newsguard_scores_expanded"], errors="coerce")
    ex = ex.dropna(subset=["score"])
    ex["lean"] = ex["newsguard_orientation"].map(_norm)

    user_lean = (ex.groupby("USER_id", observed=True)["lean"]
                 .agg(n_left=lambda s: (s == "left").sum(),
                      n_right=lambda s: (s == "right").sum(),
                      n_neutral=lambda s: (s == "neutral").sum())
                 .reset_index())
    user_lean["n_total"] = user_lean[["n_left", "n_right", "n_neutral"]].sum(axis=1)
    user_lean = user_lean[user_lean["n_total"] >= min_ng].copy()
    user_lean["right_share"]   = user_lean["n_right"]   / user_lean["n_total"]
    user_lean["left_share"]    = user_lean["n_left"]    / user_lean["n_total"]
    user_lean["neutral_share"] = user_lean["n_neutral"] / user_lean["n_total"]
    return user_lean


def _build_non_ng_frame(d, user_lean):
    """Filter base frame to non-NG tweets and merge with user_lean."""
    return (d.loc[~d["has_ng"], ["USER_id", "bucket", "log_impr"]]
              .merge(user_lean[["USER_id"] + SHARES], on="USER_id", how="inner"))


def _weighted_bucket_stats(non_ng, share_col):
    sub = non_ng[non_ng[share_col] > 0].copy()
    sub["w"] = sub[share_col]
    g = sub.groupby("bucket", observed=True)
    rows = []
    for bucket, gp in g:
        sw = gp["w"].sum()
        if sw <= 0 or len(gp) < 2: continue
        x = gp["log_impr"].values
        w = gp["w"].values
        mu = float((w * x).sum() / sw)
        n_eff = float(sw ** 2 / (w ** 2).sum())
        var = float((w * (x - mu) ** 2).sum() / sw)
        se = float(np.sqrt(var / n_eff))
        z = 1.96
        rows.append(dict(bucket=bucket, mean_log=mu, ci_lo=mu - z * se, ci_hi=mu + z * se,
                         n_tweets=len(gp), n_eff=n_eff))
    cols = ["bucket", "mean_log", "ci_lo", "ci_hi", "n_tweets", "n_eff"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("bucket").reset_index(drop=True)


def _plot_weighted_trend(frames, savepath, title=None):
    """No title. No 'Distribution' subtitle on side panel."""
    fig = plt.figure(figsize=(13, 4.5))
    gs = GridSpec(1, 2, width_ratios=[10, 1], wspace=0.30, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])

    for s in SHARES:
        sub = frames[s]
        if len(sub) == 0: continue
        m = np.expm1(sub["mean_log"]).clip(lower=1.0)
        lo = np.expm1(sub["ci_lo"]).clip(lower=1.0)
        hi = np.expm1(sub["ci_hi"]).clip(lower=1.0)
        ax.plot(sub["bucket"], m, "-o", ms=3, lw=1.8, color=SHARE_COLORS[s],
                label=f"{SHARE_LABELS[s]}: avg. impressions per tweet")
        ax.fill_between(sub["bucket"], lo, hi, color=SHARE_COLORS[s], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("Avg. impressions per tweet (95% CI) — log scale")
    ax.set_xlabel("Time")
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="lower right", frameon=True, fontsize=8, ncol=1)
    # NO TITLE

    for s in SHARES:
        sub = frames[s]
        vals = np.expm1(sub["mean_log"]).dropna().values
        vals = vals[vals > 0]
        if len(vals) < 3: continue
        log_vals = np.log(vals)
        kde = gaussian_kde(log_vals, bw_method=0.3)
        y_grid = np.linspace(log_vals.min(), log_vals.max(), 300)
        density = kde(y_grid); density = density / density.max()
        ax_dist.fill_betweenx(np.exp(y_grid), 0, density, color=SHARE_COLORS[s], alpha=0.35, linewidth=0)
        ax_dist.plot(density, np.exp(y_grid), color=SHARE_COLORS[s], lw=1.0, alpha=0.7)

    ax_dist.set_yscale("log")
    ax_dist.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_dist.set_ylim(ax.get_ylim())
    ax_dist.set_xlim(0, None)
    ax_dist.grid(False); ax_dist.grid(True, axis="y", alpha=0.2)
    ax_dist.tick_params(axis="y", labelleft=True)
    ax_dist.tick_params(axis="x", labelbottom=False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.spines["bottom"].set_visible(False)
    # NO "Distribution" subtitle

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)


def render_share_weighted_trends(result_df, savepath, *,
                                 start="2024-05-01", end="2024-11-01", min_ng=5):
    """Build the user-share-weighted trend figure (sensor or random)."""
    d = _build_base_frame(result_df, start=start, end=end, freq="3D")
    user_lean = _classify_users_for_share_weighted(d, min_ng=min_ng)
    print(f"   Pool: {len(user_lean):,} users with >={min_ng} NG-shares")
    non_ng = _build_non_ng_frame(d, user_lean)
    print(f"   Non-NG tweets matched: {len(non_ng):,}")
    frames = {s: _weighted_bucket_stats(non_ng, s) for s in SHARES}
    _plot_weighted_trend(frames, savepath)
    return frames


# ---------------------------------------------------------------------------
# Figure 10: Sensor retweet NG over time, Musk red + 3 gray + soft Sept label
# ---------------------------------------------------------------------------
def render_fig10_musk_with_soft_sept(result_sensor, savepath,
                                     start="2024-06-01", end="2024-11-01"):
    cols = ["TWEET_id", "TWEET_created_at", "newsguard_scores_expanded",
            "referenced_tweet_type", "newsguard_orientation"]
    df = result_sensor[cols].copy()
    df["TWEET_created_at"] = to_naive_utc(df["TWEET_created_at"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["TWEET_created_at"] >= start_ts) & (df["TWEET_created_at"] < end_ts)]
    df = df[df["referenced_tweet_type"].map(_is_retweet)]
    df = df[df["newsguard_orientation"].map(
        lambda x: "Right" in list(x) if isinstance(x, (list, np.ndarray)) else False)]

    scores = df[["TWEET_id", "TWEET_created_at", "newsguard_scores_expanded"]].explode(
        "newsguard_scores_expanded", ignore_index=False)
    scores["score"] = pd.to_numeric(scores["newsguard_scores_expanded"], errors="coerce")
    scores = scores.dropna(subset=["score"])

    tweet_mean = scores.groupby("TWEET_id", observed=True)["score"].mean().rename("tweet_mean")
    tweet_time = scores.groupby("TWEET_id", observed=True)["TWEET_created_at"].min()
    tm = pd.concat([tweet_mean, tweet_time], axis=1).reset_index()
    tm["bucket"] = tm["TWEET_created_at"].dt.floor("3D")
    bucketed = (tm.groupby("bucket", observed=True)["tweet_mean"]
                  .mean().rename("bucket_mean").sort_index().to_frame())
    bucketed["delta_vs_prev"] = bucketed["bucket_mean"].diff()
    deltas_all = bucketed.dropna(subset=["delta_vs_prev"]).reset_index()

    # Top 4 by absolute |Δ|
    deltas_all["abs"] = deltas_all["delta_vs_prev"].abs()
    top4 = deltas_all.nlargest(4, "abs").reset_index(drop=True)
    musk_date = pd.Timestamp("2024-07-15")
    top4["_dist"] = (pd.to_datetime(top4["bucket"]) - musk_date).abs()
    musk_idx = int(top4["_dist"].idxmin())
    musk_row = top4.loc[musk_idx]
    gray_rows = top4.drop(index=musk_idx).sort_values("bucket").reset_index(drop=True)

    # Identify "September monetization" candidate among the gray markers
    # = the marker whose bucket is closest to 2024-09-04 within ±10 days
    sept_target = pd.Timestamp("2024-09-04")
    gray_rows["_dist_sept"] = (pd.to_datetime(gray_rows["bucket"]) - sept_target).abs()
    sept_candidate_idx = int(gray_rows["_dist_sept"].idxmin())
    sept_candidate = gray_rows.loc[sept_candidate_idx]
    sept_within_window = sept_candidate["_dist_sept"] <= pd.Timedelta(days=10)

    bucketed_idx = pd.to_datetime(bucketed.reset_index()["bucket"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bucketed_idx, bucketed["bucket_mean"].values, color=NG_COLOR, lw=2, zorder=3)
    ax.set_ylabel("Avg NewsGuard (tweet-level mean)")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 100)

    # Gray markers (no event)
    for _, row in gray_rows.iterrows():
        x_g = pd.to_datetime(row["bucket"])
        delta_g = float(row["delta_vs_prev"])
        y_g = float(bucketed.reset_index().loc[(bucketed_idx - x_g).abs().idxmin(), "bucket_mean"])
        ax.axvline(x=x_g, color=GRAY_COLOR, linestyle="--", alpha=0.55, lw=1.2, zorder=2)
        ax.plot(x_g, y_g, "o", color=GRAY_COLOR, ms=5, zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(x_g, 92, f"Δ = {delta_g:+.1f}",
                rotation=0, ha="center", va="top", color=GRAY_COLOR, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=GRAY_COLOR, alpha=0.85, lw=0.6),
                zorder=5)
        # Soft September sub-label: italic gray, just below Δ box
        if int(_) == sept_candidate_idx and sept_within_window:
            ax.text(x_g, 84, "tentative — see §4.4",
                    rotation=0, ha="center", va="top",
                    color=GRAY_COLOR, fontsize=7, fontstyle="italic", alpha=0.85, zorder=5)

    # Red Musk marker
    x = pd.to_datetime(musk_row["bucket"])
    delta = float(musk_row["delta_vs_prev"])
    y_data = float(bucketed.reset_index().loc[(bucketed_idx - x).abs().idxmin(), "bucket_mean"])
    ax.axvline(x=x, color=DECAY_COLOR, linestyle="-", alpha=0.55, lw=1.8, zorder=2)
    ax.plot(x, y_data, "o", color=DECAY_COLOR, ms=7, zorder=5,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x, 18.0, f" Δ = {delta:+.1f} ", rotation=90, va="bottom", ha="center",
            color=DECAY_COLOR, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=DECAY_COLOR, alpha=0.9, lw=0.8), zorder=6)

    fig.subplots_adjust(bottom=0.30)
    ax.annotate(
        "14.–16.7.2024\nMusk – Trump endorsement.\nTwitter algorithm shift fosters\nrepublican-leaning news outlets.",
        xy=(x, y_data), xycoords="data",
        xytext=(0.18, -0.20), textcoords="axes fraction",
        ha="center", va="top", fontsize=8.5, color=TEXT_GRAY,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=DECAY_COLOR, alpha=0.95, lw=1.2),
        arrowprops=dict(arrowstyle="->", color=DECAY_COLOR, lw=1.2, shrinkA=6, shrinkB=6),
        annotation_clip=False, zorder=10,
    )
    ax.annotate(
        "gray markers: other three strongest change-points;\nno real-world event identified.",
        xy=(0.78, -0.20), xycoords="axes fraction",
        ha="center", va="top", fontsize=8.5, color=TEXT_GRAY,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY_COLOR, alpha=0.95, lw=1.0),
        annotation_clip=False, zorder=10,
    )
    ax.grid(True, axis="y", alpha=0.15)
    # NO TITLE
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: lead time of links — iter_mean_vs_median (parallel bootstrap)
# ---------------------------------------------------------------------------
START_DATE_LEAD = "2024-06-01"
END_DATE_LEAD   = "2024-11-02"


def _prep_link_mentions(df):
    d = df[["TWEET_created_at", "urls_expanded"]].copy()
    d["TWEET_created_at"] = to_naive_utc(d["TWEET_created_at"])
    d = d[(d["TWEET_created_at"] >= START_DATE_LEAD) & (d["TWEET_created_at"] < END_DATE_LEAD)]
    d["urls_expanded"] = d["urls_expanded"].apply(_to_list)
    d = d.explode("urls_expanded", ignore_index=True)
    d["link"] = d["urls_expanded"].astype(str).str.strip()
    d = d[(d["link"] != "") & d["link"].notna()]
    d["week"] = d["TWEET_created_at"].dt.to_period("W").apply(lambda r: r.start_time)
    return d[["link", "TWEET_created_at", "week"]].reset_index(drop=True)


def _first_per_link_week(df_mentions):
    return (df_mentions.groupby(["link", "week"], observed=True)["TWEET_created_at"]
            .min().rename("first_time").reset_index())


def _week_index_map(df_mentions):
    return {w: idx.to_numpy() for w, idx in df_mentions.groupby("week", observed=True).groups.items()}


def _lead_from_firsts(fr, fs):
    pairs = fs.merge(fr, on=["link", "week"], suffixes=("_sensor", "_random"))
    pairs["lead_minutes"] = (pairs["first_time_random"] - pairs["first_time_sensor"]).dt.total_seconds() / 60.0
    return pairs


# Globals for parallel workers (avoid pickling large objects per task)
_G_S = None
_G_FR = None
_G_IDX_S = None
_G_WEEKS = None
_G_WEEK_TARGETS = None
_G_RS = 42


def _set_lead_globals(prep, random_state=42):
    global _G_S, _G_FR, _G_IDX_S, _G_WEEKS, _G_WEEK_TARGETS, _G_RS
    _G_S = prep["s"]
    _G_FR = prep["fr"]
    _G_IDX_S = prep["idx_s"]
    _G_WEEKS = prep["weeks"]
    _G_WEEK_TARGETS = prep["week_targets"]
    _G_RS = random_state


def _one_iter_lead(iter_id):
    rng = np.random.default_rng(_G_RS + iter_id)
    sampled_idx = []
    for w in _G_WEEKS:
        src = _G_IDX_S.get(w, [])
        n = _G_WEEK_TARGETS[w]
        if n == 0 or len(src) == 0:
            continue
        sampled_idx.append(src if len(src) <= n else rng.choice(src, size=n, replace=False))
    if not sampled_idx:
        return {"iter": iter_id, "pairs": 0, "median_lead_minutes": np.nan,
                "mean_lead_minutes": np.nan, "sensor_earlier_share": np.nan}
    s_sampled = _G_S.iloc[np.concatenate(sampled_idx)].reset_index(drop=True)
    fs = _first_per_link_week(s_sampled)
    pairs = _lead_from_firsts(_G_FR, fs)
    if len(pairs) == 0:
        return {"iter": iter_id, "pairs": 0, "median_lead_minutes": np.nan,
                "mean_lead_minutes": np.nan, "sensor_earlier_share": np.nan}
    return {
        "iter": iter_id,
        "pairs": int(len(pairs)),
        "median_lead_minutes": float(pairs["lead_minutes"].median()),
        "mean_lead_minutes": float(pairs["lead_minutes"].mean()),
        "sensor_earlier_share": float((pairs["lead_minutes"] > 0).mean()),
    }


def run_lead_bootstrap(sensor_df, random_df, iterations=1000, n_cores=50, random_state=42):
    """Run the parallel bootstrap for the link-level lead-time analysis."""
    import sys
    sys.path.insert(0, "/home/mangermaier/friendship_paradox")
    from parallel import parallel_process

    print(f"   prep ...")
    r = _prep_link_mentions(random_df)
    s = _prep_link_mentions(sensor_df)
    fr = _first_per_link_week(r)
    idx_s = _week_index_map(s)
    idx_r = _week_index_map(r)
    week_targets = {w: min(len(idx_s.get(w, [])), len(idx_r.get(w, [])))
                    for w in set(idx_r) | set(idx_s)}
    weeks = [w for w, n in week_targets.items() if n > 0]
    prep = {"r": r, "s": s, "fr": fr, "idx_s": idx_s, "idx_r": idx_r,
            "week_targets": week_targets, "weeks": weeks}

    _set_lead_globals(prep, random_state=random_state)
    print(f"   bootstrap: {iterations} iters x {n_cores} cores ...")
    results = parallel_process(_one_iter_lead, items=list(range(iterations)),
                                n_cores=n_cores, show_progress=True, collect_results=True)
    ds = pd.DataFrame(results)

    def _ci(x):
        x = pd.Series(x).dropna()
        return (float(x.quantile(0.025)) if len(x) else np.nan,
                float(x.quantile(0.975)) if len(x) else np.nan)
    med_ci = _ci(ds["median_lead_minutes"]); mean_ci = _ci(ds["mean_lead_minutes"])
    print(f"   median lead (min): {ds['median_lead_minutes'].mean():.2f}  "
          f"95% CI [{med_ci[0]:.2f}, {med_ci[1]:.2f}]")
    print(f"   mean lead   (min): {ds['mean_lead_minutes'].mean():.2f}  "
          f"95% CI [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]")
    return ds


def plot_iter_mean_vs_median(ds_iter, savepath=None):
    """No title; same layout as cell 217."""
    vals_median = ds_iter["median_lead_minutes"].dropna()
    vals_mean   = ds_iter["mean_lead_minutes"].dropna()
    vals_share  = ds_iter["sensor_earlier_share"].dropna()

    med_mu = float(vals_median.mean())
    med_lo, med_hi = float(vals_median.quantile(0.025)), float(vals_median.quantile(0.975))
    mea_mu = float(vals_mean.mean())
    mea_lo, mea_hi = float(vals_mean.quantile(0.025)), float(vals_mean.quantile(0.975))

    xmin = min(med_lo, mea_lo); xmax = max(med_hi, mea_hi)
    xpad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    xmin, xmax = xmin - xpad, xmax + xpad

    fig, ax = plt.subplots(figsize=(10, 3.6))
    y_med, y_mea = 1.0, 0.0
    ax.errorbar(x=med_mu, y=y_med, xerr=[[med_mu - med_lo], [med_hi - med_mu]],
                fmt="o", ms=8, color="tab:blue", ecolor="tab:blue",
                elinewidth=2, capsize=6, capthick=2, zorder=3)
    ax.errorbar(x=mea_mu, y=y_mea, xerr=[[mea_mu - mea_lo], [mea_hi - mea_mu]],
                fmt="o", ms=8, color="tab:orange", ecolor="tab:orange",
                elinewidth=2, capsize=6, capthick=2, zorder=3)
    ax.text(med_mu - 0.02 * (xmax - xmin), y_med + 0.3,
            f"median = {med_mu:.2f}\n95% CI [{med_lo:.2f}, {med_hi:.2f}]",
            ha="right", va="bottom", color="#222",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f4f4f4", ec="none", alpha=0.9))
    ax.text(mea_mu - 0.02 * (xmax - xmin), y_mea + 0.3,
            f"mean = {mea_mu:.2f}\n95% CI [{mea_lo:.2f}, {mea_hi:.2f}]",
            ha="right", va="bottom", color="#222",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f4f4f4", ec="none", alpha=0.9))
    if len(vals_share) > 0:
        share_mu = float(vals_share.mean())
        ax.text(0.5, 0.9, f"Sensor earlier share\nmean = {share_mu:.3f}",
                transform=ax.transAxes, ha="center", va="top", color="#1b1b1b",
                bbox=dict(boxstyle="round,pad=0.35", fc="tab:green", ec="none", alpha=0.25))
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.5, 1.6)
    ax.set_yticks([y_mea, y_med])
    ax.set_yticklabels(["Mean lead (minutes)", "Median lead (minutes)"])
    ax.set_xlabel("Lead time (minutes)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(False, axis="y")
    if ax.legend_ is not None: ax.legend_.remove()
    # NO TITLE
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Cached data loader — reuses already-loaded sensor/random_ on re-run
# ---------------------------------------------------------------------------
def _load_data_cached():
    import sys
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        cached_s = getattr(main_mod, "sensor", None)
        cached_r = getattr(main_mod, "random_", None)
        if (cached_s is not None and cached_r is not None
            and "newsguard_orientation" in cached_r.columns):
            print("[cache] Reusing already-loaded sensor/random_ from kernel namespace.")
            return cached_s, cached_r
    print("\n[1/3] Loading parquets ...")
    sensor = load_parquet(SENSOR_PARQUET, "sensor v5")
    random_ = load_parquet(RANDOM_PARQUET, "random v5")
    if "newsguard_orientation" not in random_.columns:
        print("\n[2/3] Random parquet has no newsguard_orientation column — rebuilding it")
        print("       via domain -> orientation map derived from Sensor.")
        dom_map = build_domain_orientation_map(sensor)
        print(f"       Domain map size: {len(dom_map):,}")
        random_ = inject_orientation_from_domains(random_, dom_map)
    else:
        print("\n[2/3] Random parquet has native newsguard_orientation column — using as-is.")
    if main_mod is not None:
        setattr(main_mod, "sensor", sensor)
        setattr(main_mod, "random_", random_)
    return sensor, random_


# ---------------------------------------------------------------------------
# Main: run everything
# ---------------------------------------------------------------------------
def main():
    print("="*78)
    print("REPRODUCING ALL PAPER FIGURES (no titles, fresh from raw data)")
    print(f"Output dir: {OUT_DIR}")
    print("="*78)

    sensor, random_ = _load_data_cached()

    rendered = []
    skip_existing = bool(os.environ.get("SKIP_EXISTING", ""))

    def _try(name, fn):
        out_path = OUT_DIR / name
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            print(f"  ⊝ {name}  SKIPPED (already exists)")
            rendered.append(name)
            return
        t0 = time.time()
        try:
            fn()
            print(f"  ✓ {name}  ({time.time()-t0:.1f}s)")
            rendered.append(name)
        except Exception as e:
            print(f"  ✗ {name}  FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\n[3/3] Rendering figures ...")

    _try("ng_overall_violin.svg",
         lambda: plot_ng_overall_violin(sensor, random_,
                                        savepath=str(OUT_DIR / "ng_overall_violin.svg")))
    _try("ng_left_right_grouped_violin.svg",
         lambda: plot_ng_left_right_grouped_violin(sensor, random_,
                                                    savepath=str(OUT_DIR / "ng_left_right_grouped_violin.svg")))
    _try("ng_combined_overall_and_left_right.svg",
         lambda: plot_ng_overall_and_left_right(sensor, random_,
                                                savepath=str(OUT_DIR / "ng_combined_overall_and_left_right.svg")))
    _try("sensor_random_newsguard_3day_left_right_4panel.svg",
         lambda: plot_left_right_3day_newsguard_4panel([(sensor, "sensor"), (random_, "random")],
                                                       savepath=str(OUT_DIR / "sensor_random_newsguard_3day_left_right_4panel.svg")))
    _try("random_left_newsguard_3day.svg",
         lambda: plot_single_panel_newsguard_3day(random_, "left",
                                                   savepath=str(OUT_DIR / "random_left_newsguard_3day.svg")))
    _try("random_right_newsguard_3day.svg",
         lambda: plot_single_panel_newsguard_3day(random_, "right",
                                                   savepath=str(OUT_DIR / "random_right_newsguard_3day.svg")))
    _try("ng_impressions_left_right_neutral_log1p.svg",
         lambda: (lambda a: plot_impressions_by_orientation(
             a, include_neutral=True,
             savepath=str(OUT_DIR / "ng_impressions_left_right_neutral_log1p.svg")))(
             analyze_impressions_by_orientation(sensor, freq="3D", tweet_kind="original")))
    _try("tweet_exposure_weighted_trend.svg",
         lambda: render_share_weighted_trends(sensor,
                                              savepath=str(OUT_DIR / "tweet_exposure_weighted_trend.svg")))
    _try("tweet_exposure_weighted_trend_random.svg",
         lambda: render_share_weighted_trends(random_,
                                              savepath=str(OUT_DIR / "tweet_exposure_weighted_trend_random.svg")))
    _try("sensor_retweet_ng_musk_plus_gray_3D.svg",
         lambda: render_fig10_musk_with_soft_sept(sensor,
                                                   savepath=str(OUT_DIR / "sensor_retweet_ng_musk_plus_gray_3D.svg")))

    # Figure 3: iter_mean_vs_median (requires parallel bootstrap)
    print("\n  iter_mean_vs_median.svg (parallel bootstrap, 1000 iters x 100 cores) ...")
    _try("iter_mean_vs_median.svg",
         lambda: plot_iter_mean_vs_median(
             run_lead_bootstrap(sensor, random_, iterations=1000, n_cores=100),
             savepath=str(OUT_DIR / "iter_mean_vs_median.svg")))

    print("\n" + "="*78)
    print(f"Done. {len(rendered)} figures rendered to {OUT_DIR}")
    for f in rendered:
        print(f"  • {f}")
    print("="*78)


if __name__ == "__main__":
    main()

"""Regenerate every figure in the report from the artefacts in outputs/.

Figures are named by their number in the report, so fig3_models.pdf is Figure 3.
Figure 1 is the framework diagram and is drawn by hand, not here.

Nothing in this file contains a value copied from the report. Every number is read
from outputs/, so a figure cannot disagree with the artefact it is drawn from.

    python figures/make_figures.py            # writes PDF, the format the report uses
    python figures/make_figures.py --png      # also writes PNG, for previewing

Requires: outputs/daily_coverage.csv, prevalence.csv, model_comparison.csv,
detectability.csv, qualification.csv, trends_series.csv, and, for Figure 5,
outputs/coding_sheet.csv (not in version control; Figure 5 is skipped without it).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# embed real text in the PDF rather than outlines, so captions stay searchable
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.25
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

root = Path(__file__).resolve().parent.parent
out_folder = root / "outputs"
fig_folder = root / "figures"

accent = "#2563eb"      # emphasis in the time-series and lag figures
accent_alt = "#0f766e"  # emphasis in the model comparison, so that it does not
                        # read as a continuation of the preceding figure
neutral = "#cbd5e1"     # unemphasised bars
neutral_line = "#94a3b8"  # unemphasised series

# External onset criterion, as standard deviations above the baseline mean.
# This MUST match the value used in section 13 of the notebook. It is named
# here, and printed in the legend, so that a mismatch is visible in the figure
# rather than hidden in it.
threshold_sd = 1.5
gap_band = "#fca5a5"
ink = "#1e293b"

formats = ["pdf"]


def load(name, required=True):
    path = out_folder / name
    if not path.exists():
        if required:
            sys.exit(f"missing artefact: {path}\nrun the notebook first")
        return None
    return pd.read_csv(path)


def column(frame, *candidates):
    """Find a column by exact name, then case-insensitively, then by substring."""
    for want in candidates:
        if want in frame.columns:
            return want
    lowered = {c.lower(): c for c in frame.columns}
    for want in candidates:
        if want.lower() in lowered:
            return lowered[want.lower()]
    for want in candidates:
        for c in frame.columns:
            if want.lower() in c.lower():
                return c
    sys.exit(f"none of {candidates} found. columns are: {list(frame.columns)}")


def save(fig, stem):
    fig_folder.mkdir(exist_ok=True)
    for fmt in formats:
        path = fig_folder / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
        print(f"  {path.relative_to(root)}")
    plt.close(fig)


def shade_gaps(axis, all_days, observed, label=True):
    """Mark uncollected dates. A gap is not a zero and must not look like one."""
    missing = [d for d in all_days if d not in observed]
    if not missing:
        return
    first = True
    for day in missing:
        axis.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12),
                     color=gap_band, alpha=0.30, linewidth=0,
                     label="Not collected" if (first and label) else None)
        first = False


# ---------------------------------------------------------------------------
# Figure 2  Daily document volume
# ---------------------------------------------------------------------------
def figure_2():
    frame = load("daily_coverage.csv")
    date_col = column(frame, "Date", "day")
    doc_col = column(frame, "Documents")
    hour_col = column(frame, "Hours Collected", "hours")

    frame[date_col] = pd.to_datetime(frame[date_col])
    counts = pd.to_numeric(frame[doc_col], errors="coerce")
    hours = pd.to_numeric(frame[hour_col], errors="coerce")

    fig, axis = plt.subplots(figsize=(9, 3.6))
    axis.bar(frame[date_col], counts.fillna(0), width=0.7,
             color=accent, label="Documents after cleaning")

    shade_gaps(axis, list(frame[date_col]),
               set(frame.loc[counts.notna(), date_col]))

    for day, n, h in zip(frame[date_col], counts, hours):
        if pd.notna(n) and pd.notna(h):
            axis.annotate(f"{int(h)}h", (day, n), textcoords="offset points",
                          xytext=(0, 3), ha="center", fontsize=7, color=ink)

    axis.set_ylabel("Documents in the analysis corpus")
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_xticks(list(frame[date_col]))
    axis.set_xticklabels([d.strftime("%d %b") for d in frame[date_col]],
                         rotation=45, ha="right")
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "fig2_coverage")


# ---------------------------------------------------------------------------
# Figure 3  Model comparison
# ---------------------------------------------------------------------------
def figure_3():
    frame = load("model_comparison.csv")
    name_col = column(frame, "Model")
    labels = frame[name_col].tolist()

    panels = [
        (column(frame, "Coherence"), "Coherence\n(|UMass|, shorter is better)", True),
        (column(frame, "Diversity"), "Topic diversity\n(longer is better)", False),
        (column(frame, "Seed Consistency", "Seed consistency"),
         "Seed consistency\n(mean pairwise ARI, longer is better)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    best_row = frame[column(frame, "Coherence")].abs().idxmin()

    for axis, (col, title, use_abs) in zip(axes, panels):
        values = frame[col].astype(float)
        plotted = values.abs() if use_abs else values
        colours = [accent_alt if i == best_row else neutral for i in frame.index]
        axis.barh(labels, plotted, color=colours, height=0.6)
        for y, (shown, raw) in enumerate(zip(plotted, values)):
            axis.annotate(f"{raw:.3f}", (shown, y), textcoords="offset points",
                          xytext=(4, 0), va="center", fontsize=7.5, color=ink)
        axis.set_title(title, fontsize=8.5)
        axis.invert_yaxis()
        axis.set_xlim(0, plotted.max() * 1.25)
        axis.grid(axis="y", visible=False)

    fig.tight_layout()
    save(fig, "fig3_models")


# ---------------------------------------------------------------------------
# Figure 4  Relative prevalence of the eight candidates
# ---------------------------------------------------------------------------
def figure_4():
    frame = load("prevalence.csv")
    qualification = load("qualification.csv")

    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    all_days = list(frame[date_col])

    topic_col = column(qualification, "Topic")
    qual_col = column(qualification, "Qualifies")
    qualifying = {
        str(t) for t, q in zip(qualification[topic_col], qualification[qual_col])
        if str(q).strip().lower() in {"yes", "true", "1"}
    }

    # prevalence.csv holds every topic; the figure shows the candidates only,
    # in the order qualification.csv lists them
    topics, missing = [], []
    for t in qualification[topic_col]:
        name = str(int(t)) if str(t).replace(".0", "").isdigit() else str(t)
        if name in frame.columns:
            topics.append(name)
        elif str(t) in frame.columns:
            topics.append(str(t))
        else:
            missing.append(t)
    if missing:
        print(f"  warning: no prevalence column for topics {missing}")
    if not topics:
        sys.exit("no candidate topic matched a column in prevalence.csv; "
                 f"columns are {list(frame.columns)[:12]}...")

    rows = int(np.ceil(len(topics) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for axis, topic in zip(axes, topics):
        series = pd.to_numeric(frame[topic], errors="coerce")
        observed = set(frame.loc[series.notna(), date_col])
        is_qualifying = str(topic) in qualifying

        shade_gaps(axis, all_days, observed, label=False)
        axis.plot(frame[date_col], series, "o-",
                  color=accent if is_qualifying else neutral_line,
                  linewidth=1.8 if is_qualifying else 1.1, markersize=3.5)

        axis.set_title(f"Topic {topic}" + (" (qualifying)" if is_qualifying else ""),
                       fontsize=8.5,
                       fontweight="bold" if is_qualifying else "normal")
        axis.set_ylabel("prevalence", fontsize=7.5)
        axis.tick_params(labelsize=7)
        for label in axis.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    for axis in axes[len(topics):]:
        axis.set_visible(False)

    fig.tight_layout()
    save(fig, "fig4_prevalence")


# ---------------------------------------------------------------------------
# Figure 5  Coding disagreements
# ---------------------------------------------------------------------------
def figure_5():
    frame = load("coding_sheet.csv", required=False)
    if frame is None:
        print("  skipped fig5: outputs/coding_sheet.csv is not present")
        return

    first = column(frame, "rater_1")
    second = column(frame, "rater_2")
    topic_col = column(frame, "topic")

    frame = frame.dropna(subset=[first, second])
    frame[first] = frame[first].astype(int)
    frame[second] = frame[second].astype(int)
    disagreed = frame[frame[first] != frame[second]]

    pairs = (disagreed.groupby([first, second]).size()
             .sort_values(ascending=False))
    by_topic = disagreed[topic_col].value_counts()

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.2))

    pair_labels = [f"{a} vs {b}" for a, b in pairs.index]
    left.barh(pair_labels, pairs.values, color=accent, height=0.6)
    left.set_title("Disagreements by label pair", fontsize=9)
    left.set_xlabel("Documents", fontsize=8)
    left.invert_yaxis()
    left.grid(axis="y", visible=False)

    right.barh([f"Topic {t}" for t in by_topic.index], by_topic.values,
               color=accent, height=0.6)
    right.set_title("Disagreements by topic", fontsize=9)
    right.set_xlabel("Documents", fontsize=8)
    right.invert_yaxis()
    right.grid(axis="y", visible=False)

    for axis in (left, right):
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    save(fig, "fig5_disagreements")


# ---------------------------------------------------------------------------
# Figure 6  Prevalence against external search interest, qualifying topic
# ---------------------------------------------------------------------------
def figure_6():
    prevalence = load("prevalence.csv")
    trends = load("trends_series.csv")
    qualification = load("qualification.csv")

    topic_col = column(qualification, "Topic")
    qual_col = column(qualification, "Qualifies")
    qualifying = [str(t) for t, q in
                  zip(qualification[topic_col], qualification[qual_col])
                  if str(q).strip().lower() in {"yes", "true", "1"}]
    if not qualifying:
        print("  skipped fig6: no qualifying topic in qualification.csv")
        return
    topic = qualifying[0]

    if topic not in trends.columns:
        print(f"  skipped fig6: no column '{topic}' in trends_series.csv "
              f"(has {list(trends.columns)})")
        return

    date_col = prevalence.columns[0]
    prevalence[date_col] = pd.to_datetime(prevalence[date_col])
    series = pd.to_numeric(prevalence[topic], errors="coerce")
    observed = set(prevalence.loc[series.notna(), date_col])
    window_start = min(observed)

    trend_date = trends.columns[0]
    trends[trend_date] = pd.to_datetime(trends[trend_date])
    external = pd.to_numeric(trends[topic], errors="coerce")
    baseline = external[trends[trend_date] < window_start]

    fig, (upper, lower) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)

    shade_gaps(upper, list(prevalence[date_col]), observed)
    upper.plot(prevalence[date_col], series, "o-", color=accent, markersize=4)
    upper.set_ylabel("Relative prevalence", fontsize=8)
    upper.axvline(window_start, color=ink, linewidth=1)
    upper.legend(frameon=False, fontsize=7.5, loc="upper right")

    lower.plot(trends[trend_date], external, color="#ea580c", linewidth=1.2)
    lower.axvspan(trends[trend_date].min(), window_start,
                  color=neutral, alpha=0.22, linewidth=0, label="Baseline period")
    lower.axhline(baseline.mean(), color=ink, linestyle="-", linewidth=0.8,
                  label="Baseline mean")
    lower.axhline(baseline.mean() + threshold_sd * baseline.std(), color=ink,
                  linestyle="--", linewidth=0.8,
                  label=f"Threshold (mean + {threshold_sd} SD)")
    lower.axvline(window_start, color=ink, linewidth=1,
                  label="Observation window opens")
    lower.set_ylabel("Relative search interest\n(index 0–100)", fontsize=8)
    lower.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)

    for axis in (upper, lower):
        axis.tick_params(labelsize=7.5)
    for label in lower.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    fig.tight_layout()
    save(fig, "fig6_paired")


# ---------------------------------------------------------------------------
# Figure 7  Retrospective against prospective detectability
# ---------------------------------------------------------------------------
def figure_7():
    frame = load("detectability.csv")
    qualification = load("qualification.csv")

    topic_col = column(frame, "Topic")
    lag_col = column(frame, "Lag (days)", "lag")

    frame = frame.dropna(subset=[lag_col]).sort_values(lag_col, ascending=False)

    q_topic = column(qualification, "Topic")
    q_flag = column(qualification, "Qualifies")
    qualifying = {t for t, q in zip(qualification[q_topic], qualification[q_flag])
                  if str(q).strip().lower() in {"yes", "true", "1"}}

    labels = [f"Topic {int(t)}" for t in frame[topic_col]]
    colours = [accent if t in qualifying else neutral for t in frame[topic_col]]

    fig, axis = plt.subplots(figsize=(8, 3.2))
    axis.barh(labels, frame[lag_col].astype(float), color=colours, height=0.6)
    for y, value in enumerate(frame[lag_col].astype(float)):
        axis.annotate(f"{value:.0f}d", (value, y), textcoords="offset points",
                      xytext=(4, 0), va="center", fontsize=7.5, color=ink)
    axis.set_xlabel("Days between retrospective appearance and prospective detectability",
                    fontsize=8)
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axis.set_xlim(0, frame[lag_col].astype(float).max() * 1.15 + 0.5)

    fig.tight_layout()
    save(fig, "fig7_detectability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true",
                        help="also write PNG alongside the PDF")
    args = parser.parse_args()
    if args.png:
        formats.append("png")

    for name, function in [("Figure 2", figure_2), ("Figure 3", figure_3),
                           ("Figure 4", figure_4), ("Figure 5", figure_5),
                           ("Figure 6", figure_6), ("Figure 7", figure_7)]:
        print(name)
        function()


if __name__ == "__main__":
    main()
"""Step 8: actual vs forecast with the 80% range over the test period."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from brain.clean import ROOT
from brain.train import METRICS, TEST_PREDS

FIGURE = ROOT / "reports" / "forecast_test.png"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
FORECAST, BASELINE, BAND = "#2a78d6", "#eb6834", "#cde2fb"
ZOOM_DAYS = 7


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_ylabel("% silica concentrate (t+1)", color=INK_2, fontsize=9)


def draw(ax, p, show_baseline):
    # Plot at the hour being predicted (t+1), not the forecast origin.
    x = p.index + pd.Timedelta(hours=1)
    ax.fill_between(x, p["lo"], p["hi"], color=BAND, linewidth=0, label="80% range")
    ax.plot(x, p["actual"], color=INK, linewidth=1.2, label="Actual (lab)")
    ax.plot(x, p["forecast"], color=FORECAST, linewidth=1.6, label="Model forecast")
    if show_baseline:
        ax.plot(x, p["baseline"], color=BASELINE, linewidth=1.2, linestyle="--",
                label="Baseline (last lab value)")


def main():
    p = pd.read_csv(TEST_PREDS, index_col="origin", parse_dates=["origin"])
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    mae, cov = m["mae"], m["range_80"]["test_coverage"]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[1, 1.1])
    fig.patch.set_facecolor(SURFACE)
    for ax in (top, bottom):
        style(ax)

    draw(top, p, show_baseline=False)
    top.set_title(
        f"Test period: model MAE {mae['model_test']:.3f} vs baseline {mae['baseline_test']:.3f} "
        f"({m['test_improvement_pct']:+.1f}%) -> {m['gate']['decision']}   |   "
        f"80% range covers {cov:.0%} of actuals",
        loc="left", fontsize=10.5, color=INK,
    )
    top.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    start = p.index.min() + pd.Timedelta(days=7)
    zoom = p.loc[start : start + pd.Timedelta(days=ZOOM_DAYS)]
    draw(bottom, zoom, show_baseline=True)
    top.axvspan(zoom.index.min(), zoom.index.max(), color=GRID, alpha=0.5, linewidth=0)
    bottom.set_title(f"Zoom: {ZOOM_DAYS} days (shaded above), with the baseline", loc="left",
                     fontsize=10.5, color=INK)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    bottom.legend(loc="upper left", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2)

    fig.suptitle("Silica forecast, 1 hour ahead (iron-ore flotation plant, 2017)",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, facecolor=SURFACE)
    print(f"[plot] -> {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

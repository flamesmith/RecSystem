"""Interactive version of notebooks/review_category_pairs.ipynb §6-7.

Pick a source category (cat_2 > cat_3) and toggle which of its destination
categories are broken out versus folded into "Other" -- everything else in
that notebook is a fixed chart; this lets you explore it.

Run with:  shiny run notebooks/review_category_shiny/app.py

Data is precomputed by precompute.py into cache/c3_counts_{N}d.parquet for
N in (30, 90, 180) rather than rebuilt live -- the underlying pipeline reads
6.9M reviews and takes ~10-20s per window, far too slow for a reactive control.
Re-run precompute.py if you need a different N.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

CACHE_DIR = Path(__file__).parent / "cache"
WINDOWS = ("30", "90", "180")

# Fixed categorical order (validated for CVD/contrast) -- color is assigned by
# a category's RANK for its source, once, and never reassigned when other
# checkboxes change. "Color follows the entity, never its rank [of what's
# currently visible]": unchecking one destination must not repaint the rest.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#c9c7c1"
INK_MUTED = "#898781"
INK_TEXT = "#1a1a1a"
MAX_CANDIDATES = len(PALETTE)   # one destination per palette slot, at most


_cache: dict[str, pd.DataFrame] = {}


def load_counts(window_days: str) -> pd.DataFrame:
    if window_days not in _cache:
        _cache[window_days] = pd.read_parquet(CACHE_DIR / f"c3_counts_{window_days}d.parquet")
    return _cache[window_days]


_default = load_counts("90")
SOURCE_CHOICES = (
    _default.groupby("src_label")["edges"].sum().sort_values(ascending=False).index.tolist()
)


def build_candidates(df: pd.DataFrame, source_label: str, max_candidates: int) -> pd.DataFrame:
    """This source's top destinations, in a rank order fixed regardless of
    which of them end up checked -- so color/order never depend on the
    current selection, only on the source and the candidate-pool size."""
    rows = (df[df.src_label == source_label]
            .sort_values("share", ascending=False)
            .head(max_candidates)
            .reset_index(drop=True))
    rows["color"] = [PALETTE[i] for i in range(len(rows))]
    return rows


def build_chart(rows: pd.DataFrame, selected: set[str]):
    """Pure function: candidate rows (with a fixed `color` per row) + the set
    of dst_label values currently checked -> a matplotlib Figure. Kept free of
    any Shiny/reactive object so it can be called directly in a test."""
    rows = rows.copy()
    rows["shown"] = rows["dst_label"].isin(selected)
    shown = rows[rows["shown"]].sort_values("share", ascending=True)
    other_share = max(0.0, 1.0 - rows.loc[rows["shown"], "share"].sum())

    n_bars = len(shown) + (1 if other_share > 0 else 0)
    fig, ax = plt.subplots(figsize=(7.5, max(1.8, 0.55 * n_bars + 0.8)))
    if n_bars == 0:
        ax.text(0.5, 0.5, "Nothing selected", ha="center", va="center", color=INK_MUTED)
        ax.axis("off")
        return fig

    y = np.arange(n_bars)
    widths = list(shown["share"]) + ([other_share] if other_share > 0 else [])
    colors = list(shown["color"]) + ([OTHER_COLOR] if other_share > 0 else [])
    labels = ([l.split(" > ")[-1] for l in shown["dst_label"]]
              + (["Other"] if other_share > 0 else []))

    ax.barh(y, widths, color=colors, height=0.65)
    for yi, w in zip(y, widths):
        ax.text(w, yi, f"  {w:.1%}", va="center", ha="left", fontsize=9.5, color=INK_TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, max(widths) * 1.22 if widths else 1.0)
    ax.set_xlabel("share of this source category's follow-up reviews", color=INK_MUTED)
    ax.tick_params(axis="x", colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(axis="x", color="#e5e3dd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_radio_buttons(
            "window", "Follow-up window",
            {w: f"{w} days" for w in WINDOWS}, selected="90",
        ),
        ui.input_selectize(
            "source", "Source category (cat_2 > cat_3)",
            choices=SOURCE_CHOICES, selected=SOURCE_CHOICES[0],
        ),
        ui.input_slider(
            "max_candidates", "Candidate destinations (by share)",
            min=3, max=MAX_CANDIDATES, value=5,
        ),
        ui.output_ui("dest_checks"),
        width=380,
    ),
    ui.h4(ui.output_text("title_text")),
    ui.output_plot("chart", height="440px"),
    ui.h5("Exact numbers"),
    ui.output_data_frame("table"),
)


def server(input, output, session):

    @reactive.calc
    def counts() -> pd.DataFrame:
        return load_counts(input.window())

    @reactive.calc
    def candidates() -> pd.DataFrame:
        return build_candidates(counts(), input.source(), input.max_candidates())

    @output
    @render.ui
    def dest_checks():
        rows = candidates()
        choices = {row.dst_label: f"{row.dst_label.split(' > ')[-1]}  ({row.share:.1%})"
                  for row in rows.itertuples()}
        return ui.input_checkbox_group(
            "dest_selected", "Shown (uncheck to fold into \"Other\")",
            choices=choices, selected=list(choices.keys()),
        )

    @output
    @render.text
    def title_text():
        df = counts()
        total = int(df.loc[df.src_label == input.source(), "edges"].sum())
        return f"{input.source()}  —  {total:,} next-{input.window()}-day pairs"

    @output
    @render.plot
    def chart():
        selected = set(input.dest_selected() or [])
        return build_chart(candidates(), selected)

    @output
    @render.data_frame
    def table():
        rows = candidates().copy()
        selected = set(input.dest_selected() or [])
        rows["shown"] = rows["dst_label"].isin(selected)
        rows["share"] = (rows["share"] * 100).round(1).astype(str) + "%"
        rows = rows.rename(columns={"dst_label": "destination category"})
        return render.DataGrid(
            rows[["destination category", "edges", "share", "shown"]], width="100%",
        )


app = App(app_ui, server)

"""
eda_toolkit
===========

Trip-quality and passenger-demand diagnostics toolkit for bus routes.

A curated collection of visualization and table utilities for the
exploratory analysis of trip records (passengers, duration, status) and
of aggregated demand series (passengers per time slot) in Jupyter
notebooks.

All figures are rendered inline as vector SVG by default (resolution
independent, ideal for reports and papers), with an optional raster PNG
mode. All tables are rendered as styled HTML using a consistent visual
language.

Designed and developed in the context of the Data Challenge Pro 2026,
organized by Universidad del Norte (Barranquilla, Colombia), Sobusa S.A,
y Extreme Technologies.

Visual Language
----------------
- Black lines / bars / stems, grayscale fills (one accent red for
  thresholds and incomplete states)
- Thin (0.8 pt) black spines on every axis
- White background, no gridlines
- Scientific notation on count-like y-axes
- Default output format: SVG (vector, dpi-independent)

Data Conventions
----------------
Default values of ``col*`` parameters are the column names of the
source tables (``numeropasajeros``, ``duracion_min``, ``fechainicio``,
``pasajeros``, ``anio``, ...). Everything else (parameters, labels,
titles, returned table columns) is in English.

Main Functionalities
---------------------
Tables
    dataframe_head_table
    dataframe_integrity_blocks

Trip Quality
    plot_trip_status
    plot_passengers_distribution
    plot_passengers_histogram
    plot_duration_ecdf
    plot_short_trips_by_hour
    plot_long_trips_by_hour
    plot_zero_passenger_trips_by_hour
    plot_hour_plausibility_breakdown
    plot_passengers_by_hour

Outlier Calibration
    calibrate_hampel_threshold
    plot_hampel_boxplots

Source Consistency
    plot_ratio_vs_frequency

Demand Profiles
    plot_distribution_and_yearly_demand
    plot_demand_by_month
    plot_demand_by_week_of_month
    plot_demand_by_weekday
    plot_demand_by_seasonality
    plot_demand_by_time_slot
    plot_zeros_by_time_slot
    plot_temporal_panel

Autocorrelation
    plot_acf_pacf

Modeling Prep
    split_and_save

Route Maps
    generate_map
    list_available_routes

Dependencies
------------
numpy, pandas, matplotlib, scipy, statsmodels, IPython
(pyarrow or fastparquet for ``split_and_save``)
(shapely for ``generate_map`` / ``list_available_routes``)
"""

# Metadata ----
__version__ = "1.0.0"

__author__ = (
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Natalia Paola Alvarado Corrales <npalvarado@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)

_UNINORTE_AFFILIATION = {
    "name": "Universidad del Norte",
    "department": "Department of Mathematics, Physics and Data Science",
    "city": "Barranquilla",
    "country": "Colombia",
}

__authors_detail__ = [
    {
        "name": "Juan Camilo Mendoza Arango",
        "email": "cjarango@uninorte.edu.co",
        "orcid": "0000-0001-6872-8311",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "cjarango",
    },
    {
        "name": "Natalia Paola Alvarado Corrales",
        "email": "npalvarado@uninorte.edu.co",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "paolacorr67-ctrl",
    },
    {
        "name": "Miguel Ángel Pérez Vargas",
        "email": "vargasmiguel@uninorte.edu.co",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "miguelpvmr",
    },
]

# Standard library ----
import base64
import io
import json
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Third-party ----
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon
from matplotlib.ticker import ScalarFormatter
from scipy.stats import kendalltau, spearmanr
from shapely.geometry import shape as _shapely_shape
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Public API declaration ----
__all__ = [
    "calibrate_hampel_threshold",
    "dataframe_head_table",
    "dataframe_integrity_blocks",
    "generate_map",
    "list_available_routes",
    "plot_acf_pacf",
    "plot_demand_by_month",
    "plot_demand_by_seasonality",
    "plot_demand_by_time_slot",
    "plot_demand_by_week_of_month",
    "plot_demand_by_weekday",
    "plot_distribution_and_yearly_demand",
    "plot_duration_ecdf",
    "plot_hampel_boxplots",
    "plot_hour_plausibility_breakdown",
    "plot_long_trips_by_hour",
    "plot_passengers_by_hour",
    "plot_passengers_distribution",
    "plot_passengers_histogram",
    "plot_ratio_vs_frequency",
    "plot_short_trips_by_hour",
    "plot_temporal_panel",
    "plot_trip_status",
    "plot_zero_passenger_trips_by_hour",
    "plot_zeros_by_time_slot",
    "split_and_save",
]

# Module-level constants ----
DATA_PROCESSED = Path("../data/processed")

_DEFAULT_YEARS = (2023, 2024, 2025)
_YEAR_GRAYS = ("black", "#4D4D4D", "#B0B0B0")

# Route-map catalog: display metadata keyed by route id. Routes not in
# this catalog are still supported — they fall back to an auto title.
ROUTE_CATALOG = {
    "k54":      {"name": "CRA 54",   "title": "Trazado de la ruta CRA 54 — 56.4 km"},
    "c72":      {"name": "CL 72",    "title": "Trazado de la ruta CL 72 — 57.0 km"},
    "silencio": {"name": "SILENCIO", "title": "Trazado de la ruta SILENCIO — 62.9 km"},
}

# Candidate directories inspected (in order) when ``data_dir`` is None.
_DEFAULT_DATA_DIRS = (
    Path("../data/raw"),
    Path("data/raw"),
    Path("./data/raw"),
)

# File-name templates used by the map generator.
_GEOMETRY_TEMPLATE = "geometria_{route}.geojson"
_VIAS_TEMPLATES = ("vias_{route}.geojson", "vias.geojson")
_VIAS_GLOB = "vias_*.geojson"

# Shared HTML container styles used by every inline output (figures and
# tables) so the notebook renders with a single, consistent visual voice.
_FIGURE_CONTAINER_STYLE = (
    "display: flex; justify-content: center; align-items: center; "
    "width: 100%; box-sizing: border-box; padding: 8px 0;"
)
_TABLE_CONTAINER_STYLE = (
    "text-align: center; width: 100%; overflow-x: auto; "
    "box-sizing: border-box; padding: 8px 0;"
)


# Public API ====

# Tables ----

def dataframe_head_table(df, n=5, render=True, max_width=None):
    """Show the first ``n`` rows of a DataFrame as a styled HTML table.

    The DataFrame is shown as is, with all of its columns (no blocks, no
    variable selection), using the same visual style as the rest of the
    toolkit's tables.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        n (int): Number of rows to show. Defaults to 5.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.
        max_width (str, optional): Maximum width of the container, e.g.
            ``"100%"`` or ``"1200px"``. If ``None``, no limit is
            applied. The container scrolls horizontally on overflow.

    Returns:
        pandas.DataFrame: The plain (unstyled) head table.

    Raises:
        TypeError: If ``df`` is not a DataFrame or ``render`` is not a
            boolean.
        ValueError: If ``n`` is not a positive integer.
    """
    _require_dataframe(df)
    _require_bool(render, "render")

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("'n' must be a positive integer.")

    head = df.head(n).reset_index(drop=True)

    if render:
        _render_styled(_styled_table(head), max_width=max_width)

    return head


def dataframe_integrity_blocks(
    df,
    n_blocks=3,
    column_titles=("Variable", "Data type"),
    use_real_type=True,
    render=True,
):
    """Show the DataFrame's variables and types in N adjacent blocks.

    Each block takes two columns of a single table. Blocks are placed
    side by side, padding the missing rows of the last block with empty
    cells.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        n_blocks (int): Number of horizontal blocks. Defaults to 3.
        column_titles (tuple of str): Titles of the two columns of each
            block, in the order ``(variable, type)``. Defaults to
            ``("Variable", "Data type")``.
        use_real_type (bool): If ``True``, the first non-null value of
            each column is inspected and its real Python type
            (``type(...).__name__``) is reported. If ``False``, the
            pandas dtype is used. Defaults to ``True``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The wide plain table with ``2 * n_blocks``
        columns (empty if ``df`` has no columns).

    Raises:
        TypeError: If ``df`` is not a DataFrame, if ``column_titles`` is
            not a tuple/list of two strings, or if ``render`` or
            ``use_real_type`` is not a boolean.
        ValueError: If ``n_blocks`` is not a positive integer.
    """
    _require_dataframe(df)
    _require_bool(render, "render")
    _require_bool(use_real_type, "use_real_type")

    if not isinstance(n_blocks, int) or isinstance(n_blocks, bool) or n_blocks < 1:
        raise ValueError("'n_blocks' must be a positive integer.")

    _require_pair(column_titles, "column_titles")
    if not all(isinstance(c, str) for c in column_titles):
        raise TypeError("All elements of 'column_titles' must be strings.")

    col_var, col_type = column_titles

    if use_real_type:
        types = [_real_type_name(df[c]) for c in df.columns]
    else:
        types = [str(d) for d in df.dtypes]

    base = pd.DataFrame({col_var: df.columns, col_type: types})

    n = len(base)
    if n == 0:
        return pd.DataFrame()

    block_size = -(-n // n_blocks)  # ceil division
    padded = base.reindex(range(block_size * n_blocks)).fillna("")
    wide = pd.concat(
        [
            padded.iloc[i * block_size:(i + 1) * block_size].reset_index(drop=True)
            for i in range(n_blocks)
        ],
        axis=1,
    )

    if render:
        _render_styled(_styled_table(wide))

    return wide


# Trip quality ----

def plot_trip_status(
    df,
    col_cancelled="cancelada",
    col_finished="finalizado",
    yes_value="S",
    title="Trip status",
    xlab="",
    ylab="Trips",
    status_labels=("Completed", "Incomplete", "Cancelled"),
    status_colors=None,
    figsize=(6, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    bar_width=0.6,
    label_offset=0.03,
    y_label_pad=15,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot the distribution of trip states as a bar chart.

    Classifies each trip into one of three states: *Cancelled*
    (``col_cancelled == yes_value``), *Completed* (not cancelled and
    ``col_finished == yes_value``) or *Incomplete* (any other case).
    Bars are sorted in descending order and the percentage of the total
    is annotated above each bar. The y-axis uses scientific notation.

    Args:
        df (pandas.DataFrame): One row per trip.
        col_cancelled (str): Cancellation flag column.
        col_finished (str): Completion flag column.
        yes_value: Value that marks a flag as true. Defaults to ``"S"``.
        title (str): Plot title.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        status_labels (tuple of str): Display labels, in the order
            ``(completed, incomplete, cancelled)``.
        status_colors (dict or None): Mapping from label to color. If
            ``None``: black for completed, red for incomplete and gray
            for cancelled.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        bar_width (float): Width of each bar.
        label_offset (float): Gap between a bar and its label, as a
            fraction of the tallest bar.
        y_label_pad (float): Padding of the y-axis label.
        render (bool): If ``True``, renders inline as centered HTML.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, counts)`` where ``counts`` is a Series of
        trips per state.

    Raises:
        TypeError: If ``df`` is not a DataFrame or a column name is not
            a string.
        ValueError: If a column is not in ``df`` or ``status_labels``
            does not contain exactly three strings.
    """
    _require_dataframe(df)
    _require_columns(df, col_cancelled, col_finished)

    if not isinstance(status_labels, (tuple, list)) or len(status_labels) != 3:
        raise ValueError(
            "'status_labels' must be a tuple/list of exactly three strings."
        )

    completed, incomplete, cancelled = status_labels

    is_cancelled = (df[col_cancelled] == yes_value).to_numpy()
    is_finished = (df[col_finished] == yes_value).to_numpy()
    status = np.select(
        [is_cancelled, is_finished], [cancelled, completed], default=incomplete
    )

    counts = (
        pd.Series(status)
        .value_counts()
        .reindex(list(status_labels))
        .fillna(0)
        .astype(int)
        .sort_values(ascending=False)
    )

    if status_colors is None:
        status_colors = {
            completed: "black",
            incomplete: "#c0392b",
            cancelled: "#a0a0a0",
        }

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        range(len(counts)),
        counts.values,
        color=[status_colors[lbl] for lbl in counts.index],
        edgecolor="white",
        width=bar_width,
    )

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=tick_size)
    _label_axes(ax, xlab, ylab, title, label_size, title_size, y_pad=y_label_pad)
    _style_axes(ax, tick_size)
    _use_scientific_yaxis(ax, tick_size)

    total = counts.sum()
    ymax = counts.max()
    labels = [
        f"{(v / total * 100 if total > 0 else 0):.2f}%" for v in counts.values
    ]
    _annotate_bars(ax, bars, labels, ymax * label_offset, tick_size)
    ax.set_ylim(0, ymax * 1.20)

    _finish_figure(fig, render, fmt, dpi)

    return fig, ax, counts


def plot_passengers_distribution(
    df,
    col="numeropasajeros",
    titles=("Distribution", "Cumulative distribution"),
    xlab="Passengers per trip",
    ylab=("Frequency", "Percentage"),
    bins=50,
    color="black",
    figsize=(10, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot passengers per trip as a histogram plus an ECDF.

    Left panel: histogram (scientific notation on the y-axis). Right
    panel: empirical cumulative distribution function in percent.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col (str): Column with passengers per trip.
        titles (tuple of str): Panel titles ``(histogram, ecdf)``.
        xlab (str): Shared x-axis label.
        ylab (tuple of str): Y-axis labels ``(histogram, ecdf)``.
        bins (int): Number of histogram bins.
        color (str): Bar and line color.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for titles.
        tick_size (float): Font size for ticks.
        render (bool): If ``True``, renders inline as centered HTML.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, (ax1, ax2), stats)`` where ``stats`` is a dict
        with ``n``, ``mean``, ``median``, ``std``, ``min``, ``p25``,
        ``p75`` and ``max``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or ``titles`` / ``ylab``
            is not a tuple/list of two elements.
        ValueError: If ``col`` is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col)
    _require_pair(titles, "titles")
    _require_pair(ylab, "ylab")

    s = df[col].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.hist(s, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    _label_axes(ax1, xlab, ylab[0], titles[0], label_size, title_size)
    _style_axes(ax1, tick_size)
    _use_scientific_yaxis(ax1, tick_size)

    x_sorted = np.sort(s.to_numpy())
    ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted) * 100
    ax2.plot(x_sorted, ecdf, color=color, linewidth=1.5)
    ax2.set_ylim(0, 105)
    _label_axes(ax2, xlab, ylab[1], titles[1], label_size, title_size)
    _style_axes(ax2, tick_size)

    _finish_figure(fig, render, fmt, dpi)

    stats = {
        "n": len(s),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }

    return fig, (ax1, ax2), stats


def plot_passengers_histogram(
    df,
    col="numeropasajeros",
    title="Distribution of passengers per trip",
    xlab="Passengers per trip",
    ylab="Frequency",
    bins=50,
    bar_color="#b0b0b0",
    edge_color="black",
    edge_width=0.3,
    xlim=None,
    figsize=(8, 4.5),
    label_size=11,
    title_size=12,
    tick_size=10,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Histogram of passengers per trip (gray bars, thin black edge).

    Trips with null passengers (e.g. cancelled) are excluded.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col (str): Passengers column.
        title (str): Plot title.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        bins (int): Number of histogram bins.
        bar_color (str): Bar fill color.
        edge_color (str): Bar edge color.
        edge_width (float): Bar edge width.
        xlim (tuple, optional): X-axis limits.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` where ``table`` has the columns
        ``statistic`` and ``value`` (n, min, Q1, median, Q3, max, mean).

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``col`` is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col)

    s = df[col].dropna()

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        s, bins=bins, color=bar_color, edgecolor=edge_color,
        linewidth=edge_width,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)

    _label_axes(ax, xlab, ylab, title, label_size, title_size)
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    table = pd.DataFrame({
        "statistic": ["n", "min", "Q1", "median", "Q3", "max", "mean"],
        "value": [
            f"{len(s):,}",
            f"{s.min():.0f}",
            f"{s.quantile(0.25):.0f}",
            f"{s.median():.0f}",
            f"{s.quantile(0.75):.0f}",
            f"{s.max():.0f}",
            f"{s.mean():.2f}",
        ],
    })

    return fig, ax, table


def plot_duration_ecdf(
    df,
    col="duracion_min",
    thresholds=(30, 60, 90, 120, 150, 180, 200, 220),
    title="Cumulative distribution of trip duration",
    xlab="Duration (min)",
    ylab="Cumulative percentage (%)",
    color="black",
    figsize=(8, 4.5),
    label_size=11,
    title_size=12,
    tick_size=10,
    xlim=None,
    render=True,
    fmt="svg",
    dpi=100,
):
    """ECDF of trip duration, excluding cancelled trips.

    Cancelled trips (no valid end time, hence no duration) are removed
    before computing the ECDF. A table with the cumulative percentage
    below each threshold is also returned.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col (str): Duration column (minutes).
        thresholds (tuple of float): Thresholds for which the cumulative
            percentage is computed.
        title (str): Plot title.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        color (str): ECDF line color.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        xlim (tuple, optional): X-axis limits.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` where ``table`` has the columns
        ``threshold`` and ``cumulative_pct``.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``col`` is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col)

    s = df[col].dropna().sort_values()
    n = len(s)
    ecdf = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(s.to_numpy(), ecdf, color=color, linewidth=1.5, zorder=3)
    if xlim is not None:
        ax.set_xlim(*xlim)

    _label_axes(ax, xlab, ylab, title, label_size, title_size)
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    table = pd.DataFrame({
        "threshold": list(thresholds),
        "cumulative_pct": [round((s < u).sum() / n * 100, 3) for u in thresholds],
    })

    return fig, ax, table


def plot_short_trips_by_hour(
    df,
    col_hour="hora",
    col_duration="duracion_min",
    threshold_min=90,
    title=None,
    xlab="Hour of day",
    ylab="Share of short trips (%)",
    bar_color="black",
    edge_color="black",
    edge_width=0.0,
    figsize=(10, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Percentage of short-duration trips for each hour of the day.

    Trips with null duration (cancelled) are excluded from both the
    numerator and the denominator.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col_hour (str): Hour column with integer values 0-23.
        col_duration (str): Duration column (minutes).
        threshold_min (float): A trip is "short" if its duration is
            strictly below this value.
        title (str, optional): Plot title. If ``None``, it is generated
            from ``threshold_min``.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        bar_color (str): Bar fill color.
        edge_color (str): Bar edge color.
        edge_width (float): Bar edge width.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with the columns ``hour``,
        ``total``, ``short_trips`` and ``pct``.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col_hour, col_duration)

    valid = df[df[col_duration].notna()]

    return _plot_hourly_share(
        hours=valid[col_hour].to_numpy(),
        flag=(valid[col_duration] < threshold_min).to_numpy(),
        count_name="short_trips",
        title=title or (
            f"Hourly distribution of trips shorter than {threshold_min:g} min"
        ),
        xlab=xlab, ylab=ylab, bar_color=bar_color, edge_color=edge_color,
        edge_width=edge_width, figsize=figsize, label_size=label_size,
        title_size=title_size, tick_size=tick_size,
        render=render, fmt=fmt, dpi=dpi,
    )


def plot_long_trips_by_hour(
    df,
    col_start="fechainicio",
    col_duration="duracion_min",
    threshold_min=420,
    title=None,
    xlab="Hour of day",
    ylab="Share of long trips (%)",
    bar_color="black",
    edge_color="black",
    edge_width=0.5,
    figsize=(10, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Percentage of long-duration trips for each hour of the day.

    Cancelled trips (null duration) are excluded from both the numerator
    and the denominator. The hour is taken from the trip start time.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col_start (str): Trip start timestamp column.
        col_duration (str): Duration column (minutes).
        threshold_min (float): A trip is "long" if its duration is
            strictly above this value.
        title (str, optional): Plot title. If ``None``, it is generated
            from ``threshold_min``.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        bar_color (str): Bar fill color.
        edge_color (str): Bar edge color.
        edge_width (float): Bar edge width.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with the columns ``hour``,
        ``total``, ``long_trips`` and ``pct``.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col_start, col_duration)

    valid = df[df[col_duration].notna()]

    return _plot_hourly_share(
        hours=_hour_of(valid[col_start]).to_numpy(),
        flag=(valid[col_duration] > threshold_min).to_numpy(),
        count_name="long_trips",
        title=title or (
            f"Hourly distribution of trips longer than {threshold_min:g} min"
        ),
        xlab=xlab, ylab=ylab, bar_color=bar_color, edge_color=edge_color,
        edge_width=edge_width, figsize=figsize, label_size=label_size,
        title_size=title_size, tick_size=tick_size,
        render=render, fmt=fmt, dpi=dpi,
    )


def plot_zero_passenger_trips_by_hour(
    df,
    col_start="fechainicio",
    col_passengers="numeropasajeros",
    col_duration="duracion_min",
    title="Hourly distribution of trips with 0 passengers",
    xlab="Hour of day",
    ylab="Share of trips (%)",
    bar_color="black",
    edge_color="black",
    edge_width=0.5,
    figsize=(10, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Percentage of trips with zero passengers for each hour of the day.

    Cancelled trips (null duration) and trips with null passengers are
    excluded from both the numerator and the denominator.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col_start (str): Trip start timestamp column.
        col_passengers (str): Passengers column.
        col_duration (str): Duration column (used to exclude cancelled
            trips).
        title (str): Plot title.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        bar_color (str): Bar fill color.
        edge_color (str): Bar edge color.
        edge_width (float): Bar edge width.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with the columns ``hour``,
        ``total``, ``zero_passenger_trips`` and ``pct``.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col_start, col_passengers, col_duration)

    valid = df[df[col_duration].notna() & df[col_passengers].notna()]

    return _plot_hourly_share(
        hours=_hour_of(valid[col_start]).to_numpy(),
        flag=(valid[col_passengers] == 0).to_numpy(),
        count_name="zero_passenger_trips",
        title=title, xlab=xlab, ylab=ylab, bar_color=bar_color,
        edge_color=edge_color, edge_width=edge_width, figsize=figsize,
        label_size=label_size, title_size=title_size, tick_size=tick_size,
        render=render, fmt=fmt, dpi=dpi,
    )


def plot_hour_plausibility_breakdown(
    df,
    col_start="fechainicio",
    col_passengers="numeropasajeros",
    col_duration="duracion_min",
    hour=21,
    plausible_range=(130, 250),
    title=None,
    xlab="",
    ylab="Total passengers",
    labels=("Plausible", "Anomalous"),
    avg_label_fmt="{:.2f} pax/trip",
    plausible_color="black",
    anomalous_color="#a0a0a0",
    edge_color="black",
    edge_width=1.0,
    figsize=(5, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Compare plausible vs anomalous trips within a single hour.

    A trip is plausible if its duration lies within ``plausible_range``
    (both ends included). Cancelled trips (no duration) are excluded.
    Bars show total passengers; the label above each bar shows the
    average passengers per trip, formatted through ``avg_label_fmt``.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col_start (str): Trip start timestamp column.
        col_passengers (str): Passengers column.
        col_duration (str): Duration column (minutes).
        hour (int): Hour of the day to analyze.
        plausible_range (tuple of float): ``(min, max)`` plausible
            duration in minutes.
        title (str, optional): Plot title. If ``None``, it is generated
            from ``hour``.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        labels (tuple of str): Group labels ``(plausible, anomalous)``.
        avg_label_fmt (str): ``str.format`` template used to render the
            average passengers-per-trip value above each bar. It must
            contain a single positional placeholder for the numeric
            value, e.g. ``"{:.2f} pax/trip"`` (default),
            ``"{:.2f} pax/viaje"`` or ``"{:.0f} pasajeros/viaje"``.
        plausible_color (str): Fill color of the plausible bar.
        anomalous_color (str): Fill color of the anomalous bar.
        edge_color (str): Bar edge color.
        edge_width (float): Bar edge width.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with the columns ``group``,
        ``n_trips``, ``passengers`` and ``avg_passengers``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or ``labels`` /
            ``plausible_range`` is not a tuple/list of two elements.
        ValueError: If a column is not in ``df``, or if
            ``avg_label_fmt`` is not a valid single-placeholder
            ``str.format`` template.
    """
    _require_dataframe(df)
    _require_columns(df, col_start, col_passengers, col_duration)
    _require_pair(labels, "labels")
    _require_pair(plausible_range, "plausible_range")

    if not isinstance(avg_label_fmt, str):
        raise TypeError("'avg_label_fmt' must be a string.")
    try:
        avg_label_fmt.format(0.0)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            "'avg_label_fmt' must be a str.format template with a single "
            "positional placeholder, e.g. '{:.2f} pax/trip'."
        ) from exc

    subset = _trips_in_hours(df, col_start, col_duration, (hour,))
    is_plausible = subset[col_duration].between(*plausible_range)
    groups = (subset[is_plausible], subset[~is_plausible])

    table = pd.DataFrame({
        "group": list(labels),
        "n_trips": [len(g) for g in groups],
        "passengers": [g[col_passengers].sum() for g in groups],
        "avg_passengers": [
            round(g[col_passengers].mean(), 2) if len(g) else 0 for g in groups
        ],
    })

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        table["group"], table["passengers"],
        color=[plausible_color, anomalous_color],
        edgecolor=edge_color, linewidth=edge_width, width=0.5,
    )

    ymax = table["passengers"].max()
    _annotate_bars(
        ax, bars,
        [avg_label_fmt.format(v) for v in table["avg_passengers"]],
        ymax * 0.02, tick_size,
    )
    ax.set_ylim(0, ymax * 1.15)

    _label_axes(
        ax, xlab, ylab,
        title or f"Hour {hour} breakdown: plausible vs anomalous trips",
        label_size, title_size,
    )
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    return fig, ax, table


def plot_passengers_by_hour(
    df,
    col_start="fechainicio",
    col_passengers="numeropasajeros",
    col_duration="duracion_min",
    hours=(22, 23),
    title="Total passengers by hour",
    xlab="Hour of day",
    ylab="Total passengers",
    bar_color="black",
    edge_color="black",
    edge_width=1.0,
    figsize=(6, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Bar chart of total passengers for selected hours, with value labels.

    Cancelled trips (no duration) are excluded from the aggregation. When
    no trips fall inside the requested hours, an empty panel with an
    informative message is rendered instead of a bar chart.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col_start (str): Trip start timestamp column.
        col_passengers (str): Passengers column.
        col_duration (str): Duration column (used to exclude cancelled
            trips).
        hours (tuple of int): Hours of the day to include.
        title (str): Plot title.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        bar_color (str): Bar fill color.
        edge_color (str): Bar edge color.
        edge_width (float): Bar edge width.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)``. ``table`` has the columns ``hour``,
        ``passengers``, ``n_trips`` and ``avg_passengers``; it is empty
        when no trip matches the requested hours.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col_start, col_passengers, col_duration)

    subset = _trips_in_hours(df, col_start, col_duration, hours)

    table = (
        subset.groupby("_hour")
        .agg(
            passengers=(col_passengers, "sum"),
            n_trips=(col_passengers, "size"),
            avg_passengers=(col_passengers, "mean"),
        )
        .round(2)
        .reset_index()
        .rename(columns={"_hour": "hour"})
    )

    # Empty selection: render an informative placeholder instead of
    # failing on ``max()`` over an empty series.
    if table.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No trips in the selected hours",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=label_size, color="#808080",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        _label_axes(ax, xlab, ylab, title, label_size, title_size)
        _style_axes(ax, tick_size)
        _finish_figure(fig, render, fmt, dpi)
        return fig, ax, table

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        table["hour"].astype(str), table["passengers"],
        color=bar_color, edgecolor=edge_color, linewidth=edge_width,
        width=0.5,
    )

    ymax = float(table["passengers"].max())
    _annotate_bars(
        ax, bars, [f"{v:,.0f}" for v in table["passengers"]],
        ymax * 0.02, tick_size,
    )
    ax.set_ylim(0, ymax * 1.15)

    _label_axes(ax, xlab, ylab, title, label_size, title_size)
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    return fig, ax, table


# Outlier calibration ----

def calibrate_hampel_threshold(
    df,
    col="numeropasajeros",
    col_order="fechainicio",
    t_values=(3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0),
    k=7,
    title="Hampel threshold calibration",
    xlab="Threshold $t$ (MADs)",
    ylab="Retained observations",
    figsize=(7, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    x_label_pad=10,
    y_label_pad=15,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Calibrate the Hampel threshold by sweeping ``t``.

    Sorts ``df`` by ``col_order`` and applies a rolling Hampel filter to
    ``col`` for each value of ``t``. Plots the number of observations
    that are NOT flagged as outliers (retained) versus ``t``. The y-axis
    uses scientific notation and the curve is drawn without annotations.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col (str): Numeric column to filter.
        col_order (str): Column to sort by before filtering.
        t_values (tuple of float): Values of ``t`` to evaluate.
        k (int): Half-window size of the Hampel filter.
        title (str): Plot title.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        x_label_pad (float): Padding of the x-axis label.
        y_label_pad (float): Padding of the y-axis label.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, summary)`` where ``summary`` has the columns
        ``t``, ``n_retained``, ``n_flagged`` and ``pct_retained``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or a column name is not
            a string.
        ValueError: If a column is not in ``df``, if ``t_values`` is
            empty, or if any ``t`` is not positive.
    """
    _require_dataframe(df)
    _require_columns(df, col, col_order)

    t_values = tuple(t_values)
    if len(t_values) == 0:
        raise ValueError("'t_values' must contain at least one value.")
    if not all(isinstance(t, (int, float)) and t > 0 for t in t_values):
        raise ValueError("All elements of 't_values' must be positive numbers.")

    x = df.sort_values(col_order)[col].to_numpy(dtype=float)

    n_total = int(np.sum(~np.isnan(x)))
    n_flagged = [int(_hampel_flag(x, k=k, t=t).sum()) for t in t_values]
    n_retained = [n_total - n for n in n_flagged]

    summary = pd.DataFrame({
        "t": list(t_values),
        "n_retained": n_retained,
        "n_flagged": n_flagged,
        "pct_retained": [n / n_total * 100 for n in n_retained],
    })

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        summary["t"], summary["n_retained"],
        color="black", linewidth=1.5, marker="o", markersize=5,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.0,
    )

    _label_axes(
        ax, xlab, ylab, title, label_size, title_size,
        x_pad=x_label_pad, y_pad=y_label_pad,
    )
    _style_axes(ax, tick_size)
    _use_scientific_yaxis(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    return fig, ax, summary


def plot_hampel_boxplots(
    df,
    col_passengers="numeropasajeros",
    col_duration="duracion_min",
    t=(5.0, 5.0),
    k=7,
    titles=("Number of passengers", "Duration (min)"),
    figsize=(10, 5),
    box_color="#f2f2f2",
    edge_color="black",
    threshold_color="#c0392b",
    label_size=11,
    title_size=12,
    tick_size=10,
    showfliers=True,
    ylim_passengers=None,
    ylim_duration=None,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Two vertical boxplots with Hampel threshold lines.

    Thresholds are drawn as dashed lines at
    ``median ± t * 1.4826 * MAD``, using the global median and MAD of
    each variable as a visual reference. The right panel moves its
    y-axis to the right side for symmetry.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        col_passengers (str): Passengers column.
        col_duration (str): Duration column.
        t (tuple of float): Hampel thresholds
            ``(t_passengers, t_duration)``.
        k (int): Half-window size. Unused here (the thresholds are
            global); kept for API consistency with
            ``calibrate_hampel_threshold``.
        titles (tuple of str): Panel titles ``(left, right)``.
        figsize (tuple of float): Figure size in inches.
        box_color (str): Box fill color.
        edge_color (str): Box edge and whisker color.
        threshold_color (str): Color of the threshold lines.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for panel titles.
        tick_size (float): Font size for tick labels.
        showfliers (bool): If ``True``, draw outlier markers.
        ylim_passengers (tuple, optional): Y limits of the left panel.
        ylim_duration (tuple, optional): Y limits of the right panel.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, (ax1, ax2), stats)`` where ``stats`` maps
        ``"passengers"`` and ``"duration"`` to dicts with ``t``,
        ``median``, ``mad``, ``lo`` and ``hi``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or ``t`` / ``titles`` is
            not a tuple/list of two elements.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, col_passengers, col_duration)
    _require_pair(t, "t")
    _require_pair(titles, "titles")

    series = (df[col_passengers].dropna(), df[col_duration].dropna())
    bounds = [_robust_bounds(s, tt) for s, tt in zip(series, t)]
    ylims = (ylim_passengers, ylim_duration)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for i, (ax, s, b, title, ylim) in enumerate(
        zip(axes, series, bounds, titles, ylims)
    ):
        ax.boxplot(
            s,
            widths=0.4,
            patch_artist=True,
            showfliers=showfliers,
            boxprops={"facecolor": box_color, "edgecolor": edge_color,
                      "linewidth": 0.8},
            whiskerprops={"color": edge_color, "linewidth": 0.8},
            capprops={"color": edge_color, "linewidth": 0.8},
            medianprops={"color": edge_color, "linewidth": 1.2},
            flierprops={
                "marker": "o", "markersize": 1.5,
                "markerfacecolor": edge_color, "markeredgecolor": edge_color,
                "markeredgewidth": 0.5, "alpha": 0.25,
            },
            **_vertical_kwargs(),
        )

        for level in (b["lo"], b["hi"]):
            ax.axhline(level, color=threshold_color, linestyle="--",
                       linewidth=1.2, zorder=4)

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            y_max = s.max()
            ax.set_ylim(-0.05 * y_max, 1.05 * y_max)

        ax.set_title(title, fontsize=title_size, pad=10)
        ax.set_xticks([])
        _style_axes(ax, tick_size)

        if i == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    _finish_figure(fig, render, fmt, dpi)

    stats = {"passengers": bounds[0], "duration": bounds[1]}

    return fig, tuple(axes), stats


# Source consistency ----

def plot_ratio_vs_frequency(
    df_operational,
    df_annual,
    col_passengers="pasajeros",
    col_boardings_annual="subidas",
    col_year="anio",
    col_weekday="dia_semana",
    col_seasonality="tipo_estacionalidad",
    col_x="n_days",
    col_y="ratio",
    ref_line=1.0,
    title="Boardings ratio vs. pattern frequency",
    xlab="Days composing the pattern",
    ylab="Boardings ratio: operational / annual (log)",
    color="black",
    ref_color="red",
    marker_size=30,
    alpha=1.0,
    figsize=(8, 5),
    label_size=11,
    title_size=12,
    tick_size=10,
    xlim=None,
    ylim=None,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Scatter of the between-source boardings ratio vs. pattern frequency.

    Builds the comparison between ``df_operational`` (operational
    source) and ``df_annual`` (annual consolidated table by pattern) by
    aggregating both over the pattern key ``(year, weekday,
    seasonality, hour, minute)``. It computes the boardings ratio and
    the number of days composing each pattern, and plots the ratio
    against that frequency with a logarithmic y-axis. The horizontal
    line marks perfect consistency (ratio = 1). Spearman's rho and
    Kendall's tau between both variables are annotated in the upper-left
    corner.

    Args:
        df_operational (pandas.DataFrame): Table aggregated by time
            slot, with a ``DatetimeIndex`` and a passengers column.
        df_annual (pandas.DataFrame): Annual consolidated table with the
            columns ``ano``, ``id_dia_semana``, ``tipo_estacionalidad``,
            ``hora_num``, ``minuto_num`` and the boardings column.
        col_passengers (str): Passengers column in ``df_operational``.
        col_boardings_annual (str): Boardings column in ``df_annual``.
        col_year (str): Year column in ``df_operational``.
        col_weekday (str): Weekday column in ``df_operational``.
        col_seasonality (str): Seasonality column in
            ``df_operational``.
        col_x (str): Name of the x-axis column in the result.
        col_y (str): Name of the y-axis column in the result.
        ref_line (float): Horizontal reference value.
        title (str): Plot title (the sample size is appended).
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        color (str): Marker color.
        ref_color (str): Reference line color.
        marker_size (float): Marker size.
        alpha (float): Marker transparency.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        xlim (tuple, optional): X-axis limits.
        ylim (tuple, optional): Y-axis limits.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table, comparison)`` where ``table`` holds
        the correlation statistics (``spearman_rho``, ``spearman_p``,
        ``kendall_tau``, ``kendall_p``, ``n``) and ``comparison`` is the
        merged pattern-level DataFrame (``boardings_operational``,
        ``boardings_annual``, ``ratio``, ``n_days``).

    Raises:
        TypeError: If a table is not a DataFrame or ``df_operational``
            has no ``DatetimeIndex``.
        ValueError: If a required column is missing.
    """
    _require_dataframe(df_operational, "df_operational")
    _require_dataframe(df_annual, "df_annual")
    _require_datetime_index(df_operational, "df_operational")
    _require_columns(df_operational, col_passengers, col_year, col_weekday,
                     col_seasonality)
    _require_columns(df_annual, col_boardings_annual, "ano", "id_dia_semana",
                     "tipo_estacionalidad", "hora_num", "minuto_num")

    keys = [col_year, col_weekday, col_seasonality, "hour_num", "minute_num"]
    int_keys = [col_year, col_weekday, "hour_num", "minute_num"]

    ops = df_operational.copy()
    ops["hour_num"] = ops.index.hour.astype("int16")
    ops["minute_num"] = ops.index.minute.astype("int16")

    ops_pattern = (
        ops.groupby(keys)
        .agg(boardings_operational=(col_passengers, "sum"))
        .reset_index()
    )

    annual_pattern = (
        df_annual
        .groupby(["ano", "id_dia_semana", "tipo_estacionalidad",
                  "hora_num", "minuto_num"])
        .agg(boardings_annual=(col_boardings_annual, "sum"))
        .reset_index()
        .rename(columns={
            "ano": col_year,
            "id_dia_semana": col_weekday,
            "tipo_estacionalidad": col_seasonality,
            "hora_num": "hour_num",
            "minuto_num": "minute_num",
        })
    )

    days_pattern = (
        ops.assign(_date=ops.index.date)
        .groupby(keys)["_date"].nunique()
        .rename("n_days")
        .reset_index()
    )

    ops_pattern, annual_pattern, days_pattern = (
        _cast_pattern_keys(frame, int_keys, col_seasonality)
        for frame in (ops_pattern, annual_pattern, days_pattern)
    )

    comparison = (
        ops_pattern
        .merge(annual_pattern, on=keys, how="outer")
        .merge(days_pattern, on=keys, how="left")
    )
    comparison["ratio"] = (
        comparison["boardings_operational"]
        / comparison["boardings_annual"].replace(0, np.nan)
    )

    valid = comparison.dropna(subset=[col_y, col_x])
    valid = valid[(valid[col_x] > 0) & (valid[col_y] > 0)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        valid[col_x], valid[col_y], s=marker_size, alpha=alpha,
        color=color, edgecolor="none", zorder=3,
    )
    ax.axhline(ref_line, color=ref_color, linestyle="--", linewidth=1,
               zorder=2)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_yscale("log")

    _label_axes(ax, xlab, ylab, f"{title}",
                label_size, title_size)

    rho, p_rho = spearmanr(valid[col_x], valid[col_y])
    tau, p_tau = kendalltau(valid[col_x], valid[col_y])

    rho = cast(float, rho)
    p_rho = cast(float, p_rho)
    tau = cast(float, tau)
    p_tau = cast(float, p_tau)

    ax.text(
        0.02, 0.98, f"Spearman ρ = {rho:.3f}\nKendall τ = {tau:.3f}",
        transform=ax.transAxes, fontsize=tick_size,
        verticalalignment="top", horizontalalignment="left", zorder=4,
    )

    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    table = pd.DataFrame({
        "spearman_rho": [round(rho, 3)],
        "spearman_p": [round(p_rho, 4)],
        "kendall_tau": [round(tau, 3)],
        "kendall_p": [round(p_tau, 4)],
        "n": [len(valid)],
    })

    return fig, ax, table, comparison


# Demand profiles ----

def plot_distribution_and_yearly_demand(
    df,
    col="pasajeros",
    col_year="anio",
    bins=80,
    xlab_hist="Passengers per time slot",
    ylab_hist="Relative frequency",
    xlab_bar="Year",
    ylab_bar="Passengers per day",
    titles=("Distribution by time slot", "Daily demand by year"),
    hist_color="gray",
    bar_color="black",
    figsize=(12, 4.5),
    label_size=11,
    title_size=12,
    tick_size=10,
    ylim_top_factor=1.15,
    label_offset_frac=0.02,
    bar_width=0.45,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Two-panel figure: relative histogram and median daily demand by year.

    The left panel shows the distribution of ``col`` as relative
    frequency (sums to 1). The right panel shows the median of the daily
    total by year, with the value annotated above each bar; its y-axis
    is placed on the right and uses scientific notation.

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        col_year (str): Year column.
        bins (int): Number of histogram bins.
        xlab_hist (str): X-axis label of the histogram.
        ylab_hist (str): Y-axis label of the histogram.
        xlab_bar (str): X-axis label of the right panel.
        ylab_bar (str): Y-axis label of the right panel.
        titles (tuple of str): Panel titles ``(histogram, bars)``.
        hist_color (str): Histogram bar color.
        bar_color (str): Right-panel bar color.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for panel titles.
        tick_size (float): Font size for tick labels.
        ylim_top_factor (float): Factor extending the upper y limit of
            the right panel.
        label_offset_frac (float): Gap between a bar and its label, as a
            fraction of the y-axis maximum.
        bar_width (float): Bar width of the right panel.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, axes, tables)`` where ``tables`` is a dict with
        ``"histogram"`` (``bin_center``, ``relative_frequency``) and
        ``"yearly"`` (``median`` and ``n_days`` by year).

    Raises:
        TypeError: If ``df`` is not a DataFrame or has no
            ``DatetimeIndex``, or ``titles`` is not a pair.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col, col_year)
    _require_pair(titles, "titles")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_hist, ax_bar = axes

    counts, edges = np.histogram(df[col].dropna(), bins=bins)
    rel_freq = counts / counts.sum()
    centers = (edges[:-1] + edges[1:]) / 2

    ax_hist.bar(
        centers, rel_freq, width=np.diff(edges) * 0.95,
        color=hist_color, edgecolor="black", linewidth=0.3, zorder=3,
    )
    _label_axes(ax_hist, xlab_hist, ylab_hist, titles[0], label_size,
                title_size)
    _style_axes(ax_hist, tick_size)

    daily = _daily_totals(df, col, col_year)
    yearly = (
        daily.groupby("year")["daily_total"]
        .agg(median="median", n_days="count")
        .round(0)
    )

    x = list(range(len(yearly)))
    bars = ax_bar.bar(
        x, yearly["median"].to_numpy(), width=bar_width,
        facecolor=bar_color, edgecolor=bar_color, linewidth=1.2, zorder=3,
    )

    y_max = yearly["median"].max() * ylim_top_factor
    ax_bar.set_ylim(0, y_max)
    _annotate_bars(
        ax_bar, bars, [f"{v:,.0f}" for v in yearly["median"]],
        y_max * label_offset_frac, tick_size,
    )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(yearly.index.astype(str))
    ax_bar.yaxis.set_label_position("right")
    ax_bar.yaxis.tick_right()
    _use_scientific_yaxis(ax_bar)
    _label_axes(ax_bar, xlab_bar, ylab_bar, titles[1], label_size,
                title_size)
    _style_axes(ax_bar, tick_size)
    ax_bar.tick_params(axis="y", which="both", left=False, labelleft=False)

    _finish_figure(fig, render, fmt, dpi)

    tables = {
        "histogram": pd.DataFrame({
            "bin_center": centers.round(1),
            "relative_frequency": rel_freq.round(5),
        }),
        "yearly": yearly,
    }

    return fig, axes, tables


def plot_demand_by_month(
    df,
    col="pasajeros",
    col_month="mes",
    col_year="anio",
    xlab="Month",
    ylab="Passengers per day",
    title="Daily demand by month and year",
    month_labels=("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
    years=_DEFAULT_YEARS,
    year_labels=None,
    colors=_YEAR_GRAYS,
    line_color="black",
    text_color="black",
    figsize=(12, 4.8),
    label_size=11,
    title_size=12,
    tick_size=10,
    ylim_top_factor=1.35,
    bar_width=0.26,
    line_gap_frac=0.04,
    text_gap_frac=0.02,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Grouped bars of daily demand by month and year, with a summary line.

    Each month holds one bar per year. A bar's height is the median of
    the daily passenger total within the group. Above each month's
    group, a horizontal line is drawn over the tallest bar and annotated
    with the aggregate value (median across all years).

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        col_month (str): Month column (values 1-12).
        col_year (str): Year column.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        title (str): Plot title.
        month_labels (tuple of str): The 12 x-axis labels.
        years (tuple of int): Years to plot.
        year_labels (tuple of str, optional): Legend labels. If
            ``None``, the years themselves are used.
        colors (tuple of str): Bar colors, one per year.
        line_color (str): Summary line color.
        text_color (str): Aggregate value text color.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        ylim_top_factor (float): Factor extending the upper y limit.
        bar_width (float): Width of each bar.
        line_gap_frac (float): Gap between the tallest bar of a group
            and its summary line, as a fraction of the global maximum.
        text_gap_frac (float): Gap between the line and its text, as a
            fraction of the global maximum.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with one ``median_<year>`` column
        per year plus ``aggregate``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or has no
            ``DatetimeIndex``.
        ValueError: If a column is not in ``df`` or the label / color
            tuples do not match the requested years.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col, col_month, col_year)

    return _plot_grouped_demand(
        df, col, col_year, group=df[col_month].to_numpy(),
        categories=range(1, 13), tick_labels=month_labels,
        years=years, year_labels=year_labels, colors=colors,
        xlab=xlab, ylab=ylab, title=title, line_color=line_color,
        text_color=text_color, figsize=figsize, label_size=label_size,
        title_size=title_size, tick_size=tick_size,
        ylim_top_factor=ylim_top_factor, bar_width=bar_width,
        line_gap_frac=line_gap_frac, text_gap_frac=text_gap_frac,
        index_name="month", render=render, fmt=fmt, dpi=dpi,
    )


def plot_demand_by_week_of_month(
    df,
    col="pasajeros",
    col_year="anio",
    xlab="Week of the month",
    ylab="Passengers per day",
    title="Daily demand by week of the month and year",
    week_labels=("Week 1", "Week 2", "Week 3", "Week 4"),
    years=_DEFAULT_YEARS,
    year_labels=None,
    colors=_YEAR_GRAYS,
    line_color="black",
    text_color="black",
    figsize=(10, 5.0),
    label_size=11,
    title_size=12,
    tick_size=10,
    ylim_top_factor=1.35,
    bar_width=0.26,
    line_gap_frac=0.04,
    text_gap_frac=0.02,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Grouped bars of daily demand by week of the month and year.

    The week of the month is ``(day - 1) // 7 + 1``; days falling in the
    partial fifth week are assigned to week 4 so the axis is not
    fragmented. Each week holds one bar per year (median of the daily
    passenger total), and a summary line above each group is annotated
    with the aggregate value (median across all years).

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        col_year (str): Year column.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        title (str): Plot title.
        week_labels (tuple of str): The 4 x-axis labels.
        years (tuple of int): Years to plot.
        year_labels (tuple of str, optional): Legend labels. If
            ``None``, the years themselves are used.
        colors (tuple of str): Bar colors, one per year.
        line_color (str): Summary line color.
        text_color (str): Aggregate value text color.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        ylim_top_factor (float): Factor extending the upper y limit.
        bar_width (float): Width of each bar.
        line_gap_frac (float): Gap between the tallest bar of a group
            and its summary line, as a fraction of the global maximum.
        text_gap_frac (float): Gap between the line and its text, as a
            fraction of the global maximum.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with one ``median_<year>`` column
        per year plus ``aggregate``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or has no
            ``DatetimeIndex``.
        ValueError: If a column is not in ``df`` or the label / color
            tuples do not match the requested years.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col, col_year)

    week = np.minimum((df.index.day.to_numpy() - 1) // 7 + 1, 4)

    return _plot_grouped_demand(
        df, col, col_year, group=week,
        categories=range(1, 5), tick_labels=week_labels,
        years=years, year_labels=year_labels, colors=colors,
        xlab=xlab, ylab=ylab, title=title, line_color=line_color,
        text_color=text_color, figsize=figsize, label_size=label_size,
        title_size=title_size, tick_size=tick_size,
        ylim_top_factor=ylim_top_factor, bar_width=bar_width,
        line_gap_frac=line_gap_frac, text_gap_frac=text_gap_frac,
        index_name="week_of_month", render=render, fmt=fmt, dpi=dpi,
    )


def plot_demand_by_weekday(
    df,
    col="pasajeros",
    col_year="anio",
    col_weekday="dia_semana",
    xlab="Day of the week",
    ylab="Passengers per day",
    title="Daily demand by day of the week and year",
    weekday_labels=("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
    years=_DEFAULT_YEARS,
    year_labels=None,
    colors=_YEAR_GRAYS,
    line_color="black",
    text_color="black",
    figsize=(10, 5.0),
    label_size=11,
    title_size=12,
    tick_size=10,
    ylim_top_factor=1.35,
    bar_width=0.26,
    line_gap_frac=0.04,
    text_gap_frac=0.02,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Grouped bars of daily demand by day of the week and year.

    Each weekday holds one bar per year (median of the daily passenger
    total), and a summary line above each group is annotated with the
    aggregate value (median across all years).

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        col_year (str): Year column.
        col_weekday (str): Weekday column (values 1-7).
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        title (str): Plot title.
        weekday_labels (tuple of str): The 7 x-axis labels.
        years (tuple of int): Years to plot.
        year_labels (tuple of str, optional): Legend labels. If
            ``None``, the years themselves are used.
        colors (tuple of str): Bar colors, one per year.
        line_color (str): Summary line color.
        text_color (str): Aggregate value text color.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        ylim_top_factor (float): Factor extending the upper y limit.
        bar_width (float): Width of each bar.
        line_gap_frac (float): Gap between the tallest bar of a group
            and its summary line, as a fraction of the global maximum.
        text_gap_frac (float): Gap between the line and its text, as a
            fraction of the global maximum.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with one ``median_<year>`` column
        per year plus ``aggregate``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or has no
            ``DatetimeIndex``.
        ValueError: If a column is not in ``df`` or the label / color
            tuples do not match the requested years.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col, col_year, col_weekday)

    return _plot_grouped_demand(
        df, col, col_year, group=df[col_weekday].to_numpy(),
        categories=range(1, 8), tick_labels=weekday_labels,
        years=years, year_labels=year_labels, colors=colors,
        xlab=xlab, ylab=ylab, title=title, line_color=line_color,
        text_color=text_color, figsize=figsize, label_size=label_size,
        title_size=title_size, tick_size=tick_size,
        ylim_top_factor=ylim_top_factor, bar_width=bar_width,
        line_gap_frac=line_gap_frac, text_gap_frac=text_gap_frac,
        index_name="weekday", render=render, fmt=fmt, dpi=dpi,
    )


def plot_demand_by_seasonality(
    df,
    col="pasajeros",
    col_year="anio",
    col_seasonality="tipo_estacionalidad",
    xlab="Seasonality type",
    ylab="Passengers per day",
    title="Daily demand by seasonality type and year",
    category_order=("DIAS_CORRIENTES", "FIESTA_ATLANTICO",
                    "TEMPORADA_NACIONAL", "FESTIVO_NACIONAL"),
    category_labels=("Regular days", "Atlantic festival",
                     "National season", "National holiday"),
    years=_DEFAULT_YEARS,
    year_labels=None,
    colors=_YEAR_GRAYS,
    line_color="black",
    text_color="black",
    figsize=(11, 5.0),
    label_size=11,
    title_size=12,
    tick_size=10,
    ylim_top_factor=1.35,
    bar_width=0.26,
    line_gap_frac=0.04,
    text_gap_frac=0.02,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Grouped bars of daily demand by seasonality type and year.

    Each seasonality type holds one bar per year (median of the daily
    passenger total). Categories appear in the order given by
    ``category_order``. A summary line above each group is annotated
    with the aggregate value (median across all years).

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        col_year (str): Year column.
        col_seasonality (str): Seasonality type column.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        title (str): Plot title.
        category_order (tuple of str): Raw category values, in x-axis
            order.
        category_labels (tuple of str): Display labels matching
            ``category_order``.
        years (tuple of int): Years to plot.
        year_labels (tuple of str, optional): Legend labels. If
            ``None``, the years themselves are used.
        colors (tuple of str): Bar colors, one per year.
        line_color (str): Summary line color.
        text_color (str): Aggregate value text color.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        ylim_top_factor (float): Factor extending the upper y limit.
        bar_width (float): Width of each bar.
        line_gap_frac (float): Gap between the tallest bar of a group
            and its summary line, as a fraction of the global maximum.
        text_gap_frac (float): Gap between the line and its text, as a
            fraction of the global maximum.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with one ``median_<year>`` column
        per year plus ``aggregate``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or has no
            ``DatetimeIndex``.
        ValueError: If a column is not in ``df``, if
            ``category_labels`` does not match ``category_order``, or if
            the label / color tuples do not match the requested years.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col, col_year, col_seasonality)

    if len(category_labels) != len(category_order):
        raise ValueError(
            "'category_labels' must have the same length as "
            "'category_order'."
        )

    return _plot_grouped_demand(
        df, col, col_year, group=df[col_seasonality].to_numpy(),
        categories=category_order, tick_labels=category_labels,
        years=years, year_labels=year_labels, colors=colors,
        xlab=xlab, ylab=ylab, title=title, line_color=line_color,
        text_color=text_color, figsize=figsize, label_size=label_size,
        title_size=title_size, tick_size=tick_size,
        ylim_top_factor=ylim_top_factor, bar_width=bar_width,
        line_gap_frac=line_gap_frac, text_gap_frac=text_gap_frac,
        index_name="seasonality_type", render=render, fmt=fmt, dpi=dpi,
    )


def plot_demand_by_time_slot(
    df,
    col="pasajeros",
    col_year="anio",
    years=_DEFAULT_YEARS,
    year_labels=None,
    colors=("#222222", "#0072B2", "#D55E00"),
    line_styles=("-", "--", ":"),
    markers=("o", "s", "^"),
    xlab="Time slot",
    ylab="Passengers (median)",
    title="Demand by time slot and year",
    figsize=(12, 4.8),
    label_size=11,
    title_size=12,
    tick_size=10,
    legend_size=12,
    linewidth=1.4,
    markersize=5,
    ylim_top_factor=1.15,
    x_tick_interval=4,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Intraday demand profile by time slot and year.

    For each (time slot, year) pair, the representative value is the
    median of passengers per slot over all days of the year. Each year
    is drawn as a line with markers, using a distinct color, line style
    and marker.

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        col_year (str): Year column.
        years (tuple of int): Years to plot.
        year_labels (tuple of str, optional): Legend labels. If
            ``None``, the years themselves are used.
        colors (tuple of str): Line colors, one per year.
        line_styles (tuple of str): Line styles, one per year.
        markers (tuple of str): Markers, one per year.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        title (str): Plot title.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        legend_size (float): Font size for the legend.
        linewidth (float): Line width.
        markersize (float): Marker size.
        ylim_top_factor (float): Factor extending the upper y limit.
        x_tick_interval (int): Interval between x-axis ticks.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with one ``median_<year>`` column
        per year, indexed by ``time_slot``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or has no
            ``DatetimeIndex``.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col, col_year)

    years = tuple(years)
    year_labels = _resolve_year_labels(years, year_labels)

    labels, slots = _time_slots(df.index)
    frame = pd.DataFrame({
        "slot": labels,
        "year": df[col_year].to_numpy(),
        "value": df[col].to_numpy(),
    })
    medians = {
        y: frame.loc[frame["year"] == y].groupby("slot")["value"]
        .median().reindex(slots)
        for y in years
    }

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(slots))
    for y, color, style, marker, label in zip(
        years, colors, line_styles, markers, year_labels
    ):
        ax.plot(
            x, medians[y].to_numpy(), color=color, linestyle=style,
            marker=marker, markersize=markersize, linewidth=linewidth,
            markerfacecolor=color, markeredgecolor=color, label=label,
            zorder=3,
        )

    ax.set_ylim(0, max(s.max() for s in medians.values()) * ylim_top_factor)
    _use_scientific_yaxis(ax, tick_size)
    _apply_slot_xticks(ax, slots, x_tick_interval, tick_size)
    _label_axes(ax, xlab, ylab, title, label_size, title_size)
    ax.legend(frameon=False, fontsize=legend_size, loc="upper right")
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    table = pd.DataFrame({f"median_{y}": medians[y] for y in years}).round(1)
    table.index.name = "time_slot"

    return fig, ax, table


def plot_zeros_by_time_slot(
    df,
    col="pasajeros",
    col_n_trips="n_viajes",
    xlab="Time slot",
    ylab="Time slots with zero passengers",
    title="Composition of zeros by time slot",
    type_labels=("No dispatch", "Zero with trips"),
    colors=("black", "#B0B0B0"),
    figsize=(12, 4.8),
    label_size=11,
    title_size=12,
    tick_size=10,
    legend_size=12,
    ylim_top_factor=1.25,
    x_tick_interval=4,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Stacked composition of zero-passenger slots by time slot.

    Each slot with ``col == 0`` is classified using the operational
    record as either *no dispatch* (``col_n_trips == 0``) or *zero with
    trips* (``col_n_trips > 0``). Counts per time slot are stacked by
    type.

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        col_n_trips (str): Number-of-trips column.
        xlab (str): X-axis label.
        ylab (str): Y-axis label.
        title (str): Plot title.
        type_labels (tuple of str): Legend labels
            ``(no_dispatch, zero_with_trips)``.
        colors (tuple of str): Fill colors of the two types.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for the title.
        tick_size (float): Font size for tick labels.
        legend_size (float): Font size for the legend.
        ylim_top_factor (float): Factor extending the upper y limit.
        x_tick_interval (int): Interval between x-axis ticks.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax, table)`` with the columns ``no_dispatch`` and
        ``zero_with_trips``, indexed by ``time_slot``.

    Raises:
        TypeError: If ``df`` is not a DataFrame, has no
            ``DatetimeIndex``, or ``type_labels`` / ``colors`` is not a
            pair.
        ValueError: If a column is not in ``df``.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col, col_n_trips)
    _require_pair(type_labels, "type_labels")
    _require_pair(colors, "colors")

    labels, slots = _time_slots(df.index)

    is_zero = (df[col] == 0).to_numpy()
    no_dispatch = is_zero & (df[col_n_trips] == 0).to_numpy()
    frame = pd.DataFrame({
        "slot": labels,
        "no_dispatch": no_dispatch,
        "zero_with_trips": is_zero & ~no_dispatch,
    })
    table = (
        frame.groupby("slot")[["no_dispatch", "zero_with_trips"]]
        .sum().reindex(slots).astype(int)
    )
    table.index.name = "time_slot"

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(slots))
    bottom = np.zeros(len(slots))
    for kind, color, label in zip(table.columns, colors, type_labels):
        values = table[kind].to_numpy()
        ax.bar(
            x, values, bottom=bottom, width=0.85, color=color,
            edgecolor="black", linewidth=0.4, label=label, zorder=3,
        )
        bottom += values

    ax.set_ylim(0, bottom.max() * ylim_top_factor)
    _apply_slot_xticks(ax, slots, x_tick_interval, tick_size)
    _label_axes(ax, xlab, ylab, title, label_size, title_size)
    ax.legend(frameon=False, fontsize=legend_size, loc="upper left")
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    return fig, ax, table


def plot_temporal_panel(
    df,
    col="pasajeros",
    daily_freq="D",
    zoom_left=("2024-03-04", "2024-03-10"),
    zoom_right=("2024-09-02", "2024-09-08"),
    text_left="March 2024",
    text_right="September 2024",
    title="Passenger demand (2023-2025)",
    xlab_zoom="Day of month",
    ylab_daily="Passengers per day",
    ylab_zoom="Passengers per time slot",
    color="black",
    linewidth_daily=0.7,
    linewidth_zoom=1.0,
    figsize=(13, 7),
    label_size=11,
    title_size=12,
    tick_size=10,
    height_ratios=(1.2, 1.0),
    hspace=0.28,
    text_pos=(0.98, 0.95),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Composite panel: daily series (top) and two weekly zooms (bottom).

    Args:
        df (pandas.DataFrame): DataFrame with a ``DatetimeIndex``.
        col (str): Passengers column.
        daily_freq (str): Resampling frequency of the top panel.
        zoom_left (tuple of str): ``(start, end)`` of the left zoom.
        zoom_right (tuple of str): ``(start, end)`` of the right zoom.
        text_left (str): Annotation of the left zoom.
        text_right (str): Annotation of the right zoom.
        title (str): Panel title (placed on the top axes).
        xlab_zoom (str): X-axis label of the zoom panels.
        ylab_daily (str): Y-axis label of the top panel.
        ylab_zoom (str): Y-axis label of the left zoom (the right zoom
            shares the y-axis and hides it).
        color (str): Line color.
        linewidth_daily (float): Line width of the top panel.
        linewidth_zoom (float): Line width of the zoom panels.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for annotations (the title uses
            ``title_size + 2``).
        tick_size (float): Font size for tick labels.
        height_ratios (tuple of float): Row height ratios.
        hspace (float): Vertical spacing passed to the grid.
        text_pos (tuple of float): Annotation position in axes
            coordinates.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, (ax_top, ax_left, ax_right))``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or has no
            ``DatetimeIndex``.
        ValueError: If ``col`` is not in ``df``.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, col)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=height_ratios, hspace=hspace)

    ax_top = fig.add_subplot(grid[0, :])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_left)

    daily = df[col].resample(daily_freq).sum()
    ax_top.plot(daily.index, daily.values, color=color,
                linewidth=linewidth_daily, zorder=3)
    # The title goes on the axes to avoid colliding with the x10^n offset text.
    ax_top.set_title(title, fontsize=title_size + 2, pad=15)
    _label_axes(ax_top, None, ylab_daily, None, label_size, title_size)
    _style_axes(ax_top, tick_size)

    for ax, (start, end), text in (
        (ax_left, zoom_left, text_left),
        (ax_right, zoom_right, text_right),
    ):
        sub = df.loc[start:end]
        ax.plot(sub.index, sub[col].to_numpy(), color=color,
                linewidth=linewidth_zoom, zorder=3)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.text(
            text_pos[0], text_pos[1], text, transform=ax.transAxes,
            ha="right", va="top", fontsize=title_size, color="black",
            zorder=5,
        )
        _label_axes(ax, xlab_zoom, None, None, label_size, title_size)
        _style_axes(ax, tick_size)

    _label_axes(ax_left, None, ylab_zoom, None, label_size, title_size)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)

    for ax in (ax_top, ax_left):
        _use_scientific_yaxis(ax)
        ax.yaxis.get_offset_text().set_visible(True)

    _finish_figure(fig, render, fmt, dpi, tight=False)

    return fig, (ax_top, ax_left, ax_right)


# Autocorrelation ----

def plot_acf_pacf(
    series,
    titles=("Autocorrelation (ACF)", "Partial autocorrelation (PACF)"),
    xlab="Lag",
    ylab="Autocorrelation",
    lags=100,
    ylim=(-1.1, 1.1),
    figsize=(10, 4),
    label_size=11,
    title_size=12,
    tick_size=10,
    color="black",
    band_color="lightgray",
    band_alpha=0.3,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot the ACF and PACF of a univariate series side by side.

    Draws both functions with ``statsmodels`` using the toolkit's style:
    black stems, light-gray confidence bands, no gridlines. A lighter
    sibling of ``ts_toolkit.plot_acf_pacf`` (same name, fewer options).

    Args:
        series (pandas.Series): Univariate time series. NaNs are
            dropped.
        titles (tuple of str): Titles of the ACF and PACF subplots.
        xlab (str): X-axis label of both subplots.
        ylab (str): Y-axis label of the left subplot.
        lags (int): Number of lags to display.
        ylim (tuple of float): Y-axis limits of both subplots.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for titles.
        tick_size (float): Font size for tick labels.
        color (str): Stem and marker color.
        band_color (str): Confidence band color.
        band_alpha (float): Confidence band transparency.
        render (bool): If ``True``, renders inline.
        fmt ({"svg", "png"}): Output format.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple: ``(fig, (ax1, ax2))``.

    Raises:
        TypeError: If ``series`` has no ``dropna`` method or ``titles``
            is not a pair.
        ValueError: If ``series`` is empty after dropping NaNs.
    """
    if not hasattr(series, "dropna"):
        raise TypeError(
            "'series' must be a pandas Series or similar object with a "
            "'dropna' method."
        )
    _require_pair(titles, "titles")

    series = series.dropna()
    if len(series) == 0:
        raise ValueError("'series' is empty after dropping NaNs.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    common = {
        "lags": lags, "zero": False, "color": color,
        "vlines_kwargs": {"color": color, "linewidth": 0.8},
    }
    plot_acf(series, ax=ax1, title=titles[0], **common)
    plot_pacf(series, ax=ax2, title=titles[1], **common)

    for ax in (ax1, ax2):
        _style_axes(ax, tick_size)

        for collection in ax.collections:
            collection.set_facecolor(band_color)
            collection.set_alpha(band_alpha)

        for line in ax.lines:
            line.set_markerfacecolor(color)
            line.set_markeredgecolor(color)
            line.set_markersize(1.8)
            line.set_linewidth(0.8)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(ax.get_title(), fontsize=title_size, pad=10)
        ax.set_xlabel(xlab, fontsize=label_size, labelpad=10)

    ax1.set_ylabel(ylab, fontsize=label_size, labelpad=12)
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

    _finish_figure(fig, render, fmt, dpi)

    return fig, (ax1, ax2)


# Modeling prep ----

def split_and_save(
    df,
    train_ratio=0.80,
    train_filename="train.parquet",
    holdout_filename="holdout.parquet",
    output_dir=None,
    verbose=True,
):
    """Split a dataset chronologically and save train / holdout parquets.

    Args:
        df (pandas.DataFrame): DataFrame with a temporal index.
        train_ratio (float): Fraction used for training, in ``(0, 1)``.
        train_filename (str): File name of the training set.
        holdout_filename (str): File name of the holdout set.
        output_dir (str or pathlib.Path, optional): Output directory
            (created if missing). If ``None``, ``DATA_PROCESSED`` is
            used.
        verbose (bool): If ``True``, prints a summary of the split.

    Returns:
        tuple: ``(train, test, cutoff_date)``.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``train_ratio`` is not in ``(0, 1)`` or the split
            produces an empty train or test set.
    """
    _require_dataframe(df)

    if (
        not isinstance(train_ratio, (int, float))
        or isinstance(train_ratio, bool)
        or not (0 < train_ratio < 1)
    ):
        raise ValueError("'train_ratio' must be a number in the interval (0, 1).")

    out_dir = Path(output_dir) if output_dir is not None else DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_index()
    n_total = len(df)
    n_train = int(n_total * train_ratio)

    if n_train == 0 or n_train == n_total:
        raise ValueError("'train_ratio' produces an empty train or test set.")

    cutoff_date = df.index[n_train]
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()

    train.to_parquet(out_dir / train_filename)
    test.to_parquet(out_dir / holdout_filename)

    if verbose:
        print("Saved:")
        print(f"  {out_dir / train_filename}    — {len(train):,} time slots")
        print(f"  {out_dir / holdout_filename}  — {len(test):,} time slots")
        print()
        print(f"Cutoff date: {cutoff_date}")
        print(f"Train: {train.index.min()} → {train.index.max()}")
        print(f"Test:  {test.index.min()} → {test.index.max()}")
        print()
        print(f"Train share:   {len(train) / n_total * 100:.1f} %")
        print(f"Holdout share: {len(test) / n_total * 100:.1f} %")

    return train, test, cutoff_date


# Route maps ----

def list_available_routes(data_dir=None):
    """Return the route ids that have a geometry file in ``data_dir``.

    Scans for ``geometria_{route}.geojson`` files.

    Args:
        data_dir (str or pathlib.Path, optional): Directory to scan.
            When ``None``, the default candidates in
            ``_DEFAULT_DATA_DIRS`` are tried in order.

    Returns:
        list of str: Sorted route ids (e.g. ``["c72", "k54", "silencio"]``).

    Raises:
        FileNotFoundError: If no candidate data directory exists.
    """
    data_dir = _resolve_data_dir(data_dir)
    prefix = "geometria_"
    suffix = ".geojson"
    routes = [
        path.stem[len(prefix):-len(suffix)]
        for path in data_dir.glob(f"{prefix}*{suffix}")
    ]
    return sorted(routes)


def generate_map(
    route,
    title=None,
    data_dir=None,
    figsize=(5.0, 5.5),
    save_to=None,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Build and return the map of a bus route from local GeoJSON files.

    Reads ``geometria_{route}.geojson`` (mandatory) and, when available,
    a shared roads layer (``vias_{route}.geojson``, ``vias.geojson`` or
    any ``vias_*.geojson``). No database connection is required, and all
    input files are discovered automatically inside ``data_dir``.

    The rendered output (title, legend, compass card) is intentionally
    in Spanish to match the operational context of the source data.

    Args:
        route (str): Route id (e.g. ``"k54"``, ``"c72"``, ``"silencio"``).
            Any id for which a geometry file exists is accepted.
        title (str, optional): Custom title. Falls back to
            ``ROUTE_CATALOG[route]["title"]`` or, if the route is not in
            the catalog, to a title built from the route id.
        data_dir (str or pathlib.Path, optional): Directory holding the
            GeoJSON files. When ``None``, ``_DEFAULT_DATA_DIRS`` is
            searched in order.
        figsize (tuple of float): Figure size in inches.
        save_to (str or pathlib.Path, optional): Output path. A ``.pdf``
            extension produces a vector PDF; any other extension
            produces a raster image at 300 dpi.
        render (bool): If ``True`` (default), the figure is rendered
            inline as centered HTML using the toolkit's shared visual
            container. Set to ``False`` when the caller wants to keep
            the figure alive (e.g. to call ``plt.show()`` manually or
            to post-process the axes before display).
        fmt ({"svg", "png"}): Output format forwarded to the inline
            renderer. ``"svg"`` (default) is vector and
            dpi-independent; ``"png"`` rasterizes at ``dpi``.
        dpi (int): Resolution used only when ``fmt="png"``.

    Returns:
        tuple: ``(fig, ax)``. When ``render=True`` the figure has
        already been closed by the inline renderer — use
        ``render=False`` if you need to manipulate it afterwards.

    Raises:
        ValueError: If ``route`` is not a non-empty string, or if the
            geometry file contains no usable line geometries.
        FileNotFoundError: If the geometry file or the data directory
            cannot be found.
    """
    route_id = route.strip().lower() if isinstance(route, str) else ""
    if not route_id:
        raise ValueError("'route' must be a non-empty string.")

    data_dir = _resolve_data_dir(data_dir)
    lines = _load_route_geometry(route_id, data_dir)
    road_segments = _load_roads(route_id, data_dir)

    return_index = _find_return_index(lines)
    west, east, south, north = _bounding_box(lines)

    fig, ax = plt.subplots(figsize=figsize)

    if road_segments:
        ax.add_collection(LineCollection(
            road_segments, colors="#b0b0b0", linewidths=0.25, zorder=1,
        ))

    _draw_route(ax, lines, return_index)
    _draw_markers(ax, lines, return_index)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or _resolve_default_title(route_id), fontsize=10, pad=9)

    plt.tight_layout()
    
    _draw_compass(ax, fig)
    _draw_legend(ax)

    if save_to is not None:
        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".pdf":
            fig.savefig(out_path, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Guardado: {out_path}")

    if render:
        _render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, ax


# Private helpers ====

# Validation ----

def _require_dataframe(df, name="df"):
    """Raise ``TypeError`` unless ``df`` is a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'{name}' must be a pandas DataFrame.")


def _require_datetime_index(df, name="df"):
    """Raise ``TypeError`` unless ``df`` has a ``DatetimeIndex``."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"'{name}' must have a pandas DatetimeIndex.")


def _require_columns(df, *columns):
    """Validate that every name in ``columns`` is a string column of ``df``."""
    for c in columns:
        if not isinstance(c, str):
            raise TypeError(
                f"Column names must be strings, got '{type(c).__name__}'."
            )
        if c not in df.columns:
            raise ValueError(f"Column '{c}' is not in 'df'.")


def _require_bool(value, name):
    """Raise ``TypeError`` unless ``value`` is a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"'{name}' must be a boolean.")


def _require_pair(value, name):
    """Raise ``TypeError`` unless ``value`` is a tuple/list of length 2."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"'{name}' must be a tuple or list of two elements.")


# Figure rendering ----

def _render_figure(fig, fmt="svg", dpi=100):
    """Render a matplotlib figure inline as centered HTML.

    Internal rendering engine shared by every ``plot_*`` function. It is
    not part of the public API: callers reach it through the ``render``,
    ``fmt`` and ``dpi`` parameters of the plotting functions.

    Args:
        fig (matplotlib.figure.Figure): The figure to render. It is
            closed after encoding, so it will not also pop up as a
            separate matplotlib window.
        fmt ({"svg", "png"}): Output format. ``"svg"`` (default) embeds
            a vector image directly in the notebook output; ``dpi`` has
            no effect in this mode. ``"png"`` rasterizes the figure and
            embeds it as a Base64-encoded image.
        dpi (int): Resolution used only when ``fmt="png"``.

    Returns:
        None: The figure is displayed as a side effect.

    Raises:
        ValueError: If ``fmt`` is not one of ``{"svg", "png"}``.
    """
    if fmt not in ("svg", "png"):
        raise ValueError(f"'fmt' must be 'svg' or 'png', got '{fmt}'.")

    buf = io.BytesIO()

    if fmt == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg_data = buf.getvalue().decode("utf-8")
        display(
            HTML(
                f'<div style="{_FIGURE_CONTAINER_STYLE}">{svg_data}</div>'
            )
        )
    else:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        encoded = base64.b64encode(buf.getbuffer()).decode("ascii")
        display(
            HTML(
                f'<div style="{_FIGURE_CONTAINER_STYLE}">'
                f'<img src="data:image/png;base64,{encoded}"></div>'
            )
        )


# Axes styling and figure plumbing ----

def _style_spines(ax, color="black", linewidth=0.8):
    """Apply the toolkit's spine style to every spine of ``ax``."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)


def _style_ticks(ax, tick_size):
    """Apply the toolkit's tick style to the x and y axes of ``ax``."""
    for axis, side in (("x", "bottom"), ("y", "left")):
        ax.tick_params(
            axis=axis, which="both", labelsize=tick_size,
            direction="out", length=4, width=0.8, colors="black",
            **{side: True},
        )


def _style_axes(ax, tick_size):
    """Apply the toolkit's standard axis style (spines, ticks, no grid)."""
    _style_spines(ax)
    _style_ticks(ax, tick_size)
    ax.grid(False)


def _label_axes(ax, xlab, ylab, title, label_size, title_size,
                x_pad=10, y_pad=12, title_pad=10):
    """Set axis labels and title; ``None`` leaves the element untouched."""
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=label_size, labelpad=x_pad)
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=label_size, labelpad=y_pad)
    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=title_pad)


def _use_scientific_yaxis(ax, tick_size=None):
    """Use scientific notation (math text) on the y-axis of ``ax``."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    if tick_size is not None:
        ax.yaxis.get_offset_text().set_fontsize(tick_size)


def _annotate_bars(ax, bars, labels, offset, fontsize):
    """Write one centered label above each bar, separated by ``offset``."""
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center", va="bottom", fontsize=fontsize, zorder=4,
        )


def _finish_figure(fig, render, fmt, dpi, tight=True):
    """Apply ``tight_layout`` (optional) and render the figure if asked."""
    if tight:
        fig.tight_layout()
    if render:
        _render_figure(fig, fmt=fmt, dpi=dpi)


def _vertical_kwargs():
    """Boxplot kwarg selecting vertical orientation across Matplotlib versions."""
    if mpl.__version_info__ >= (3, 10):
        return {"orientation": "vertical"}
    return {"vert": True}


# HTML tables ----

def _table_style_rules():
    """Return the shared HTML style rules used by every styled table."""
    return [
        {
            "selector": "",
            "props": [
                ("margin-left", "auto"),
                ("margin-right", "auto"),
                ("width", "auto"),
                ("border-collapse", "collapse"),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("background-color", "#f2f2f2"),
                ("color", "black"),
                ("font-weight", "bold"),
                ("border", "1px solid black"),
                ("padding", "10px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "center"),
                ("border", "1px solid black"),
                ("padding", "10px"),
            ],
        },
        {"selector": "th:first-child", "props": [("text-align", "left")]},
        {"selector": "td:first-child", "props": [("text-align", "left")]},
    ]


def _styled_table(df):
    """Return ``df`` as a ``Styler`` with the toolkit's table style."""
    return df.style.hide(axis="index").set_table_styles(_table_style_rules())


def _render_styled(styler, max_width=None):
    """Render a ``Styler`` as centered, horizontally scrollable HTML."""
    container = _TABLE_CONTAINER_STYLE
    if max_width is not None:
        container += f" max-width: {max_width};"
    display(HTML(f"<div style='{container}'>{styler.to_html()}</div>"))


def _real_type_name(series):
    """Return the Python type name of the first non-null value of ``series``.

    Falls back to the pandas dtype when the series is empty or all null.
    """
    non_null = series.dropna()
    if non_null.empty:
        return str(series.dtype)
    return type(non_null.iloc[0]).__name__


# Trip-level helpers ----

def _hour_of(timestamps):
    """Return the hour of day (0-23) of a timestamp-like Series."""
    return pd.to_datetime(timestamps).dt.hour


def _trips_in_hours(df, col_start, col_duration, hours):
    """Return non-cancelled trips starting in ``hours``, with an ``_hour`` column."""
    hour = _hour_of(df[col_start])
    mask = (hour.isin(hours) & df[col_duration].notna()).to_numpy()
    subset = df.loc[mask].copy()
    subset["_hour"] = hour.to_numpy()[mask]
    return subset


def _plot_hourly_share(
    hours, flag, count_name, title, xlab, ylab, bar_color, edge_color,
    edge_width, figsize, label_size, title_size, tick_size, render, fmt, dpi,
):
    """Bar chart of the percentage of flagged trips for each hour (0-23).

    Shared engine of the ``plot_*_trips_by_hour`` functions. ``hours``
    and ``flag`` are aligned arrays with the hour of each trip and a
    boolean marking whether it counts in the numerator.

    Returns:
        tuple: ``(fig, ax, table)`` with the columns ``hour``,
        ``total``, ``<count_name>`` and ``pct``.
    """
    work = pd.DataFrame({"hour": hours, "flag": np.asarray(flag, dtype="int8")})

    table = (
        work.groupby("hour")
        .agg(total=("flag", "size"), **{count_name: ("flag", "sum")})
        .reset_index()
    )
    table["pct"] = (table[count_name] / table["total"] * 100).round(2)
    table = table[table["total"] > 0].sort_values("hour").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        table["hour"], table["pct"], color=bar_color, edgecolor=edge_color,
        linewidth=edge_width, width=0.7,
    )
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=tick_size - 1)

    _label_axes(ax, xlab, ylab, title, label_size, title_size)
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    return fig, ax, table


# Hampel helpers ----

def _hampel_flag(x, k=7, t=3.0):
    """Return a boolean mask flagging Hampel outliers in the array ``x``."""
    n = len(x)
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        window = x[max(0, i - k):min(n, i + k + 1)]
        window = window[~np.isnan(window)]
        if len(window) < 3:
            continue
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad == 0:
            continue
        if abs(x[i] - med) > t * 1.4826 * mad:
            flags[i] = True
    return flags


def _robust_bounds(series, t):
    """Return global ``median ± t * 1.4826 * MAD`` bounds of ``series``."""
    med = series.median()
    mad = (series - med).abs().median()
    return {
        "t": t,
        "median": float(med),
        "mad": float(mad),
        "lo": float(med - t * 1.4826 * mad),
        "hi": float(med + t * 1.4826 * mad),
    }


# Demand helpers ----

def _resolve_year_labels(years, year_labels):
    """Return legend labels for ``years`` (the years themselves by default)."""
    if year_labels is None:
        return tuple(str(y) for y in years)
    if len(year_labels) != len(years):
        raise ValueError("'year_labels' must have one label per year.")
    return tuple(year_labels)


def _daily_totals(df, col, col_year, group=None):
    """Sum ``col`` by calendar day, year and (optionally) a grouping array.

    Returns:
        pandas.DataFrame: Columns ``date``, ``year``, ``group`` (only if
        ``group`` is given) and ``daily_total``.
    """
    data = {"date": df.index.date, "year": df[col_year].to_numpy()}
    if group is not None:
        data["group"] = np.asarray(group)
    data["value"] = df[col].to_numpy()

    frame = pd.DataFrame(data)
    return (
        frame.groupby([c for c in frame.columns if c != "value"])["value"]
        .sum()
        .rename("daily_total")
        .reset_index()
    )


def _plot_grouped_demand(
    df, col, col_year, group, categories, tick_labels, years, year_labels,
    colors, xlab, ylab, title, line_color, text_color, figsize, label_size,
    title_size, tick_size, ylim_top_factor, bar_width, line_gap_frac,
    text_gap_frac, index_name, render, fmt, dpi,
):
    """Grouped bars of median daily demand per category and year.

    Shared engine of the ``plot_demand_by_*`` functions (month, week of
    month, weekday, seasonality). ``group`` is an array aligned with the
    rows of ``df`` holding the category of each row. Above each
    category's group of bars, a horizontal line is drawn over the
    tallest bar and annotated with the aggregate median across years.

    Returns:
        tuple: ``(fig, ax, table)`` with one ``median_<year>`` column per
        year plus ``aggregate``, indexed by ``index_name``.
    """
    years = tuple(years)
    year_labels = _resolve_year_labels(years, year_labels)
    if len(colors) < len(years):
        raise ValueError("'colors' must have at least one color per year.")

    categories = list(categories)
    daily = _daily_totals(df, col, col_year, group)

    by_year = {
        y: daily.loc[daily["year"] == y]
        .groupby("group")["daily_total"].median().reindex(categories)
        for y in years
    }
    aggregate = (
        daily.groupby("group")["daily_total"].median().reindex(categories)
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(categories))
    series = [by_year[y] for y in years]

    for i, (serie, color, label) in enumerate(zip(series, colors, year_labels)):
        offset = (i - (len(series) - 1) / 2) * bar_width
        ax.bar(
            x + offset, serie.to_numpy(), width=bar_width, facecolor=color,
            edgecolor="black", linewidth=0.6, label=label, zorder=3,
        )

    y_bar_max = max(s.max() for s in series)
    ax.set_ylim(0, y_bar_max * ylim_top_factor)

    tops = np.nanmax(np.vstack([s.to_numpy() for s in series]), axis=0)
    line_gap = y_bar_max * line_gap_frac
    text_gap = y_bar_max * text_gap_frac
    half = len(series) * bar_width / 2

    for xi, value, top in zip(x, aggregate.to_numpy(), tops):
        y_line = top + line_gap
        ax.plot([xi - half, xi + half], [y_line, y_line],
                color=line_color, linewidth=1.2, zorder=4)
        ax.text(
            xi, y_line + text_gap, f"{value:,.0f}", ha="center",
            va="bottom", fontsize=tick_size, color=text_color, zorder=5,
        )

    _use_scientific_yaxis(ax, tick_size)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_size)
    _label_axes(ax, xlab, ylab, title, label_size, title_size)
    ax.legend(frameon=False, fontsize=tick_size, loc="upper right")
    _style_axes(ax, tick_size)
    _finish_figure(fig, render, fmt, dpi)

    table = pd.DataFrame(
        {**{f"median_{y}": by_year[y] for y in years}, "aggregate": aggregate}
    ).round(0)
    table.index.name = index_name

    return fig, ax, table


def _time_slots(index):
    """Return ``(labels, slots)``: ``HH:MM`` label per row and sorted unique slots."""
    labels = np.asarray(index.strftime("%H:%M"))
    return labels, sorted(set(labels))


def _apply_slot_xticks(ax, slots, interval, tick_size):
    """Place a tick every ``interval`` time slots on the x-axis of ``ax``."""
    ticks = list(range(0, len(slots), interval))
    ax.set_xticks(ticks)
    ax.set_xticklabels([slots[i] for i in ticks], rotation=0, fontsize=tick_size)


def _cast_pattern_keys(frame, int_keys, str_key):
    """Cast pattern keys to ``int64`` / ``str`` so frames merge cleanly."""
    frame = frame.copy()
    frame[int_keys] = frame[int_keys].astype("int64")
    frame[str_key] = frame[str_key].astype(str)
    return frame


# Route-map helpers ----

def _resolve_data_dir(data_dir=None):
    """Return an existing data directory, searching the default candidates.

    Args:
        data_dir (str or pathlib.Path, optional): Explicit directory.
            When ``None``, the first existing candidate in
            ``_DEFAULT_DATA_DIRS`` is used.

    Returns:
        pathlib.Path: The resolved, existing directory.

    Raises:
        FileNotFoundError: If no candidate directory exists.
    """
    if data_dir is not None:
        path = Path(data_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {path}")
        return path

    for candidate in _DEFAULT_DATA_DIRS:
        if candidate.is_dir():
            return candidate

    tried = ", ".join(str(c) for c in _DEFAULT_DATA_DIRS)
    raise FileNotFoundError(
        f"Could not locate a data directory. Tried: {tried}. "
        "Pass 'data_dir' explicitly to override."
    )


def _resolve_geometry_file(route, data_dir):
    """Return the geometry GeoJSON path for ``route``.

    Raises:
        ValueError: If ``route`` is not a non-empty string.
        FileNotFoundError: If the expected file does not exist.
    """
    if not isinstance(route, str) or not route.strip():
        raise ValueError("'route' must be a non-empty string.")

    route = route.strip().lower()
    path = Path(data_dir) / _GEOMETRY_TEMPLATE.format(route=route)
    if not path.exists():
        available = list_available_routes(data_dir)
        hint = (
            f" Available routes: {available}." if available
            else " No route geometries were found in that directory."
        )
        raise FileNotFoundError(
            f"Route geometry not found for '{route}'. Expected: {path}.{hint}"
        )
    return path


def _resolve_vias_file(route, data_dir):
    """Return the roads GeoJSON path, or ``None`` if no candidate exists.

    Search order:
        1. ``vias_{route}.geojson``
        2. ``vias.geojson``
        3. First match of ``vias_*.geojson`` (alphabetical)
    """
    data_dir = Path(data_dir)
    for template in _VIAS_TEMPLATES:
        candidate = data_dir / template.format(route=route)
        if candidate.exists():
            return candidate

    for candidate in sorted(data_dir.glob(_VIAS_GLOB)):
        if candidate.exists():
            return candidate

    return None


def _load_route_geometry(route, data_dir):
    """Load and validate the route geometry as a list of ``LineString``.

    ``MultiLineString`` features are exploded into individual
    ``LineString`` geometries so every downstream operation can rely on
    ``.xy`` and ``.coords``.

    Raises:
        ValueError: If the file contains no usable line geometries.
    """
    path = _resolve_geometry_file(route, data_dir)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    features = payload.get("features") or []
    if not features:
        raise ValueError(f"No features found in {path}.")

    lines = []
    for feat in features:
        geom_json = feat.get("geometry") if isinstance(feat, dict) else None
        if not geom_json:
            continue
        geom = _shapely_shape(geom_json)
        if geom.geom_type == "LineString":
            lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            lines.extend(geom.geoms)

    if not lines:
        raise ValueError(
            f"No LineString or MultiLineString geometries found in {path}."
        )
    return lines


def _load_roads(route, data_dir):
    """Load the optional base road network as a list of coordinate arrays.

    Returns an empty list when no roads file is available, so callers can
    always treat the result uniformly.
    """
    path = _resolve_vias_file(route, data_dir)
    if path is None:
        return []

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    segments = []
    for feat in payload.get("features") or []:
        geom_json = feat.get("geometry") if isinstance(feat, dict) else None
        if not geom_json:
            continue
        try:
            geom = _shapely_shape(geom_json)
        except Exception:  # noqa: BLE001, S112
            continue

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            segments.append(np.asarray(line.coords))
    return segments


def _bounding_box(lines, margin=0.008):
    """Return ``(west, east, south, north)`` for a list of ``LineString``."""
    coords = []
    for line in lines:
        coords.extend(line.coords)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons) - margin, max(lons) + margin,
            min(lats) - margin, max(lats) + margin)


def _find_return_index(lines):
    """Return the index of the segment whose midpoint sits farthest north."""
    lats = np.array([g.interpolate(0.5, normalized=True).y for g in lines])
    return int(np.argmax(lats))


def _draw_route(ax, lines, return_index):
    """Draw the outward leg as a solid line and the return leg as dashed."""
    for i, line in enumerate(lines):
        x, y = line.xy
        if i <= return_index:
            ax.plot(x, y, color="black", linewidth=1.8,
                    solid_capstyle="round", zorder=4)
        else:
            ax.plot(x, y, color="black", linewidth=1.2,
                    linestyle=(0, (6, 4)), zorder=3)


def _draw_markers(ax, lines, return_index):
    """Draw the start, return and end markers of the route."""
    start = lines[0].coords[0]
    turnaround = lines[return_index].coords[-1]
    end = lines[-1].coords[-1]

    ax.scatter(*start, color="white", s=55, zorder=6,
               edgecolor="black", linewidth=1.0, marker="o")
    ax.scatter(*turnaround, color="white", s=55, zorder=6,
               edgecolor="black", linewidth=1.0, marker="o")
    ax.scatter(*end, color="black", s=55, zorder=6,
               edgecolor="black", linewidth=1.0, marker="o")


def _draw_compass(ax, fig, cx=0.88, cy=0.90, radius=0.055):
    """Draw a stylized north-pointing compass rose in axes coordinates."""
    fig_w, fig_h = fig.get_size_inches()
    bbox = ax.get_position()
    ratio = (bbox.width * fig_w) / (bbox.height * fig_h)

    ax.add_patch(Ellipse(
        (cx, cy), width=2 * radius, height=2 * radius * ratio,
        transform=ax.transAxes, fill=False,
        edgecolor="black", linewidth=0.8, zorder=10,
    ))
    ax.add_patch(Ellipse(
        (cx, cy),
        width=2 * radius * 0.35,
        height=2 * radius * 0.35 * ratio,
        transform=ax.transAxes, fill=False,
        edgecolor="black", linewidth=0.5, zorder=10,
    ))

    radius_y = radius * ratio
    ax.plot([cx, cx], [cy - radius_y, cy + radius_y],
            transform=ax.transAxes, color="black",
            linewidth=0.7, zorder=10)
    ax.plot([cx - radius, cx + radius], [cy, cy],
            transform=ax.transAxes, color="black",
            linewidth=0.7, zorder=10)

    ax.add_patch(Polygon(
        [(cx, cy + radius_y + 0.012),
         (cx - 0.006, cy + radius_y),
         (cx + 0.006, cy + radius_y)],
        closed=True, transform=ax.transAxes,
        facecolor="black", edgecolor="black",
        linewidth=0.5, zorder=11,
    ))

    offset_x = radius + 0.028
    offset_y = radius_y + 0.028
    ax.text(cx, cy + offset_y, "N", transform=ax.transAxes,
            ha="center", va="center", fontsize=8,
            fontweight="bold", zorder=12)
    ax.text(cx, cy - offset_y, "S", transform=ax.transAxes,
            ha="center", va="center", fontsize=7, zorder=12)
    ax.text(cx + offset_x, cy, "E", transform=ax.transAxes,
            ha="center", va="center", fontsize=7, zorder=12)
    ax.text(cx - offset_x, cy, "W", transform=ax.transAxes,
            ha="center", va="center", fontsize=7, zorder=12)


def _draw_legend(ax):
    """Draw the route legend in Spanish (operational output language)."""
    handles = [
        Line2D([0], [0], color="black", linewidth=1.8, label="Ida"),
        Line2D([0], [0], color="black", linewidth=1.2,
               linestyle=(0, (6, 4)), label="Vuelta"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markersize=6, markeredgecolor="black", label="Fin/Retorno"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=6, markeredgecolor="black", label="Inicio"),
    ]
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.04, 0.01),
        fontsize=8, framealpha=0.95, edgecolor="black", fancybox=False,
        borderpad=0.4, labelspacing=0.3, handlelength=2.0,
        handletextpad=0.5, borderaxespad=0.4, ncol=1,
    )


def _resolve_default_title(route):
    """Return the catalog title for ``route`` or an auto-generated one."""
    entry = ROUTE_CATALOG.get(route)
    if entry is not None:
        return entry["title"]
    return f"Trazado de la ruta {route.upper()}"
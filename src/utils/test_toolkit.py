"""
test_toolkit
============

Evaluation and comparison toolkit for final testing.

This module provides functions to visualize the model's fit and compare
in-sample (Train) versus out-of-sample (Test) performance for passenger demand.
"""

__version__ = "2.5.0"

import base64
import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from eval.scorer import DemandScorer
from IPython.display import HTML, display
from matplotlib.patches import ConnectionPatch, Rectangle

__all__ = [
    "display_train_test_comparison",
    "plot_demand_fit",
    "plot_temporal_fit_panel",
]


# ===========================================================================
# PUBLIC API
# ===========================================================================


def plot_temporal_fit_panel(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    agg_freq: str = "D",
    zoom_left: tuple[str, str] = ("2024-09-02", "2024-09-08"),
    zoom_right: tuple[str, str] = ("2025-02-12", "2025-02-12"),
    text_left: str = "Septiembre 2024",
    text_right: str = "12 Febrero 2025",
    title: str = "Demanda Real vs Predicha",
    xlab_left: str = "Día del mes",
    xlab_right: str = "Hora del día",
    ylab_top: str = "Pasajeros acumulados por día",
    ylab_zoom: str = "Pasajeros por franja",
    label_realized: str = "Demanda Real ($y_t$)",
    label_pred: str = "Predicción LightGBM ($\\hat{y}_t$)",
    colors: tuple[str, str] = ("#7f7f7f", "#0072B2"),
    alphas: tuple[float, float] = (0.75, 0.95),
    linewidth_top: float | tuple[float, float] = 1.0,
    linewidth_zoom: float | tuple[float, float] = 1.2,
    top_ylim: tuple[float, float] | None = None,
    zoom_ylim: tuple[float, float] | None = None,
    top_ylim_pad: tuple[float, float] = (0.06, 0.20),
    zoom_ylim_pad: tuple[float, float] = (0.08, 0.12),
    figsize: tuple[float, float] = (13, 7),
    label_size: float = 11,
    title_size: float = 12,
    tick_size: float = 10,
    legend_size: float = 11,
    legend_loc: str = "upper left",
    height_ratios: tuple[float, float] = (1.2, 1.0),
    hspace: float = 0.04,
    wspace: float = 0.02,
    text_pos: tuple[float, float] = (0.98, 0.95),
    scientific_yaxis: bool = True,
    sci_threshold: float = 10000.0,
    render: bool = True,
    fmt: str = "svg",
    dpi: int = 100,
) -> tuple | None:
    """Composite panel comparing realized vs predicted demand: aggregated series
    (top) and two detail zooms (bottom) with automatic hourly formatting for single-day slices.

    Args:
        dates (pd.DatetimeIndex): Chronological timestamps matching ``y_true`` and ``y_pred``.
        y_true (np.ndarray): Realized demand values.
        y_pred (np.ndarray): Predicted demand values.
        agg_freq (str): Resampling frequency of the top panel (e.g., ``"D"``).
        zoom_left (tuple of str): ``(start, end)`` slice for the left zoom panel.
        zoom_right (tuple of str): ``(start, end)`` slice for the right zoom panel.
        text_left (str): Corner annotation for the left zoom panel.
        text_right (str): Corner annotation for the right zoom panel.
        title (str): Panel title (placed on the top axes).
        xlab_left (str): X-axis label for the bottom-left zoom panel.
        xlab_right (str): X-axis label for the bottom-right zoom panel.
        ylab_top (str): Y-axis label for the top aggregated panel.
        ylab_zoom (str): Y-axis label for the left zoom panel (shared with right).
        label_realized (str): Legend label for realized demand.
        label_pred (str): Legend label for predicted demand.
        colors (tuple of str): ``(color_realized, color_pred)`` line colors.
        alphas (tuple of float): ``(alpha_realized, alpha_pred)`` line transparencies.
        linewidth_top (float or tuple of float): Line width(s) ``(real, pred)`` on the top panel.
        linewidth_zoom (float or tuple of float): Line width(s) ``(real, pred)`` on the bottom zoom panels.
        top_ylim (tuple of float, optional): Explicit ``(ymin, ymax)`` for the top panel.
        zoom_ylim (tuple of float, optional): Explicit ``(ymin, ymax)`` for the bottom zoom panels.
        top_ylim_pad (tuple of float): ``(bottom_pad, top_pad)`` fractions for ``ax_top``.
        zoom_ylim_pad (tuple of float): ``(bottom_pad, top_pad)`` fractions for ``ax_left`` / ``ax_right``.
        figsize (tuple of float): Figure size in inches.
        label_size (float): Font size for axis labels.
        title_size (float): Font size for zoom annotations (title uses ``title_size + 2``).
        tick_size (float): Font size for tick labels.
        legend_size (float): Font size for the legend.
        legend_loc (str): Legend placement on the top axes.
        height_ratios (tuple of float): Row height ratios ``(top, bottom)``.
        hspace (float): Vertical padding fraction between top and bottom rows (via ``constrained_layout``).
        wspace (float): Horizontal padding fraction between bottom-left and bottom-right panels.
        text_pos (tuple of float): Corner annotation position in axes coordinates.
        scientific_yaxis (bool): Whether to format Y axes with scientific notation.
        sci_threshold (float): Minimum upper Y-limit required to trigger scientific notation.
        render (bool): If ``True``, renders inline in Jupyter and returns ``None``.
        fmt ({"svg", "png"}): Output format when rendering inline.
        dpi (int): Resolution for ``fmt="png"``.

    Returns:
        tuple or None: ``(fig, (ax_top, ax_left, ax_right))`` if ``render=False``,
        otherwise ``None``.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    color_realized, color_pred = colors
    alpha_realized, alpha_pred = alphas
    lw_top_real, lw_top_pred = (
        linewidth_top if isinstance(linewidth_top, tuple) else (linewidth_top, linewidth_top)
    )
    lw_zoom_real, lw_zoom_pred = (
        linewidth_zoom if isinstance(linewidth_zoom, tuple) else (linewidth_zoom, linewidth_zoom)
    )

    df = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)},
        index=dates,
    ).sort_index()

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, hspace=hspace, wspace=wspace)
    grid = fig.add_gridspec(2, 2, height_ratios=height_ratios)

    ax_top = fig.add_subplot(grid[0, :])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_left)

    # 1. Top Panel: Aggregated series (Real vs Predicted)
    agg = df.resample(agg_freq).sum()
    ax_top.plot(
        agg.index,
        agg["y_true"].to_numpy(),
        color=color_realized,
        linewidth=lw_top_real,
        alpha=alpha_realized,
        label=label_realized,
        zorder=3,
    )
    ax_top.plot(
        agg.index,
        agg["y_pred"].to_numpy(),
        color=color_pred,
        linewidth=lw_top_pred,
        alpha=alpha_pred,
        label=label_pred,
        zorder=4,
    )

    if top_ylim is not None:
        ax_top.set_ylim(top_ylim)
    else:
        y_min_top = min(agg["y_true"].min(), agg["y_pred"].min())
        y_max_top = max(agg["y_true"].max(), agg["y_pred"].max())
        span_top = y_max_top - y_min_top
        pad_bot, pad_top = top_ylim_pad
        ax_top.set_ylim(
            bottom=y_min_top - span_top * pad_bot,
            top=y_max_top + span_top * pad_top,
        )

    ax_top.set_title(title, fontsize=title_size + 2, pad=12)
    ax_top.set_ylabel(ylab_top, fontsize=label_size, labelpad=10)
    ax_top.legend(frameon=False, fontsize=legend_size, loc=legend_loc)
    _style_axes(ax_top, tick_size)

    # 2. Bottom Panels: Two detail zooms (multi-day or single-day intraday)
    zoom_mins, zoom_maxs = [], []
    for ax, (start, end), text, xlab in (
        (ax_left, zoom_left, text_left, xlab_left),
        (ax_right, zoom_right, text_right, xlab_right),
    ):
        sub = df.loc[start:end]
        if not sub.empty:
            zoom_mins.append(min(sub["y_true"].min(), sub["y_pred"].min()))
            zoom_maxs.append(max(sub["y_true"].max(), sub["y_pred"].max()))

        ax.plot(
            sub.index,
            sub["y_true"].to_numpy(),
            color=color_realized,
            linewidth=lw_zoom_real,
            alpha=alpha_realized,
            zorder=3,
        )
        ax.plot(
            sub.index,
            sub["y_pred"].to_numpy(),
            color=color_pred,
            linewidth=lw_zoom_pred,
            alpha=alpha_pred,
            zorder=4,
        )

        span_hours = (
            (sub.index.max() - sub.index.min()).total_seconds() / 3600.0
            if len(sub) > 1
            else 0
        )
        if span_hours <= 48:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))

        ax.text(
            text_pos[0],
            text_pos[1],
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=title_size,
            color="black",
            zorder=5,
        )
        ax.set_xlabel(xlab, fontsize=label_size, labelpad=8)
        _style_axes(ax, tick_size)

    if zoom_ylim is not None:
        ax_left.set_ylim(zoom_ylim)
    elif zoom_mins:
        y_min_z = min(zoom_mins)
        y_max_z = max(zoom_maxs)
        span_z = y_max_z - y_min_z
        z_pad_bot, z_pad_top = zoom_ylim_pad
        ax_left.set_ylim(
            bottom=y_min_z - span_z * z_pad_bot,
            top=y_max_z + span_z * z_pad_top,
        )

    ax_left.set_ylabel(ylab_zoom, fontsize=label_size, labelpad=10)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)

    if scientific_yaxis:
        for ax in (ax_top, ax_left):
            _use_scientific_yaxis(ax, threshold=sci_threshold)
            ax.yaxis.get_offset_text().set_visible(ax.get_ylim()[1] >= sci_threshold)

    if render:
        _finish_figure(fig, fmt=fmt, dpi=dpi, tight=False)
        return None

    return fig, (ax_top, ax_left, ax_right)


def plot_demand_fit(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Demanda Real vs Predicha",
    xlab: str = "Fecha",
    ylab: str = "Pasajeros por franja",
    label_realized: str = "Demanda Real ($y_t$)",
    label_pred: str = "Predicción LightGBM ($\\hat{y}_t$)",
    colors: tuple[str, str] = ("#7f7f7f", "#0072B2"),
    alphas: tuple[float, float] = (0.75, 0.95),
    linewidths: float | tuple[float, float] = 1.2,
    legend_loc: str = "upper left",
    legend_bbox: tuple = (0.04, 0.98),
    legend_size: float = 12,
    ylim: tuple | None = None,
    ylim_pad: tuple[float, float] = (0.06, 0.18),
    y_step: float | None = None,
    zoom_slice: np.ndarray | None = None,
    zoom_title: str = "Zoom: Período de Interés",
    zoom_bbox: list | tuple = (0.15, 0.30, 0.35, 0.35),
    zoom_pad_frac: float = 0.15,
    pads: tuple = (12, 12, 15),
    label_size: float = 12,
    title_size: float = 12,
    tick_size: float = 11,
    figsize: tuple = (12, 6),
    fmt: str = "svg",
    dpi: int = 100,
) -> None:
    """Plots realized vs predicted demand, with optional inset zoom."""
    color_realized, color_pred = colors
    alpha_realized, alpha_pred = alphas
    lw_real, lw_pred = (
        linewidths if isinstance(linewidths, tuple) else (linewidths, linewidths)
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        dates,
        y_true,
        color=color_realized,
        linewidth=lw_real,
        linestyle="-",
        alpha=alpha_realized,
        label=label_realized,
        zorder=3,
    )
    ax.plot(
        dates,
        y_pred,
        color=color_pred,
        linewidth=lw_pred,
        linestyle="-",
        alpha=alpha_pred,
        label=label_pred,
        zorder=4,
    )

    pad_x, pad_y, pad_title = pads

    ax.set_title(title, fontsize=title_size, pad=pad_title, loc="center")
    ax.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)
    ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)

    _style_axes(ax, tick_size)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        y_min = min(np.min(y_true), np.min(y_pred))
        y_max = max(np.max(y_true), np.max(y_pred))
        span = y_max - y_min
        pad_bot, pad_top = ylim_pad
        ax.set_ylim(bottom=y_min - span * pad_bot, top=y_max + span * pad_top)

    if y_step is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))

    ax.legend(frameon=False, fontsize=legend_size, loc=legend_loc, bbox_to_anchor=legend_bbox)

    if zoom_slice is not None:
        ax_ins = ax.inset_axes(zoom_bbox)

        ax_ins.plot(
            dates[zoom_slice],
            y_true[zoom_slice],
            color=color_realized,
            linewidth=lw_real,
            linestyle="-",
            alpha=alpha_realized,
            zorder=3,
        )
        ax_ins.plot(
            dates[zoom_slice],
            y_pred[zoom_slice],
            color=color_pred,
            linewidth=lw_pred,
            linestyle="-",
            alpha=alpha_pred,
            zorder=4,
        )

        _style_axes(ax_ins, tick_size - 2)
        ax_ins.set_title(zoom_title, fontsize=tick_size, pad=10)

        xmin_data = dates[zoom_slice].min()
        xmax_data = dates[zoom_slice].max()
        ymin = min(y_true[zoom_slice].min(), y_pred[zoom_slice].min())
        ymax = max(y_true[zoom_slice].max(), y_pred[zoom_slice].max())

        zoom_pad = (ymax - ymin) * zoom_pad_frac
        rect_ymin = ymin - zoom_pad
        rect_ymax = ymax + zoom_pad
        ax_ins.set_ylim(rect_ymin, rect_ymax)

        xmin_num = mdates.date2num(xmin_data)
        xmax_num = mdates.date2num(xmax_data)

        rect = Rectangle(
            (xmin_num, rect_ymin),
            xmax_num - xmin_num,
            rect_ymax - rect_ymin,
            fill=False,
            edgecolor="black",
            linewidth=0.9,
            alpha=1.0,
        )
        ax.add_patch(rect)

        con_top = ConnectionPatch(
            xyA=(1.0, 1.0),
            coordsA=ax_ins.transAxes,
            xyB=(xmin_num, rect_ymax),
            coordsB=ax.transData,
            color="black",
            linewidth=0.9,
            alpha=1.0,
            zorder=10,
        )
        con_bot = ConnectionPatch(
            xyA=(1.0, 0.0),
            coordsA=ax_ins.transAxes,
            xyB=(xmin_num, rect_ymin),
            coordsB=ax.transData,
            color="black",
            linewidth=0.9,
            alpha=1.0,
            zorder=10,
        )

        fig.add_artist(con_top)
        fig.add_artist(con_bot)

    _finish_figure(fig, fmt=fmt, dpi=dpi, tight=True)


def display_train_test_comparison(
    y_train: np.ndarray,
    preds_train: np.ndarray,
    y_naive_train: np.ndarray,
    y_test: np.ndarray,
    preds_test: np.ndarray,
    y_naive_test: np.ndarray,
    p: float = 1.28,
    decimals: int = 4,
    column_titles: tuple | None = None,
    metrics: list | None = None,
    show_diff: bool = True,
    error_relativo: bool = False,
) -> None:
    """Generates a styled comparison table of Train (In-Sample) vs Test (Out-of-Sample) metrics.

    Args:
        y_train (np.ndarray): Realized target values for the training set.
        preds_train (np.ndarray): In-sample predicted values.
        y_naive_train (np.ndarray): Seasonal naive baseline for the training set.
        y_test (np.ndarray): Realized target values for the test/holdout set.
        preds_test (np.ndarray): Out-of-sample predicted values.
        y_naive_test (np.ndarray): Seasonal naive baseline for the test/holdout set.
        p (float): Tweedie variance power parameter.
        decimals (int): Number of decimal places to display.
        column_titles (tuple of str, optional): Custom column headers. If ``None``,
            defaults to 3 or 4 columns depending on ``show_diff`` and ``error_relativo``.
        metrics (list of str, optional): Subset/order of metric keys to display.
        show_diff (bool): Whether to include a difference column comparing Test vs Train.
        error_relativo (bool): If ``True``, computes the percentage relative error
            ``((test - train) / |train|) * 100``; otherwise computes the absolute
            error ``test - train``.
    """
    expected_cols = 4 if show_diff else 3
    if column_titles is None:
        if show_diff:
            diff_header = "Error Relativo (%)" if error_relativo else "Error Absoluto"
            column_titles = (
                "Métrica",
                "Entrenamiento (In-Sample)",
                "Prueba (Out-of-Sample)",
                diff_header,
            )
        else:
            column_titles = (
                "Métrica",
                "Entrenamiento (In-Sample)",
                "Prueba (Out-of-Sample)",
            )
    elif len(column_titles) != expected_cols:
        raise ValueError(f"column_titles must be a tuple of exactly {expected_cols} strings.")

    scorer = DemandScorer()

    train_metrics = scorer.calculate_all(y_train, preds_train, y_naive_train, p=p)
    test_metrics = scorer.calculate_all(y_test, preds_test, y_naive_test, p=p)

    metric_labels = {
        "tweedie_deviance": "Tweedie Deviance",
        "mae": "MAE",
        "rmse": "RMSE",
        "mse": "MSE",
        "bias": "Bias",
        "smape": "SMAPE",
        "mase": "MASE",
        "pinball_50": "Pinball (τ=0.50)",
    }

    if metrics is None:
        metrics = list(train_metrics.keys())

    labels = [metric_labels.get(m, m) for m in metrics]
    train_vals = [train_metrics[m] for m in metrics]
    test_vals = [test_metrics[m] for m in metrics]

    data = {
        column_titles[0]: labels,
        column_titles[1]: [f"{v:.{decimals}f}" for v in train_vals],
        column_titles[2]: [f"{v:.{decimals}f}" for v in test_vals],
    }

    if show_diff:
        if error_relativo:
            diff_vals = [
                ((t - tr) / abs(tr) * 100.0) if tr != 0 else np.nan
                for t, tr in zip(test_vals, train_vals)
            ]
            data[column_titles[3]] = [
                f"{v:+.{decimals}f}%" if np.isfinite(v) else "N/A"
                for v in diff_vals
            ]
        else:
            diff_vals = [t - tr for t, tr in zip(test_vals, train_vals)]
            data[column_titles[3]] = [f"{v:+.{decimals}f}" for v in diff_vals]

    df = pd.DataFrame(data)

    styled_table = (
        df.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules())
    )

    _render_styled(styled_table)


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================


def _style_axes(ax, tick_size):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        labelsize=tick_size,
        direction="out",
        length=4,
        width=0.8,
        colors="black",
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        labelsize=tick_size,
        direction="out",
        length=4,
        width=0.8,
        colors="black",
    )


def _use_scientific_yaxis(ax, threshold: float = 10000.0):
    if ax.get_ylim()[1] < threshold:
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="plain")
        return

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)


def _finish_figure(fig, fmt: str = "svg", dpi: int = 100, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()

    buf = io.BytesIO()
    if fmt == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        display(
            HTML(
                f'<div style="display: flex; justify-content: center; width: 100%;">{buf.getvalue().decode("utf-8")}</div>'
            )
        )
    else:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        display(
            HTML(
                f'<div style="text-align: center; width: 100%;"><img src="data:image/png;base64,{base64.b64encode(buf.getbuffer()).decode("ascii")}"></div>'
            )
        )


def _table_style_rules() -> list:
    return [
        {"selector": "", "props": [("margin-left", "auto"), ("margin-right", "auto"), ("width", "auto"), ("border-collapse", "collapse")]},
        {"selector": "th", "props": [("text-align", "center"), ("background-color", "#f2f2f2"), ("color", "black"), ("font-weight", "bold"), ("border", "1px solid black"), ("padding", "10px")]},
        {"selector": "td", "props": [("text-align", "center"), ("border", "1px solid black"), ("padding", "10px")]},
        {"selector": "th:first-child", "props": [("text-align", "left")]},
        {"selector": "td:first-child", "props": [("text-align", "left")]},
    ]


def _render_styled(styler) -> None:
    display(
        HTML(
            "<div style='text-align: center; width: 100%;'>"
            + styler.to_html()
            + "</div>"
        )
    )
"""
trainer_toolkit
===============
"""

__version__ = "2.4.0"

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from scipy.stats import gaussian_kde  # <-- NUEVO IMPORT
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import bds

__all__ = [
    "display_styled_df",
    "display_top_results",
    "plot_learning_curve",
    "plot_metric_evolution",
    "plot_tweedie_fit",
    "run_model_diagnostics",
]


def display_styled_df(
    df: pd.DataFrame,
    decimals: int = 3,
    p_value_cols: list | None = None,
    align_left_first: bool = True,
) -> None:
    if p_value_cols is None:
        p_value_cols = []

    df_display = df.copy()
    format_dict = {}

    for col in df.columns:
        is_p_value = any(p_col in col for p_col in p_value_cols)

        if is_p_value:
            format_dict[col] = lambda x: _format_apa_pvalue(x)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                format_dict[col] = "{:,.0f}"
            else:
                format_dict[col] = f"{{:.{decimals}f}}"

    styled = (
        df_display.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(align_left_first=align_left_first))
        .format(format_dict)
    )

    _render_styled(styled)


def display_top_results(
    results_path: str,
    top_n: int = 5,
    decimals: int = 3,
    metric: str = "tweedie_deviance",
    penalty_value: float = 1e10,
    column_names: list | tuple | None = None,
    align_left_first: bool = True,
) -> None:
    csv_files = list(results_path.glob("*_results.csv"))
    if not csv_files:
        print(f"No results found in {results_path}.")
        return

    df_raw = pd.read_csv(csv_files[0])

    df_raw["is_failure"] = df_raw[metric] >= 1e9
    df_raw.loc[df_raw["is_failure"], metric] = penalty_value

    ignore_cols = [
        "fold", "train_size", "val_size", "trial", "train_start_idx", 
        "train_end_idx", "val_start_idx", "val_end_idx",
        "val_start_ts", "val_end_ts", "converged", "best_iteration",
        "train_time_sec", "is_failure", "tweedie_p",
        "tweedie_deviance", "mae", "rmse", "mse", "bias", "smape", "mase", "pinball_50"
    ]

    group_cols = [c for c in df_raw.columns if c not in ignore_cols]

    df_agg = df_raw.groupby(group_cols)[metric].agg(["mean", "std"]).reset_index()
    
    df_agg["Métrica (Mean ± SD)"] = df_agg.apply(
        lambda r: f"{r['mean']:.{decimals}f} ± {r['std']:.{decimals}f}", axis=1
    )

    df_top = df_agg.sort_values(by="mean", ascending=True).head(top_n).drop(columns=["mean", "std"])

    if column_names is not None:
        if len(column_names) != len(df_top.columns):
            raise ValueError(
                f"Provided column_names has {len(column_names)} elements, "
                f"but the table has {len(df_top.columns)} columns."
            )
        df_top.columns = column_names

    display_styled_df(df_top, align_left_first=align_left_first, decimals=decimals)


def run_model_diagnostics(
    std_residuals: np.ndarray,
    lags: tuple = (10, 20, 30),
    alpha: float = 0.05,
    p_adjust: str = "holm",
    include_bds: bool = False,
    column_titles: tuple | None = None,
) -> pd.DataFrame:
    rows = []

    lb_res = acorr_ljungbox(std_residuals, lags=list(lags), return_df=True)
    for lag in lags:
        rows.append({
            "Test": f"Ljung-Box (Lag {lag})",
            "Stat": lb_res.loc[lag, "lb_stat"],
            "p_raw": lb_res.loc[lag, "lb_pvalue"],
            "Family": "LB",
        })

    for lag in lags:
        arch_res = het_arch(std_residuals, nlags=lag)
        rows.append({
            "Test": f"ARCH-LM (Lag {lag})",
            "Stat": arch_res[0],
            "p_raw": arch_res[1],
            "Family": "ARCH",
        })

    if include_bds:
        bds_res = bds(std_residuals, max_dim=2)
        rows.append({
            "Test": "BDS (Dim 2)",
            "Stat": bds_res[0],
            "p_raw": bds_res[1],
            "Family": "BDS",
        })

    df = pd.DataFrame(rows)
    df["p_adj"] = np.nan

    for fam in df["Family"].unique():
        mask = df["Family"] == fam
        if fam in ["LB", "ARCH"] and mask.sum() > 1:
            _, p_adj, _, _ = multipletests(df.loc[mask, "p_raw"], method=p_adjust)
            df.loc[mask, "p_adj"] = p_adj.tolist()

    conclusions = []
    for _, row in df.iterrows():
        p = row["p_adj"] if pd.notna(row["p_adj"]) else row["p_raw"]
        test = row["Test"]

        if "Ljung-Box" in test:
            concl = "Sin Autocorrelación" if p > alpha else "Autocorrelación Significativa"
        elif "ARCH-LM" in test:
            concl = "Sin Efectos ARCH" if p > alpha else "Efectos ARCH Significativos"
        elif "BDS" in test:
            concl = "Sin Estructura No Lineal" if p > alpha else "Estructura No Lineal Significativa"
        else:
            concl = ""

        conclusions.append(concl)

    df["Conclusion"] = conclusions
    df_out = df[["Test", "Stat", "p_raw", "p_adj", "Conclusion"]].copy()

    if column_titles is None:
        column_titles = (
            "Prueba",
            "Estadístico",
            "p-valor",
            f"p-ajustado ({p_adjust})",
            f"Conclusión (α={alpha})",
        )

    if len(column_titles) != 5:
        raise ValueError("column_titles must be a tuple of exactly 5 strings.")

    df_out.columns = column_titles

    return df_out


def plot_metric_evolution(
    results_path: str,
    x_col: str = "window",
    metric: str = "tweedie_deviance",
    aggregate_mode: str = "mean",
    title: str = "Evolución de la Métrica por Hiperparámetro",
    xlab: str | None = None,
    ylab: str | None = None,
    ylim: tuple | None = None,
    y_step: float | None = None,
    pads: tuple = (9, 9, 12),
    label_size: float = 12,
    title_size: float = 12,
    tick_size: float = 11,
    figsize: tuple = (10, 6),
    fmt: str = "svg",
    dpi: int = 100,
) -> None:
    csv_files = list(results_path.glob("*_results.csv"))
    if not csv_files:
        print(f"No results found in {results_path}.")
        return

    df_raw = pd.read_csv(csv_files[0])

    if xlab is None:
        xlab = x_col
    if ylab is None:
        ylab = f"{metric.replace('_', ' ').title()} ({'Mean' if aggregate_mode == 'mean' else 'Median'})"

    if aggregate_mode == "mean":
        df_plot = df_raw.groupby(x_col)[metric].agg(center="mean", spread="std").reset_index()
        df_plot["err"] = df_plot["spread"] / np.sqrt(df_raw.groupby(x_col)[metric].count().values)
    elif aggregate_mode == "median":
        df_plot = df_raw.groupby(x_col)[metric].agg(
            center="median",
            spread=lambda x: x.quantile(0.75) - x.quantile(0.25),
        ).reset_index()
        df_plot["err"] = df_plot["spread"]
    else:
        raise ValueError("aggregate_mode must be 'mean' or 'median'.")

    df_plot = df_plot.sort_values(by=x_col)

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        x=df_plot[x_col], y=df_plot["center"], yerr=df_plot["err"],
        color="#0072B2", marker="o", linestyle="-",
        markersize=6, linewidth=1.5, capsize=4, capthick=1.0,
        elinewidth=1.0, ecolor="#0072B2",
    )

    pad_x, pad_y, pad_title = pads

    ax.set_title(title, fontsize=title_size, pad=pad_title, loc="center")
    ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)
    ax.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)

    if ylim is not None:
        ax.set_ylim(ylim)
    if y_step is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))

    _style_axes(ax, tick_size)
    ax.grid(False)

    ax.margins(x=0.05, y=0.10)
    plt.tight_layout()

    _render_figure(fig, fmt, dpi)


def plot_learning_curve(
    evals_result: dict,
    title: str = "Curva de Aprendizaje (LightGBM)",
    xlab: str = "Iteración (Árboles)",
    ylab: str = "Pérdida Tweedie",
    label_train: str = "Entrenamiento",
    label_valid: str = "Validación",
    label_best: str = "Mejor iteración",
    best_iteration: int | None = None,
    figsize: tuple = (12, 6),
    fmt: str = "svg",
    dpi: int = 100,
    title_size: float = 14,
    label_size: float = 14,
    tick_size: float = 12,
    legend_size: float = 14,
    line_width: float = 1.2,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    legend_loc: str = "upper right",
    title_pad: float = 15,
    xlab_pad: float = 12,
    ylab_pad: float = 12,
) -> None:
    if not evals_result or "train" not in evals_result or "valid" not in evals_result:
        raise ValueError("evals_result must contain 'train' and 'valid' keys.")

    metric_name = next(iter(evals_result["train"].keys()))
    train_loss = evals_result["train"][metric_name]
    val_loss = evals_result["valid"][metric_name]

    iterations = range(1, len(train_loss) + 1)
    
    if best_iteration is None:
        best_iteration = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_iteration - 1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(iterations, train_loss, label=label_train, color="#0072B2", linewidth=line_width)
    ax.plot(iterations, val_loss, label=label_valid, color="#D50000", linewidth=line_width)

    ax.axvline(
        x=best_iteration, color="black", linestyle="--", linewidth=line_width * 0.5,
        label=f"{label_best} ({best_iteration})"
    )
    ax.scatter(best_iteration, best_val_loss, color="black", s=50, zorder=5)

    ax.set_title(title, fontsize=title_size, pad=title_pad, loc="center")
    ax.set_xlabel(xlab, fontsize=label_size, labelpad=xlab_pad)
    ax.set_ylabel(ylab, fontsize=label_size, labelpad=ylab_pad)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    _style_axes(ax, tick_size)
    ax.grid(False)

    margins_y = 0.10 if ylim is None else 0
    ax.margins(x=0.05, y=margins_y)
    
    ax.legend(frameon=False, fontsize=legend_size, loc=legend_loc)
    plt.tight_layout()

    _render_figure(fig, fmt, dpi)


def plot_tweedie_fit(
    series: pd.Series,
    p: float | None = None,
    titles: tuple | None = None,
    xlab: str = "Valor",
    ylab_left: str = "Densidad",
    xlab_right: str = "Cuantiles Teóricos (Tweedie)",
    ylab_right: str = "Cuantiles Muestrales",
    legend_loc: str = "upper right",
    legend_bbox: tuple | None = None,   # <-- NUEVO: Para mover la leyenda
    xlim_left: tuple | None = None,
    ylim_left: tuple | None = None,
    xlim_right: tuple | None = None,
    ylim_right: tuple | None = None,
    y_step_left: float | None = None,
    pads: tuple = (9, 9, 12),           # <-- Puedes aumentar esto desde el notebook
    label_size: float = 12,
    title_size: float = 12,
    tick_size: float = 11,
    bins: int = 80,
    n_sim: int = 20000,
    line_width: float = 1.5,
    marker_size: float = 10,
    marker_alpha: float = 0.4,
    ref_line_color: str = "black",
    ref_line_linestyle: str = "--",
    ref_line_width: float = 1.0,
    right_yaxis: bool = True,
    figsize: tuple = (10, 4.5),
    render: bool = True,
    fmt: str = "svg",
    dpi: int = 100,
):
    series = series.dropna()
    
    if p is None:
        p = _estimate_tweedie_p_internal(series)
        
    mu = series.mean()
    var = series.var()
    phi = var / (mu ** p)
    
    alpha = (2 - p) / (p - 1)
    lam = (mu**(2 - p)) / (phi * (2 - p))
    theta = phi * (p - 1) * (mu**(p - 1))
    
    np.random.seed(42)
    N = np.random.poisson(lam, size=n_sim)
    tweedie_sim = np.zeros(n_sim)
    for i in range(n_sim):
        if N[i] > 0:
            tweedie_sim[i] = np.sum(np.random.gamma(shape=alpha, scale=theta, size=N[i]))
            
    p_emp = (np.arange(1, len(series) + 1) - 0.5) / len(series)
    theoretical_q = np.quantile(tweedie_sim, p_emp)
    sample_q = np.sort(series.values)
    
    # Formato de p a 1 decimal
    if titles is None:
        titles = (f"Histograma y Tweedie (p={p:.1f})", f"Q-Q Plot (Tweedie p={p:.1f})")

    pad_x, pad_y, pad_title = pads
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for ax in (ax1, ax2):
        _style_axes(ax, tick_size)

    ax1.hist(
        series, bins=bins, density=True, color="white", edgecolor="black",
        linewidth=0.5, alpha=0.7, label="Datos"
    )
    
    kde = gaussian_kde(tweedie_sim)
    x_vals = np.linspace(0, max(series.max(), tweedie_sim.max()), 1000)
    pdf_vals = kde(x_vals)
    
    # Formato de p a 1 decimal en la leyenda
    ax1.plot(x_vals, pdf_vals, color="black", linewidth=line_width, label=f"Tweedie (p = {p:.1f})")
    
    ax1.set_title(titles[0], fontsize=title_size, pad=pad_title)
    ax1.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)
    ax1.set_ylabel(ylab_left, fontsize=label_size, labelpad=pad_y)
    
    # Aplicar el bbox a la leyenda si se proporciona
    if legend_bbox is not None:
        ax1.legend(frameon=False, fontsize=tick_size, loc=legend_loc, bbox_to_anchor=legend_bbox)
    else:
        ax1.legend(frameon=False, fontsize=tick_size, loc=legend_loc)
    
    if xlim_left is not None:
        ax1.set_xlim(xlim_left)
    if ylim_left is not None:
        ax1.set_ylim(ylim_left)
    if y_step_left is not None:
        ax1.yaxis.set_major_locator(plt.MultipleLocator(y_step_left))

    ax2.scatter(
        theoretical_q, sample_q, color="black", s=marker_size, alpha=marker_alpha
    )
    
    q_min = min(theoretical_q.min(), sample_q.min())
    q_max = max(theoretical_q.max(), sample_q.max())
    ax2.plot(
        [q_min, q_max], [q_min, q_max],
        color=ref_line_color, linestyle=ref_line_linestyle, linewidth=ref_line_width
    )
    
    ax2.set_title(titles[1], fontsize=title_size, pad=pad_title)
    ax2.set_xlabel(xlab_right, fontsize=label_size, labelpad=pad_x)
    ax2.set_ylabel(ylab_right, fontsize=label_size, labelpad=pad_y)

    if xlim_right is not None:
        ax2.set_xlim(xlim_right)
    if ylim_right is not None:
        ax2.set_ylim(ylim_right)
    
    if right_yaxis:
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
    else:
        ax2.tick_params(axis="y", which="both", right=False, labelright=False)
        
    plt.tight_layout()
    if render:
        _render_figure(fig, fmt, dpi)
        
    return fig, (ax1, ax2)


def _estimate_tweedie_p_internal(pasajeros: pd.Series, min_count: int = 30) -> float:
    if not isinstance(pasajeros.index, pd.DatetimeIndex):
        return 1.5
    df = pd.DataFrame({
        "hora": pasajeros.index.hour,
        "dow": pasajeros.index.dayofweek,
        "y": pasajeros.values,
    })
    groups = (
        df.groupby(["hora", "dow"])["y"]
        .agg(["mean", "var", "count"])
        .query("count >= @min_count")
        .dropna()
    )
    groups = groups[(groups["mean"] > 0) & (groups["var"] > 0)]
    if len(groups) < 5:
        return 1.5
    x = np.log(groups["mean"].values)
    y = np.log(groups["var"].values)
    return float(np.clip(np.polyfit(x, y, 1)[0], 1.01, 1.99))


def _style_axes(ax, tick_size: float) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(
        axis="x", which="both", bottom=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )
    ax.tick_params(
        axis="y", which="both", left=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )


def _render_figure(fig, fmt: str = "svg", dpi: int = 100) -> None:
    if fmt == "svg":
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        display(HTML(
            f'<div style="display: flex; justify-content: center; width: 100%;">'
            f'{buf.getvalue().decode("utf-8")}</div>'
        ))
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        display(HTML(
            f'<div style="text-align: center; width: 100%;">'
            f'<img src="data:image/png;base64,{base64.b64encode(buf.getbuffer()).decode("ascii")}"></div>'
        ))


def _format_apa_pvalue(val: float) -> str:
    if pd.isna(val):
        return ""
    if val < 0.001:
        return "< .001"
    formatted = f"{val:.3f}"
    return formatted[1:] if formatted.startswith("0.") else formatted


def _table_style_rules(align_left_first: bool = True) -> list:
    rules = [
        {"selector": "", "props": [
            ("margin-left", "auto"),
            ("margin-right", "auto"),
            ("width", "auto"),
            ("border-collapse", "collapse"),
        ]},
        {"selector": "th", "props": [
            ("text-align", "center"),
            ("background-color", "#f2f2f2"),
            ("color", "black"),
            ("font-weight", "bold"),
            ("border", "1px solid black"),
            ("padding", "10px"),
        ]},
        {"selector": "td", "props": [
            ("text-align", "center"),
            ("border", "1px solid black"),
            ("padding", "10px"),
        ]},
    ]
    if align_left_first:
        rules.extend([
            {"selector": "th:first-child", "props": [("text-align", "left")]},
            {"selector": "td:first-child", "props": [("text-align", "left")]},
        ])
    return rules


def _render_styled(styler) -> None:
    display(
        HTML(
            "<div style='text-align: center; width: 100%;'>"
            + styler.to_html()
            + "</div>"
        )
    )
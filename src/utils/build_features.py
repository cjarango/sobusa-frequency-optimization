"""
build_features
==============

Feature engineering for SOBUSA demand forecasting.

Builds calendar, cyclical, seasonality and lag features on a split
(train or holdout) produced by ``eda_toolkit.split_and_save(df_agre)``.

The output is the model-ready dataset: the ``franja_ts`` DatetimeIndex
(full date + time) is preserved, and the DataFrame carries 24 columns
in a fixed order.

    1.      ``franja``               (HH:MM:SS tag from the consolidated)
    2-23.   the 22 feature columns   (calendar, cyclical, seasonality, lags)
    24.     ``pasajeros``            (response)

All other metadata and intermediate columns from the consolidation
(``anio``, ``mes``, ``dia``, ``minuto``, ``tipo_estacionalidad``,
``nombre_evento``, ``n_viajes``) are dropped.

The response column (``pasajeros``) is **not** transformed by this
module. It is returned exactly as it comes in, so the caller decides
how to aggregate it into the target of each horizon (30, 60, 120 min)
according to the modeling strategy.

Design Decisions
----------------
- Only operationally safe lags are used (32, 33, 65, 455). Shorter lags
  are not reliably available at decision time due to trip duration
  (Q3 ~ 300 min) plus device synchronization delay.
- The timestamp is read from the DataFrame index (``franja_ts``, a
  DatetimeIndex) and preserved as the index of the output.
- ``dia_semana`` is assumed to follow the ISO convention (1 = Monday,
  7 = Sunday), consistent with the SOBUSA consolidated tables.
- Lag columns are stored with the nullable integer dtype (``Int64``)
  so that warmup NaNs are preserved while values keep an integer
  representation.
- NaNs from lags are preserved by default. Pass ``dropna=True`` to
  remove rows with incomplete features.
- Seasonality is one-hot encoded against the five canonical categories,
  so the feature columns exist even when a split does not contain all
  of them.
- Excluded on purpose: ``anio`` (risk of extrapolation with tree
  models), ``n_viajes`` (optimizer decision variable), ``duracion_min``
  (unknown at dispatch time), ``nombre_evento``.

Expected Input
--------------
A split produced by ``split_and_save`` with a ``DatetimeIndex``
(``franja_ts``) and at least the columns::

    franja, mes, dia_semana, tipo_estacionalidad, pasajeros

Any other column from the consolidated table is treated as metadata
and discarded.

Main Functionalities
--------------------
Pipeline
    build_features
Calendar
    add_calendar_features
Cyclical
    add_cyclical_features
Seasonality
    add_seasonality_features
Lags
    add_lag_features
Column Lists
    feature_columns

Dependencies
------------
numpy, pandas
"""
from __future__ import annotations

# Third-party ----
import numpy as np
import pandas as pd

# Metadata ----
__version__ = "1.3.0"

__author__ = (
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Natalia Paola Alvarado Corrales <npalvarado@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)

# Public API declaration ----
__all__ = [
    "LAGS",
    "LAG_PREFIX",
    "SEASON_CATEGORIES",
    "SEASON_COL",
    "TARGET",
    "TIMESTAMP_COL",
    "add_calendar_features",
    "add_cyclical_features",
    "add_lag_features",
    "add_seasonality_features",
    "build_features",
    "feature_columns",
]

# Module-level constants ----
LAGS = (32, 33, 65, 455)
LAG_PREFIX = "lag"
SEASON_COL = "tipo_estacionalidad"
TARGET = "pasajeros"
TIMESTAMP_COL = "franja"
INDEX_NAME = "franja_ts"

SEASON_CATEGORIES = (
    "DIAS_CORRIENTES",
    "FESTIVO_NACIONAL",
    "FIESTA_ATLANTICO",
    "TEMPORADA_NACIONAL",
    "FIESTAS_PATRONALES",
)


# Public API ====

# Calendar ----

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``hora``, ``minuto`` and ``es_fin_de_semana`` from the index.

    Assumes ``dia_semana`` follows the ISO convention: 1 = Monday,
    7 = Sunday. Weekend rows are therefore those with ``dia_semana``
    in ``{6, 7}``.

    Args:
        df (pandas.DataFrame): Split with a ``DatetimeIndex`` named
            ``franja_ts`` and a ``dia_semana`` column.

    Returns:
        pandas.DataFrame: Copy of ``df`` with the three calendar
        columns added as ``int8``.

    Raises:
        TypeError: If ``df`` is not a DataFrame or lacks a DatetimeIndex.
        ValueError: If ``dia_semana`` is not a column of ``df``.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(df, "dia_semana")

    df = df.copy()
    ts = pd.to_datetime(df.index)
    df["hora"] = ts.hour.astype("int8")
    df["minuto"] = ts.minute.astype("int8")
    df["es_fin_de_semana"] = df["dia_semana"].isin([6, 7]).astype("int8")
    return df


# Cyclical ----

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for hour (fundamental + 2nd harmonic),
    day of week, month and semester.

    Args:
        df (pandas.DataFrame): DataFrame with ``hora``, ``minuto``,
            ``dia_semana`` and ``mes`` columns.

    Returns:
        pandas.DataFrame: Copy of ``df`` with ten ``float64`` columns.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If any required column is missing.
    """
    _require_dataframe(df)
    _require_columns(df, "hora", "minuto", "dia_semana", "mes")

    df = df.copy()
    hora_dec = df["hora"] + df["minuto"] / 60

    df["sin_hora"]   = np.sin(2 * np.pi * hora_dec / 24)
    df["cos_hora"]   = np.cos(2 * np.pi * hora_dec / 24)
    df["sin_hora_2"] = np.sin(4 * np.pi * hora_dec / 24)
    df["cos_hora_2"] = np.cos(4 * np.pi * hora_dec / 24)

    df["sin_dow"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    df["sin_mes"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["cos_mes"] = np.cos(2 * np.pi * df["mes"] / 12)

    df["sin_semestre"] = np.sin(2 * np.pi * df["mes"] / 6)
    df["cos_semestre"] = np.cos(2 * np.pi * df["mes"] / 6)
    return df


# Seasonality ----

def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ``tipo_estacionalidad`` against the five canonical
    categories.

    Forcing the categories guarantees that all five ``est_*`` columns
    are present even when a split does not contain every category, so
    train and holdout share an identical feature schema. The categorical
    dtype is applied on the Series (not on a bare ``pd.Categorical``) so
    the DataFrame index is preserved through ``pd.get_dummies``.

    Args:
        df (pandas.DataFrame): DataFrame with ``tipo_estacionalidad``.

    Returns:
        pandas.DataFrame: Copy of ``df`` with five ``int8`` ``est_*``
        columns appended.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``tipo_estacionalidad`` is not in ``df``.
    """
    _require_dataframe(df)
    _require_columns(df, SEASON_COL)

    df = df.copy()
    cat_dtype = pd.CategoricalDtype(categories=list(SEASON_CATEGORIES))
    season = df[SEASON_COL].astype(cat_dtype)
    dummies = pd.get_dummies(season, prefix="est", dtype="int8")
    return pd.concat([df, dummies], axis=1)


# Lags ----

def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = LAGS,
    source_col: str = TARGET,
    prefix: str = LAG_PREFIX,
) -> pd.DataFrame:
    """Add safe lags of ``source_col`` as new columns ``{prefix}_{k}``.

    The DataFrame is assumed to be sorted chronologically; the caller
    is responsible for the ordering. Lag columns are stored with the
    nullable integer dtype (``Int64``) so that warmup NaNs are preserved
    while values keep an integer representation.

    Args:
        df (pandas.DataFrame): DataFrame with the source column.
        lags (tuple of int): Lag offsets, in slots of 15 min. Defaults
            to ``LAGS``.
        source_col (str): Column to lag. Defaults to ``TARGET``
            (``"pasajeros"``).
        prefix (str): Prefix of the lag columns. Defaults to
            ``LAG_PREFIX`` (``"lag"``).

    Returns:
        pandas.DataFrame: Copy of ``df`` with one ``{prefix}_{k}``
        column per requested lag, of dtype ``Int64``.

    Raises:
        TypeError: If ``df`` is not a DataFrame, ``lags`` is not a
            tuple/list of positive integers, or ``prefix`` is not a
            non-empty string.
        ValueError: If ``source_col`` is not in ``df`` or any lag is
            not a positive integer.
    """
    _require_dataframe(df)
    _require_columns(df, source_col)

    if not isinstance(lags, (tuple, list)) or len(lags) == 0:
        raise TypeError("'lags' must be a non-empty tuple or list.")

    if not isinstance(prefix, str) or not prefix.strip():
        raise TypeError("'prefix' must be a non-empty string.")

    for k in lags:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(
                f"Every lag must be a positive integer, got {k!r}."
            )

    df = df.copy()
    for k in lags:
        df[f"{prefix}_{k}"] = (
            df[source_col].shift(k).astype("Int64")
        )
    return df


# Pipeline ----

def build_features(
    df: pd.DataFrame,
    dropna: bool = False,
    source_col: str = TARGET,
    lag_prefix: str = LAG_PREFIX,
) -> pd.DataFrame:
    """Apply the full feature pipeline and return the model-ready dataset.

    The DatetimeIndex (``franja_ts``) is preserved. The output carries
    exactly 24 columns in this order:

    1.      ``franja`` (HH:MM:SS tag from the consolidated)
    2-23.   the 22 feature columns (calendar, cyclical, seasonality,
            lags)
    24.     ``source_col`` (the response)

    All other metadata and intermediate columns from the consolidation
    (``anio``, ``mes``, ``dia``, ``minuto``, ``tipo_estacionalidad``
    raw, ``nombre_evento``, ``n_viajes``) are dropped. The response
    column is left untouched.

    Args:
        df (pandas.DataFrame): Split produced by ``split_and_save``.
            Must have a ``DatetimeIndex`` and at least the columns
            ``franja``, ``mes``, ``dia_semana``, ``tipo_estacionalidad``
            and ``source_col``.
        dropna (bool): If ``True``, drop every row that contains a NaN
            in any feature. If ``False`` (default), NaNs from the lag
            warmup are preserved so the caller can inspect or handle
            them manually.
        source_col (str): Column used as response and as the source of
            the lags. Defaults to ``TARGET`` (``"pasajeros"``).
        lag_prefix (str): Prefix of the lag columns. Defaults to
            ``LAG_PREFIX`` (``"lag"``).

    Returns:
        pandas.DataFrame: Model-ready dataset with the ``franja_ts``
        DatetimeIndex, ``franja`` as the first column, the 22 features
        in canonical order, and ``source_col`` as the last column.

    Raises:
        TypeError: If ``df`` is not a DataFrame, lacks a DatetimeIndex,
            or ``dropna`` is not a boolean.
        ValueError: If any required column is missing.
    """
    _require_dataframe(df)
    _require_datetime_index(df)
    _require_columns(
        df, TIMESTAMP_COL, "dia_semana", "mes", SEASON_COL, source_col,
    )
    _require_bool(dropna, "dropna")

    df = df.sort_index().copy()
    df = (
        df
        .pipe(add_calendar_features)
        .pipe(add_cyclical_features)
        .pipe(add_seasonality_features)
        .pipe(
            add_lag_features,
            source_col=source_col,
            prefix=lag_prefix,
        )
    )

    if dropna:
        cols = feature_columns(df, lag_prefix=lag_prefix)
        df = df.dropna(subset=cols).copy()

    # Keep only the timestamp tag, the 22 features and the response.
    # The DatetimeIndex is preserved as-is.
    keep = (
        [TIMESTAMP_COL]
        + feature_columns(df, lag_prefix=lag_prefix)
        + [source_col]
    )
    return df[keep].copy()


# Column lists ----

def feature_columns(
    df: pd.DataFrame,
    lag_prefix: str = LAG_PREFIX,
) -> list[str]:
    """Return the feature columns entering the model.

    The list excludes the timestamp tag, the response column, and any
    metadata (``anio``, ``mes``, ``dia``, ``minuto``,
    ``tipo_estacionalidad`` raw, ``nombre_evento``).

    Args:
        df (pandas.DataFrame): DataFrame with the feature columns.
        lag_prefix (str): Prefix used for the lag columns. Must match
            the prefix passed to ``build_features``. Defaults to
            ``LAG_PREFIX``.

    Returns:
        list of str: Feature column names in canonical order.

    Raises:
        TypeError: If ``df`` is not a DataFrame or ``lag_prefix`` is
            not a non-empty string.
    """
    _require_dataframe(df)

    if not isinstance(lag_prefix, str) or not lag_prefix.strip():
        raise TypeError("'lag_prefix' must be a non-empty string.")

    calendar = ["hora", "dia_semana", "es_fin_de_semana"]
    cyclical = [
        "sin_hora", "cos_hora", "sin_hora_2", "cos_hora_2",
        "sin_dow", "cos_dow",
        "sin_mes", "cos_mes",
        "sin_semestre", "cos_semestre",
    ]
    season = sorted(c for c in df.columns if c.startswith("est_"))
    lags = [f"{lag_prefix}_{k}" for k in LAGS]
    return calendar + cyclical + season + lags


# Private helpers ====

# Validation ----

def _require_dataframe(df, name: str = "df") -> None:
    """Raise ``TypeError`` unless ``df`` is a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'{name}' must be a pandas DataFrame.")


def _require_datetime_index(df, name: str = "df") -> None:
    """Raise ``TypeError`` unless ``df`` has a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"'{name}' must have a pandas DatetimeIndex.")


def _require_columns(df, *columns: str) -> None:
    """Raise unless every name in ``columns`` is a string column of ``df``."""
    for c in columns:
        if not isinstance(c, str):
            raise TypeError(
                f"Column names must be strings, got '{type(c).__name__}'."
            )
        if c not in df.columns:
            raise ValueError(f"Column '{c}' is not in 'df'.")


def _require_bool(value, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"'{name}' must be a boolean.")
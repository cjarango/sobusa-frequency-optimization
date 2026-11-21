from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from utils.build_features import build_features, feature_columns

TZ_BOGOTA = ZoneInfo("America/Bogota")


def get_inference_row(
    engine: Engine,
    route: str,
    target_dt: datetime,
    tipo_estacionalidad: str = "DIAS_CORRIENTES",
) -> pd.DataFrame:
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=TZ_BOGOTA)
    else:
        target_dt = target_dt.astimezone(TZ_BOGOTA)

    query = text("""
        SELECT
            franja_ts AT TIME ZONE 'America/Bogota' AS franja_ts,
            pasajeros,
            tipo_estacionalidad
        FROM historico
        WHERE route = :route
          AND franja_ts < :target_dt
        ORDER BY franja_ts DESC
        LIMIT 500
    """)

    with engine.connect() as con:
        rows = con.execute(query, {"route": route, "target_dt": target_dt}).fetchall()

    if len(rows) < 456:
        raise ValueError(
            f"Historial insuficiente para '{route}' antes de {target_dt}: "
            f"se requieren al menos 456 slots, disponibles {len(rows)}."
        )

    df_hist = pd.DataFrame(rows, columns=["franja_ts", "pasajeros", "tipo_estacionalidad"])
    df_hist["franja_ts"] = pd.to_datetime(df_hist["franja_ts"])
    df_hist = df_hist.sort_values("franja_ts").set_index("franja_ts")

    target_naive = pd.Timestamp(target_dt.replace(tzinfo=None))
    row_target = pd.DataFrame(
        {"pasajeros": [0.0], "tipo_estacionalidad": [tipo_estacionalidad]},
        index=pd.DatetimeIndex([target_naive], name="franja_ts"),
    )

    df_full = pd.concat([df_hist, row_target])
    df_full["franja"] = df_full.index.strftime("%H:%M:%S")
    df_full["mes"] = df_full.index.month
    df_full["dia_semana"] = df_full.index.isocalendar().day.astype("int64")

    df_feat = build_features(df_full, dropna=False)
    feat_cols = feature_columns(df_feat)
    return df_feat.loc[[target_naive], feat_cols]
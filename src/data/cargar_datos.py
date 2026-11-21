from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
ENGINE = create_engine("postgresql+psycopg://postgres:postgres123@localhost:5432/frecuencias")

ARCHIVOS = {
    "KRA 54": {
        "train": DATA_DIR / "train_k54.parquet",
        "test": DATA_DIR / "holdout_k54.parquet",
    },
    "Silencio": {
        "train": DATA_DIR / "train_silencio.parquet",
        "test": DATA_DIR / "holdout_silencio.parquet",
    },
    "Calle 72": {
        "train": DATA_DIR / "train_c72.parquet",
        "test": DATA_DIR / "holdout_c72.parquet",
    },
}

COLUMNAS_DB = [
    "route",
    "franja_ts",
    "pasajeros",
    "n_viajes",
    "tipo_estacionalidad",
    "nombre_evento",
]


def preparar_df(path: Path, route: str) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index()
    ts = pd.to_datetime(df["franja_ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/Bogota")
    else:
        ts = ts.dt.tz_convert("America/Bogota")

    df["franja_ts"] = ts
    df["route"] = route
    return df[COLUMNAS_DB]


def main() -> None:
    with ENGINE.begin() as con:
        con.execute(text("TRUNCATE TABLE historico, test_set;"))

    for route, rutas_archivos in ARCHIVOS.items():
        df_train = preparar_df(rutas_archivos["train"], route)
        df_test = preparar_df(rutas_archivos["test"], route)

        df_train.to_sql("historico", ENGINE, if_exists="append", index=False, method="multi", chunksize=5000)
        df_test.to_sql("test_set", ENGINE, if_exists="append", index=False, method="multi", chunksize=5000)

        print(f"[{route}] historico: {len(df_train):,} filas | test_set: {len(df_test):,} filas")


if __name__ == "__main__":
    main()
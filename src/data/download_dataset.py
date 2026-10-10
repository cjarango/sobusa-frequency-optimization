"""
Descarga datasets desde PostgreSQL y los guarda en Parquet en data/raw.

Tablas disponibles:
    timbradas        -> timbradas_anio (agregado por patrón)
    progvehiculos    -> progvehiculos (despachos individuales, sin agregar)
    both             -> ambas (default)

Rutas disponibles:
    silencio  -> id_ruta = 2
    k54       -> id_ruta = 16
    c72       -> id_ruta = 21
    todas     -> las 3 rutas

Uso:
    python src/data/download_dataset.py --ruta k54
    python src/data/download_dataset.py --ruta k54 --tablas progvehiculos
    python src/data/download_dataset.py --ruta todas --tablas both
"""
import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

RUTAS = {
    "silencio": 2,
    "k54":      16,
    "c72":      21,
}

TABLAS_VALIDAS = {"timbradas", "progvehiculos", "both"}

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def log(msg: str, icono: str = "·") -> None:
    print(f"[{icono}] {msg}")


def parsear_ruta(arg: str) -> tuple[str, list[int]]:
    arg = arg.lower().strip()
    if arg == "todas":
        return "todas", list(RUTAS.values())
    if arg in RUTAS:
        return arg, [RUTAS[arg]]
    raise ValueError(
        f"Ruta '{arg}' no reconocida. "
        f"Opciones válidas: {list(RUTAS.keys())} o 'todas'."
    )


def parsear_tablas(arg: str) -> list[str]:
    arg = arg.lower().strip()
    if arg not in TABLAS_VALIDAS:
        raise ValueError(
            f"Tabla '{arg}' no reconocida. "
            f"Opciones válidas: {sorted(TABLAS_VALIDAS)}."
        )
    if arg == "both":
        return ["timbradas", "progvehiculos"]
    return [arg]


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga datasets desde PostgreSQL a Parquet."
    )
    parser.add_argument(
        "--ruta", "-r",
        required=True,
        help="Ruta a descargar: silencio, k54, c72 o todas.",
    )
    parser.add_argument(
        "--tablas", "-t",
        default="both",
        help="Tablas a descargar: timbradas, progvehiculos o both (default).",
    )
    parser.add_argument("--host",     default=os.getenv("DB_HOST"))
    parser.add_argument("--port",     default=os.getenv("DB_PORT", "5432"))
    parser.add_argument("--db",       default=os.getenv("DB_NAME", "dataset"))
    parser.add_argument("--user",     default=os.getenv("DB_USER"))
    parser.add_argument("--password", default=os.getenv("DB_PASSWORD"))
    return parser.parse_args()


def validar_credenciales(args: argparse.Namespace) -> None:
    faltantes = [
        nombre for nombre, valor in (
            ("--host",     args.host),
            ("--user",     args.user),
            ("--password", args.password),
        )
        if not valor
    ]
    if faltantes:
        raise EnvironmentError(
            f"Faltan credenciales: {faltantes}. "
            "Pásalas por CLI o defínelas en el .env."
        )


def crear_conexion(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    log("Instalando y cargando extensión postgres...", "ok")
    con.execute("INSTALL postgres;")
    con.execute("LOAD postgres;")
    con.execute(f"""
        ATTACH 'host={args.host} port={args.port} dbname={args.db}
                user={args.user} password={args.password}'
        AS mi_pg (TYPE POSTGRES, READ_ONLY);
    """)
    return con


TIMBRADAS_QUERY = """
SELECT
    t.id_ruta,
    ru.nombre                                          AS nombre_ruta,
    t.ano,
    t.id_dia_semana,
    t.id_estacionalidad_anio,
    t.hora_inicio,
    t.hora_fin,
    EXTRACT(HOUR   FROM t.hora_inicio)::int            AS hora_num,
    EXTRACT(MINUTE FROM t.hora_inicio)::int            AS minuto_num,
    e.estacionalidad                                   AS tipo_estacionalidad,
    e.nombre                                           AS nombre_evento,
    ea.fecha_inicio,
    ea.fecha_fin,
    (ea.fecha_fin - ea.fecha_inicio + 1)               AS duracion_evento_dias,
    t.subidas,
    t.bajadas,
    t.nro_viajes,
    t.promedio_subidas,
    t.promedio_bajadas
FROM mi_pg.public.timbradas_anio t
JOIN mi_pg.public.rutas ru
     ON ru.idruta = t.id_ruta
LEFT JOIN mi_pg.public.estacionalidades_anio ea
     ON ea.id_estacionalidad_anio = t.id_estacionalidad_anio
LEFT JOIN mi_pg.public.estacionalidades e
     ON e.id_estacionalidad = ea.id_estacionalidad
WHERE t.id_ruta IN ({rutas_placeholder})
  AND t.proceso = 'sobusa'
  AND t.ano BETWEEN 2023 AND 2026
ORDER BY t.ano, t.id_ruta, t.id_dia_semana,
         t.id_estacionalidad_anio, t.hora_inicio;
"""


PROGVEHICULOS_QUERY = """
SELECT
    p.*,
    ru.nombre AS nombre_ruta
FROM mi_pg.public.progvehiculos p
JOIN mi_pg.public.rutas ru
     ON ru.idruta = p.idruta
WHERE p.idruta IN ({rutas_placeholder})
  AND p.proceso = 'sobusa'
  AND p.fechainicio >= '2023-01-01'
  AND p.fechainicio <  '2027-01-01'
ORDER BY p.fechainicio;
"""


def _normalizar_timbradas(df: pd.DataFrame) -> pd.DataFrame:
    df["hora_inicio"] = pd.to_datetime(
        df["hora_inicio"], format="%H:%M:%S"
    ).dt.time
    df["hora_fin"] = pd.to_datetime(
        df["hora_fin"], format="%H:%M:%S"
    ).dt.time

    df["nombre_ruta"]         = df["nombre_ruta"].astype("category")
    df["tipo_estacionalidad"] = df["tipo_estacionalidad"].astype("category")
    df["nombre_evento"]       = df["nombre_evento"].astype("category")

    df["ano"]                  = df["ano"].astype("int16")
    df["id_dia_semana"]        = df["id_dia_semana"].astype("int8")
    df["hora_num"]             = df["hora_num"].astype("int8")
    df["minuto_num"]           = df["minuto_num"].astype("int8")
    df["duracion_evento_dias"] = df["duracion_evento_dias"].astype("int16")
    df["id_ruta"]              = df["id_ruta"].astype("int16")

    df.columns = [c.lower() for c in df.columns]
    return df


def _normalizar_progvehiculos(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("fechasalida", "fecharegistro", "fechainicio",
                "fechafin", "fechaliq"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "id_ruta" in df.columns:
        df = df.rename(columns={"idruta": "id_ruta"}) \
            if "idruta" in df.columns else df

    if "idruta" in df.columns:
        df["idruta"] = df["idruta"].astype("int16")

    for col in ("finalizado", "cancelada", "liquidado", "activa",
                "excedelazosalida", "excedeplazosalida", "exportado"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in ("numeropasajeros", "numerobajadas", "numerobloqueos",
                "puntoini", "puntofin", "ordenpunto", "ordenpuntofin"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    for col in ("acumulada_subida", "acumulada_bajada", "acumulada_bloqueo"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df.columns = [c.lower() for c in df.columns]
    return df


def descargar(con, tabla, nombre, ids) -> tuple[pd.DataFrame, float]:
    ids_str = ", ".join(str(i) for i in ids)

    if tabla == "timbradas":
        query = TIMBRADAS_QUERY.format(rutas_placeholder=ids_str)
        df = con.execute(query).df()
        df = _normalizar_timbradas(df)
        out = DATA_RAW / f"timbradas_anio_{nombre}.parquet"
    else:
        query = PROGVEHICULOS_QUERY.format(rutas_placeholder=ids_str)
        df = con.execute(query).df()
        df = _normalizar_progvehiculos(df)
        out = DATA_RAW / f"progvehiculos_{nombre}.parquet"

    if df.empty:
        raise ValueError(
            f"La consulta de '{tabla}' no devolvió datos "
            f"para la ruta '{nombre}'."
        )

    df.to_parquet(out, index=False, compression="snappy")
    if not out.exists() or out.stat().st_size == 0:
        raise IOError(f"El archivo {out} no se generó correctamente.")

    df_check = pd.read_parquet(out)
    if len(df_check) != len(df):
        raise IOError(
            f"Discrepancia en {out.name}: "
            f"{len(df)} filas vs {len(df_check)} leídas."
        )

    size_kb = out.stat().st_size / 1024
    return df, size_kb


def main(args: argparse.Namespace) -> None:
    t_inicio = time.perf_counter()

    log("Validando argumentos...", "1/4")
    nombre, ids = parsear_ruta(args.ruta)
    tablas = parsear_tablas(args.tablas)
    log(f"Ruta: {nombre} | IDs: {ids} | Tablas: {tablas}", "ok")

    log("Validando credenciales...", "2/4")
    validar_credenciales(args)
    log(f"Host: {args.host} | DB: {args.db} | Usuario: {args.user}", "ok")

    log("Conectando a PostgreSQL vía DuckDB...", "3/4")
    try:
        con = crear_conexion(args)
    except duckdb.Error as e:
        raise ConnectionError(
            f"No se pudo conectar a la base de datos. Detalle: {e}"
        )
    log("Conexión establecida.", "ok")

    log("Descargando tablas...", "4/4")
    resumen = []
    for tabla in tablas:
        df, size_kb = descargar(con, tabla, nombre, ids)
        resumen.append((tabla, out_name := (
            f"timbradas_anio_{nombre}.parquet" if tabla == "timbradas"
            else f"progvehiculos_{nombre}.parquet"
        ), len(df), size_kb))
        log(f"{tabla:14s} → {len(df):>8,} filas | "
            f"{size_kb:>8.1f} KB | {out_name}", "ok")

    con.close()

    t_fin = time.perf_counter()
    duracion = t_fin - t_inicio

    print()
    print("-" * 72)
    print(f"  Descarga completada en {duracion:.2f} segundos")
    print(f"  Ruta: {nombre}")
    print("-" * 72)
    for tabla, archivo, filas, kb in resumen:
        print(f"  {tabla:14s} {filas:>10,} filas  {kb:>8.1f} KB  {archivo}")
    print("-" * 72)


if __name__ == "__main__":
    try:
        main(parsear_argumentos())
    except ValueError as e:
        print(f"\n[error] {e}"); sys.exit(2)
    except EnvironmentError as e:
        print(f"\n[error] {e}"); sys.exit(3)
    except ConnectionError as e:
        print(f"\n[error] {e}"); sys.exit(4)
    except IOError as e:
        print(f"\n[error] {e}"); sys.exit(5)
    except KeyboardInterrupt:
        print("\n[error] Descarga cancelada por el usuario."); sys.exit(130)
    except Exception as e:
        print(f"\n[error] Error inesperado: {e}"); sys.exit(99)
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_route_params(engine: Engine, route: str) -> dict[str, Any]:
    query = text("""
        SELECT
            idruta,
            capacidad_nominal,
            ciclo_min,
            flota_disponible,
            headway_min,
            headway_max,
            ocupacion_objetivo
        FROM rutas
        WHERE idruta = :route
    """)
    with engine.connect() as con:
        row = con.execute(query, {"route": route}).fetchone()
    if row is None:
        raise KeyError(f"Ruta '{route}' no encontrada en la tabla 'rutas'.")
    return dict(row._mapping)


def calcular_despacho(
    demanda_horizonte: float,
    horizon_min: int,
    capacidad_bus: int,
    tiempo_ciclo_min: float,
    flota_disponible: int,
    factor_ocupacion: float = 0.75,
    headway_min: float = 5.0,
    headway_max: float = 30.0,
) -> dict[str, Any]:
    """
    Convierte demanda acumulada en el horizonte H a tasa horaria (pax/h),
    calcula la frecuencia óptima f = D_h / (C * rho) y el headway h = 60 / f.
    """
    demanda_hora = demanda_horizonte * (60.0 / horizon_min)
    capacidad_efectiva = capacidad_bus * factor_ocupacion

    freq_por_demanda = demanda_hora / capacidad_efectiva if capacidad_efectiva > 0 else 0.0
    freq_min_operativa = 60.0 / headway_max
    freq_max_operativa = 60.0 / headway_min

    freq_objetivo = min(max(freq_por_demanda, freq_min_operativa), freq_max_operativa)
    headway_optimo = round(60.0 / freq_objetivo, 2)

    buses_requeridos = math.ceil(freq_objetivo * (tiempo_ciclo_min / 60.0))
    deficit_flota = max(0, buses_requeridos - flota_disponible)

    buses_asignados = min(buses_requeridos, flota_disponible)
    freq_efectiva = buses_asignados / (tiempo_ciclo_min / 60.0) if tiempo_ciclo_min > 0 else 0.0
    headway_efectivo = round(60.0 / freq_efectiva, 2) if freq_efectiva > 0 else headway_max
    capacidad_ofertada_hora = freq_efectiva * capacidad_bus
    ocupacion_estimada = (
        round(demanda_hora / capacidad_ofertada_hora, 3) if capacidad_ofertada_hora > 0 else 0.0
    )

    return {
        "demanda_horizonte_pax": round(demanda_horizonte, 2),
        "demanda_horaria_pax_h": round(demanda_hora, 2),
        "frecuencia_objetivo_buses_h": round(freq_objetivo, 2),
        "headway_optimo_min": headway_optimo,
        "buses_requeridos": buses_requeridos,
        "buses_asignados": buses_asignados,
        "deficit_flota": deficit_flota,
        "headway_efectivo_min": headway_efectivo,
        "ocupacion_estimada": ocupacion_estimada,
        "saturacion_alerta": deficit_flota > 0 or ocupacion_estimada > 1.0,
    }


def optimize_route_dispatch(
    engine: Engine,
    prediction_result: dict[str, Any],
    factor_ocupacion_override: float | None = None,
) -> dict[str, Any]:
    route = prediction_result["route"]
    horizon_min = prediction_result["horizon_min"]
    params = get_route_params(engine, route)

    rho = factor_ocupacion_override or float(params.get("ocupacion_objetivo") or 0.75)
    cap = int(params["capacidad_nominal"])
    ciclo = float(params["ciclo_min"])
    flota = int(params["flota_disponible"])
    hw_min = float(params.get("headway_min") or 5.0)
    hw_max = float(params.get("headway_max") or 30.0)

    plan_nominal = calcular_despacho(
        demanda_horizonte=prediction_result["predicted_passengers"],
        horizon_min=horizon_min,
        capacidad_bus=cap,
        tiempo_ciclo_min=ciclo,
        flota_disponible=flota,
        factor_ocupacion=rho,
        headway_min=hw_min,
        headway_max=hw_max,
    )

    plan_robusto_95 = calcular_despacho(
        demanda_horizonte=prediction_result["conformal_interval"]["upper"],
        horizon_min=horizon_min,
        capacidad_bus=cap,
        tiempo_ciclo_min=ciclo,
        flota_disponible=flota,
        factor_ocupacion=rho,
        headway_min=hw_min,
        headway_max=hw_max,
    )

    return {
        "route": route,
        "horizon_min": horizon_min,
        "target_datetime": prediction_result["target_datetime"],
        "stale": prediction_result["stale"],
        "warning": prediction_result["warning"],
        "parametros_ruta": {
            "capacidad_nominal": cap,
            "ciclo_min": ciclo,
            "flota_disponible": flota,
            "ocupacion_objetivo": rho,
        },
        "prediccion": {
            "puntual": prediction_result["predicted_passengers"],
            "ic_95_inferior": prediction_result["conformal_interval"]["lower"],
            "ic_95_superior": prediction_result["conformal_interval"]["upper"],
        },
        "plan_nominal": plan_nominal,
        "plan_robusto_p95": plan_robusto_95,
    }
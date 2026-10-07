# SOBUSA Data Challenge 2026 — Predicción y Optimización de Frecuencias

Prototipo de un sistema inteligente para predecir la demanda de pasajeros y recomendar frecuencias de despacho en las rutas de SOBUSA (Sistema de Transporte Masivo de Barranquilla), desarrollado en el marco del Data Challenge 2026 propuesto por Extreme Technologies en alianza con la Universidad del Norte.

## Contexto

SOBUSA opera un sistema de transporte masivo cuyas decisiones de despacho dependen hoy, en gran medida, de la experiencia operativa y de reglas fijas. La información sobre subidas, bajadas, programaciones, segmentos de ruta y estacionalidades ya existe en PostgreSQL, alimentada por triggers y procesos internos de Extreme Technologies.

El reto consiste en transformar esa información en un sistema de decisión anticipada: predecir la demanda por ruta y franja horaria, traducirla en una frecuencia de despacho recomendada (headway, buses por hora, vehículos requeridos) y generar alertas operativas explicables cuando la operación real se desvía de lo esperado.

El sistema no debe limitarse a un modelo entrenado en un notebook. Se requiere un producto mínimo viable con API REST, persistencia trazable en PostgreSQL y ejecución automática, sin modificar las tablas, triggers ni scripts existentes.

Pregunta que resuelve el sistema:

> Con base en los datos históricos y recientes, ¿cada cuánto despachar buses para optimizar la flota y el recaudo? ¿Cuántos pasajeros y vehículos se esperan y qué situaciones requieren atención inmediata?


## Objetivos

### Objetivo general

Diseñar e implementar un prototipo basado en inteligencia artificial y analítica de datos que consuma las tablas existentes en PostgreSQL, prediga la demanda por ruta y franja horaria, recomiende frecuencias de despacho, detecte desviaciones y genere alertas operativas explicables, usando Python como lenguaje principal.

### Objetivos específicos

- Predecir la demanda de pasajeros por ruta y franja horaria para horizontes de 30, 60 y 120 minutos.
- Calcular headway, buses por hora y vehículos requeridos bajo restricciones operativas de capacidad, flota, headway mínimo y nivel de servicio.
- Detectar sobreocupación, baja utilización, caídas anómalas de demanda, desviaciones del pronóstico y datos incompletos.
- Exponer las variables que más influyen en cada predicción mediante importancia global y local.
- Publicar predicciones, recomendaciones y alertas mediante una API REST versionada.
- Registrar cada predicción, recomendación y alerta con versión de modelo, corte de datos y métricas asociadas.

## Alcance

| Módulo | Responsabilidad |
|---|---|
| Acceso y calidad | Leer tablas existentes; validar completitud, validez, unicidad y consistencia. |
| Predicción de demanda | Estimar subidas por ruta y franja con intervalo de confianza. |
| Optimización | Traducir la demanda en headway, buses por hora y vehículos requeridos. |
| Anomalías y alertas | Evaluar reglas operativas, priorizar y registrar eventos. |
| API y tablero | Exponer indicadores, predicciones, recomendaciones y alertas. |

Fuera del alcance: control directo de despachos, modificación de triggers y tablas existentes, despliegue productivo definitivo.


## Enfoque de modelado

El problema se formula como un pronóstico supervisado de demanda agregada por ruta y ventana temporal.

- Unidad de observación: (id_ruta, fecha, franja_horaria).
- Variable objetivo principal: subidas_total, correspondiente a la suma de subidas en la franja (demanda de abordajes).
- Variables objetivo secundarias: bajadas_total y subidas_por_viaje.
- Variables predictoras: hora, minuto, franja, día de la semana, mes, estacionalidad, indicadores cíclicos, lags temporales y variables de oferta (n_viajes).
- Modelo candidato: Gradient Boosting con LightGBM, exportable a ONNX.
- Línea base: promedio por día-hora y promedio de la última semana.
- Validación: división cronológica (entrenamiento, validación y prueba) con métricas MAE, RMSE y WAPE, desagregadas por hora pico y hora valle.

### Optimización de frecuencia

La recomendación se construye sobre la predicción de demanda como un problema de programación lineal entera mixta resuelto con PuLP, con la siguiente función objetivo de referencia:

$$
\min \; \alpha \cdot \text{costo\_operativo} + \beta \cdot \text{tiempo\_espera} + \gamma \cdot \text{penalización\_sobreocupación} + \delta \cdot \text{desviación\_frecuencia}
$$

Sujeto a:

- capacidad disponible
- flota disponible
- headway mínimo y máximo
- duración de ciclo
- nivel de servicio

Valores iniciales de referencia:

| Parámetro | Valor inicial |
|---|---|
| Ocupación objetivo | 65 % – 85 % |
| Ocupación crítica | mayor o igual a 90 % |
| Baja utilización | menor a 30 % |
| Espera máxima | 5 minutos |
| Error de pronóstico tolerable | mayor o igual a 15 % |

---

## Estructura del proyecto

```bash
.
├── .github/
│   └── workflows/
│       └── ci.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_baseline_and_features.ipynb
│   └── 03_modeling_lightgbm.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── optimization/
│   ├── alerts/
│   └── utils/
├── tests/
├── api/
│   └── v1/
├── alembic/
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```


## Instalación y configuración

**1. Clonar el repositorio:**

```bash
git clone https://github.com/cjarango/sobusa-frequency-optimization.git
cd sobusa-data-challenge-2026
```

**2. Recrear el entorno de Conda:**

```bash
conda env create -f environment.yml
```

**3. Activar el entorno:**

```bash
conda activate sobusa_data_challenge_2026
```

## API REST

Endpoints mínimos:

| Método | Ruta | Finalidad |
|---|---|---|
| POST | /api/v1/frequency/executions | Ejecutar lectura incremental, predicción y optimización. |
| GET | /api/v1/frequency/executions/{id} | Consultar corte, estado, métricas y errores. |
| POST | /api/v1/predictions/run | Solicitar inferencia para ruta y horizonte. |
| GET | /api/v1/predictions | Consultar pronósticos por filtros. |
| POST | /api/v1/recommendations/calculate | Calcular frecuencia y flota. |
| GET | /api/v1/recommendations | Consultar recomendaciones. |
| GET | /api/v1/alerts | Listar alertas por estado, nivel, ruta y fecha. |
| PATCH | /api/v1/alerts/{id}/status | Reconocer, cerrar o descartar una alerta. |
| GET | /api/v1/models | Consultar registro de modelos. |
| POST | /api/v1/models/{id}/activate | Activar versión aprobada con auditoría. |

Ejemplo de respuesta de predicción:

```bash
{
  "predictionId": "8fd7...",
  "routeId": "RUTA-16",
  "generatedAt": "2026-07-22T16:00:00-05:00",
  "horizonMinutes": 60,
  "predictedPassengers": 185,
  "confidenceInterval": { "lower": 164, "upper": 207 },
  "modelVersion": "demand-gb-1.0.0",
  "recommendation": {
    "headwayMinutes": 10,
    "busesPerHour": 6,
    "vehiclesRequired": 8
  },
  "explanation": ["hora_pico", "demanda_ultimos_7_dias", "headway_real"]
}
```

## Modelo de datos

Las tablas nuevas se crean en un esquema aislado denominado ai_frecuencias, con el fin de no interferir con las tablas operativas existentes:

- model_registry: registro versionado de modelos.
- prediction: pronósticos con intervalo de confianza y metadatos.
- recommendation: headway, buses por hora y vehículos requeridos.
- alert: alertas con severidad, causa, evidencia y estado.
- alert_history: bitácora de cambios de estado.
- execution_log: registro de cada ejecución del pipeline.

Reglas de no interferencia:

- Usuario de solo lectura sobre tablas operativas.
- Usuario separado con escritura únicamente en ai_frecuencias.
- Transacciones de consulta en modo READ ONLY.
- Sin operaciones DDL sobre el esquema operacional.
- Sin modificación de triggers ni scripts existentes.


## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Gestión de entorno | Conda (canal conda-forge) |
| Análisis de datos | pandas, polars, DuckDB, PyArrow, SciPy |
| Machine Learning | scikit-learn, LightGBM, XGBoost, SHAP, Optuna |
| Optimización | PuLP |
| Deep Learning | PyTorch (CPU) |
| Exportación de modelos | ONNX, ONNX Runtime, skl2onnx |
| API | FastAPI, Uvicorn, Pydantic |
| Base de datos | PostgreSQL 14 con PostGIS |
| ORM y migraciones | SQLAlchemy, Alembic |
| Pruebas | pytest, pytest-cov |
| Calidad de código | Ruff, Black, SonarQube |
| Observabilidad | structlog, Prometheus, OpenTelemetry |

## Plan de trabajo

| Fase | Actividades | Resultado |
|---|---|---|
| 1. Descubrimiento | Entendimiento operativo, diccionario y perfilamiento. | Problema y datos validados. |
| 2. Línea base | Modelo simple, reglas iniciales y API mínima. | Referencia medible. |
| 3. Modelado | Features, modelos, validación temporal y registro. | Modelo candidato. |
| 4. Producto | Persistencia, inferencia, optimización, alertas y tablero. | MVP integrado. |
| 5. Integración | Lectura incremental, pruebas SQL y validación de triggers. | Acceso estable y no invasivo. |
| 6. Endurecimiento | Pruebas, observabilidad y contenedores. | Versión demostrable. |
| 7. Cierre | Prueba reservada, documentación y presentación. | Entrega final. |

## Equipo

| Integrante | Rol |
|---|---|
| Natalia Paola Alvarado Corrales | Científica de Datos |
| Miguel Ángel Pérez Vargas | Científico de Datos |
| Juan Camilo Mendoza Arango | Científico de Datos |

Proyecto desarrollado en el marco del Data Challenge 2026 — Universidad del Norte.


## Referencias

- Especificación técnica del Reto de Frecuencias, Data Challenge 2026, Extreme Technologies.

## Licencia

Distribuido bajo la Licencia MIT. Consultar el archivo LICENSE para más detalles.

## Contacto

Para dudas técnicas o coordinación con Extreme Technologies, contactar a través del canal oficial definido para el Data Challenge 2026.
# SOBUSA Data Challenge 2026 — Predicción y Optimización de Frecuencias

Prototipo de un sistema inteligente contenerizado para predecir la demanda de pasajeros en intervalos de 15 minutos y recomendar frecuencias de despacho (*headways*) y flota requerida en las rutas de SOBUSA (Barranquilla, Colombia), desarrollado en el marco del Data Challenge 2026 propuesto por Extreme Technologies en alianza con la Universidad del Norte.

## Contexto

SOBUSA opera un sistema de transporte urbano cuyas decisiones de despacho dependen en gran medida de la experiencia operativa y de reglas fijas. La información sobre abordajes en intervalos de 15 minutos, ventanas de evaluación y parámetros operativos de ruta reside en PostgreSQL.

El reto consiste en transformar esa información en un sistema de decisión anticipada: predecir la demanda por ruta y horizonte temporal (30, 60 y 120 minutos) con bandas de incertidumbre conformal, traducirla en una frecuencia de despacho recomendada (*headway*, buses por hora, vehículos requeridos) y generar diagnósticos operativos explicables frente a la flota disponible, sin modificar el esquema operacional existente.

Pregunta que resuelve el sistema:

> Con base en los datos históricos y recientes, ¿cada cuánto despachar buses para optimizar la flota y el nivel de servicio? ¿Cuántos pasajeros y vehículos se esperan y qué situaciones requieren atención inmediata?

---

## Objetivos

### Objetivo general

Diseñar e implementar un prototipo basado en inteligencia artificial y analítica de datos que consuma las tablas existentes en PostgreSQL, prediga la demanda por ruta y horizonte temporal, recomiende frecuencias de despacho, detecte déficit o saturación de capacidad y exponga los resultados mediante una API REST y un tablero web contenerizado.

### Objetivos específicos

- Predecir la demanda de pasajeros por ruta y franja de 15 minutos para horizontes de 30, 60 y 120 minutos mediante modelos LightGBM Tweedie con predicción conformal.
- Calcular *headway*, buses por hora y vehículos requeridos bajo restricciones operativas de capacidad nominal, ocupación objetivo, duración de ciclo y flota disponible.
- Diagnosticar automáticamente estados de operación normal, alerta de capacidad o déficit de flota según la demanda esperada y su banda superior de seguridad.
- Publicar predicciones, recomendaciones y diagnósticos operativos mediante una API REST documentada (OpenAPI/Swagger) y una interfaz web interactiva en Jinja2.
- Garantizar la reproducibilidad completa del entorno mediante Docker Compose con autoinicialización de PostgreSQL.

---

## Alcance

| Módulo | Responsabilidad |
|---|---|
| Acceso a datos | Consultar tablas operativas e históricas en PostgreSQL mediante SQLAlchemy en modo lectura. |
| Predicción de demanda | Estimar abordajes por ruta y horizonte con bandas superiores de seguridad conformales. |
| Optimización de despacho | Traducir la demanda pronosticada en *headway*, buses por hora y flota requerida. |
| Diagnóstico operativo | Evaluar capacidad efectiva frente a flota disponible y clasificar alertas operativas. |
| API y tablero web | Exponer endpoints REST y tablero interactivo para simulación por corte temporal. |

Fuera del alcance: control directo de despachos en tiempo real, modificación de triggers o tablas operativas existentes, despliegue productivo definitivo.

---

## Enfoque de modelado

El problema se formula como un pronóstico supervisado de demanda agregada por ruta y ventana temporal de 15 minutos.

- **Unidad de observación:** `(ruta, ventana de 15 minutos)`.
- **Variable objetivo principal:** `pasajeros` (demanda de abordajes en la ventana de 15 minutos, no negativa y con sobredispersión).
- **Horizontes de predicción:** `30 min` ($h = 2$), `60 min` ($h = 4$) y `120 min` ($h = 8$).
- **Variables predictoras:** rezagos autorregresivos temporales, estadísticas móviles, codificación cíclica de hora y día de la semana, indicadores de franja pico/valle y atributos de ruta.
- **Modelo seleccionado:** Gradient Boosting con función de pérdida **Tweedie** (`LightGBMTweedieWrapper`), calibrado mediante validación cruzada temporal con ventana deslizante (*sliding-window*) y optimización bayesiana de hiperparámetros con Optuna.
- **Cuantificación de incertidumbre:** intervalos de predicción conformales calibrados sobre conjunto independiente (`calibration`), reportando la estimación puntual (`y_pred`) y el límite superior de seguridad (`y_upper`).

### Optimización de frecuencia y flota

A partir de la demanda horaria proyectada $D_h$ y los parámetros físicos almacenados por ruta, la recomendación de despacho se formula bajo las restricciones operativas del corredor:

$$
\begin{aligned}
C_{ef} &= C_{nom} \times \rho \\
B_h &= \left\lceil \frac{D_h}{C_{ef}} \right\rceil \\
H &= \frac{60}{B_h} \\
F_{req} &= \left\lceil \frac{T_{ciclo}}{H} \right\rceil
\end{aligned}
$$

Donde cada símbolo corresponde a un campo de la tabla `rutas` o a una variable calculada:

| Símbolo | Significado | Campo / origen |
|---|---|---|
| $D_h$ | Demanda horaria proyectada (pasajeros/hora) | Salida del modelo (`y_pred` o `y_upper`) |
| $C_{nom}$ | Capacidad nominal del vehículo | `capacidad_nominal` |
| $\rho$ | Ocupación objetivo | `ocupacion_objetivo` |
| $C_{ef}$ | Capacidad efectiva por vehículo | Calculada |
| $B_h$ | Buses por hora | Calculada |
| $H$ | *Headway* (minutos entre despachos) | Calculada |
| $T_{ciclo}$ | Duración total del ciclo de ruta (min) | `ciclo_min` |
| $F_{req}$ | Flota requerida | Calculada |
| $F_{disp}$ | Flota disponible | `flota_disponible` |

Sujeto a:

- capacidad efectiva disponible por vehículo ($C_{ef}$)
- flota disponible en patio/corredor ($F_{req} \le F_{disp}$)
- límites operativos de *headway* mínimo y máximo
- duración total de ciclo de ruta ($T_{ciclo}$)
- nivel de servicio y ocupación objetivo ($\rho$)

Valores operativos de referencia:

| Parámetro | Valor de referencia |
|---|---|
| Resolución temporal | 15 minutos |
| Horizontes operativos | 30, 60 y 120 minutos |
| Ocupación objetivo | Parametrizada por ruta en PostgreSQL |
| Diagnóstico `OPTIMO` | $F_{req} \le F_{disp}$ bajo demanda esperada y banda superior |
| Diagnóstico `ALERTA_CAPACIDAD` | Demanda puntual cubierta, banda superior exige flota adicional |
| Diagnóstico `DEFICIT_FLOTA` | $F_{req} > F_{disp}$ bajo demanda esperada |

---

## Estructura del proyecto

```text
.
├── db/
│   └── init.sql
├── models/
├── notebooks/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── db/
│   │   └── connection.py
│   ├── models/
│   │   ├── base.py
│   │   ├── lgbm_tweedie_wrapper.py
│   │   └── predictor_service.py
│   └── optimization/
│       └── dispatch.py
├── templates/
│   └── dashboard.html
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Instalación y puesta en marcha

### Opción A: Despliegue contenerizado con Docker Compose (Recomendada)

**Requisitos previos:**
- Docker Desktop con Docker Compose v2 activo.
- Puertos locales `8000` (API y tablero web) y `5432` (PostgreSQL) disponibles.

**1. Construir y levantar los servicios:**

```bash
docker compose up --build -d
```

En el primer arranque, el contenedor `sobusa-db` inicializa automáticamente la base de datos y carga los registros desde `db/init.sql` en aproximadamente 10–15 segundos.

**2. Verificar estado de los contenedores:**

```bash
docker compose ps
```

**3. Acceder a la aplicación:**
- **Tablero Web Interactivo:** http://localhost:8000/
- **Documentación API (Swagger UI):** http://localhost:8000/docs
- **Healthcheck del Servicio:** http://localhost:8000/health

**4. Detener los servicios:**

```bash
docker compose down
```

### Opción B: Ejecución local con Conda

**1. Recrear y activar el entorno:**

```bash
conda env create -f environment.yml
conda activate sobusa_data_challenge_2026
```

**2. Iniciar el servidor Uvicorn (requiere instancia local de PostgreSQL activa):**

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API REST y rutas web

Endpoints implementados en FastAPI:

| Método | Ruta | Finalidad |
|---|---|---|
| `GET` | `/` | Renderiza el tablero web operativo con rutas y cortes disponibles. |
| `POST` | `/ui/forecast` | Ejecuta predicción y optimización desde el formulario web y muestra resultados. |
| `GET` | `/health` | Verifica estado de la API, conexión a PostgreSQL y cantidad de modelos cargados. |
| `GET` | `/api/v1/routes` | Lista las rutas configuradas y sus parámetros operativos de flota y ciclo. |
| `POST` | `/api/v1/predict` | Ejecuta inferencia puntual, banda superior conformal y recomendación de *headway* y flota. |

Ejemplo de respuesta de `POST /api/v1/predict`:

```json
{
  "ruta": "RUTA K54",
  "horizonte_min": 60,
  "cutoff_index": 120,
  "timestamp_corte": "2026-07-22T16:00:00",
  "prediccion": {
    "demanda_esperada": 185.4,
    "banda_superior_conformal": 207.1
  },
  "recomendacion_despacho": {
    "headway_min": 10.0,
    "buses_por_hora": 6,
    "flota_requerida": 8,
    "flota_disponible": 10,
    "estado_operativo": "OPTIMO"
  }
}
```

---

## Modelo de datos

La base de datos `sobusa_db` en PostgreSQL 16 se autoinicializa mediante `db/init.sql` con las siguientes tablas:

- `rutas`: parámetros operativos por corredor (`capacidad_nominal`, `ciclo_min`, `flota_disponible`, `ocupacion_objetivo`).
- `historico`: registros históricos de abordajes en intervalos de 15 minutos para construcción de contexto temporal.
- `test_set`: ventanas temporales de evaluación utilizadas para simular cortes operativos desde la API y el tablero.
- `calibration`: cuantiles residuales calibrados por ruta y horizonte para la construcción de intervalos conformales.

Reglas de no interferencia:

- Conexiones de consulta ejecutadas sin alterar la estructura de las tablas base.
- Separación entre el motor de inferencia (`sobusa-api`) y la persistencia en base de datos (`sobusa-db`).
- Configuración mediante variables de entorno (`DATABASE_URL`) compatible con despliegue local y contenerizado.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Gestión de entorno | Conda (`conda-forge`) y Docker |
| Análisis y procesamiento | pandas, NumPy, PyArrow, Joblib |
| Machine Learning y Tuning | LightGBM (objetivo Tweedie), scikit-learn, Optuna |
| API y Frontend Web | FastAPI, Uvicorn, Pydantic, Jinja2 |
| Base de Datos y ORM | PostgreSQL 16 (Alpine), SQLAlchemy 2, Psycopg 3 |
| Contenerización | Docker, Docker Compose |

---

## Plan de trabajo

| Fase | Actividades | Resultado |
|---|---|---|
| 1. Descubrimiento | Entendimiento operativo, perfilamiento de ventanas de 15 min y validación de reglas. | Datos y restricciones operativas validadas. |
| 2. Línea base | Construcción de features temporales, rezagos y benchmarks de referencia. | Línea base cronológica establecida. |
| 3. Modelado | Entrenamiento LightGBM Tweedie, validación *sliding-window* y tuning con Optuna. | Modelos serializados por ruta y horizonte. |
| 4. Calibración | Calibración conformal sobre conjunto independiente para bandas de seguridad. | Intervalos conformales integrados. |
| 5. Optimización | Modelado de capacidad efectiva, *headway*, buses por hora y diagnóstico de flota. | Motor de recomendación de despacho. |
| 6. Producto | Desarrollo de API REST en FastAPI y tablero web operativo en Jinja2. | MVP funcional integrado a PostgreSQL. |
| 7. Contenerización | Empaquetado con Docker Compose, volcado `db/init.sql` y documentación. | Entrega reproducible lista para evaluación. |

---

## Equipo

| Integrante | Código | Rol |
|---|---|---|
| Natalia Paola Alvarado Corrales | 200150416 | Científica de Datos |
| Miguel Ángel Pérez Vargas | 200150359 | Científico de Datos |
| Juan Camilo Mendoza Arango | 200135583 | Científico de Datos |

Proyecto desarrollado en el marco del **Data Challenge 2026** — Universidad del Norte & Extreme Technologies.

## Referencias

- Especificación técnica del Reto de Frecuencias, Data Challenge 2026, Extreme Technologies & Universidad del Norte.

## Licencia

Distribuido bajo la Licencia MIT. Consultar el archivo `LICENSE` para más detalles.

## Contacto

Para dudas técnicas o coordinación con Extreme Technologies, contactar a través del canal oficial definido para el Data Challenge 2026.

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from models.predictor_service import DemandPredictorService
from optimization.headway_optimizer import optimize_route_dispatch

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
TEMPLATES_DIR = ROOT_DIR / "templates"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres123@localhost:5432/frecuencias",
)
ENGINE = create_engine(DATABASE_URL)

app = FastAPI(
    title="SOBUSA Demand Forecasting & Headway Optimization API",
    version="1.0.0",
    description="Predicción de demanda con intervalos conformes Tweedie (95%) y optimización de despacho.",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
predictor = DemandPredictorService(models_dir=MODELS_DIR, engine=ENGINE)

ROUTES_LIST = ["KRA 54", "Silencio", "Calle 72"]
DEFAULT_FORM = {
    "route": "Silencio",
    "horizon_min": 30,
    "target_datetime": "2025-10-15T13:30",
    "tipo_estacionalidad": "DIAS_CORRIENTES",
    "factor_ocupacion_objetivo": 0.85,
}


class PredictRequest(BaseModel):
    route: Literal["KRA 54", "Silencio", "Calle 72"]
    horizon_min: Literal[30, 60, 120]
    target_datetime: datetime = Field(examples=["2025-05-26T17:30:00"])
    tipo_estacionalidad: str = Field(default="DIAS_CORRIENTES")


class OptimizeRequest(PredictRequest):
    factor_ocupacion_objetivo: float | None = Field(default=None, gt=0.1, le=1.3)


@app.get("/", response_class=HTMLResponse)
def dashboard_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "routes": ROUTES_LIST,
            "form": DEFAULT_FORM,
            "result": None,
            "error": None,
        },
    )


@app.post("/ui/optimize", response_class=HTMLResponse)
def dashboard_optimize(
    request: Request,
    route: str = Form(...),
    horizon_min: int = Form(...),
    target_datetime: str = Form(...),
    tipo_estacionalidad: str = Form("DIAS_CORRIENTES"),
    factor_ocupacion_objetivo: float = Form(0.85),
) -> HTMLResponse:
    form_state = {
        "route": route,
        "horizon_min": horizon_min,
        "target_datetime": target_datetime,
        "tipo_estacionalidad": tipo_estacionalidad,
        "factor_ocupacion_objetivo": factor_ocupacion_objetivo,
    }
    try:
        dt_obj = datetime.fromisoformat(target_datetime)
        pred = predictor.predict(
            route=route,
            horizon_min=horizon_min,
            target_dt=dt_obj,
            tipo_estacionalidad=tipo_estacionalidad,
        )
        result = optimize_route_dispatch(
            engine=ENGINE,
            prediction_result=pred,
            factor_ocupacion_override=factor_ocupacion_objetivo,
        )
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "routes": ROUTES_LIST,
                "form": form_state,
                "result": result,
                "error": None,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "routes": ROUTES_LIST,
                "form": form_state,
                "result": None,
                "error": str(exc),
            },
        )


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "loaded_bundles": len(predictor.bundles)}


@app.get("/models")
def list_models() -> list[dict[str, object]]:
    return [
        {
            "route": b["route"],
            "horizon": b["horizon"],
            "version": b["version"],
            "trained_at": b["trained_at"],
            "valid_until": b["valid_until"],
            "metrics": b["metrics"],
        }
        for b in predictor.bundles.values()
    ]


@app.post("/predict")
def predict_demand(req: PredictRequest) -> dict[str, object]:
    try:
        return predictor.predict(
            route=req.route,
            horizon_min=req.horizon_min,
            target_dt=req.target_datetime,
            tipo_estacionalidad=req.tipo_estacionalidad,
        )
    except (KeyError, ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/optimize")
def optimize_dispatch(req: OptimizeRequest) -> dict[str, object]:
    try:
        pred = predictor.predict(
            route=req.route,
            horizon_min=req.horizon_min,
            target_dt=req.target_datetime,
            tipo_estacionalidad=req.tipo_estacionalidad,
        )
        return optimize_route_dispatch(
            engine=ENGINE,
            prediction_result=pred,
            factor_ocupacion_override=req.factor_ocupacion_objetivo,
        )
    except (KeyError, ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
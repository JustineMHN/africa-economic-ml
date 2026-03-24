"""
api.py
------
API FastAPI — AfricaInvest Intelligence
Endpoints : /health, /predict, /metrics, /predict/batch

Usage local :
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import model
from app.schemas import (
    GrowthCategory,
    HealthResponse,
    ModelMetrics,
    PredictionRequest,
    PredictionResponse,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (chargement du modèle au démarrage) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage — chargement du modèle...")
    try:
        model.load()
        logger.info("Modèle chargé avec succès ✅")
    except FileNotFoundError as e:
        logger.error("Impossible de charger le modèle : %s", e)
    yield
    logger.info("Arrêt de l'application.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AfricaInvest Intelligence API",
    description=(
        "API de prédiction de la croissance économique africaine.\n\n"
        "Classifie le potentiel de croissance en **low / medium / high** "
        "à partir d'indicateurs macro-économiques."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Santé"])
async def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    return HealthResponse(
        status="ok",
        model_loaded=model.is_loaded,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict(request: PredictionRequest):
    """
    Prédit la catégorie de croissance économique pour un pays / une période donnée.

    - **low**    : Croissance faible  (< 1.5 score composite)
    - **medium** : Croissance modérée (1.5 – 4.0)
    - **high**   : Croissance élevée  (> 4.0)
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    features = request.model_dump()

    try:
        result = model.predict(features)
    except Exception as exc:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        prediction=GrowthCategory(result["prediction"]),
        probabilities=result["probabilities"],
        confidence=result["confidence"],
        country=request.country,
        region=request.region.value,
    )


@app.post("/predict/batch", tags=["Prédiction"])
async def predict_batch(requests: list[PredictionRequest]):
    """
    Prédictions sur plusieurs pays en une seule requête (max 100).
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    if len(requests) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 prédictions par batch.")

    results = []
    for req in requests:
        try:
            result = model.predict(req.model_dump())
            results.append({
                "country":       req.country,
                "region":        req.region.value,
                "prediction":    result["prediction"],
                "probabilities": result["probabilities"],
                "confidence":    result["confidence"],
                "error":         None,
            })
        except Exception as exc:
            results.append({
                "country": req.country,
                "region":  req.region.value if req.region else None,
                "error":   str(exc),
            })

    return JSONResponse(content={"results": results, "count": len(results)})


@app.get("/metrics", response_model=ModelMetrics, tags=["Modèle"])
async def get_metrics():
    """Retourne les métriques d'entraînement du modèle actif."""
    metrics = model.get_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="Métriques non disponibles.")
    return metrics


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "message":     "AfricaInvest Intelligence API",
        "docs":        "/docs",
        "health":      "/health",
    }

"""
schemas.py
----------
Schémas Pydantic pour la validation des données de l'API AfricaInvest.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Region(str, Enum):
    west_africa    = "West Africa"
    east_africa    = "East Africa"
    north_africa   = "North Africa"
    central_africa = "Central Africa"
    southern_africa = "Southern Africa"


class GrowthCategory(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"


# ── Requête de prédiction ─────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Corps de la requête POST /predict."""

    country:                  Optional[str]  = Field(None, example="Kenya")
    region:                   Region         = Field(..., example="East Africa")
    gdp_growth:               float          = Field(..., ge=-15.0, le=20.0,
                                                     example=5.2,
                                                     description="Taux de croissance du PIB (%)")
    inflation_rate:           float          = Field(..., ge=0.0, le=100.0,
                                                     example=6.1,
                                                     description="Taux d'inflation (%)")
    unemployment_rate:        float          = Field(..., ge=0.0, le=60.0,
                                                     example=10.5,
                                                     description="Taux de chômage (%)")
    fdi_pct_gdp:              float          = Field(..., ge=0.0, le=30.0,
                                                     example=3.8,
                                                     description="Investissements directs étrangers (% PIB)")
    trade_openness:           float          = Field(..., ge=0.0, le=200.0,
                                                     example=55.0,
                                                     description="Ouverture commerciale (% PIB)")
    literacy_rate:            float          = Field(..., ge=0.0, le=100.0,
                                                     example=72.0,
                                                     description="Taux d'alphabétisation (%)")
    population_growth:        float          = Field(..., ge=0.0, le=10.0,
                                                     example=2.4,
                                                     description="Croissance démographique (%)")
    internet_penetration:     float          = Field(..., ge=0.0, le=100.0,
                                                     example=35.0,
                                                     description="Pénétration Internet (%)")
    government_debt_pct:      float          = Field(..., ge=0.0, le=200.0,
                                                     example=52.0,
                                                     description="Dette publique (% PIB)")
    natural_resources_rents:  float          = Field(..., ge=0.0, le=60.0,
                                                     example=4.5,
                                                     description="Rentes ressources naturelles (% PIB)")

    @field_validator("gdp_growth")
    @classmethod
    def validate_gdp(cls, v: float) -> float:
        return round(v, 4)

    class Config:
        json_schema_extra = {
            "example": {
                "country":                "Kenya",
                "region":                 "East Africa",
                "gdp_growth":             5.2,
                "inflation_rate":         6.1,
                "unemployment_rate":      10.5,
                "fdi_pct_gdp":            3.8,
                "trade_openness":         55.0,
                "literacy_rate":          72.0,
                "population_growth":      2.4,
                "internet_penetration":   35.0,
                "government_debt_pct":    52.0,
                "natural_resources_rents": 4.5,
            }
        }


# ── Réponse de prédiction ─────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Corps de la réponse POST /predict."""

    prediction:   GrowthCategory            = Field(..., description="Catégorie prédite")
    probabilities: dict[str, float]         = Field(..., description="Probabilités par classe")
    confidence:   float                     = Field(..., description="Confiance max (0–1)")
    country:      Optional[str]             = None
    region:       str                       = Field(..., description="Région saisie")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction":    "high",
                "probabilities": {"low": 0.05, "medium": 0.20, "high": 0.75},
                "confidence":    0.75,
                "country":       "Kenya",
                "region":        "East Africa",
            }
        }


# ── Santé de l'API ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str  = "ok"
    model_loaded: bool = False
    version:      str  = "1.0.0"


# ── Métriques du modèle ───────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    model:         str
    f1_macro_test: float
    cv_f1_mean:    float
    cv_f1_std:     float
    n_train:       int
    n_test:        int
    features:      list[str]
    label_order:   list[str]

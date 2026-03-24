"""
model.py
--------
Chargement et inférence du modèle AfricaInvest.
Singleton thread-safe pour usage dans FastAPI.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH  = ROOT / "artifacts" / "metrics.json"

FEATURE_COLUMNS = [
    "gdp_growth", "inflation_rate", "unemployment_rate", "fdi_pct_gdp",
    "trade_openness", "literacy_rate", "population_growth",
    "internet_penetration", "government_debt_pct", "natural_resources_rents",
    "region",
]

LABEL_ORDER = ["low", "medium", "high"]


class PredictionModel:
    """Encapsule le pipeline scikit-learn et expose une méthode predict propre."""

    _instance: Optional["PredictionModel"] = None

    def __new__(cls) -> "PredictionModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, path: Path = ARTIFACT_PATH) -> None:
        """Charge le modèle depuis le disque."""
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact introuvable : {path}\n"
                "→ Exécute : python scripts/train_model.py"
            )
        self._pipeline = joblib.load(path)
        self._loaded = True
        logger.info("Modèle chargé depuis %s", path)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features: dict) -> dict:
        """
        Effectue une prédiction à partir d'un dictionnaire de features.

        Parameters
        ----------
        features : dict
            Doit contenir toutes les clés de FEATURE_COLUMNS.

        Returns
        -------
        dict avec keys : prediction, probabilities, confidence
        """
        if not self._loaded:
            raise RuntimeError("Modèle non chargé — appelez load() d'abord.")

        row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])

        pred_label    = self._pipeline.predict(row)[0]
        pred_proba    = self._pipeline.predict_proba(row)[0]
        classes       = self._pipeline.classes_

        probabilities = {str(cls): round(float(p), 4)
                         for cls, p in zip(classes, pred_proba)}
        confidence    = round(float(pred_proba.max()), 4)

        return {
            "prediction":    str(pred_label),
            "probabilities": probabilities,
            "confidence":    confidence,
        }

    def get_metrics(self) -> Optional[dict]:
        """Lit le fichier metrics.json généré lors de l'entraînement."""
        if not METRICS_PATH.exists():
            return None
        with open(METRICS_PATH) as f:
            return json.load(f)


# ── Instance globale (singleton) ──────────────────────────────────────────────
model = PredictionModel()

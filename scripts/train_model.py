"""
train_model.py
--------------
Entraîne un pipeline scikit-learn de classification de la croissance économique
et sauvegarde le modèle dans artifacts/model.joblib.

Usage :
    python scripts/train_model.py
    python scripts/train_model.py --data data/africa_economic_data.csv --output artifacts/model.joblib
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ── Chemins par défaut ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA     = ROOT / "data" / "africa_economic_data.csv"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "model.joblib"
METRICS_PATH     = ROOT / "artifacts" / "metrics.json"

# ── Features ──────────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "gdp_growth", "inflation_rate", "unemployment_rate", "fdi_pct_gdp",
    "trade_openness", "literacy_rate", "population_growth",
    "internet_penetration", "government_debt_pct", "natural_resources_rents",
]
CATEGORICAL_FEATURES = ["region"]
TARGET = "growth_category"
LABEL_ORDER = ["low", "medium", "high"]


def build_pipeline(model_name: str = "random_forest") -> Pipeline:
    """Construit un pipeline sklearn : préprocessing + classificateur."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0,
                                                   class_weight="balanced",
                                                   random_state=42),
        "random_forest":       RandomForestClassifier(n_estimators=300,
                                                       max_depth=15,
                                                       class_weight="balanced",
                                                       random_state=42,
                                                       n_jobs=-1),
        "gradient_boosting":   GradientBoostingClassifier(n_estimators=200,
                                                           max_depth=5,
                                                           learning_rate=0.05,
                                                           random_state=42),
    }

    if model_name not in classifiers:
        raise ValueError(f"Modèle inconnu : {model_name}. Choix : {list(classifiers)}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifiers[model_name]),
    ])


def train(data_path: Path, artifact_path: Path, model_name: str = "random_forest") -> dict:
    """Entraîne le modèle et retourne les métriques."""
    print(f"\n{'='*55}")
    print(f"  AfricaInvest — Entraînement du modèle")
    print(f"{'='*55}")
    print(f"  Dataset  : {data_path}")
    print(f"  Modèle   : {model_name}")
    print(f"  Artifact : {artifact_path}\n")

    # ── Chargement ────────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    print(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}\n")

    # ── Entraînement ──────────────────────────────────────────────────────────
    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)

    # ── Évaluation ────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print("=== Rapport de classification ===")
    print(classification_report(y_test, y_pred, target_names=LABEL_ORDER))

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"CV F1-macro (5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    metrics = {
        "model":          model_name,
        "f1_macro_test":  round(float(f1_macro), 4),
        "cv_f1_mean":     round(float(cv_scores.mean()), 4),
        "cv_f1_std":      round(float(cv_scores.std()), 4),
        "n_train":        int(X_train.shape[0]),
        "n_test":         int(X_test.shape[0]),
        "features":       NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "label_order":    LABEL_ORDER,
    }

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"\n✅ Modèle sauvegardé : {artifact_path}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Métriques sauvegardées : {METRICS_PATH}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle AfricaInvest")
    parser.add_argument("--data",   type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--model",  type=str,  default="random_forest",
                        choices=["logistic_regression", "random_forest", "gradient_boosting"])
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERREUR] Dataset introuvable : {args.data}")
        print("  → Exécute d'abord : python data/generate_data.py")
        sys.exit(1)

    train(args.data, args.output, args.model)


if __name__ == "__main__":
    main()

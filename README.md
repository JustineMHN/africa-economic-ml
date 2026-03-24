# 🌍 AfricaInvest Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Plateforme MLOps de prédiction de la croissance économique africaine.  
> Classifie le potentiel de croissance (**low / medium / high**) à partir d'indicateurs macro-économiques, exposé via une **API FastAPI** et un **dashboard Streamlit**.

---

## 📋 Table des matières

- [Architecture](#-architecture)
- [Stack technique](#-stack-technique)
- [Installation](#-installation)
- [Démarrage rapide](#-démarrage-rapide)
- [API Reference](#-api-reference)
- [Métriques du modèle](#-métriques-du-modèle)
- [Déploiement](#-déploiement)

---

## 📁 Architecture

```
africa-economic-ml/
│
├── app/                         # Logique applicative (API + modèle)
│   ├── __init__.py
│   ├── api.py                   # FastAPI — routes & lifespan
│   ├── model.py                 # Singleton de chargement/inférence
│   └── schemas.py               # Pydantic — validation entrées/sorties
│
├── ui/
│   └── streamlit_app.py         # Dashboard interactif
│
├── artifacts/
│   ├── model.joblib             # Pipeline scikit-learn sérialisé
│   └── metrics.json             # Métriques d'entraînement
│
├── scripts/
│   └── train_model.py           # Entraînement + sauvegarde de l'artifact
│
├── data/
│   ├── africa_economic_data.csv # Dataset (2 000 observations, 50 pays)
│   └── generate_data.py         # Générateur de données synthétiques
│
├── notebooks/
│   └── eda_analysis.ipynb       # Analyse exploratoire complète
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Stack technique

| Couche | Technologies |
|--------|-------------|
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **ML** | scikit-learn (RandomForest, GradientBoosting, LogReg), joblib |
| **UI** | Streamlit |
| **Data** | pandas, numpy |
| **Viz** | matplotlib, seaborn, plotly |

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<your-username>/africa-economic-ml.git
cd africa-economic-ml

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Démarrage rapide

### Étape 1 — Générer le dataset

```bash
python data/generate_data.py
```

### Étape 2 — Entraîner le modèle

```bash
python scripts/train_model.py
# Options : --model gradient_boosting | random_forest | logistic_regression
```

### Étape 3 — Lancer l'API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### Étape 4 — Lancer le dashboard

```bash
streamlit run ui/streamlit_app.py
```

| Service | URL |
|---------|-----|
| API Docs (Swagger) | http://localhost:8000/docs |
| API ReDoc | http://localhost:8000/redoc |
| Dashboard Streamlit | http://localhost:8501 |

---

## 📡 API Reference

### `POST /predict`

Prédit la catégorie de croissance pour un pays donné.

**Payload :**
```json
{
  "country": "Kenya",
  "region": "East Africa",
  "gdp_growth": 5.2,
  "inflation_rate": 6.1,
  "unemployment_rate": 10.5,
  "fdi_pct_gdp": 3.8,
  "trade_openness": 55.0,
  "literacy_rate": 72.0,
  "population_growth": 2.4,
  "internet_penetration": 35.0,
  "government_debt_pct": 52.0,
  "natural_resources_rents": 4.5
}
```

**Réponse :**
```json
{
  "prediction": "high",
  "probabilities": { "low": 0.05, "medium": 0.20, "high": 0.75 },
  "confidence": 0.75,
  "country": "Kenya",
  "region": "East Africa"
}
```

### Autres endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Statut de l'API et du modèle |
| `GET` | `/metrics` | Métriques d'entraînement |
| `POST` | `/predict/batch` | Prédictions multiples (max 100) |

---

## 📈 Métriques du modèle

| Modèle | F1-macro (test) | CV F1-macro (5-fold) |
|--------|----------------|---------------------|
| Random Forest *(défaut)* | ~75% | ~75% ± 4% |
| Gradient Boosting | ~74% | ~73% ± 3% |
| Logistic Regression | ~68% | ~67% ± 3% |

---

## 🌐 Déploiement

### Railway / Render (API)

```bash
# Procfile
web: uvicorn app.api:app --host 0.0.0.0 --port $PORT
```

### Streamlit Cloud (UI)

Déploiement direct depuis GitHub sur [share.streamlit.io](https://share.streamlit.io) — pointe vers `ui/streamlit_app.py`.

---

## 📄 Licence

[MIT](LICENSE) — Justine MHN, 2026

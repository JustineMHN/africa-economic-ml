"""
streamlit_app.py
----------------
Interface Streamlit — AfricaInvest Intelligence
Dashboard de prédiction de la croissance économique africaine.

Usage :
    streamlit run ui/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL  = "http://localhost:8000"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "africa_economic_data.csv"

st.set_page_config(
    page_title="AfricaInvest Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"]          { background: #161b22; }
    .metric-card {
        background: #21262d; border: 1px solid #30363d;
        border-radius: 10px; padding: 1.2rem; margin: 0.4rem 0;
    }
    .pred-badge {
        display: inline-block; padding: 0.5rem 1.4rem;
        border-radius: 20px; font-size: 1.3rem; font-weight: 700;
    }
    .high   { background: #1f6b3a; color: #56d364; border: 1px solid #56d364; }
    .medium { background: #5c4a00; color: #e3b341; border: 1px solid #e3b341; }
    .low    { background: #6b1d1d; color: #f85149; border: 1px solid #f85149; }
    .section-title { color: #58a6ff; font-weight: 700; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


def badge(label: str) -> str:
    return f'<span class="pred-badge {label}">{label.upper()}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 AfricaInvest")
    st.markdown("*Intelligence économique africaine*")
    st.divider()

    api_ok = check_api()
    st.markdown(
        f"**API** : {'🟢 Connectée' if api_ok else '🔴 Hors ligne'}",
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.warning("Démarre l'API :\n```\nuvicorn app.api:app --reload\n```")

    st.divider()
    page = st.radio("Navigation", ["🔮 Prédiction", "📊 Exploration", "📈 Métriques"])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Prédiction":
    st.markdown("# 🔮 Prédiction de Croissance")
    st.markdown("Renseigne les indicateurs macro-économiques pour obtenir une prédiction.")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<p class="section-title">📍 Identification</p>', unsafe_allow_html=True)
        country = st.text_input("Pays (optionnel)", placeholder="ex : Kenya")
        region  = st.selectbox("Région *", [
            "West Africa", "East Africa", "North Africa",
            "Central Africa", "Southern Africa",
        ])

        st.markdown('<p class="section-title">💹 Indicateurs économiques</p>', unsafe_allow_html=True)
        gdp_growth        = st.slider("PIB — Croissance (%)",         -10.0, 15.0,  4.5, 0.1)
        inflation_rate    = st.slider("Inflation (%)",                  0.0, 50.0,  6.0, 0.1)
        unemployment_rate = st.slider("Chômage (%)",                    0.0, 40.0, 12.0, 0.1)
        fdi_pct_gdp       = st.slider("IDE (% PIB)",                    0.0, 15.0,  3.5, 0.1)
        trade_openness    = st.slider("Ouverture commerciale (% PIB)", 10.0,120.0, 60.0, 0.5)

    with col2:
        st.markdown('<p class="section-title">🏛️ Indicateurs sociaux & institutionnels</p>', unsafe_allow_html=True)
        literacy_rate           = st.slider("Alphabétisation (%)",          20.0, 99.0, 65.0, 0.5)
        population_growth       = st.slider("Croissance démographique (%)",  0.0,  6.0,  2.5, 0.1)
        internet_penetration    = st.slider("Pénétration Internet (%)",      0.0, 80.0, 30.0, 0.5)
        government_debt_pct     = st.slider("Dette publique (% PIB)",       10.0,130.0, 55.0, 0.5)
        natural_resources_rents = st.slider("Rentes ressources (% PIB)",     0.0, 30.0,  5.0, 0.1)

    st.divider()

    if st.button("🚀 Lancer la prédiction", use_container_width=True, type="primary"):
        if not api_ok:
            st.error("L'API n'est pas disponible. Démarre le serveur FastAPI.")
        else:
            payload = {
                "country": country or None,
                "region":  region,
                "gdp_growth": gdp_growth,
                "inflation_rate": inflation_rate,
                "unemployment_rate": unemployment_rate,
                "fdi_pct_gdp": fdi_pct_gdp,
                "trade_openness": trade_openness,
                "literacy_rate": literacy_rate,
                "population_growth": population_growth,
                "internet_penetration": internet_penetration,
                "government_debt_pct": government_debt_pct,
                "natural_resources_rents": natural_resources_rents,
            }
            with st.spinner("Prédiction en cours..."):
                result = predict(payload)

            if result:
                pred = result["prediction"]
                conf = result["confidence"]
                proba = result["probabilities"]

                st.markdown(f"### Résultat : {badge(pred)}", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Prédiction",  pred.upper())
                c2.metric("Confiance",   f"{conf*100:.1f}%")
                c3.metric("Pays",        country if country else "—")

                st.markdown("#### Probabilités par classe")
                prob_df = pd.DataFrame(
                    {"Classe": list(proba.keys()), "Probabilité (%)": [v*100 for v in proba.values()]}
                ).sort_values("Probabilité (%)", ascending=False)
                st.bar_chart(prob_df.set_index("Classe"))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploration":
    st.markdown("# 📊 Exploration des données")
    df = load_data()

    if df.empty:
        st.warning("Dataset introuvable. Exécute : `python data/generate_data.py`")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", f"{len(df):,}")
        c2.metric("Pays", df["country"].nunique())
        c3.metric("Régions", df["region"].nunique())
        c4.metric("Années", f"{df['year'].min()}–{df['year'].max()}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribution des classes")
            st.bar_chart(df["growth_category"].value_counts())

        with col2:
            st.markdown("#### Distribution par région")
            st.bar_chart(df["region"].value_counts())

        st.markdown("#### Statistiques descriptives")
        num_cols = [
            "gdp_growth", "inflation_rate", "unemployment_rate",
            "fdi_pct_gdp", "literacy_rate", "internet_penetration",
        ]
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

        st.markdown("#### Aperçu du dataset")
        st.dataframe(df.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Métriques":
    st.markdown("# 📈 Métriques du modèle")

    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        if r.status_code == 200:
            m = r.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("F1-macro (test)", f"{m['f1_macro_test']*100:.1f}%")
            c2.metric("CV F1-macro (5-fold)", f"{m['cv_f1_mean']*100:.1f}%")
            c3.metric("Modèle", m["model"].replace("_", " ").title())

            st.divider()
            st.markdown("#### Configuration")
            st.json(m)
        else:
            st.warning("Métriques non disponibles via l'API.")
    except Exception:
        # Lecture directe du fichier si l'API est hors ligne
        metrics_path = Path(__file__).resolve().parent.parent / "artifacts" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            st.markdown("*(lecture locale — API hors ligne)*")
            c1, c2, c3 = st.columns(3)
            c1.metric("F1-macro (test)", f"{m['f1_macro_test']*100:.1f}%")
            c2.metric("CV F1-macro (5-fold)", f"{m['cv_f1_mean']*100:.1f}%")
            c3.metric("Modèle", m["model"].replace("_", " ").title())
            st.json(m)
        else:
            st.error("Métriques non disponibles. Lance d'abord : `python scripts/train_model.py`")

"""
generate_data.py
----------------
Génère un dataset synthétique d'indicateurs économiques africains.
Usage : python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
N_ROWS = 2_000
OUTPUT_PATH = Path(__file__).parent / "africa_economic_data.csv"

COUNTRIES = [
    "Nigeria", "Ethiopia", "Egypt", "DRC", "Tanzania", "Kenya",
    "Uganda", "Algeria", "Sudan", "Morocco", "Ghana", "Mozambique",
    "Côte d'Ivoire", "Madagascar", "Cameroon", "Angola", "Niger",
    "Burkina Faso", "Mali", "Malawi", "Zambia", "Senegal", "Zimbabwe",
    "Rwanda", "Guinea", "Benin", "Burundi", "Tunisia", "South Africa",
    "Togo", "Sierra Leone", "Libya", "Congo", "Liberia", "Mauritania",
    "Eritrea", "Gambia", "Botswana", "Gabon", "Lesotho", "Guinea-Bissau",
    "Equatorial Guinea", "Mauritius", "Eswatini", "Djibouti", "Comoros",
    "Cabo Verde", "Sao Tome and Principe", "Seychelles", "Namibia",
]

REGIONS = {
    "West Africa":   ["Nigeria", "Ghana", "Senegal", "Côte d'Ivoire", "Mali",
                      "Burkina Faso", "Niger", "Guinea", "Benin", "Togo",
                      "Sierra Leone", "Liberia", "Gambia", "Guinea-Bissau",
                      "Cabo Verde", "Mauritania"],
    "East Africa":   ["Ethiopia", "Tanzania", "Kenya", "Uganda", "Rwanda",
                      "Burundi", "Djibouti", "Eritrea", "Comoros", "Seychelles",
                      "Mauritius"],
    "North Africa":  ["Egypt", "Algeria", "Morocco", "Sudan", "Tunisia", "Libya"],
    "Central Africa":["DRC", "Cameroon", "Congo", "Gabon",
                      "Equatorial Guinea", "Sao Tome and Principe"],
    "Southern Africa":["South Africa", "Mozambique", "Madagascar", "Zambia",
                       "Zimbabwe", "Malawi", "Angola", "Botswana", "Lesotho",
                       "Eswatini", "Namibia"],
}

COUNTRY_TO_REGION = {c: r for r, countries in REGIONS.items() for c in countries}


def generate_dataset(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    countries = rng.choice(COUNTRIES, size=n)
    years     = rng.integers(2000, 2024, size=n)

    # ── Features économiques ──────────────────────────────────────────────────
    gdp_growth         = rng.normal(loc=4.5,  scale=3.0,  size=n).clip(-10, 15)
    inflation_rate     = rng.exponential(scale=7.0, size=n).clip(0, 50)
    unemployment_rate  = rng.normal(loc=12.0, scale=6.0,  size=n).clip(1, 40)
    fdi_pct_gdp        = rng.normal(loc=3.5,  scale=2.5,  size=n).clip(0, 15)
    trade_openness     = rng.normal(loc=60.0, scale=20.0, size=n).clip(10, 120)
    literacy_rate      = rng.normal(loc=65.0, scale=18.0, size=n).clip(20, 99)
    population_growth  = rng.normal(loc=2.5,  scale=1.0,  size=n).clip(0, 6)
    internet_penetration = rng.beta(2, 5, size=n) * 80
    government_debt_pct  = rng.normal(loc=55.0, scale=20.0, size=n).clip(10, 130)
    natural_resources_rents = rng.exponential(scale=5.0, size=n).clip(0, 30)

    # ── Cible : croissance économique (3 classes) ─────────────────────────────
    score = (
        0.4  * gdp_growth
        - 0.15 * inflation_rate
        - 0.10 * unemployment_rate
        + 0.20 * fdi_pct_gdp
        + 0.10 * (literacy_rate / 10)
        + 0.05 * internet_penetration
        + rng.normal(0, 0.5, size=n)
    )
    growth_category = pd.cut(
        score,
        bins=[-np.inf, 1.5, 4.0, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    df = pd.DataFrame({
        "country":               countries,
        "region":                [COUNTRY_TO_REGION.get(c, "Other") for c in countries],
        "year":                  years,
        "gdp_growth":            gdp_growth.round(2),
        "inflation_rate":        inflation_rate.round(2),
        "unemployment_rate":     unemployment_rate.round(2),
        "fdi_pct_gdp":           fdi_pct_gdp.round(2),
        "trade_openness":        trade_openness.round(2),
        "literacy_rate":         literacy_rate.round(2),
        "population_growth":     population_growth.round(2),
        "internet_penetration":  internet_penetration.round(2),
        "government_debt_pct":   government_debt_pct.round(2),
        "natural_resources_rents": natural_resources_rents.round(2),
        "growth_category":       growth_category,
    })

    return df.reset_index(drop=True)


if __name__ == "__main__":
    print("Génération du dataset...")
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset sauvegardé : {OUTPUT_PATH}")
    print(f"Shape : {df.shape}")
    print(df["growth_category"].value_counts())

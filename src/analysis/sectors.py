"""Helper pour grouper les tickers par secteur.

Le mapping vient de ``config.yaml`` (clé ``sectors``). Le code en aval est
totalement data-driven : ajouter un secteur ou un ticker dans le yaml suffit
pour que les plots et tableaux d'analyse en tiennent compte.
"""

from __future__ import annotations
import pandas as pd


def invert_sector_map(sector_map: dict[str, list[str]]) -> dict[str, str]:
    """{sector: [tickers]}  ->  {ticker: sector}."""
    out: dict[str, str] = {}
    for sector, tickers in sector_map.items():
        for t in tickers:
            out[t] = sector
    return out


def attach_sector(
    df: pd.DataFrame,
    sector_map: dict[str, list[str]],
    ticker_col: str = "ticker",
    fillna: str = "Other",
) -> pd.DataFrame:
    """Ajoute (ou écrase) une colonne 'sector' au DataFrame."""
    inv = invert_sector_map(sector_map)
    df = df.copy()
    df["sector"] = df[ticker_col].map(inv).fillna(fillna)
    return df


def sectors_present(
    df: pd.DataFrame,
    sector_map: dict[str, list[str]],
    min_tickers: int = 1,
) -> list[str]:
    """Liste des secteurs qui ont au moins ``min_tickers`` tickers calibrés
    dans le DataFrame. Retourne dans l'ordre du yaml.
    """
    if "sector" not in df.columns:
        df = attach_sector(df, sector_map)
    cnt = df.groupby("sector")["ticker"].nunique()
    return [s for s in sector_map.keys() if cnt.get(s, 0) >= min_tickers]


def tickers_in_sector(
    sector: str,
    df: pd.DataFrame,
    sector_map: dict[str, list[str]],
) -> list[str]:
    """Tickers du secteur ``sector`` réellement présents dans le DataFrame."""
    declared = set(sector_map.get(sector, []))
    return sorted(set(df["ticker"].unique()) & declared)


def sector_coverage(
    df: pd.DataFrame,
    sector_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Petit tableau récap : par secteur, nb de tickers déclarés vs calibrés."""
    rows = []
    available = set(df["ticker"].unique())
    for sector, tickers in sector_map.items():
        declared = set(tickers)
        calibrated = declared & available
        rows.append({
            "sector": sector,
            "n_declared": len(declared),
            "n_calibrated": len(calibrated),
            "tickers": ", ".join(sorted(calibrated)) if calibrated else "—",
        })
    return pd.DataFrame(rows)

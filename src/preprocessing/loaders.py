"""Loaders for option chains and spot history. Multi-ticker, date-range aware.

Raises explicit ``DataMissingError`` early — before launching expensive batch
jobs — when requested tickers/dates are not available on disk.
"""

from __future__ import annotations
from pathlib import Path
from datetime import date, datetime
import numpy as np
import pandas as pd


class DataMissingError(RuntimeError):
    """Raised when requested ticker / date range is not available on disk."""


# ----------------------------------------------------------------------------
# Availability check
# ----------------------------------------------------------------------------

def list_available_dates(ticker: str, data_root: str | Path) -> pd.DatetimeIndex:
    """List all trade dates we have option-chain data for."""
    folder = Path(data_root) / ticker
    if not folder.exists():
        return pd.DatetimeIndex([])
    dates = set()
    for f in folder.rglob("*_call.parquet"):
        try:
            dates.add(pd.Timestamp(f.stem.split("_")[0]))
        except Exception:
            continue
    return pd.DatetimeIndex(sorted(dates))


def validate_data_availability(
    tickers: list[str],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    data_root: str | Path,
    min_coverage: float = 0.5,
) -> dict[str, pd.DatetimeIndex]:
    """Check that each ticker has decent coverage on [start, end].

    Returns a dict {ticker -> available dates within the range}.

    Raises DataMissingError if any ticker has < min_coverage fraction of
    expected trading days, or no folder at all.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    expected_days = pd.bdate_range(start, end)
    n_expected = len(expected_days)
    out = {}
    missing = []

    for t in tickers:
        avail = list_available_dates(t, data_root)
        in_range = avail[(avail >= start) & (avail <= end)]
        out[t] = in_range
        cov = len(in_range) / n_expected if n_expected else 0
        if cov < min_coverage:
            missing.append(f"{t}: {len(in_range)}/{n_expected} jours ({cov:.0%})")

    if missing:
        msg = (
            f"Données manquantes ou trop incomplètes sur [{start.date()}, {end.date()}] :\n  "
            + "\n  ".join(missing)
            + f"\n(seuil min_coverage = {min_coverage:.0%}). Télécharge les données avant."
        )
        raise DataMissingError(msg)

    return out


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_chain(
    trade_date: str | pd.Timestamp,
    ticker: str,
    data_root: str | Path,
    rights: tuple[str, ...] = ("call", "put"),
) -> pd.DataFrame:
    """Load the option chain for one (ticker, date)."""
    d = pd.Timestamp(trade_date)
    folder = Path(data_root) / ticker / f"{d.year:04d}" / f"{d.month:02d}"
    dfs = []
    for r in rights:
        f = folder / f"{d.strftime('%Y-%m-%d')}_{r}.parquet"
        if f.exists():
            dfs.append(pd.read_parquet(f))
    if not dfs:
        raise DataMissingError(f"No data: {ticker} {d.date()}")
    df = pd.concat(dfs, ignore_index=True)
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    df["dte"] = (df["expiration"] - df["trade_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot"]
    df["log_moneyness"] = np.log(df["moneyness"])
    df["ticker"] = ticker
    return df


def load_spot_history(
    ticker: str,
    data_root: str | Path,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Spot prices reconstructed from option-chain files. Adds log_return."""
    root = Path(data_root) / ticker
    if not root.exists():
        raise DataMissingError(f"No folder for {ticker} in {data_root}")

    records = []
    for f in sorted(root.rglob("*_call.parquet")):
        tmp = pd.read_parquet(f, columns=["trade_date", "spot"])
        if len(tmp):
            records.append(tmp.iloc[0])

    df = (
        pd.DataFrame(records)
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ticker"] = ticker
    df["log_return"] = np.log(df["spot"] / df["spot"].shift(1))

    if start_date is not None:
        df = df[df["trade_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["trade_date"] <= pd.Timestamp(end_date)]
    return df.reset_index(drop=True)

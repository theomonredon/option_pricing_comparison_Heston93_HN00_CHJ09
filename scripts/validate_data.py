"""Vérifie la disponibilité des données avant de lancer le batch.

Usage:
    python scripts/validate_data.py [--config config.yaml]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.preprocessing import validate_data_availability, DataMissingError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Validation des données pour {len(cfg.tickers)} tickers sur [{cfg.start_date.date()}, {cfg.end_date.date()}]")
    print(f"  data_root = {cfg.data_root.resolve()}")
    print(f"  tickers   = {cfg.tickers}")

    try:
        avail = validate_data_availability(
            cfg.tickers, cfg.start_date, cfg.end_date, cfg.data_root,
            min_coverage=args.min_coverage,
        )
    except DataMissingError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\nOK. Coverage by ticker:")
    for t, dates in avail.items():
        print(f"  {t:6s}: {len(dates):4d} jours disponibles "
              f"({dates.min().date()} -> {dates.max().date()})")


if __name__ == "__main__":
    main()

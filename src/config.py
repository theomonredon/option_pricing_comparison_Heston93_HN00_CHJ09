"""Config loader. Single source of truth for the batch jobs and notebook."""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import pandas as pd


@dataclass
class Config:
    raw: dict
    data_root: Path
    results_root: Path
    tickers: list[str]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    models: list[str]

    @property
    def filters(self) -> dict:
        return self.raw["filters"]

    @property
    def calibration(self) -> dict:
        return self.raw["calibration"]

    @property
    def hn_warmup_days(self) -> int:
        return int(self.raw["hn_warmup"]["window_days"])

    @property
    def oos(self) -> dict:
        return self.raw["oos"]


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return Config(
        raw=raw,
        data_root=Path(raw["data_root"]),
        results_root=Path(raw["results_root"]),
        tickers=list(raw["tickers"]),
        start_date=pd.Timestamp(raw["start_date"]),
        end_date=pd.Timestamp(raw["end_date"]),
        models=list(raw["models"]),
    )

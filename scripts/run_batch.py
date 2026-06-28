"""Batch unifié : pour chaque (date, ticker, model) du protocole rolling OOS :
   1. Calibre le modèle au jour d (in-sample)
   2. Avec les mêmes params, pricer les options aux jours d+1, d+2, d+5 (OOS)
   3. Persiste le tout dans results/batch_results.parquet et results/params_long.parquet

Reprenable : si batch_results.parquet existe, on saute les (date, ticker, model)
déjà calculés. Pour repartir de zéro : supprimer le fichier.

Usage:
    python scripts/run_batch.py [--config config.yaml] [--horizons 1 2 5]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import load_config
from src.preprocessing import (
    validate_data_availability, list_available_dates,
    load_chain, load_spot_history, FilterConfig, filter_chain, add_implied_vol,
    DataMissingError,
)
from src.models import (
    MODEL_REGISTRY, HestonPricer, HNPricer, CHJPricer,
    HestonParameters, HNParameters, CHJParameters,
    garch_filter,
)
from src.calibration import calibrate, evaluate_metrics
from src.analysis.metrics import classify_regime, realized_vol_series


# -------------------------------------------------------------------- utility

def initial_pricer(model_name: str, h_next: float | None = None):
    """Default starting point for the calibration."""
    if model_name == "heston93":
        return HestonPricer(
            HestonParameters(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.5, V_0=0.04),
            r_f=0.0,
        )
    elif model_name == "hn2000":
        return HNPricer(
            HNParameters(omega=5e-6, alpha=2e-6, beta=0.85, gamma=150, lam=2.0),
            h_next=h_next if h_next is not None else 0.20 ** 2 / 252,
            r_f=0.0,
        )
    elif model_name == "chj2009":
        return CHJPricer(
            CHJParameters(
                kappa_1=8.0, theta_1=0.02, sigma_1=0.5, rho_1=-0.5, V_1_0=0.02,
                kappa_2=0.5, theta_2=0.03, sigma_2=0.3, rho_2=-0.5, V_2_0=0.03,
            ),
            r_f=0.0,
        )
    raise ValueError(f"Unknown model {model_name}")


def hn_filter_h_next(spot_history: pd.DataFrame, target_date: pd.Timestamp,
                     warmup_days: int, params: HNParameters,
                     min_warmup: int = 60) -> float:
    """Filtre h_t via fenêtre rolling de `warmup_days` returns avant target_date.

    Si moins de `warmup_days` sont disponibles, dégrade à `min_warmup` (utilise
    tout ce qui est dispo). Si même `min_warmup` n'est pas atteint, raise.
    Returns h_{target_date + 1 day} (la variance qui s'applique au jour suivant).
    """
    hist = spot_history[spot_history.trade_date < target_date].dropna(subset=["log_return"])
    n = len(hist)
    if n < min_warmup:
        raise DataMissingError(
            f"Historique insuffisant pour HN warmup à {target_date.date()} "
            f"(dispo {n}, floor {min_warmup})"
        )
    effective = min(n, warmup_days)
    rets = hist["log_return"].iloc[-effective:].to_numpy()
    h, _ = garch_filter(rets, params, h0=params.unconditional_variance)
    return float(h[-1])


def trading_days_in_month(month_start: pd.Timestamp, available_dates: pd.DatetimeIndex):
    """Returns the available trading days in the calendar month of month_start."""
    end_of_month = month_start + pd.offsets.MonthEnd(0)
    return available_dates[(available_dates >= month_start) & (available_dates <= end_of_month)]


def build_test_calendar(cfg, available_dates: dict[str, pd.DatetimeIndex],
                        regime_classifier: dict[pd.Timestamp, str] | None = None,
                        n_days_override: int | None = None):
    """Returns list of (date, regime) tuples — calibration days for the batch.

    A date is included only if (a) it falls in one of the test months
    (b) at least one ticker has data on (d, d+max_horizon).

    n_days_per_period (from config or n_days_override): max trading days taken
    per test period. None = full month."""
    rows = []
    test_months_cfg = cfg.oos["test_months"]
    horizons = cfg.oos["horizons"]
    max_h = max(horizons)

    n_days = n_days_override
    if n_days is None:
        n_days = cfg.oos.get("n_days_per_period", None)

    # Union of available dates across tickers (each ticker filtered later)
    all_dates = sorted(set().union(*[set(d) for d in available_dates.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    for regime, months in test_months_cfg.items():
        for m_str in months:
            m = pd.Timestamp(m_str)
            month_days = trading_days_in_month(m, all_dates)
            if n_days is not None:
                month_days = month_days[:int(n_days)]
            for d in month_days:
                # need at least `max_h` business days after d
                future = all_dates[(all_dates > d) & (all_dates <= d + pd.Timedelta(days=int(max_h * 1.6 + 2)))]
                if len(future) < max_h:
                    continue
                r = regime
                if regime_classifier is not None:
                    r = regime_classifier.get(pd.Timestamp(d.date()), regime)
                rows.append((d, r))
    # dedup, keep first regime if same date appears twice
    seen = {}
    for d, r in rows:
        if d not in seen:
            seen[d] = r
    return sorted(seen.items())


def horizon_dates(d: pd.Timestamp, horizons: list[int],
                  available_dates: pd.DatetimeIndex) -> dict[int, pd.Timestamp]:
    """For each horizon h, returns the h-th available trading day after d."""
    future = available_dates[available_dates > d]
    out = {}
    for h in horizons:
        if len(future) >= h:
            out[h] = future[h - 1]
    return out


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-metrics", default="results/batch_results.parquet")
    ap.add_argument("--out-params", default="results/params_long.parquet")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of (date, ticker, model) to compute (debug).")
    ap.add_argument("--n-days", type=int, default=None,
                    help="Override n_days_per_period from config (max trading days per test period).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {len(cfg.tickers)} tickers, {len(cfg.models)} models, "
          f"[{cfg.start_date.date()} -> {cfg.end_date.date()}]")

    # 1. Data availability
    try:
        available = validate_data_availability(
            cfg.tickers, cfg.start_date, cfg.end_date, cfg.data_root, min_coverage=0.3,
        )
    except DataMissingError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # 2. Spot histories (warmup HN + regime classification)
    print("Loading spot histories...")
    histories = {t: load_spot_history(t, cfg.data_root) for t in cfg.tickers}
    spy_hist = histories.get("SPY")

    regime_classifier = None
    if spy_hist is not None and cfg.oos.get("auto_classify_regime_via") == "SPY":
        rv = realized_vol_series(spy_hist["log_return"], window=21)
        regime_classifier = {
            pd.Timestamp(r.trade_date.date()): classify_regime(r_val)
            for (_, r), r_val in zip(spy_hist.iterrows(), rv)
            if not pd.isna(r_val)
        }
        print(f"  regime classifier via SPY RV21: {len(regime_classifier)} dates")

    # 3. Build calibration calendar
    n_days_override = args.n_days if args.n_days is not None else None
    calendar = build_test_calendar(cfg, available, regime_classifier, n_days_override=n_days_override)
    n_days_eff = n_days_override or cfg.oos.get("n_days_per_period", None)
    print(f"Calibration calendar: {len(calendar)} dates "
          f"(n_days_per_period={'full month' if n_days_eff is None else n_days_eff})")

    # 4. Resume from existing output if any
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if Path(args.out_metrics).exists():
        existing = pd.read_parquet(args.out_metrics)
        print(f"Found existing results: {len(existing)} rows. Skipping already-done triples.")

    done_keys = set()
    if not existing.empty:
        done_keys = set(
            zip(
                pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d"),
                existing["ticker"], existing["model"],
            )
        )

    filt_cfg = FilterConfig(**cfg.filters, r_annual=cfg.calibration["r_annual"])
    horizons = cfg.oos["horizons"]
    new_metric_rows = []
    new_param_rows = []
    n_processed = 0

    # Pré-calcul du nombre total de travaux pour la progress bar
    n_starts_cfg = cfg.calibration.get("n_starts", 1)
    def _n_starts(m):
        if isinstance(n_starts_cfg, dict):
            return int(n_starts_cfg.get(m, 1))
        return int(n_starts_cfg)

    total_jobs = sum(
        1 for d, _ in calendar for t in cfg.tickers
        if d in available[t]
        for m in cfg.models
        if (d.strftime("%Y-%m-%d"), t, m) not in done_keys
    )
    avg_starts = sum(_n_starts(m) for m in cfg.models) / len(cfg.models)
    print(f"\nTravaux à effectuer : {total_jobs} calibrations "
          f"(~{avg_starts:.1f} starts/calib en moyenne)\n")

    pbar = tqdm(total=total_jobs, unit="calib", ncols=90,
                bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]")

    # 5. Outer loop
    for d, regime in calendar:
        for ticker in cfg.tickers:
            ticker_dates = available[ticker]
            if d not in ticker_dates:
                continue
            try:
                chain = load_chain(d, ticker, cfg.data_root)
            except Exception as e:
                print(f"  ERROR: {d.date()} {ticker} chain load: {e}")
                continue
            try:
                clean = filter_chain(chain, filt_cfg)
                if len(clean) < 30:
                    continue
                clean = add_implied_vol(clean, cfg.calibration["r_annual"])
                if len(clean) < 30:
                    continue
            except Exception as e:
                print(f"  ERROR: {d.date()} {ticker} filter: {e}")
                continue

            # OOS dates
            hdates = horizon_dates(d, horizons, ticker_dates)
            future_chains = {}
            for h, dh in hdates.items():
                try:
                    fc = load_chain(dh, ticker, cfg.data_root)
                    fc = filter_chain(fc, filt_cfg)
                    if len(fc) >= 10:
                        future_chains[h] = add_implied_vol(fc, cfg.calibration["r_annual"])
                except Exception:
                    pass

            for model_name in cfg.models:
                key = (d.strftime("%Y-%m-%d"), ticker, model_name)
                if key in done_keys:
                    continue

                try:
                    cls = MODEL_REGISTRY[model_name]
                    # HN needs h_next from filter
                    if model_name == "hn2000":
                        p_init_hn = HNParameters(omega=5e-6, alpha=2e-6, beta=0.85, gamma=150, lam=2.0)
                        h_next = hn_filter_h_next(histories[ticker], d, cfg.hn_warmup_days, p_init_hn)
                        x0 = initial_pricer(model_name, h_next=h_next)
                    else:
                        x0 = initial_pricer(model_name)

                    # Multi-start : on peut spécifier soit un dict
                    # {model: n_starts} soit un int global dans le config.
                    n_starts_cfg = cfg.calibration.get("n_starts", 1)
                    if isinstance(n_starts_cfg, dict):
                        n_starts = int(n_starts_cfg.get(model_name, 1))
                    else:
                        n_starts = int(n_starts_cfg)

                    t0 = time.time()
                    pricer_opt, info = calibrate(
                        cls, clean, x0,
                        r_f=cfg.calibration["r_annual"],
                        loss=cfg.calibration["loss"],
                        method=cfg.calibration["method"],
                        maxiter=cfg.calibration["maxiter"],
                        n_starts=n_starts,
                        seed=cfg.calibration.get("seed", 42),
                        verbose=False,
                    )
                    elapsed = time.time() - t0

                    in_sample = evaluate_metrics(pricer_opt, clean, cfg.calibration["r_annual"])
                    row = {
                        "date": d, "ticker": ticker, "model": model_name,
                        "regime": regime,
                        "n_options": in_sample["n_options"],
                        "in_sample_iv_vega_rmse": info["final_loss"],
                        "in_sample_rmse_price": in_sample["rmse_price"],
                        "in_sample_rmse_iv": in_sample["rmse_iv"],
                        "in_sample_mae_iv": in_sample["mae_iv"],
                        "calibration_time_sec": elapsed,
                        "converged": info["converged"],
                        "n_starts": info.get("n_starts", 1),
                        "best_start_idx": info.get("best_start_idx", 0),
                        "n_converged_starts": int(sum(info.get("all_start_converged", [info["converged"]]))),
                    }

                    for h, fc in future_chains.items():
                        # For HN, update h_next by continuing the filter for h steps
                        if model_name == "hn2000":
                            # extend filter using the realised returns between d and d+h
                            extra_hist = histories[ticker][
                                (histories[ticker].trade_date > d) &
                                (histories[ticker].trade_date <= hdates[h])
                            ].dropna(subset=["log_return"])
                            extra_ret = extra_hist["log_return"].to_numpy()
                            if len(extra_ret) > 0:
                                h_t, _ = garch_filter(
                                    extra_ret, pricer_opt.params_p, h0=pricer_opt.h_next
                                )
                                future_h_next = float(h_t[-1])
                            else:
                                future_h_next = pricer_opt.h_next
                            pricer_for_oos = HNPricer(
                                pricer_opt.params_p, h_next=future_h_next, r_f=pricer_opt.r_f
                            )
                        else:
                            pricer_for_oos = pricer_opt

                        oos_metrics = evaluate_metrics(pricer_for_oos, fc, cfg.calibration["r_annual"])
                        row[f"oos_iv_vega_rmse_J{h}"] = oos_metrics["iv_vega_rmse"]
                        row[f"oos_rmse_price_J{h}"]   = oos_metrics["rmse_price"]
                        row[f"oos_rmse_iv_J{h}"]      = oos_metrics["rmse_iv"]
                        row[f"oos_n_options_J{h}"]    = oos_metrics["n_options"]

                    # Params: long format
                    pvec = pricer_opt.get_calibration_vector()
                    pnames = _param_names(model_name)
                    for name, val in zip(pnames, pvec):
                        new_param_rows.append({
                            "date": d, "ticker": ticker, "model": model_name,
                            "param_name": name, "param_value": float(val),
                        })

                    new_metric_rows.append(row)
                    n_processed += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "last": f"{ticker}/{model_name[:3]}",
                        "loss": f"{row['in_sample_iv_vega_rmse']:.3f}",
                        "t": f"{elapsed:.0f}s",
                    })

                    if args.limit and n_processed >= args.limit:
                        print(f"Limit reached ({args.limit})")
                        _persist(args, existing, new_metric_rows, new_param_rows)
                        return

                except Exception as e:
                    pbar.update(1)
                    pbar.write(f"  ERROR: {d.date()} {ticker} {model_name}: {e}")

                # Persist every 20 rows
                if n_processed > 0 and n_processed % 20 == 0:
                    _persist(args, existing, new_metric_rows, new_param_rows)

    pbar.close()
    _persist(args, existing, new_metric_rows, new_param_rows)
    print(f"\nOK Done. {n_processed} new rows. Total: see {args.out_metrics}")


def _param_names(model_name: str) -> list[str]:
    if model_name == "heston93":
        return ["log_kappa", "log_theta", "log_sigma", "rho", "log_V_0"]
    if model_name == "hn2000":
        return ["log_omega", "log_alpha", "beta", "gamma", "lam", "log_h_next"]
    if model_name == "chj2009":
        return ["log_kappa_1", "log_theta_1", "log_sigma_1", "rho_1", "log_V_1_0",
                "log_kappa_2", "log_theta_2", "log_sigma_2", "rho_2", "log_V_2_0"]
    return []


def _persist(args, existing, new_metric_rows, new_param_rows):
    if new_metric_rows:
        new_metrics = pd.DataFrame(new_metric_rows)
        merged = pd.concat([existing, new_metrics], ignore_index=True)
        merged.to_parquet(args.out_metrics, index=False)
    if new_param_rows:
        new_params = pd.DataFrame(new_param_rows)
        existing_params = (
            pd.read_parquet(args.out_params) if Path(args.out_params).exists()
            else pd.DataFrame()
        )
        merged_params = pd.concat([existing_params, new_params], ignore_index=True)
        merged_params.to_parquet(args.out_params, index=False)


if __name__ == "__main__":
    main()

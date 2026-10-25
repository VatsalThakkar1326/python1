"""End-to-end backtest: Mark I forecasts -> decision engine -> our simulator.

Forecasts for every decision time are computed up front in batches (much faster than one
at a time), then the simulator replays the period minute by minute and asks the decision
engine what to do at each decision step. Everything the engine sees at time t comes from
data up to t.

Three runs share the same period and simulator:
  * strategy:          real costs
  * strategy (no fees): same decisions logic with zero fees/spread/impact, to show how much
                        trading costs take
  * buy and hold:      equal weight in every coin, bought once, same costs
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mark1.data.store import load_panel
from mark1.decide.engine import DecisionConfig, DecisionEngine
from mark1.features.build import FeatureSet, load_features
from mark1.model.forecaster import Forecaster
from mark1.model.train import load_meta
from mark1.sim import Decision, SimConfig, buy_and_hold_benchmark, run_backtest

log = logging.getLogger(__name__)

DECISION_TAUS = [5, 15, 30, 60, 120, 240, 480, 720, 1440]


@dataclass
class BacktestOutput:
    out_dir: Path
    metrics: dict
    report_path: Path | None


def _aligned_panel(fs: FeatureSet):
    start = pd.Timestamp(fs.times[0])
    end = pd.Timestamp(fs.times[-1]) + pd.Timedelta(minutes=1)
    panel = load_panel(fs.symbols, start, end)
    if len(panel.times) != len(fs.times) or (panel.times != fs.times).any():
        raise ValueError("Bars on disk no longer match the feature dataset; rebuild with `mark1 build`.")
    return panel


def one_way_cost(sim_cfg: SimConfig, n_assets: int, impact_bps: float = 2.0) -> np.ndarray:
    """Expected one-way cost per coin as a log return: taker fee + half spread + typical impact."""
    hs = np.broadcast_to(np.asarray(sim_cfg.half_spread_bps, dtype=float), (n_assets,))
    return (sim_cfg.taker_fee_bps + hs + impact_bps) * 1e-4


def _strategy(engine: DecisionEngine, rows: np.ndarray, fc: dict, levels: list[float],
              cost: np.ndarray, symbols: list[str], times: np.ndarray, decisions: list | None):
    index = {int(t): k for k, t in enumerate(rows)}
    taus = fc["taus"]

    def strategy(t, market, portfolio):
        k = index.get(int(t))
        if k is None:
            return None
        target, info = engine.decide(fc["quantiles"][k], taus, levels, portfolio.weights, cost,
                                     fc["ready"][k], market.close[-1], portfolio.equity)
        if decisions is not None:
            for i, act in enumerate(info.action):
                if act in ("enter", "exit", "dip_exit", "stop"):
                    kb = int(np.argmax(info.expected[:, i] / info.risk[:, i]))
                    decisions.append({
                        "time": str(pd.Timestamp(times[t])), "coin": symbols[i].replace("USDT", ""),
                        "action": act, "weight_before": round(float(portfolio.weights[i]), 4),
                        "weight_after": round(float(target[i]), 4),
                        "best_tau": int(info.best_tau[i]),
                        "expected_pct": round(100 * float(info.expected[kb, i]), 3),
                        "worst_near_pct": round(100 * float(info.expected[np.asarray(taus) <= 240, i].min()), 3),
                    })
        return Decision(target_weights=target)

    return strategy


def run_model_backtest(
    run_dir: Path,
    split: str = "test",
    decision_every: int = 5,
    taker_fee_bps: float | None = None,
    maker_fee_bps: float | None = None,
    decision_cfg: DecisionConfig | None = None,
    sim_cfg: SimConfig | None = None,
    report: bool = True,
) -> BacktestOutput:
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    fs = load_features(Path(meta["train_config"]["dataset_dir"]), mmap=True)
    panel = _aligned_panel(fs)
    sp = meta["splits"][split]
    start, end = int(sp["start"]), int(sp["end"])

    sim_cfg = sim_cfg or SimConfig()
    changes = {k: v for k, v in (("taker_fee_bps", taker_fee_bps), ("maker_fee_bps", maker_fee_bps)) if v is not None}
    if changes:
        sim_cfg = sim_cfg.replace(**changes)
    decision_cfg = decision_cfg or DecisionConfig()

    forecaster = Forecaster.load(run_dir)
    log.info("forecasting every %d minutes x %d horizons (cached after the first run) ...", decision_every, len(DECISION_TAUS))
    fc = forecaster.predict_grid(fs, run_dir, start, end, decision_every, DECISION_TAUS)
    rows = fc["rows"]

    levels = forecaster.quantiles
    cost = one_way_cost(sim_cfg, panel.n_assets)
    decisions: list[dict] = []

    log.info("simulating %s split: %s -> %s", split, pd.Timestamp(fs.times[start]), pd.Timestamp(fs.times[end - 1]))
    strat = run_backtest(panel, _strategy(DecisionEngine(decision_cfg), rows, fc, levels, cost, fs.symbols, fs.times, decisions),
                         sim_cfg, start, end, decision_every)
    free_cfg = sim_cfg.replace(taker_fee_bps=0.0, maker_fee_bps=0.0, half_spread_bps=0.0, impact_coef=0.0)
    free = run_backtest(panel, _strategy(DecisionEngine(decision_cfg), rows, fc, levels, np.zeros(panel.n_assets),
                                         fs.symbols, fs.times, None), free_cfg, start, end, decision_every)
    bh = buy_and_hold_benchmark(panel, sim_cfg, start, end)

    print(strat.summary("Mark I strategy (real costs)"))
    print(free.summary("Mark I strategy (no costs)"))
    print(bh.summary("Buy and hold, equal weight"))

    out_dir = run_dir / f"backtest_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"strategy": strat.metrics, "strategy_no_costs": free.metrics, "buy_and_hold": bh.metrics,
               "sim_config": _jsonable(asdict(sim_cfg)), "decision_config": asdict(decision_cfg),
               "decision_every": decision_every, "split": split,
               "period": [str(pd.Timestamp(fs.times[start])), str(pd.Timestamp(fs.times[end - 1]))]}
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    strat.fills.to_parquet(out_dir / "fills.parquet")
    pd.DataFrame({"strategy": strat.equity, "strategy_no_costs": free.equity, "buy_and_hold": bh.equity}).to_parquet(out_dir / "equity.parquet")
    (out_dir / "decisions.json").write_text(json.dumps(decisions))

    report_path = None
    if report:
        from mark1.report import build_report_data, write_report
        data = build_report_data(run_dir, meta, fs, panel, start, end, strat, free, bh, decisions, forecaster, metrics)
        report_path = write_report(data, out_dir / "report.html")
        log.info("report written to %s", report_path)
    return BacktestOutput(out_dir, metrics, report_path)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


__all__ = ["run_model_backtest", "one_way_cost", "DECISION_TAUS"]

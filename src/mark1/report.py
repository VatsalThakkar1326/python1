"""Build the interactive backtest dashboard (a single self-contained HTML file)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).with_name("report_template.html")
CONE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
METRIC_KEYS = ["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "ann_vol", "fees_paid",
               "slippage_cost", "total_costs", "n_trades", "avg_exposure", "turnover_annual",
               "win_rate", "final_equity", "initial_equity"]


def _epoch(times) -> list[int]:
    return (pd.DatetimeIndex(times).astype("int64") // 1_000_000_000).tolist()


def _clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _r(a: np.ndarray, nd: int = 4) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, dtype=float)]


def _fills(df: pd.DataFrame, coins: list[str]) -> list[dict]:
    out = []
    for rec in df.itertuples():
        out.append({
            "t": int(pd.Timestamp(rec.time).value // 1_000_000_000),
            "coin": coins[int(rec.asset)], "side": rec.side, "price": float(rec.price),
            "qty": float(rec.qty), "notional": round(float(rec.qty * rec.price), 2), "fee": round(float(rec.fee), 2),
        })
    return out


def build_report_data(run_dir, meta, fs, panel, start, end, strat, free, bh, decisions, forecaster, metrics,
                      price_step: int = 15, n_cones: int = 160, rl: dict | None = None) -> dict:
    coins = [s.replace("USDT", "") for s in fs.symbols]
    results_path = Path(run_dir) / "results.json"
    history_path = Path(run_dir) / "history.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    history = json.loads(history_path.read_text()).get("history", []) if history_path.exists() else []

    # Equity and exposure, sampled every price_step minutes.
    eq = pd.DataFrame({"strategy": strat.equity, "no_costs": free.equity, "buy_and_hold": bh.equity})
    eq = eq.iloc[::price_step]
    exposure = strat.exposure.reindex(eq.index, method="ffill").fillna(0.0)

    # Prices for every coin on the same grid.
    rows = np.arange(start, end, price_step)
    prices = {c: _r(panel.close[rows, j], 8) for j, c in enumerate(coins)}

    fills = _fills(strat.fills, coins) if len(strat.fills) else []

    # Forecast cones at evenly spaced times, with what actually happened afterwards.
    taus = np.array([5, 15, 30, 60, 120, 240, 480, 720, 1440])
    cone_rows = np.linspace(start, end - 2, n_cones).astype(np.int64)
    fc = forecaster.predict(fs, cone_rows, taus)
    lvl_idx = [forecaster.quantiles.index(q) for q in CONE_LEVELS]
    q = fc["quantiles"][..., lvl_idx]                                    # [R, K, N, 5] log returns
    lc = np.asarray(fs.log_close)
    after = np.minimum(cone_rows[:, None] + taus[None, :], end - 1)       # stay inside the period
    realized = lc[after] - lc[cone_rows][:, None, :]                       # [R, K, N]
    realized[(cone_rows[:, None] + taus[None, :]) >= end] = np.nan
    cones = {"taus": taus.tolist(), "levels": CONE_LEVELS, "times": _epoch(fs.times[cone_rows]), "coins": {}}
    for j, c in enumerate(coins):
        cq = np.expm1(q[:, :, j, :]) * 100
        cr = np.expm1(realized[:, :, j]) * 100
        ready = fc["ready"][:, j]
        cones["coins"][c] = {
            "q": [[_r(cq[r, k], 3) for k in range(len(taus))] if ready[r] else None for r in range(len(cone_rows))],
            "realized": [_r(cr[r], 3) for r in range(len(cone_rows))],
            "price": _r(panel.close[cone_rows, j], 8),
        }

    test = results.get("test", {})
    eval_rows = []
    for h, r in sorted(test.get("horizons", {}).items(), key=lambda kv: int(kv[0])):
        cov = r.get("coverage", [])
        eval_rows.append({"h": int(h), "skill": _clean(r["skill"]), "ic": _clean(r["rank_ic"]), "hit": _clean(r["hit_rate"]),
                          "cov05": cov[0] if cov else None, "cov50": cov[len(cov) // 2] if cov else None,
                          "cov95": cov[-1] if cov else None})
    hist = [{"step": h["step"], "train": h["train_loss"], "val_skill": h["val"]["skill"]} for h in history]

    rl_block = None
    if rl is not None:
        m0 = rl["metrics"].get("initial_equity")
        rl_eq = rl["equity"].reindex(eq.index, method="ffill").fillna(m0)
        rl_block = {"label": f"RL agent, stage {rl.get('stage') or '?'}", "decision_every": rl.get("decision_every"),
                    "metrics": {m: _clean(rl["metrics"].get(m)) for m in METRIC_KEYS},
                    "equity": _r(rl_eq.to_numpy(), 2),
                    "fills": _fills(rl["fills"], coins) if len(rl["fills"]) else []}

    sim = metrics["sim_config"]
    return {
        "rl": rl_block,
        "run": {"name": Path(run_dir).name, "params": results.get("n_params"), "best_step": results.get("best_step"),
                "test_skill": _clean(test.get("skill")), "train_period": [meta["splits"]["train"]["start"], meta["splits"]["train"]["end"]],
                "d_model": meta["model_config"]["d_model"], "layers": meta["model_config"]["n_layers"]},
        "period": metrics["period"],
        "split": metrics["split"],
        "costs": {"taker_bps": sim["taker_fee_bps"], "maker_bps": sim["maker_fee_bps"],
                  "half_spread_bps": sim["half_spread_bps"], "impact_coef": sim["impact_coef"],
                  "participation_cap": sim["participation_cap"], "decision_every": metrics["decision_every"]},
        "metrics": {k: {m: _clean(v.get(m)) for m in METRIC_KEYS} for k, v in
                    (("strategy", metrics["strategy"]), ("no_costs", metrics["strategy_no_costs"]), ("buy_and_hold", metrics["buy_and_hold"]))},
        "equity": {"t": _epoch(eq.index), "strategy": _r(eq["strategy"], 2), "no_costs": _r(eq["no_costs"], 2),
                   "buy_and_hold": _r(eq["buy_and_hold"], 2), "exposure": _r(exposure.to_numpy(), 4)},
        "coins": coins,
        "prices": {"t": _epoch(fs.times[rows]), **prices},
        "fills": fills,
        "cones": cones,
        "decisions": decisions[-4000:],
        "decisions_total": len(decisions),
        "model_eval": {"test": eval_rows, "history": hist},
    }


def render(data: dict, standalone: bool = True) -> str:
    """Fill the template. `standalone` adds the HTML document wrapper for opening the file locally."""
    body = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    if not standalone:
        return body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')


def write_report(data: dict, path: Path, standalone: bool = True) -> Path:
    """Write the report. Also writes `<name>.body.html` without the document wrapper (for publishing)."""
    path = Path(path)
    path.write_text(render(data, standalone))
    path.with_suffix(".body.html").write_text(render(data, standalone=False))
    return path

"""Train the RL decision layer on top of Mark I, in two stages.

Stage 1 (frozen): Mark I's forecasts are precomputed; PPO trains the allocation policy in our
simulator on random 3-day episodes from the training period.

Stage 2 (fused): Mark I is placed inside the policy and fine-tuned together with it. Policy
gradients teach the forecaster what matters for profit; a quantile loss on real future
returns (the anchor) keeps it a calibrated forecaster. Its gradients are scaled down so it
moves slowly.

After each evaluation the policy trades the whole validation period once, deterministically,
and is compared with the rule engine and buy-and-hold on the same period and costs. The best
policy by validation Sharpe is kept and finally scored on the test period.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from mark1.backtest import DECISION_TAUS, _aligned_panel, _strategy, one_way_cost
from mark1.decide.engine import DecisionConfig, DecisionEngine
from mark1.features.build import load_features
from mark1.model.dataset import Normalizer
from mark1.model.forecaster import Forecaster
from mark1.model.train import load_meta, resolve_device
from mark1.rl.env import EnvConfig, Forecasts, TradingEnv, round_trip_cost
from mark1.rl.policy import AllocationPolicy, FusedForecaster, PolicyConfig
from mark1.rl.ppo import PPO, PPOConfig
from mark1.sim import SimConfig, buy_and_hold_benchmark, compute_metrics, run_backtest

log = logging.getLogger(__name__)


@dataclass
class RLConfig:
    run_dir: str                       # trained Mark I run
    out_dir: str | None = None         # defaults to <run_dir>/rl_stage<stage>
    stage: int = 1
    init_from: str | None = None       # stage 2: path to the stage-1 policy checkpoint
    total_steps: int = 200_000
    eval_every: int = 5                # PPO iterations between validation runs
    device: str = "auto"
    seed: int = 0
    forecaster_lr_mult: float = 0.1    # stage 2
    aux_coef: float = 1.0              # stage 2: weight of the quantile anchor
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(n_envs=8, n_steps=128, minibatch_size=256,
                                                              reward_scale=100.0, ent_coef=0.0, lr=3e-4))
    d_model: int = 64
    n_layers: int = 2


class Setup:
    """Everything shared by training, evaluation and the dashboard."""

    def __init__(self, run_dir: Path, decision_every: int, device: str = "auto"):
        self.run_dir = Path(run_dir)
        self.meta = load_meta(self.run_dir)
        self.fs = load_features(Path(self.meta["train_config"]["dataset_dir"]), mmap=True)
        self.panel = _aligned_panel(self.fs)
        self.splits = {k: (int(v["start"]), int(v["end"])) for k, v in self.meta["splits"].items()}
        self.forecaster = Forecaster.load(self.run_dir, device)
        self.sigma = np.asarray(self.fs.sigma)
        self.every = decision_every
        self._fc: dict[str, Forecasts] = {}

    def forecasts(self, split: str) -> Forecasts:
        if split not in self._fc:
            s, e = self.splits[split]
            out = self.forecaster.predict_grid(self.fs, self.run_dir, s, e, self.every, DECISION_TAUS)
            self._fc[split] = Forecasts(out["rows"], out["quantiles"], out["ready"], out["taus"], self.forecaster.quantiles)
        return self._fc[split]

    def env(self, split: str, sim_cfg: SimConfig, env_cfg: EnvConfig, seed: int = 0, full: bool = False) -> TradingEnv:
        s, e = self.splits[split]
        return TradingEnv(self.panel, self.forecasts(split), self.sigma, s, e, sim_cfg, env_cfg, seed=seed, full_period=full)


def policy_config(env: TradingEnv, d_model: int, n_layers: int) -> PolicyConfig:
    return PolicyConfig(n_assets=env.n, n_taus=len(env.fc.taus), n_quantiles=env.fc.quantiles.shape[-1],
                        n_coin_features=TradingEnv.N_COIN_FEATURES, n_portfolio_features=TradingEnv.N_PORTFOLIO_FEATURES,
                        d_model=d_model, n_layers=n_layers)


@torch.no_grad()
def run_policy_period(policy: AllocationPolicy, env: TradingEnv, device: torch.device) -> dict:
    """Trade one full period deterministically. Returns metrics and the equity series."""
    policy.eval()
    obs = env.reset()
    done = False
    while not done:
        batch = {k: torch.as_tensor(v[None]).to(device) for k, v in obs.items()}
        dist, _ = policy(batch)
        obs, _, done, _ = env.step(dist.mode[0].cpu().numpy())
    policy.train()
    sim = env.sim
    st = sim.stats()
    eq = sim.equity_series()
    m = compute_metrics(eq, exposure=sim.exposure_series(), fees_paid=st["fees_paid"], slippage_cost=st["slippage_cost"],
                        traded_notional=st["traded_notional"], n_trades=st["n_fills"])
    return {"metrics": m, "equity": eq, "fills": sim.fills_frame()}


def benchmark_period(setup: Setup, split: str, sim_cfg: SimConfig) -> dict:
    """Rule engine and buy-and-hold on the same period, decision grid and costs."""
    s, e = setup.splits[split]
    fc = setup.forecasts(split)
    fcd = {"quantiles": fc.quantiles, "ready": fc.ready, "taus": fc.taus}
    cost = one_way_cost(sim_cfg, setup.panel.n_assets)
    strat = _strategy(DecisionEngine(DecisionConfig()), fc.rows, fcd, fc.levels, cost, setup.fs.symbols, setup.fs.times, None)
    t0 = int(fc.rows[0])
    rules = run_backtest(setup.panel, strat, sim_cfg, t0, e, setup.every)
    bh = buy_and_hold_benchmark(setup.panel, sim_cfg, t0, e)
    return {"rules": rules, "buy_and_hold": bh}


def _brief(m: dict) -> str:
    return (f"return {100 * m['total_return']:+.2f}%  sharpe {m['sharpe']:+.2f}  maxdd {100 * m['max_drawdown']:.1f}%  "
            f"costs ${m['total_costs']:,.0f}  fills {m['n_trades']}")


def train_rl(cfg: RLConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.ppo.device = str(device)
    cfg.ppo.seed = cfg.seed
    out_dir = Path(cfg.out_dir or Path(cfg.run_dir) / f"rl_stage{cfg.stage}")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup = Setup(Path(cfg.run_dir), cfg.env.decision_every, str(device))
    sim_cfg = SimConfig()
    for split in ("train", "val", "test"):          # precompute (and cache) Mark I forecasts up front
        log.info("forecasts for %s: %d decision times", split, len(setup.forecasts(split).rows))
    if device.type == "cpu" and cfg.stage == 1:
        torch.set_num_threads(1)                      # small policy nets run much faster single-threaded

    probe = setup.env("train", sim_cfg, cfg.env)
    pcfg = policy_config(probe, cfg.d_model, cfg.n_layers)

    fused = None
    extra = None
    aux = None
    if cfg.stage == 2:
        model = setup.forecaster.model
        model.train()
        norm = Normalizer.from_dict(setup.meta["normalizer"])
        rt = round_trip_cost(sim_cfg, setup.panel.n_assets, cfg.env.impact_estimate_bps)
        fused = FusedForecaster(model, setup.fs, norm, setup.meta["model_config"]["views"], np.asarray(DECISION_TAUS),
                                rt, label_end=setup.splits["train"][1], lr_mult=cfg.forecaster_lr_mult).to(device)
        aux = lambda obs: cfg.aux_coef * fused.aux_loss(obs)
    policy = AllocationPolicy(pcfg, fused).to(device)
    if cfg.init_from:
        state = torch.load(cfg.init_from, map_location=device, weights_only=True)
        missing, unexpected = policy.load_state_dict(state["policy"], strict=False)
        log.info("initialised policy from %s (missing %d, unexpected %d keys)", cfg.init_from, len(missing), len(unexpected))

    make_env = lambda i: setup.env("train", sim_cfg, cfg.env, seed=cfg.seed * 1000 + i)
    ppo = PPO(policy, make_env, cfg.ppo, aux_loss_fn=aux, extra_params=extra)

    bench_val = benchmark_period(setup, "val", sim_cfg)
    log.info("VAL rules        %s", _brief(bench_val["rules"].metrics))
    log.info("VAL buy and hold %s", _brief(bench_val["buy_and_hold"].metrics))
    val_env = setup.env("val", sim_cfg, cfg.env, full=True)

    best = {"sharpe": -np.inf, "iteration": 0}
    evals = []

    def callback(it: int, stats: dict):
        if it % cfg.eval_every != 0:
            return None
        res = run_policy_period(policy, val_env, device)
        m = res["metrics"]
        log.info("iter %d  VAL rl %s", it, _brief(m))
        evals.append({"iteration": it, "val": m, "train": {k: v for k, v in stats.items() if isinstance(v, (int, float))}})
        if np.isfinite(m["sharpe"]) and m["sharpe"] > best["sharpe"]:
            best.update(sharpe=m["sharpe"], iteration=it)
            torch.save({"policy": policy.state_dict(), "policy_config": pcfg.to_dict(), "iteration": it}, out_dir / "policy.pt")
        (out_dir / "evals.json").write_text(json.dumps(evals, indent=2, default=float))
        return None

    ppo.learn(cfg.total_steps, callback=callback)
    callback(cfg.eval_every * 10**9, {})  # final evaluation (iteration number only used as a label)

    # Score the best policy on the untouched test period.
    state = torch.load(out_dir / "policy.pt", map_location=device, weights_only=True)
    policy.load_state_dict(state["policy"])
    test_env = setup.env("test", sim_cfg, cfg.env, full=True)
    rl_test = run_policy_period(policy, test_env, device)
    bench_test = benchmark_period(setup, "test", sim_cfg)
    summary = {
        "stage": cfg.stage, "best_iteration": best["iteration"], "best_val_sharpe": best["sharpe"],
        "val": {"rules": bench_val["rules"].metrics, "buy_and_hold": bench_val["buy_and_hold"].metrics},
        "test": {"rl": rl_test["metrics"], "rules": bench_test["rules"].metrics, "buy_and_hold": bench_test["buy_and_hold"].metrics},
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
    }
    for name, m in summary["test"].items():
        log.info("TEST %-14s %s", name, _brief(m))
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    rl_test["equity"].to_frame("rl").to_parquet(out_dir / "test_equity.parquet")
    rl_test["fills"].to_parquet(out_dir / "test_fills.parquet")
    return summary

"""Train the fused Mark I Agent in a single run: forecaster + all dimension pathways + RL policy.

    mark1 joint --preset gpu-12gb --out-dir runs/joint_v1
    mark1 joint --preset cpu-smoke --out-dir runs/joint_smoke

Everything starts from random weights and trains together. Every PPO update also optimises the
five per-dimension auxiliary tasks (see agent.py); their weight starts high, because early on
the trading reward is mostly noise and the encoder needs dense signal, and decays to a floor so
profit increasingly shapes the representation.

Regularisation against memorising: weight decay, dropout and input noise on the supervised rows,
random dimension hiding in the environment, label purging between splits, and keeping the
checkpoint with the best validation Sharpe.

Outputs in out_dir:
  agent.pt        the whole agent (best validation Sharpe)
  model.pt, meta.json, results.json, history.json
                  its forecaster in the standard Mark I format, so `mark1 evaluate` and
                  `mark1 backtest --run-dir out_dir --rl-policy out_dir/agent.pt` work on it
  joint_results.json  test-period trading vs buy and hold, forecast skill, dimension importance
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from mark1 import config
from mark1.backtest import _aligned_panel
from mark1.features.build import load_features
from mark1.model.dataset import Normalizer, WindowSampler, make_splits, save_json
from mark1.model.feature_store import TorchFeatureStore, pick_store_device
from mark1.model.network import ModelConfig
from mark1.model.train import Baseline, evaluate, format_report, resolve_device
from mark1.rl.agent import DIMENSIONS, AgentConfig, AuxWeights, DimensionBlind, Labels, MarkOneAgent
from mark1.rl.env import EnvConfig, TradingEnv, round_trip_cost
from mark1.rl.ppo import PPO, PPOConfig
from mark1.rl.train_rl import _brief, run_policy_period
from mark1.sim import SimConfig, buy_and_hold_benchmark

log = logging.getLogger(__name__)


@dataclass
class JointConfig:
    dataset_dir: str = str(config.DATASET_DIR)
    out_dir: str = str(config.RUNS_DIR / "joint")
    # forecaster / encoder
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_experts: int = 4
    dropout: float = 0.1
    use_asset_embedding: bool = True
    views: tuple[tuple[int, int], ...] = ((120, 1), (96, 15), (84, 120))
    # policy
    d_policy: int = 64
    n_policy_layers: int = 2
    # auxiliary tasks
    aux: AuxWeights = field(default_factory=AuxWeights)
    aux_start: float = 1.0
    aux_end: float = 0.3
    aux_decay_frac: float = 0.5
    sup_batch: int = 32
    sup_taus: int = 8
    input_noise: float = 0.05
    # environment and PPO
    env: EnvConfig = field(default_factory=lambda: EnvConfig(decision_every=15, episode_minutes=3 * 1440, dim_dropout=0.15))
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(n_envs=8, n_steps=128, minibatch_size=256, lr=3e-4, reward_scale=100.0,
                                                              max_grad_norm=1.0, weight_decay=0.01, target_kl=0.03))
    # run
    total_steps: int = 200_000
    eval_every: int = 5
    forecast_eval_batches: int = 10
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    gap: int = config.MAX_HORIZON
    device: str = "auto"
    store_device: str = "auto"
    seed: int = 0


PRESETS: dict[str, dict] = {
    # A few minutes on a laptop CPU: checks the whole loop, learns nothing useful.
    "cpu-smoke": dict(d_model=32, n_layers=1, n_heads=2, n_experts=2, d_policy=32, n_policy_layers=1,
                      sup_batch=8, total_steps=2048, eval_every=2, forecast_eval_batches=2,
                      ppo=dict(n_envs=4, n_steps=64, minibatch_size=64)),
    # RTX 5070 Ti / any 12 GB card, 32 GB RAM, 2021+ data. About 10M parameters, roughly 8 h.
    "gpu-12gb": dict(d_model=160, n_layers=3, n_heads=8, n_experts=6, dropout=0.15, d_policy=128, n_policy_layers=2,
                     sup_batch=32, input_noise=0.05, total_steps=5_000_000, eval_every=10, forecast_eval_batches=20,
                     ppo=dict(n_envs=32, n_steps=128, minibatch_size=128, lr=2e-4, weight_decay=0.05)),
    # Same model, shorter run (about 2-3 h) to check things before the overnight run.
    "gpu-12gb-quick": dict(d_model=160, n_layers=3, n_heads=8, n_experts=6, dropout=0.15, d_policy=128, n_policy_layers=2,
                           sup_batch=32, total_steps=1_000_000, eval_every=5, forecast_eval_batches=10,
                           ppo=dict(n_envs=32, n_steps=128, minibatch_size=128, lr=2e-4, weight_decay=0.05)),
}


def apply_preset(cfg: JointConfig, name: str) -> JointConfig:
    p = PRESETS[name]
    for k, v in p.items():
        if k == "ppo":
            for pk, pv in v.items():
                setattr(cfg.ppo, pk, pv)
        elif k == "env":
            for ek, ev in v.items():
                setattr(cfg.env, ek, ev)
        else:
            setattr(cfg, k, v)
    return cfg


def _aux_scale(progress: float, cfg: JointConfig) -> float:
    if cfg.aux_decay_frac <= 0:
        return cfg.aux_end
    f = min(1.0, progress / cfg.aux_decay_frac)
    return cfg.aux_start + (cfg.aux_end - cfg.aux_start) * f


class JointRun:
    """Everything a joint run needs, also used to reload a trained agent for backtests."""

    def __init__(self, cfg: JointConfig, device: torch.device, norm: Normalizer | None = None):
        self.cfg = cfg
        self.device = device
        self.fs = load_features(Path(cfg.dataset_dir), mmap=True)
        self.splits = make_splits(len(self.fs.times), cfg.views, cfg.fractions, cfg.gap)
        self.norm = norm or Normalizer.fit(self.fs, self.splits["train"], seed=cfg.seed)
        self.store = TorchFeatureStore(self.fs, self.norm, cfg.views, pick_store_device(self.fs, device, cfg.store_device))
        self.panel = _aligned_panel(self.fs)
        self.sim_cfg = SimConfig()
        self.sigma = np.asarray(self.fs.sigma)
        self.ready = np.asarray(self.fs.ready)
        self.round_trip = round_trip_cost(self.sim_cfg, self.panel.n_assets, cfg.env.impact_estimate_bps)

    def agent_config(self) -> AgentConfig:
        c = self.cfg
        mc = ModelConfig(n_asset_features=len(self.fs.asset_names), n_market_features=len(self.fs.market_names),
                         n_assets=len(self.fs.symbols), d_model=c.d_model, n_heads=c.n_heads, n_layers=c.n_layers,
                         n_experts=c.n_experts, dropout=c.dropout, use_asset_embedding=c.use_asset_embedding,
                         views=c.views, masked_coin_token=True)
        return AgentConfig(model=mc, d_policy=c.d_policy, n_policy_layers=c.n_policy_layers, aux=c.aux,
                           sup_batch=c.sup_batch, sup_taus=c.sup_taus, input_noise=c.input_noise)

    def env(self, split: str, seed: int = 0, full: bool = False, env_cfg: EnvConfig | None = None) -> TradingEnv:
        s = self.splits[split]
        return TradingEnv(self.panel, None, self.sigma, s.start, s.end, self.sim_cfg, env_cfg or self.cfg.env,
                          seed=seed, full_period=full, ready=self.ready)

    def buy_and_hold(self, split: str) -> dict:
        s = self.splits[split]
        first = self.env(split, full=True).grid[0]
        return buy_and_hold_benchmark(self.panel, self.sim_cfg, int(first), s.end).metrics


def dimension_importance(agent: MarkOneAgent, run: JointRun, split: str, full_metrics: dict) -> dict:
    """Trade the period with each dimension hidden; the drop in Sharpe and return is its importance."""
    out = {}
    for i, name in enumerate(DIMENSIONS):
        m = run_policy_period(DimensionBlind(agent, i), run.env(split, full=True), run.device)["metrics"]
        out[name] = {"sharpe_without": m["sharpe"], "return_without": m["total_return"],
                     "sharpe_drop": full_metrics["sharpe"] - m["sharpe"],
                     "return_drop": full_metrics["total_return"] - m["total_return"]}
        log.info("  without %-13s %s", name, _brief(m))
    return out


def _write_markone_files(agent: MarkOneAgent, run: JointRun, out_dir: Path, baseline: Baseline, step: int) -> None:
    """Save the agent's forecaster in the standard Mark I run format."""
    mc = agent.cfg.model
    meta = {
        "train_config": {"dataset_dir": run.cfg.dataset_dir, "views": [list(v) for v in run.cfg.views],
                         "eval_batches": run.cfg.forecast_eval_batches, "batch_size": 64,
                         "report_horizons": list(config.REPORT_HORIZONS)},
        "model_config": mc.to_dict(), "normalizer": run.norm.to_dict(), "baseline": baseline.to_dict(),
        "splits": {k: asdict(v) for k, v in run.splits.items()}, "symbols": run.fs.symbols,
        "asset_names": run.fs.asset_names, "market_names": run.fs.market_names, "joint": True,
    }
    save_json(meta, out_dir / "meta.json")
    torch.save({"model": agent.core.state_dict(), "step": step}, out_dir / "model.pt")


def save_agent(agent: MarkOneAgent, run: JointRun, path: Path, iteration: int) -> None:
    torch.save({"agent": agent.state_dict(), "agent_config": agent.cfg.to_dict(),
                "joint_config": json.loads(json.dumps(asdict(run.cfg), default=list)),
                "normalizer": run.norm.to_dict(), "iteration": iteration}, path)


def load_agent(path: Path, device: str = "auto") -> tuple[MarkOneAgent, JointRun]:
    """Rebuild a saved agent and its data plumbing (for backtests and the dashboard)."""
    dev = resolve_device(device)
    state = torch.load(path, map_location=dev, weights_only=False)
    jc = state["joint_config"]
    cfg = JointConfig(**{k: v for k, v in jc.items() if k not in ("aux", "env", "ppo", "views", "fractions")})
    cfg.views = tuple(tuple(v) for v in jc["views"])
    cfg.fractions = tuple(jc["fractions"])
    cfg.env = EnvConfig(**jc["env"])
    cfg.aux = AuxWeights(**jc["aux"])
    run = JointRun(cfg, dev, norm=Normalizer.from_dict(state["normalizer"]))
    agent = MarkOneAgent(AgentConfig.from_dict(state["agent_config"]), run.round_trip).to(dev)
    agent.load_state_dict(state["agent"])
    agent.attach(run.store, None, None, run.fs.asset_names)
    return agent.eval(), run


def train_joint(cfg: JointConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.ppo.device, cfg.ppo.seed = str(device), cfg.seed
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "joint_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=list))

    run = JointRun(cfg, device)
    for s in run.splits.values():
        log.info("%-5s %s -> %s", s.name, np.datetime_as_string(run.fs.times[s.start], "m"), np.datetime_as_string(run.fs.times[s.end - 1], "m"))
    train = run.splits["train"]
    labels = Labels(run.fs, label_end=train.end)
    train_rows = np.arange(train.start, train.end - config.MAX_HORIZON)
    train_rows = train_rows[np.asarray(run.fs.ready[train_rows]).sum(axis=1) >= 2]

    acfg = run.agent_config()
    agent = MarkOneAgent(acfg, run.round_trip).to(device).attach(run.store, labels, train_rows, run.fs.asset_names, cfg.seed)
    n_params = sum(p.numel() for p in agent.parameters())
    log.info("Mark I Agent: %.2fM parameters (forecaster %.2fM)", n_params / 1e6, sum(p.numel() for p in agent.core.parameters()) / 1e6)

    # Forecast-quality yardstick (volatility-only baseline) and fixed validation batches.
    train_sampler = WindowSampler(run.fs, run.norm, cfg.views, train, seed=cfg.seed)
    quantiles = list(acfg.model.quantiles)
    baseline = Baseline.fit(train_sampler, quantiles, n_batches=20)
    val_fc = WindowSampler(run.fs, run.norm, cfg.views, run.splits["val"]).fixed_batches(cfg.forecast_eval_batches, 64, list(config.REPORT_HORIZONS))

    make_env = lambda i: run.env("train", seed=cfg.seed * 1000 + i)
    ppo = PPO(agent, make_env, cfg.ppo, aux_loss_fn=agent.aux_total)
    iters_total = max(1, cfg.total_steps // (cfg.ppo.n_envs * cfg.ppo.n_steps))

    bh_val = run.buy_and_hold("val")
    log.info("VAL buy and hold %s", _brief(bh_val))
    val_env = run.env("val", full=True)
    best = {"sharpe": -np.inf, "iteration": 0}
    history, evals, t0 = [], [], time.time()

    def evaluate_now(it: int, stats: dict, final: bool = False) -> None:
        res = run_policy_period(agent, val_env, device)
        m = res["metrics"]
        fc = evaluate(agent.core, val_fc, baseline, device, quantiles)
        log.info("iter %d%s  VAL agent %s  forecast skill %+.2f%%  (%.0f min elapsed)", it, " final" if final else "",
                 _brief(m), 100 * fc["skill"], (time.time() - t0) / 60)
        evals.append({"iteration": it, "val": m, "forecast_skill": fc["skill"], "aux": dict(agent.last_aux)})
        history.append({"step": it, "train_loss": agent.last_aux.get("forecast", float("nan")), "val": fc})
        if np.isfinite(m["sharpe"]) and m["sharpe"] > best["sharpe"]:
            best.update(sharpe=m["sharpe"], iteration=it)
            save_agent(agent, run, out_dir / "agent.pt", it)
            _write_markone_files(agent, run, out_dir, baseline, it)
        save_json({"history": history}, out_dir / "history.json")
        (out_dir / "evals.json").write_text(json.dumps(evals, indent=2, default=float))

    last = [0]

    def callback(it: int, stats: dict):
        last[0] = it
        agent.aux_scale = _aux_scale(it / iters_total, cfg)
        if it % max(1, cfg.eval_every // 2 or 1) == 0:
            a = agent.last_aux
            log.info("iter %d  aux x%.2f  own_vol %.3f  rank %.3f  market %.3f  hidden %.3f  forecast %.3f", it, agent.aux_scale,
                     a.get("own_vol", np.nan), a.get("rank", np.nan), a.get("market", np.nan), a.get("hidden_coin", np.nan), a.get("forecast", np.nan))
        if it % cfg.eval_every == 0:
            evaluate_now(it, stats)
        return None

    agent.aux_scale = cfg.aux_start
    ppo.learn(cfg.total_steps, callback=callback)
    if last[0] % cfg.eval_every != 0:
        evaluate_now(last[0], {}, final=True)

    # Final scoring of the best checkpoint on the untouched test period.
    state = torch.load(out_dir / "agent.pt", map_location=device, weights_only=False)
    agent.load_state_dict(state["agent"])
    agent.eval()
    test_res = run_policy_period(agent, run.env("test", full=True), device)
    bh_test = run.buy_and_hold("test")
    test_fc = WindowSampler(run.fs, run.norm, cfg.views, run.splits["test"]).fixed_batches(cfg.forecast_eval_batches, 64, list(config.REPORT_HORIZONS))
    fc_rep = evaluate(agent.core, test_fc, baseline, device, quantiles)
    log.info("TEST agent        %s", _brief(test_res["metrics"]))
    log.info("TEST buy and hold %s", _brief(bh_test))
    log.info("%s", format_report(fc_rep, "TEST forecast"))
    log.info("dimension importance on the test period (Sharpe and return drop when hidden):")
    importance = dimension_importance(agent, run, "test", test_res["metrics"])

    save_json({"best_step": best["iteration"], "best_val_loss": float("nan"), "test": fc_rep, "n_params": n_params}, out_dir / "results.json")
    summary = {"best_iteration": best["iteration"], "best_val_sharpe": best["sharpe"], "n_params": n_params,
               "test": {"agent": test_res["metrics"], "buy_and_hold": bh_test, "forecast_skill": fc_rep["skill"]},
               "val_buy_and_hold": bh_val, "dimension_importance_test": importance}
    (out_dir / "joint_results.json").write_text(json.dumps(summary, indent=2, default=float))
    test_res["equity"].to_frame("agent").to_parquet(out_dir / "test_equity.parquet")
    return summary

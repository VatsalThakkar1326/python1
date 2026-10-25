"""Command line for Mark I.

    mark1 download  --start 2021-01-01 --end 2026-10-24
    mark1 download-extras --start 2021-01-01 --end 2026-10-24
    mark1 build     --start 2021-01-01
    mark1 train     --run-dir runs/v1 --steps 50000 --d-model 128
    mark1 evaluate  --run-dir runs/v1
    mark1 backtest  --run-dir runs/v1
    mark1 joint     --preset gpu-12gb --out-dir runs/joint_v1
    mark1 rl        --run-dir runs/v1 --stage 1
    mark1 rl        --run-dir runs/v1 --stage 2 --init-from runs/v1/rl_stage1/policy.pt
    mark1 info
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from mark1 import config


def _yesterday() -> str:
    return str(datetime.now(timezone.utc).date() - timedelta(days=1))


def cmd_download(a: argparse.Namespace) -> None:
    from mark1.data.binance_archive import coverage, download_universe
    download_universe(a.symbols, date.fromisoformat(a.start), date.fromisoformat(a.end), workers=a.workers)
    print(coverage().to_string(index=False))


def cmd_download_extras(a: argparse.Namespace) -> None:
    from mark1.data.extras import download_extras, extras_coverage
    try:
        download_extras(a.symbols, date.fromisoformat(a.start), date.fromisoformat(a.end), workers=a.workers, kinds=a.kinds)
    finally:  # show what is on disk even if some downloads failed; re-running resumes from the zip cache
        cov = extras_coverage()
        print(cov.to_string(index=False) if len(cov) else "No extras on disk yet.")


def cmd_build(a: argparse.Namespace) -> None:
    from mark1.data.store import load_panel
    from mark1.features.build import build_features, save_features
    from mark1.features.extras import has_extras
    panel = load_panel(a.symbols, a.start, a.end)
    logging.info("panel: %d minutes x %d coins", panel.n_times, panel.n_assets)
    use = a.extras == "on" or (a.extras == "auto" and has_extras(config.EXTRAS_DIR))
    if a.extras == "on" and not has_extras(config.EXTRAS_DIR):
        raise SystemExit(f"No extras in {config.EXTRAS_DIR}. Run `mark1 download-extras` first.")
    logging.info("extra dimensions (positioning, options vol, macro, events): %s", "on" if use else "off")
    fs = build_features(panel, extras_dir=config.EXTRAS_DIR if use else None)
    save_features(fs, Path(a.out))
    logging.info("features saved to %s (%d asset + %d market features)", a.out, len(fs.asset_names), len(fs.market_names))


def cmd_train(a: argparse.Namespace) -> None:
    from mark1.model.train import TrainConfig, train
    names = {f.name for f in fields(TrainConfig)}
    overrides = {k: v for k, v in vars(a).items() if k in names and v is not None}
    train(TrainConfig(**overrides))


def cmd_evaluate(a: argparse.Namespace) -> None:
    from mark1.model.train import evaluate_run
    evaluate_run(Path(a.run_dir), split=a.split, device=a.device, eval_batches=a.eval_batches)


def cmd_backtest(a: argparse.Namespace) -> None:
    from mark1.backtest import run_model_backtest
    run_model_backtest(Path(a.run_dir), split=a.split, decision_every=a.decision_every,
                       taker_fee_bps=a.taker_fee_bps, maker_fee_bps=a.maker_fee_bps,
                       rl_policy=Path(a.rl_policy) if a.rl_policy else None)


def cmd_rl(a: argparse.Namespace) -> None:
    from mark1.rl.env import EnvConfig
    from mark1.rl.ppo import PPOConfig
    from mark1.rl.train_rl import RLConfig, train_rl
    env = EnvConfig(decision_every=a.decision_every, episode_minutes=a.episode_days * 1440)
    ppo = PPOConfig(n_envs=a.n_envs, n_steps=a.n_steps, minibatch_size=a.minibatch_size, lr=a.lr,
                    reward_scale=a.reward_scale, ent_coef=a.ent_coef)
    train_rl(RLConfig(run_dir=a.run_dir, out_dir=a.out_dir, stage=a.stage, init_from=a.init_from,
                      total_steps=a.total_steps, eval_every=a.eval_every, device=a.device, seed=a.seed,
                      forecaster_lr_mult=a.forecaster_lr_mult, aux_coef=a.aux_coef, env=env, ppo=ppo,
                      d_model=a.d_model, n_layers=a.n_layers))


def cmd_joint(a: argparse.Namespace) -> None:
    from mark1.rl.joint import JointConfig, apply_preset, train_joint
    cfg = JointConfig()
    if a.preset:
        apply_preset(cfg, a.preset)
    for k in ("dataset_dir", "out_dir", "total_steps", "eval_every", "d_model", "n_layers", "n_heads", "n_experts",
              "dropout", "d_policy", "sup_batch", "device", "store_device", "seed"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    for k in ("n_envs", "n_steps", "minibatch_size", "lr", "weight_decay"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg.ppo, k, v)
    if a.dim_dropout is not None:
        cfg.env.dim_dropout = a.dim_dropout
    train_joint(cfg)


def cmd_info(a: argparse.Namespace) -> None:
    from mark1.data.binance_archive import coverage
    cov = coverage()
    print(cov.to_string(index=False) if len(cov) else "No data yet. Run `mark1 download`.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mark1", description="Mark I research CLI")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download 1-minute bars from the Binance archive")
    d.add_argument("--symbols", nargs="+", default=config.DEFAULT_UNIVERSE)
    d.add_argument("--start", default="2021-01-01")
    d.add_argument("--end", default=_yesterday())
    d.add_argument("--workers", type=int, default=8)
    d.set_defaults(fn=cmd_download)

    de = sub.add_parser("download-extras", help="download funding, futures metrics, premium, BVOL, context spot and FRED data")
    de.add_argument("--symbols", nargs="+", default=config.DEFAULT_UNIVERSE)
    de.add_argument("--start", default="2021-01-01")
    de.add_argument("--end", default=_yesterday())
    de.add_argument("--workers", type=int, default=8)
    de.add_argument("--kinds", nargs="+", default=None, choices=["funding", "metrics", "premium", "bvol", "spot", "fred"],
                    help="only these kinds (default: all)")
    de.set_defaults(fn=cmd_download_extras)

    b = sub.add_parser("build", help="build the feature dataset from downloaded bars")
    b.add_argument("--symbols", nargs="+", default=config.DEFAULT_UNIVERSE)
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--out", default=str(config.DATASET_DIR))
    b.add_argument("--extras", choices=["auto", "on", "off"], default="auto",
                   help="add positioning, options vol, macro and event features (auto: when downloaded)")
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train", help="train Mark I")
    t.add_argument("--dataset-dir", dest="dataset_dir")
    t.add_argument("--run-dir", dest="run_dir")
    t.add_argument("--steps", type=int)
    t.add_argument("--batch-size", dest="batch_size", type=int)
    t.add_argument("--n-tau", dest="n_tau", type=int)
    t.add_argument("--d-model", dest="d_model", type=int)
    t.add_argument("--n-layers", dest="n_layers", type=int)
    t.add_argument("--n-heads", dest="n_heads", type=int)
    t.add_argument("--n-experts", dest="n_experts", type=int)
    t.add_argument("--dropout", type=float)
    t.add_argument("--lr", type=float)
    t.add_argument("--eval-every", dest="eval_every", type=int)
    t.add_argument("--eval-batches", dest="eval_batches", type=int)
    t.add_argument("--num-workers", dest="num_workers", type=int)
    t.add_argument("--device")
    t.add_argument("--seed", type=int)
    t.set_defaults(fn=cmd_train)

    ev = sub.add_parser("evaluate", help="score a saved checkpoint on the val or test period")
    ev.add_argument("--run-dir", required=True)
    ev.add_argument("--split", default="test", choices=["val", "test"])
    ev.add_argument("--device", default="auto")
    ev.add_argument("--eval-batches", dest="eval_batches", type=int, default=None)
    ev.set_defaults(fn=cmd_evaluate)

    bt = sub.add_parser("backtest", help="run a trained model through the simulator")
    bt.add_argument("--run-dir", required=True)
    bt.add_argument("--split", default="test", choices=["val", "test"])
    bt.add_argument("--decision-every", type=int, default=5)
    bt.add_argument("--taker-fee-bps", type=float, default=None)
    bt.add_argument("--maker-fee-bps", type=float, default=None)
    bt.add_argument("--rl-policy", default=None, help="also trade the period with this RL policy.pt and show it in the report")
    bt.set_defaults(fn=cmd_backtest)

    r = sub.add_parser("rl", help="train the RL decision layer on top of a trained Mark I run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--out-dir", default=None)
    r.add_argument("--stage", type=int, default=1, choices=[1, 2], help="1: frozen Mark I, 2: fused fine-tune")
    r.add_argument("--init-from", default=None, help="stage 2: stage-1 policy.pt to start from")
    r.add_argument("--total-steps", type=int, default=200_000)
    r.add_argument("--decision-every", type=int, default=15)
    r.add_argument("--episode-days", type=int, default=3)
    r.add_argument("--n-envs", type=int, default=8)
    r.add_argument("--n-steps", type=int, default=128)
    r.add_argument("--minibatch-size", type=int, default=256)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--reward-scale", type=float, default=100.0)
    r.add_argument("--ent-coef", type=float, default=0.0)
    r.add_argument("--eval-every", type=int, default=5)
    r.add_argument("--d-model", type=int, default=64)
    r.add_argument("--n-layers", type=int, default=2)
    r.add_argument("--forecaster-lr-mult", type=float, default=0.1)
    r.add_argument("--aux-coef", type=float, default=1.0)
    r.add_argument("--device", default="auto")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_rl)

    j = sub.add_parser("joint", help="train forecaster + RL policy together in one run (the fused Mark I Agent)")
    j.add_argument("--preset", choices=["cpu-smoke", "gpu-12gb", "gpu-12gb-quick"], default=None)
    j.add_argument("--dataset-dir", dest="dataset_dir")
    j.add_argument("--out-dir", dest="out_dir")
    j.add_argument("--total-steps", dest="total_steps", type=int)
    j.add_argument("--eval-every", dest="eval_every", type=int)
    j.add_argument("--d-model", dest="d_model", type=int)
    j.add_argument("--n-layers", dest="n_layers", type=int)
    j.add_argument("--n-heads", dest="n_heads", type=int)
    j.add_argument("--n-experts", dest="n_experts", type=int)
    j.add_argument("--dropout", type=float)
    j.add_argument("--d-policy", dest="d_policy", type=int)
    j.add_argument("--sup-batch", dest="sup_batch", type=int)
    j.add_argument("--n-envs", dest="n_envs", type=int)
    j.add_argument("--n-steps", dest="n_steps", type=int)
    j.add_argument("--minibatch-size", dest="minibatch_size", type=int)
    j.add_argument("--lr", type=float)
    j.add_argument("--weight-decay", dest="weight_decay", type=float)
    j.add_argument("--dim-dropout", dest="dim_dropout", type=float)
    j.add_argument("--device")
    j.add_argument("--store-device", dest="store_device")
    j.add_argument("--seed", type=int)
    j.set_defaults(fn=cmd_joint)

    i = sub.add_parser("info", help="show downloaded data coverage")
    i.set_defaults(fn=cmd_info)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()

"""Command line for Mark I.

    mark1 download  --start 2021-01-01 --end 2026-10-24
    mark1 build     --start 2021-01-01
    mark1 train     --run-dir runs/v1 --steps 50000 --d-model 128
    mark1 backtest  --run-dir runs/v1
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


def cmd_build(a: argparse.Namespace) -> None:
    from mark1.data.store import load_panel
    from mark1.features.build import build_features, save_features
    panel = load_panel(a.symbols, a.start, a.end)
    logging.info("panel: %d minutes x %d coins", panel.n_times, panel.n_assets)
    fs = build_features(panel)
    save_features(fs, Path(a.out))
    logging.info("features saved to %s (%d asset + %d market features)", a.out, len(fs.asset_names), len(fs.market_names))


def cmd_train(a: argparse.Namespace) -> None:
    from mark1.model.train import TrainConfig, train
    names = {f.name for f in fields(TrainConfig)}
    overrides = {k: v for k, v in vars(a).items() if k in names and v is not None}
    train(TrainConfig(**overrides))


def cmd_backtest(a: argparse.Namespace) -> None:
    from mark1.backtest import run_model_backtest
    run_model_backtest(Path(a.run_dir), split=a.split, decision_every=a.decision_every,
                       taker_fee_bps=a.taker_fee_bps, maker_fee_bps=a.maker_fee_bps)


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

    b = sub.add_parser("build", help="build the feature dataset from downloaded bars")
    b.add_argument("--symbols", nargs="+", default=config.DEFAULT_UNIVERSE)
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--out", default=str(config.DATASET_DIR))
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

    bt = sub.add_parser("backtest", help="run a trained model through the simulator")
    bt.add_argument("--run-dir", required=True)
    bt.add_argument("--split", default="test", choices=["val", "test"])
    bt.add_argument("--decision-every", type=int, default=5)
    bt.add_argument("--taker-fee-bps", type=float, default=None)
    bt.add_argument("--maker-fee-bps", type=float, default=None)
    bt.set_defaults(fn=cmd_backtest)

    i = sub.add_parser("info", help="show downloaded data coverage")
    i.set_defaults(fn=cmd_info)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()

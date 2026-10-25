"""Project-wide defaults: paths, the trading universe, and forecast horizons."""

from __future__ import annotations

import os
from pathlib import Path

# Root for downloaded data and training outputs. Override with MARK1_HOME.
HOME = Path(os.environ.get("MARK1_HOME", Path.cwd()))
DATA_DIR = HOME / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded zip files from the Binance archive
BARS_DIR = DATA_DIR / "bars"        # one Parquet file per symbol, 1-minute bars
EXTRAS_DIR = DATA_DIR / "extras"    # extra market data: funding, futures metrics, premium, BVOL, context spot, FRED
DATASET_DIR = DATA_DIR / "dataset"  # aligned feature tensors ready for training
RUNS_DIR = HOME / "runs"            # checkpoints, metrics, backtest reports

# Liquid USDT spot pairs that have traded on Binance since at least early 2021.
# All of them have USD pairs on Coinbase and/or Kraken, where live trading would happen.
DEFAULT_UNIVERSE: list[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "TRXUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT",
    "ETCUSDT", "XLMUSDT", "NEARUSDT", "FILUSDT", "AAVEUSDT",
]

# Horizons (minutes) used for reporting and decisions. The model itself is trained on
# continuous horizons, so it can be queried at any value between MIN and MAX.
REPORT_HORIZONS: list[int] = [1, 5, 15, 30, 60, 120, 240, 480, 1440]
MIN_HORIZON = 1
MAX_HORIZON = 1440

# Quantile levels the model predicts for each horizon.
QUANTILES: list[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

MINUTES_PER_YEAR = 365 * 24 * 60  # crypto trades 24/7

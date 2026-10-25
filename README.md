# Mark I

A crypto trading research system built from scratch:

1. **Data**: 1-minute bars for 20 liquid coins from the public Binance archive (free, no account).
2. **Features**: 26 per-coin and 21 market features covering price action, volume, volatility,
   order flow (aggressive buyers vs sellers), other coins, market mood, and clock patterns.
   Built with no look-ahead (tested).
3. **Mark I model** (PyTorch): reads every coin at once and predicts, for **any horizon from
   1 minute to 1 day**, a range of likely future returns (7 quantiles) for every coin.
4. **Decision engine**: turns those forecast curves into buy / hold / sell / dip-exit / stop
   decisions, net of trading costs, with fractional-Kelly sizing and risk limits.
5. **Simulator**: our own minute-bar exchange simulator with fees, spread, slippage, market
   impact, latency, partial fills, limit and stop orders.

## How the model maps to the design

| Your dimension | Where it lives |
|---|---|
| 1. The coin's own history | Three causal TCN encoders per coin: last 2 hours at 1-min, last day at 15-min, last week at 2-hour steps |
| 2. Other coins | Self-attention across all coin tokens, plus cross-sectional rank features |
| 3. Market pattern | A market encoder (equal-weight market returns, breadth, dispersion, correlation, volume, order flow, clock) whose token picks a soft mix of regime "experts" |
| 4. Everything affecting everything | Stacked cross blocks; lead-lag attention lets each coin look at every coin's state 0/5/15/30 minutes ago |
| 5. Continuous multi-horizon prediction with confidence | Horizon decoder: ask for any tau in minutes, get monotone quantiles (5%...95%) of the volatility-scaled return |

## Setup on your machine (NVIDIA GPU)

Needs Python 3.11+ and an NVIDIA driver.

```bash
git clone https://github.com/VatsalThakkar1326/python1.git
cd python1
git checkout vatsal/keen-johnson-dolehf

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

# 1) PyTorch with CUDA: pick the command for your CUDA version at https://pytorch.org/get-started/locally/
#    e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2) The project
pip install -e ".[dev]"

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run it

```bash
# 1) Download 1-minute bars (20 coins, 2021 to yesterday: ~3 GB, ~15-30 min)
mark1 download --start 2021-01-01

# 2) Build features (~2.4M minutes x 20 coins; needs ~16 GB RAM, writes ~6 GB to data/dataset)
mark1 build --start 2021-01-01

# 3) Train (uses the GPU automatically). Start medium, then scale up.
mark1 train --run-dir runs/v1 --d-model 128 --n-layers 3 --batch-size 128 --steps 50000 --num-workers 6

# 4) Backtest the trained model through the simulator on the held-out test period
mark1 backtest --run-dir runs/v1
```

Short on RAM? Use fewer years (`--start 2024-01-01` for both download and build).

## Reading the training output

Every evaluation prints, per horizon:

- **skill%**: how much better the model's forecast ranges are than a baseline that only knows
  current volatility. 0 means it learned nothing beyond volatility; positive is real
  information. Expect small numbers (a few percent is good for markets).
- **rank_ic**: correlation between predicted and actual ranking of coins at the same time.
  0.02 to 0.05 is meaningful in finance.
- **hit%**: how often the predicted direction is right. 50% is a coin flip.
- **cov05 / cov50 / cov95**: calibration. Should be close to 0.05 / 0.50 / 0.95, which means
  "90% confident" really is right 90% of the time.

All numbers come from data the model never trained on (validation and a final test period,
separated by a 1-day gap so no future returns leak).

## Project layout

```
src/mark1/
  config.py              universe, horizons, quantiles, paths
  data/                  Binance archive downloader, minute-grid loader, Panel container
  features/              causal feature builder, continuous-horizon targets
  model/                 network, dataset/splits/normalisation, training, inference
  decide/                decision engine (forecast curves -> target weights)
  sim/                   our market simulator
  cli.py                 `mark1` command
tests/                   pytest suite (look-ahead checks included)
research_notes/, reports/  the background research behind the design
```

Run tests with `pytest`.

## Important caveats

- Trading costs dominate at minute level. BTC moves about 5 bps per minute while a retail
  round trip costs roughly 50-100 bps, so the decision engine only trades when the predicted
  move over some horizon beats that.
- Binance data is used for research only (Binance does not serve Canada). Live trading would
  go through Canada-registered venues such as Coinbase or Kraken; set the simulator's fees to
  the venue's actual tier.
- Nothing here is financial advice; treat every backtest as an upper bound until it survives
  live paper trading.

# Mark I

A crypto trading research system built from scratch:

1. **Data**: 1-minute bars for 20 liquid coins from the public Binance archive (free, no account).
2. **Features**: 26 per-coin and 21 market features covering price action, volume, volatility,
   order flow (aggressive buyers vs sellers), other coins, market mood, and clock patterns.
   Built with no look-ahead (tested).
3. **Mark I model** (PyTorch): reads every coin at once and predicts, for **any horizon from
   1 minute to 1 day**, a range of likely future returns (7 quantiles) for every coin.
4. **RL decision layer** (our own PPO): learns portfolio weights from Mark I's forecast curves
   and the portfolio state, rewarded by real equity growth after costs in the simulator.
   Hard safety rails (position caps, stop-loss, drawdown brake) wrap every decision.
5. **Rule-based decision engine**: the baseline the RL layer must beat. Cost-aware
   buy / hold / sell / dip-exit / stop rules with fractional-Kelly sizing.
6. **Simulator**: our own minute-bar exchange simulator with fees, spread, slippage, market
   impact, latency, partial fills, limit and stop orders.
7. **Dashboard**: every backtest writes an interactive `report.html`.

## How the model maps to the design

| Your dimension | Where it lives |
|---|---|
| 1. The coin's own history | Three causal TCN encoders per coin: last 2 hours at 1-min, last day at 15-min, last week at 2-hour steps |
| 2. Other coins | Self-attention across all coin tokens, plus cross-sectional rank features |
| 3. Market pattern | A market encoder (equal-weight market returns, breadth, dispersion, correlation, volume, order flow, clock) whose token picks a soft mix of regime "experts" |
| 4. Everything affecting everything | Stacked cross blocks; lead-lag attention lets each coin look at every coin's state 0/5/15/30 minutes ago |
| 5. Continuous multi-horizon prediction with confidence | Horizon decoder: ask for any tau in minutes, get monotone quantiles (5%...95%) of the volatility-scaled return |

## The fused Mark I Agent (one network, one training run)

`mark1 joint` trains the forecaster and the RL trading policy together, from scratch, in a
single run. Nothing is pretrained first.

| Dimension | What the policy reads directly | Auxiliary task that forces it to be learned |
|---|---|---|
| 1. Coin's own history | own-history pathway (before any cross-coin mixing) | predict the coin's own future volatility from that pathway alone |
| 2. Other coins | what attention over the other coins added | rank which coins will do best over the next 1 h and 4 h |
| 3. Market pattern | market token + regime experts' contribution | quantiles of the whole market's move at every horizon |
| 4. Butterfly effect | what lead-lag attention added (reactions 0-30 min later) | hide one coin's inputs and recover its recent moves from the others |
| 5. Multi-horizon forecast | quantile curves at 9 horizons, in units of trading cost | quantile loss on real future returns |

The coin representation is exactly the sum of pathways 1-4 (all blocks are residual), so each
group is a separate signal. During training the environment hides whole dimensions at random
(15% each by default), so the policy has to use all of them; at the end each dimension's
importance is measured by hiding it on the test period and recording the drop in Sharpe.
Auxiliary weights start high (dense signal while the trading reward is still noise) and decay
to 30% so that profit increasingly shapes the network.

Against memorising: model size scaled with data (about 10M parameters for 5+ years), weight
decay, dropout and input noise on supervised rows, random dimension hiding, purged splits,
and the best checkpoint picked by validation Sharpe.

```bash
mark1 joint --preset gpu-12gb-quick --out-dir runs/joint_quick   # ~2-3 h sanity run
mark1 joint --preset gpu-12gb --out-dir runs/joint_v1            # ~8 h overnight run
mark1 backtest --run-dir runs/joint_v1 --decision-every 15 --rl-policy runs/joint_v1/agent.pt
```

Out of GPU memory? Lower `--minibatch-size` (128 -> 64) or `--sup-batch` (32 -> 16). The
features sit on the GPU in float16 (about 2.7 GB for 5 years); `--store-device cpu` keeps them
in RAM instead, which is slower but frees VRAM. Results land in `runs/joint_v1/joint_results.json`.

## Setup on your machine (NVIDIA GPU)

Needs Python 3.11+ and an NVIDIA driver.

```bash
git clone https://github.com/VatsalThakkar1326/python1.git
cd python1
git checkout vatsal/keen-johnson-dolehf

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

# 1) PyTorch with CUDA. RTX 50-series (Blackwell, e.g. 5070 Ti) needs a CUDA 12.8+ build:
#    pip install torch --index-url https://download.pytorch.org/whl/cu128
#    (other cards: pick the command at https://pytorch.org/get-started/locally/)
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

# 4) Score the best checkpoint on the held-out test period
mark1 evaluate --run-dir runs/v1

# 5) Backtest model + rule engine through the simulator; opens as runs/v1/backtest_test/report.html
mark1 backtest --run-dir runs/v1 --decision-every 15

# 6) RL decision layer, stage 1: PPO on top of the frozen forecaster
mark1 rl --run-dir runs/v1 --stage 1 --total-steps 2000000 --n-envs 16 --n-steps 256 --d-model 128

# 7) RL stage 2: fuse forecaster and policy, fine-tune together (quantile loss keeps the forecaster honest)
mark1 rl --run-dir runs/v1 --stage 2 --init-from runs/v1/rl_stage1/policy.pt --total-steps 500000 \
  --lr 1e-4 --forecaster-lr-mult 0.05

# 8) Put the RL agent on the dashboard next to the rules engine and buy and hold
mark1 backtest --run-dir runs/v1 --decision-every 15 --rl-policy runs/v1/rl_stage2/policy.pt
```

Stage 2 moves the policy fast because every forecast feature shifts when Mark I updates, so it
uses a lower learning rate than stage 1 (the first CPU smoke run hit PPO's KL early stop at 3e-4).

RL results land in `runs/v1/rl_stage1/results.json`: the policy's test-period metrics next to the
rule engine and buy-and-hold on the same period and costs.

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

## How the RL layer is set up

| Piece | Choice |
|---|---|
| Observation | Mark I quantiles for every coin and horizon (in units of round-trip cost), each coin's weight, P&L and time held, cash, exposure, drawdown, time left |
| Action | N+1 logits (coins + cash), turned into long-only portfolio weights by a softmax |
| Reward | log equity growth per step after every fee, spread and impact cost, minus a drawdown penalty |
| Safety rails | per-coin cap, gross cap, stop-loss with cooldown, drawdown brake, minimum trade size; the agent cannot override them |
| Algorithm | our own PPO (clipped objective, GAE with time-limit bootstrapping, KL early stop) |
| Stage 1 | Mark I frozen; forecasts precomputed and cached |
| Stage 2 | Mark I inside the policy; policy gradients flow into it (slowed 10x) while a quantile loss on real future returns anchors it |
| Selection | best validation Sharpe over full-period deterministic runs, then one test-period score |

## Project layout

```
src/mark1/
  config.py              universe, horizons, quantiles, paths
  data/                  Binance archive downloader, minute-grid loader, Panel container
  features/              causal feature builder, continuous-horizon targets
  model/                 network, dataset/splits/normalisation, training, inference
  decide/                rule-based decision engine (forecast curves -> target weights)
  rl/                    PPO, trading environment, allocation policy, staged training
  sim/                   our market simulator
  backtest.py, report.py end-to-end backtest and the HTML dashboard
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

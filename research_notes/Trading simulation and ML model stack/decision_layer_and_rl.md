# Decision Layer for Forecast-Driven Trading (buy/sell/hold/stop, sizing, portfolio management) and RL Tooling in Python — state as of 2026-10-25

Method notes for the report writer:
- Release dates below are **PyPI "latest release" upload dates** pulled from the PyPI JSON API on 2026-10-25. GitHub last-commit dates could not be retrieved (the `gh`/GitHub API was blocked in this session and the GitHub HTML pages omit commit dates). A stale PyPI release does not prove a repo is dead (some repos ship from `master`), and a fresh one does not prove quality.
- Evidence quality is flagged inline. Several sources are search-engine summaries of abstracts rather than full texts; those are marked "(abstract/summary only)".
- The scope here is the decision layer, RL tooling, risk and architecture. Forecasting models and market simulators are covered by other researchers.

---

## 1. Approaches: (a) predict-then-optimize, (b) end-to-end deep RL, (c) hybrid (forecasts + uncertainty as RL state), (d) supervised / meta-labeling for bet sizing

### Takeaway
The best-documented, most robust pattern is a separated pipeline (forecast -> sizing/optimizer -> risk overlay -> execution) with a weight/position vector as the interface; end-to-end deep RL for alpha has weak, often poorly-tested evidence, and recent careful tests find learned policies roughly *match* simple rules net of costs. Hybrid designs (forecast as extra state) and meta-labeling are plausible and widely recommended, but I found little rigorous independent evidence for either in trading.

### Cited Findings
**(a) Predict-then-optimize / separated layers**
- Boyd et al. frame trading as a convex optimization each period that balances expected return, risk, transaction costs and holding costs (e.g., short borrow); the multi-period version plans a sequence of trades but executes only the first (receding horizon = model predictive control). The paper explicitly does not address forecasting; it treats the optimizer as a way to exploit predictions "however they are produced". — [Boyd et al., Multi-Period Trading via Convex Optimization (arXiv 1705.00109)](https://arxiv.org/pdf/1705.00109); [Stanford page](https://stanford.edu/~boyd/papers/cvx_portfolio.html)
- Gârleanu & Pedersen derive a closed-form optimal dynamic policy when trading is costly and returns are predictable by signals that mean-revert at different speeds: "aim in front of the target" and "trade partially towards the current aim"; the aim is a blend of the current Markowitz portfolio and those expected in the future, and slower-decaying signals get more weight. Applied to commodity futures, the strategy had higher net-of-cost returns than simpler benchmarks. — [NBER w15205](https://www.nber.org/papers/w15205)
- FinRL-X (AI4Finance, arXiv March 2026, accepted at PAKDD 2026 DMO-FinTech workshop) uses a "weight-centric" interface: "the target portfolio weight vector is the sole interface contract between strategy logic and downstream evaluation or execution modules," so backtest and live consume the same representation; it has four layers (data, strategy, backtesting, broker execution) and a volatility-aware (VIX-style) risk overlay that scales aggregate exposure while keeping relative allocations unchanged. — [FinRL-X paper](https://arxiv.org/html/2603.21330v1)
- An arXiv preprint on prediction-market agents argues for separating the forecasting layer from the trading ("belief-to-trade") layer; in its bootstrap results the full separated system had a positive ROI interval while a forecast-only policy's interval spanned zero. The source itself notes sizing/risk components are confounded with the separation (single paper, different domain). — [arXiv 2607.03015](https://arxiv.org/html/2607.03015v1)

**(b) End-to-end deep RL (state -> action)**
- Moody & Saffell (NIPS 1998) trained trading systems directly on financial objectives (profit, Sharpe, differential Sharpe); for a long/short trader, maximizing the differential Sharpe ratio gave more consistent results than maximizing profit, and both beat a forecast-based system trained to minimize MSE; data were monthly S&P 500 from 1970–1994. They argue optimal decisions under transaction costs depend on current system state, which motivates recurrent/RL learning. — [NIPS paper](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading); [differential Sharpe description](https://arxiv.org/pdf/2110.04745)
- A July 2026 walk-forward study of end-to-end *neural policy* allocation (trained on a differentiable negative Sharpe, i.e., not value-based RL) on 16 CME futures (2011–2024 OOS) found the best model (Portfolio Transformer) had cross-asset Sharpe 0.55 vs 0.52 for equal-weight 1/N; the paper concludes it "matches rather than significantly beats" 1/N (probability it beats 1/N only 0.59; beats TSMOM 0.68; beats risk parity 0.99). Only the equity-index results survived Bonferroni correction. The LSTM's net Sharpe fell from 0.50 to −0.38 as costs rose 0 -> 10 bp, while the transformer went 0.55 -> 0.50 (turnover ~0.02/day vs ~0.17/day). Mixture-of-experts, larger feature sets, ensembles and regime features did not reliably help. — [arXiv 2607.00475](https://arxiv.org/html/2607.00475v1)
- TensorTrade's own README reports, for a PPO agent on BTC/USD, test P&L of +$239 at 0% commission vs −$650 at 0.1% commission (buy-and-hold −$355) and states "The primary challenge is trading frequency—commission costs currently exceed prediction profits." — [tensortrade-org/tensortrade](https://github.com/tensortrade-org/tensortrade)
- A replication of Jiang et al.'s EIIE portfolio framework reportedly matched strong results on crypto but showed only average performance on stocks, with the agent converging to a UCRP-like (uniform constant rebalanced) strategy (secondary summary; the source itself contradicts itself on crypto details). — [arXiv 2409.08426](https://arxiv.org/pdf/2409.08426) via [AlphaXiv summary](https://www.alphaxiv.org/abs/2409.08426)

**(c) Hybrid (forecasts/uncertainty as state; RL + classical optimizer)**
- A 2026 energy-trading preprint found that adding a forecast to the DDQN state helped, but did not replace the need to give the agent current price context directly; the best state combined absolute, relative and forecast features (different domain; abstract-level). — [arXiv 2606.27032](https://arxiv.org/html/2606.27032)
- FinRL-X lists RL allocators as one swappable module inside a rule-based pipeline (stock selection -> allocation -> timing -> risk overlay), and in one use case a DRL allocator over the top 25% of NASDAQ-100 names with "timing" had Sharpe 0.89 vs 0.55 without timing; SPY Sharpe 0.84, QQQ 0.95 (backtest Jan 2018–Oct 2025, 10 bps/side). The paper's SPY/QQQ figures differ between its Table 2 and Table 3 without explanation, and the authors say results are "not intended to establish statistically significant alpha". — [FinRL-X paper](https://arxiv.org/html/2603.21330v1)
- A paper titled "Deep Reinforcement Learning and Convex Mean-Variance Optimisation for Portfolio Management" exists (arXiv 2203.11318) combining a DRL stage with convex optimization; I could not extract its results (PDF unreadable via fetch). — [arXiv 2203.11318](https://arxiv.org/pdf/2203.11318)
- A 2025 systematic review (arXiv 2512.10913) reports hybrid RL Sharpe 1.57 vs pure RL 1.35 vs traditional 0.95, **but the page itself states several of its figures come from a synthetic dataset built to mirror literature patterns, not from the original studies — do not rely on these numbers**. Its qualitative advice: prioritize implementation quality and domain knowledge over algorithm choice, use hybrids, monitor drift, use walk-forward validation. — [arXiv 2512.10913](https://arxiv.org/html/2512.10913v1)

**(d) Supervised / meta-labeling for bet sizing**
- Lopez de Prado's meta-labeling: a primary model picks the side; a secondary binary classifier (0 = pass, 1 = take the bet) decides whether to act, and its predicted probability can be used to scale position size; the book argues bet size should follow model confidence and favors log-loss over accuracy for tuning. — [Jesse docs on meta-labeling](https://docs.jesse.trade/docs/research/ml/meta-labeling); [summary of AFML](https://www.shortform.com/pdf/advances-in-financial-machine-learning-pdf-marcos-lopez-de-prado)
- A Journal of Financial Data Science paper "Meta-Labeling: Calibration and Position Sizing" tests uncalibrated vs calibrated probabilities across six sizing algorithms (including a new "sigmoid optimal position sizing"), judged by Sharpe and max drawdown; calibration helps fixed sizing rules noticeably, but methods that estimate sizing functions from training data see no significant gain (abstract only; author list not confirmed in my results). — [abstract as listed on a profile page](https://ae.linkedin.com/in/jacquesjoubert)
- A QuantConnect forum thread is titled "why meta labeling is not a silver bullet" (only the intro was seen). — [QuantConnect forum](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)

### Inferences
- For a solo developer, (a)+(d) are the same family: the forecaster's multi-horizon means/quantiles and calibrated confidence become inputs to an explicit, auditable sizing/optimization rule; (d) is just a learned calibration/gating step on top. These are debuggable; RL policies are not.
- End-to-end RL has to learn costs, risk, sizing and timing from noisy, low-signal data at once; the 2026 futures study and TensorTrade's README both suggest cost-awareness dominates outcomes, so putting costs/risk in explicit code (not in the reward) is lower-variance engineering.
- (c) is the only RL variant that matches the user's example ("hold today, sell tomorrow noon"), because it can learn *when to wait* given forecast term-structure plus position/cost state; but it should be a constrained overlay, not the alpha source.

### Gaps
- No rigorous, independent head-to-head of predict-then-optimize vs hybrid RL vs end-to-end RL under identical leakage-free protocol and realistic costs was found.
- No peer-reviewed 2024–2026 out-of-sample test of meta-labeling bet sizing against a clear benchmark was found (only the calibration/sizing abstract and blog-level material).
- The DRL + convex MVO paper's numbers were not retrievable.

---

## 2. RL environments / frameworks and their maintenance status (as of 2026-10-25)

### Takeaway
Most trading-RL environments are research/educational code with stale releases (2022–2024) and simplistic action/cost models; the AI4Finance ecosystem (FinRL, FinRL-Meta, ElegantRL) has stale PyPI releases and is explicitly redirecting users to FinRL-X for production; ABIDES-Gym is archived; only TensorTrade (PyPI Feb 2026) and the SB3/Ray/TorchRL/Gymnasium base libraries are actively released. For a solo developer, writing a small custom Gymnasium env is probably less risky than inheriting one of these.

### Cited Findings
**Maintenance snapshot (PyPI latest release unless noted)**

| Project | Latest PyPI release | Other status evidence | Source |
|---|---|---|---|
| FinRL | 0.3.7, 2024-04-12 | GitHub: 16.8k stars; README "Important" banner says repo "preserves the original end-to-end educational and research framework" and points to FinRL-X as "next-generation evolution"; its version history ends at 0.3.5 (2022-06-25); original has only "basic Alpaca support"; "Nothing herein constitutes financial advice" | [PyPI](https://pypi.org/pypi/finrl/json); [GitHub](https://github.com/AI4Finance-Foundation/FinRL) |
| FinRL-Meta | 0.3.6, 2023-02-07 | 2.0k stars; README directs production users to FinRL-X; no deprecation banner; data connectors incl. Alpaca (US, 1-min), Binance/CCXT (crypto), Yahoo (1-min, rate limited), WRDS, QuantConnect; README Python badge says 3.6 | [PyPI](https://pypi.org/pypi/finrl-meta/json); [GitHub](https://github.com/AI4Finance-Foundation/FinRL-Meta) |
| FinRL-X / FinRL-Trading | PyPI package named `finrl-trading` 2.0.2, 2025-11-23 (I did not verify that it is the official AI4Finance release) | arXiv:2603.21330 (March 2026); Alpaca execution, FMP/Yahoo/WRDS data, `bt` backtesting engine; SOTAVerified shows no independent reproductions | [PyPI](https://pypi.org/pypi/finrl-trading/json); [paper](https://arxiv.org/abs/2603.21330); [summary](https://sotaverified.org/papers/260321330) |
| ElegantRL | 0.3.6, 2023-02-07 | 4.4k stars; README says maintainers are "actively working on refactoring"; requirements list dated (Python 3.6+, PyTorch 1.6+, gym 0.17); supports DDPG/TD3/SAC/PPO/REDQ/DQN variants/MARL, Isaac Gym-style GPU vectorized envs; most recent dated material on page is a Nov 2022 tutorial | [PyPI](https://pypi.org/pypi/elegantrl/json); [GitHub](https://github.com/AI4Finance-Foundation/ElegantRL) |
| TensorTrade | 1.0.4, 2026-02-06 (Python >=3.12) | 7.2k stars, Apache-2.0; no archive notice; README: default BSH (buy/sell/hold) action scheme and PBR (position-based returns) reward scheme; RLlib + Optuna recommended; README inconsistent on Python (3.12+ vs 3.11/3.12) and links issues to a legacy repo; README says commission-aware rewards/alternate action spaces are "priorities", i.e., not done | [PyPI](https://pypi.org/pypi/tensortrade/json); [GitHub](https://github.com/tensortrade-org/tensortrade) |
| tensortrade-ng (fork) | 2.0.0a2, 2024-08-26 (alpha) | — | [PyPI](https://pypi.org/pypi/tensortrade-ng/json) |
| gym-anytrading | 2.0.0, 2023-08-27 | 2.4k stars, MIT; `StocksEnv`/`ForexEnv`; **only two actions (Sell/Buy) -> Short/Long positions, deliberately no Hold, no sizing**; constant fees; no budget/multi-asset modelling; README points to gym-mtsim for experts; uses Gymnasium API | [PyPI](https://pypi.org/pypi/gym-anytrading/json); [GitHub](https://github.com/AminHP/gym-anytrading) |
| gym-mtsim | 2.0.0, 2024-03-20 | 525 stars; MetaTrader 5 simulator (Forex/stocks/crypto/futures); multi-asset flattened 1-D action vector (close-probabilities, hold/new, signed volume via expit) to work around SB3's lack of Dict action support; hedging, margin/stop-out; fees approximated as constants; **historical data download works only on Windows** (MetaTrader5 package) | [PyPI](https://pypi.org/pypi/gym-mtsim/json); [GitHub](https://github.com/AminHP/gym-mtsim) |
| Qlib (RL module: order execution) | pyqlib 0.9.7, 2025-08-15 | 49.7k stars; README says "active and continuing development" but its newest dated feature is May 2023 and newest named release v0.9.0 (Dec 2022) in the captured page; RL order execution examples: TWAP baseline, PPO (IJCAI 2020), OPDS (AAAI 2021); nested decision framework (Oct 2021) and RL framework (Nov 10, 2022); **official dataset temporarily unavailable** (community data suggested; Yahoo-derived data "may not be perfect"); unreleased main-branch change restricts loading recorders unless `trusted=True`; RD-Agent LLM quant factory (Aug 2024) | [PyPI](https://pypi.org/pypi/pyqlib/json); [GitHub](https://github.com/microsoft/qlib) |
| ABIDES-Gym (ABIDES-JPMC) | no PyPI package found (lookup failed) | **Repo archived by owner 2 June 2025**, read-only, no support; wraps ABIDES as a Gym env (`markets-daily_investor-v0`, older Gym API); paper arXiv 2110.14771 | [GitHub](https://github.com/jpmorganchase/abides-jpmc-public) |
| mbt_gym | no PyPI package found (lookup failed) | 181 stars, BSD-3; 440 commits, 1 open issue; gym envs for model-based market making and optimal execution (vectorized); paper "mbt-gym: RL for model-based limit order book trading", ICAIF 2023 / arXiv 2209.07823; no releases listed | [GitHub](https://github.com/JJJerome/mbt_gym) |
| PGPortfolio (EIIE) | not on PyPI (source repo) | 1.9k stars, GPL-3.0, 65 commits; **TensorFlow >=1.0 + tflearn** (legacy stack); README warns market efficiency may have improved since 2017 so the algorithm "may no longer work", that backtests ignore slippage/impact, and that the arXiv v2 test period was ~30% shorter than stated so the volume-based asset-selection window overlapped backtest data (a look-ahead leak) | [GitHub](https://github.com/ZhengyaoJiang/PGPortfolio) |

- Gymnasium itself is current (1.4.0, 2026-10-05) — [PyPI](https://pypi.org/pypi/gymnasium/json). d3rlpy pins gymnasium 1.0.0 — [GitHub](https://github.com/takuseno/d3rlpy).
- **FinRL Contests (2023–2025)**: tasks were Data-Centric Stock Trading (2023, 30 DJIA stocks, 46 teams), LLM-engineered signals with RLMF and Crypto trading with ensemble learning (2024, 21 teams), FinRL-DeepSeek stock trading, FinRL-AlphaSeek crypto trading and a crypto FinAI contest (2025, 85 teams); 200+ participants / 100+ institutions / 20+ countries. Starter kits provide GPU-vectorized market envs (PPO on 30 stocks, A100: ~227,000 samples/s with 2,048 parallel envs, ~1,650x one env). — [FinRL Contests paper](https://arxiv.org/html/2504.02281v4)
- Contest documentation lists "FinAI Contest 2025" as the ongoing crypto-sentiment track. — [FinAI contest docs](https://finrl-contest.readthedocs.io/)

### Inferences
- The **"near-exact replica of the live platform"** idea is only well supported by the *execution/microstructure* tools (ABIDES-Gym — now archived; mbt_gym; Qlib RL's order-execution simulator), not by the FinRL/gym-anytrading style bar-replay environments, which model fees as constants and ignore market impact/queue position. For daily/hourly swing decisions on liquid names, bar replay with a conservative cost model is the pragmatic choice; for HFT you would need an LOB-level simulator (other researchers' scope).
- Because gym-anytrading/gym-mtsim/TensorTrade bake in restrictive action spaces and constant costs, the lowest-risk path is a ~200-line custom Gymnasium env whose observation/action/cost model you fully control and unit-test, borrowing ideas (not code) from these.
- Treat FinRL (original) as a tutorial/reference; do not build on FinRL-Meta/ElegantRL PyPI releases (2023). FinRL-X is the only AI4Finance project with a 2026 paper, but its evidence base is thin (see sec. 5).

### Gaps
- GitHub last-commit dates and issue-response times for all repos were not retrievable (API blocked). The "stale" labels rely on PyPI release dates plus README text.
- No PyPI package found for ABIDES-Gym or mbt_gym (lookups failed — could be 404 or different package names); install is probably from source.
- No documentation found for CLEAN-style or Qlib RL environments' realism relative to TSX/crypto venues.

---

## 3. RL algorithm libraries (SB3, RLlib, CleanRL, TorchRL, SB3-contrib, Dreamer/model-based, offline RL) — status and fit

### Takeaway
SB3 + SB3-contrib (RecurrentPPO, MaskablePPO, TQC) is the best-maintained, simplest choice for a solo developer (releases June 2026); RLlib and TorchRL are active but heavier; CleanRL's last PyPI release is 2023; d3rlpy (offline RL) is usable but with a slow release cadence (last 2025-03); Dreamer/model-based RL has no credible trading evidence found.

### Cited Findings
| Library | Latest PyPI release | Notes | Source |
|---|---|---|---|
| Stable-Baselines3 | 2.9.0, 2026-06-15 (Python >=3.10) | active | [PyPI](https://pypi.org/pypi/stable-baselines3/json) |
| SB3-Contrib | 2.9.0, 2026-06-15 | algorithms listed: ARS, CrossQ, Maskable PPO, Recurrent PPO, QR-DQN, TQC, TRPO; described as "Experimental code" | [PyPI](https://pypi.org/pypi/sb3-contrib/json); [docs](https://sb3-contrib.readthedocs.io/en/master/) |
| SBX (SB3 in JAX) | 0.28.0, 2026-07-24 | — | [PyPI](https://pypi.org/pypi/sbx-rl/json) |
| Ray / RLlib | ray 2.59.0, 2026-10-02 | New API stack is default since Ray 2.40, PyTorch-only (TF dropped); old stack still usable "beyond Ray 3.0" per docs; opt-out via `api_stack()` flags | [PyPI](https://pypi.org/pypi/ray/json); [RLlib docs](https://docs.ray.io/en/latest/rllib/rllib-new-api-stack.html) |
| TorchRL | 0.14.1, 2026-10-10 | docs include a DreamerV3 reference implementation | [PyPI](https://pypi.org/pypi/torchrl/json); [docs](https://docs.pytorch.org/rl/stable/reference/dreamer_v3.html) |
| CleanRL | 1.2.0, 2023-05-22 (requires Python <3.11 on PyPI) | stale release | [PyPI](https://pypi.org/pypi/cleanrl/json) |
| skrl | 2.1.1, 2026-10-21 | — | [PyPI](https://pypi.org/pypi/skrl/json) |
| d3rlpy | 2.8.1, 2025-03-02 | MIT; PyTorch >=2.5; algorithms: CQL (discrete+continuous), IQL, Decision Transformer, TD3+BC, BCQ, BC, SAC, AWAC, PLAS, BEAR, etc.; README warns v2 has breaking changes vs v1 | [PyPI](https://pypi.org/pypi/d3rlpy/json); [GitHub](https://github.com/takuseno/d3rlpy) |
| ElegantRL / FinRL | see sec. 2 | | |

- Offline RL in trading: Yun (arXiv Nov 2024) adapts a GPT-2-initialized Decision Transformer with LoRA for quantitative trading and reports it is "competitive" with CQL, IQL and BC, "superior" only in some scenarios (code: github.com/syyunn/finrl-dt). — [arXiv 2411.17900](https://arxiv.org/html/2411.17900v1) ; Lee & Moon (AAMAS 2023 extended abstract) report their decision-transformer actor-critic with BC regularizer (TACR) beats other state-of-the-art methods on Sharpe and profit on stock data (abstract only). — [AAMAS 2023 PDF](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p2815.pdf)
- A theory overview notes Decision Transformer can only reproduce/interpolate behaviors in the dataset and cannot "stitch" suboptimal trajectories into better-than-data policies, unlike Q-learning-style methods (secondary source); a 2025 paper asks "Should We Ever Prefer Decision Transformer for Offline RL?". — [theorempath](https://theorempath.com/topics/offline-reinforcement-learning); [arXiv 2507.10174](https://arxiv.org/pdf/2507.10174)
- DreamerV3 (single configuration across 150+ tasks) learns a world model and trains actor/critic in latent imagination; my search found **no** trading application. Millea's 2021 survey names model-based RL (learning a world model of the trading environment) and hierarchical RL as promising directions, focused mainly on crypto. — [DreamerV3 paper](https://arxiv.org/abs/2301.04104); [Millea survey abstract](https://ideas.repec.org/a/gam/jdataj/v6y2021i11p119-d680602.html)
- A 2026 ScienceDirect paper on portfolio optimization with path-dependent transaction costs reports LSTM-SAC converging more stably than GRU, Transformer and Recurrent PPO variants (abstract-level). — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590005626003140)
- A 2025 PLOS ONE paper comparing static vs dynamic factor-based reward designs for asset allocation with DDPG/SAC/PPO finds no universal winner; adaptability depends on context. — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0332779)

### Inferences
- Practical stack: SB3 (PPO / SAC / TQC) for baselines; `MaskablePPO` is attractive for trading because you can mask illegal actions (e.g., "sell" when flat, orders exceeding risk limits); `RecurrentPPO` for partially observable state; both are in SB3-Contrib. RLlib's distributed power is unnecessary at solo scale and its API has been in flux (old->new stack).
- Offline RL (CQL/IQL via d3rlpy) is attractive in principle because you cannot explore live, but the trading evidence is thin and the "behavior policy" in historical price data is not an expert — there is no expert dataset like in robotics, so offline RL mostly reduces to fitting the historical return stream; its fit is much better for *execution* logs from your own broker than for alpha.
- Since the GPU-vectorized envs in FinRL contests reached ~227k samples/s, sample *throughput* is cheap but sample *diversity* is not: one stock-year is only ~252 daily bars, so more epochs mostly means more overfitting (this is an inference from the contest numbers and the overfitting literature below).

### Gaps
- No verified GitHub release/commit info for SB3-contrib RecurrentPPO vs. changes in 2.x; I did not read the SB3 changelog.
- No trading-specific evidence found for Dreamer/world-model RL; no controlled comparison of SB3 vs RLlib vs CleanRL on trading tasks.

---

## 4. RL design details for trading: state, actions, reward, costs, non-stationarity, overfitting, sample efficiency, simulation approach

### Takeaway
The literature is dominated by PnL-reward agents with no statistical testing and flat/no transaction costs; best practices that are actually documented are (i) explicit cost-aware rewards or cost penalties, (ii) risk-adjusted/differential-Sharpe or drawdown-aware rewards, (iii) hypothesis-test-based overfitting rejection, purged/walk-forward validation and (iv) training on simulated or augmented market data rather than a single historical path. Evidence for domain randomization specifically is thin.

### Cited Findings
- A 2026 critical survey (SSRN, 124 papers 2018–2026 screened from 318) reports PnL-based rewards dominate (65 papers, 52.4%), risk-adjusted rewards appear in 16, multi-objective formulations in 20; **86.3% of papers report outperformance with no formal statistical test and 89.5% model no transaction costs or only a flat proportional fee**; named key challenges: distribution shift, sample efficiency, sim-to-reality gap, risk-constrained optimization, interpretability, offline RL (abstract/search snippet only; full text 403). — [SSRN 7363738](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7363738)
- Moody & Saffell's differential Sharpe ratio tracks exponentially weighted first and second moments of returns so a Sharpe-style reward can be computed one step at a time. — [description in arXiv 2110.04745](https://arxiv.org/pdf/2110.04745); [NIPS paper](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading)
- PGPortfolio/EIIE casts portfolio management as immediate-reward optimization with a transaction-cost penalty (no Monte Carlo/bootstrapped gradient needed), with continuous portfolio-weight actions. — [PGPortfolio GitHub](https://github.com/ZhengyaoJiang/PGPortfolio)
- Action-space examples: discrete 2-action long/short (gym-anytrading), discrete BSH (TensorTrade), multi-asset continuous volume vectors with expit squashing (gym-mtsim), continuous target weights (EIIE; FinRL-X's weight-vector interface). — [gym-anytrading](https://github.com/AminHP/gym-anytrading); [TensorTrade](https://github.com/tensortrade-org/tensortrade); [gym-mtsim](https://github.com/AminHP/gym-mtsim); [FinRL-X](https://arxiv.org/html/2603.21330v1)
- **Backtest-overfitting control for DRL**: Gort et al. formulate overfitting detection as a hypothesis test: train many agents, estimate the probability of overfitting, reject overfitted ones; on 10 cryptocurrencies, 2022-05-01 to 2022-06-27 (two crashes), less-overfitted agents outperformed more-overfitted ones (authors' own, short single-window evidence). They also argue a single validation set overfits and that going back and forth between train and test leaks information. — [arXiv 2209.05559](https://arxiv.org/abs/2209.05559)
- In the FinRL 2025 contest one team used that approach (reject agents at 10% significance); its PPO agent returned −34.96% over a period when the S&P crypto index fell 50.78% (i.e., rejection reduced but did not eliminate losses). Also, reported contest metrics contained errors (a 2023 Sharpe of 9.56 flagged as incorrectly reported; a Rachev ratio and 2025 Sharpe values inconsistent with text). — [FinRL Contests paper](https://arxiv.org/html/2504.02281v4)
- The FinRL contests report that policy instability and sampling cost motivated ensembles and GPU-vectorized environments, that generalization to unseen markets remained a challenge, and that crypto ensembles lacked diversity (individual agents performed almost identically to the ensembles). — [FinRL Contests paper](https://arxiv.org/html/2504.02281v4)
- **Training on simulated vs historical data**: an RL-trading paper on multi-agent virtual markets reports agents trained on historical series often performed very differently out-of-sample than in training (limited market conditions -> overfitting) and that agents trained on simulated market data beat historically trained agents out of sample (abstract-level; the paper's own claim). — [ScienceDirect S1568494623000030](https://www.sciencedirect.com/science/article/abs/pii/S1568494623000030)
- General backtest-overfitting evidence: a study of 888 Quantopian algorithms found in-sample and out-of-sample performance only weakly correlated, and the more backtesting done, the larger the gap (secondary summary). — [Wiecki et al., "All That Glitters Is Not Gold"](https://www.researchgate.net/publication/307553701_All_That_Glitters_Is_Not_Gold_Comparing_Backtest_and_Out-of-Sample_Performance_on_a_Large_Cohort_of_Trading_Algorithms)
- Tools for measuring overfitting: Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, corrects for multiple trials, non-normality and sample length) and Probability of Backtest Overfitting via combinatorially symmetric CV (Bailey, Borwein, Lopez de Prado, Zhu); combinatorial purged CV yields a distribution of OOS paths but is computationally costly. — [PBO on SSRN](https://papers.ssrn.com/abstract=2326253); [DSR overview](https://www.luxalgo.com/library/concept/deflated-sharpe-ratio.md); [Purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation). skfolio ships `CombinatorialPurgedCV` and `WalkForward`. — [skfolio](https://skfolio.org/)
- Non-stationarity: reviews cite Jiang et al. (2017) that DRL models trained on history degrade significantly when run live; and PGPortfolio's README itself warns the 2017 algorithm may no longer work. — [arXiv 2512.10913 (qualitative text only; see synthetic-data warning in sec. 1)](https://arxiv.org/html/2512.10913v1); [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio)
- Lopez de Prado attributes failure of ML funds largely to methodological errors (backtest overfitting, cross-validation leakage), and is quoted on the extremely low signal-to-noise ratio in markets (the latter is a press attribution). — [GARP white paper page](https://www.garp.org/white-paper/the-10-reasons-most-machine-learning-funds-fail); [Bloomberg](https://prod.cm.bloomberg.com/news/articles/2018-10-09/the-big-problem-with-machine-learning-algorithms)
- Seed sensitivity: in the 2026 futures study, averaging three seeds reliably stabilized results; more seeds gave no further gain. — [arXiv 2607.00475](https://arxiv.org/html/2607.00475v1)

### Inferences (design recommendations synthesized from the above; not directly sourced)
- **State**: forecast mean/quantiles/uncertainty for each horizon, current position and unrealized PnL, time-in-trade, spread/volume/volatility, cost estimates, and a small regime vector; keep it low-dimensional (daily-bar data has few effective samples). Avoid feeding raw long price windows when you already have a forecaster.
- **Actions**: prefer *position-scaling* actions (e.g., target exposure in {0, 0.25, 0.5, 1}x of an optimizer/Kelly-suggested cap, plus "exit now") over raw buy/sell, so the policy can only act inside risk limits; mask illegal actions (MaskablePPO).
- **Reward**: per-step net PnL after *modelled* costs (spread + fee + slippage proportional to size/volatility) minus a drawdown/variance or turnover penalty (a differential-Sharpe or CVaR-style term is the documented alternative); never reward gross PnL.
- **Anti-overfit protocol**: purged/embargoed walk-forward, many seeds, report DSR/PBO, include a "do-nothing/buy-and-hold/1-N/vol-target" baseline and require the RL overlay to beat the *same pipeline without RL* net of costs.
- **Domain randomization** (random costs, latency, partial fills, volatility scaling, bootstrapped/block-resampled or generated price paths, randomized regime start dates) is the standard sim-to-real remedy in RL generally and is consistent with the virtual-market result above, but I found no trading paper that isolates its effect.

### Gaps
- No quantified study isolating reward-design effects (PnL vs differential Sharpe vs drawdown penalty) under realistic costs; the PLOS ONE paper says no universal winner.
- No source found specifically on domain randomization in trading RL, or on how closely agent-based simulators (ABIDES) predict live fills.
- The SSRN critical survey's full text (403) and Millea's full text (403) were not read.

---

## 5. Evidence: does deep RL work out-of-sample/live? Is RL used in industry mostly for execution/market making?

### Takeaway
Evidence for RL as an *alpha source* is weak: contest/benchmark results are mixed and mostly short windows, critical surveys find missing statistical tests and costs, and the few rigorous out-of-sample tests show learned policies only matching simple baselines. Documented industry success stories are in execution and market making, and optimization-based portfolio construction remains the default.

### Cited Findings
- FinRL contest 2023 (DJIA 30): top teams had better risk-adjusted returns and drawdown than the DJIA in the first test window (10/25–11/14/2023) but lower profitability; two teams had negative returns in the second window (11/15–11/22/2023). Ensemble experiments (2021–2023 rolling): Ensemble-1 62.60% return, Sharpe 1.48, max drawdown −8.98% vs **standalone PPO 63.37%, Sharpe 1.55**; mean-variance 13.9% and DJIA 18.95%, Sharpe ~0.47–0.48. — [FinRL Contests paper](https://arxiv.org/html/2504.02281v4)
- Crypto 2024 ensembles (Bitcoin LOB, OOS): 0.66% cumulative return, Sharpe 0.28, max DD −0.73% vs BTC buy-and-hold 0.74% (Sharpe 0.20, DD −1.3%); 2025 crypto winners: Fermion 0.23% (Sharpe 0.0037), Mt. Everest −0.05% vs BTC −7.35% (max DD −17.02%). 2025 stock (Nasdaq-100, 2019–2023 test, LLM signals): Otago Alpha (PPO) 191.14% return, Sharpe 1.08, max DD −28.16%; Ruijian & Sally (GRPO) 335.57%, Sharpe 0.95, max DD −50.24%; Queen's Gambit (PPO) 342.65% with max DD −92.47%; baselines S&P 500 90.03%, Nasdaq-100 164.52%. The paper notes some 2025 Sharpe values look inconsistent with the text. — [FinRL Contests paper](https://arxiv.org/html/2504.02281v4)
- FinRL-X paper trading (Alpaca, Oct 26, 2025–Mar 12, 2026, daily rebalancing, ensemble of two rule-based strategies — not an RL policy): +19.76% vs SPY −2.51% and QQQ −4.79%, Sharpe 1.96, max DD −12.22%; ~4.5 months; authors state evaluation horizon is limited and results are "not intended to establish statistically significant alpha". — [FinRL-X paper](https://arxiv.org/html/2603.21330v1)
- Critical survey 2026 (see sec. 4): 86.3% of 124 papers no formal statistical test; 89.5% no/flat costs. — [SSRN 7363738](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7363738)
- Millea (Data, Nov 2021) critical survey: aims to identify promising directions and show "how inconsistency in the research community holds back DRL trading work"; mainly crypto. — [RePEc abstract](https://ideas.repec.org/a/gam/jdataj/v6y2021i11p119-d680602.html)
- EIIE replication: strong on crypto, average on stocks (see sec. 1). — [arXiv 2409.08426](https://arxiv.org/pdf/2409.08426)
- End-to-end learned policy vs simple rules (futures, 2011–2024 OOS): learned policy ≈ 1/N net of costs; cost sensitivity sharply cuts the high-turnover model. — [arXiv 2607.00475](https://arxiv.org/html/2607.00475v1)
- A broad evidence base on backtest overfitting: 888 automated strategies show weak IS/OOS correlation; in another comparison of 80 models, 44 beat buy-and-hold but none significantly (secondary summary of search results). — [Wiecki et al.](https://www.researchgate.net/publication/307553701_All_That_Glitters_Is_Not_Gold_Comparing_Backtest_and_Out-of-Sample_Performance_on_a_Large_Cohort_of_Trading_Algorithms)
- **Industry use**: Hambly, Xu & Yang's survey notes the finance industry has seen "a number of recent successes" applying RL in order execution, market making and portfolio optimization; Gašperov et al. (Mathematics 2021) conclude RL methods generally outperform standard, analytically derived market-making strategies on risk-adjusted return (academic reviews, not live P&L). — [arXiv 2112.04553](https://www.arxiv.org/pdf/2112.04553); [DOAJ record](https://doaj.org/article/86c142702878470a833bb38d0424c979)
- JPMorgan's LOXM: reported in 2017 as a deep-RL execution system trained on billions of historical and simulated trades, tested in Europe from Q1 2017, constrained during learning and run within the bank's electronic risk framework; claims of better pricing than benchmark are the bank's own, with no independent verification in my sources. — [MIT Technology Review](https://www.technologyreview.com/f/608467/jpmorgan-will-use-ai-to-carry-out-trades-across-the-world/); [Markets Media](https://marketsmedia.com/j-p-morgan-ai-trading-global/)
- Qlib's RL module is specifically about order execution (TWAP baseline, PPO, OPDS) within a nested decision framework (portfolio level above execution level). — [microsoft/qlib](https://github.com/microsoft/qlib)
- Kolm & Ritter present RL as a way to solve multi-period optimal control problems, listing trading under transaction costs, hedging and market making as applications (a 2025 Journal of Portfolio Management article by the same authors is a later treatment). — [Kolm & Ritter, Modern Perspectives on RL in Finance](https://cims.nyu.edu/~ritter/kolm2020modern.pdf)

### Inferences
- There are no verified live, audited deployments of end-to-end deep RL *alpha* in my sources; the strongest "live" item (FinRL-X paper trading) is not an RL policy and covers <5 months. Absence of evidence is not proof of failure, but the burden of proof is on RL for alpha.
- The FinRL contest numbers show large dispersion between returns and drawdowns (e.g., −92% max DD alongside +342% return) — contest rankings partly reward risk-taking in a bull test window, so they are poor guides to a risk-managed live system.
- RL's comparative advantage appears where the environment's dynamics are well modelled and *you* affect outcomes (execution, inventory/market making), and where rewards are dense and short-horizon. This maps to the user's HFT desire only if they have LOB data and a simulator with impact; for swing/intraday decisions it adds little over explicit expected-utility rules.

### Gaps
- No audited live track record of deep-RL alpha strategies, and no practitioner survey of RL usage in funds, was found. Search for "RL for alpha in industry" returned nothing substantive.
- No independent replication of FinRL's ensemble paper or of FinRL-DeepSeek contest winners was found.
- The LOXM source is press-era (2017); current status unknown.

---

## 6. Classical decision tools: portfolio optimization, Kelly/fractional Kelly, volatility targeting, stops and take-profits, drawdown circuit breakers, exposure limits

### Takeaway
Well-maintained Python libraries cover all classical needs (PyPortfolioOpt, Riskfolio-Lib, skfolio, cvxpy/cvxportfolio). Kelly is fragile to input error, so use fractional Kelly with shrunk edge estimates; volatility targeting helps tail risk everywhere and Sharpe mainly for risk assets; stop-losses help when returns have momentum/regime-switching and hurt under a random walk, so they are a risk-management tool not an alpha tool; drawdown control has theory but is hard with costs.

### Cited Findings
**Libraries (PyPI latest release)**
- PyPortfolioOpt 1.6.0, 2026-02-26 — [PyPI](https://pypi.org/pypi/pyportfolioopt/json); Riskfolio-Lib 7.4.0, 2026-10-04 — [PyPI](https://pypi.org/pypi/riskfolio-lib/json); skfolio 1.11.0, 2026-10-24 — [PyPI](https://pypi.org/pypi/skfolio/json); cvxpy 1.9.3, 2026-09-19 (Python >=3.11) — [PyPI](https://pypi.org/pypi/cvxpy/json); cvxportfolio 1.5.1, 2025-07-06 — [PyPI](https://pypi.org/pypi/cvxportfolio/json).
- skfolio (sklearn-style) offers: MeanRisk, HRP, risk budgeting/risk parity, NCO, stacking, max-diversification, distributionally robust CVaR, HERC; priors include Black–Litterman, factor models, entropy pooling and opinion pooling (i.e., forecasts as "views"); uncertainty sets via bootstrap; constraints for turnover, tracking error, weights, groups, cardinality; per-asset `transaction_costs` parameter in MeanRisk; CVaR risk measures; WalkForward and CombinatorialPurgedCV. — [skfolio.org](https://skfolio.org/)
- cvxportfolio (GPL-3.0; 1.3k stars): policies take forecasts (returns, covariances, risk-forecast error) as inputs, multi-period optimization with a `planning_horizon`, transaction/holding cost models (defaults aimed at US stocks), back-test simulator where the policy sees only past data; README says public API changes are reserved for a future 2.0.0; runs daily example strategies since late 2023. — [cvxgrp/cvxportfolio](https://github.com/cvxgrp/cvxportfolio)

**Kelly / fractional Kelly**
- Continuous-investment Kelly: f* = (μ − r)/σ² (maximizes expected log wealth / long-run geometric growth); binary form f = p/l − q/g. Example: 6% excess return and 15% vol gives Sharpe 0.40 and a Kelly fraction of 267%. Fractional Kelly (half/quarter) is used to reduce ruin risk, volatility and model error, at the cost of slower growth; overestimating the edge pushes sizing past optimal and raises ruin risk. — [Wikipedia: Kelly criterion](https://en.wikipedia.org/wiki/Kelly_criterion)
- Estimation risk is the main weakness: a 2025 arXiv paper says Kelly's two main criticisms are large short-term risk and high sensitivity to input errors, that fractional Kelly mainly addresses the first, and that estimation-risk suboptimality remains open. Chopra & Ziemba (1993) show errors in means cost far more (certainty-equivalent loss) than errors in variances/covariances. — [arXiv 2508.18868](https://arxiv.org/html/2508.18868v2); [Chopra & Ziemba table](https://escholarship.org/content/qt5mr5k8qj/qt5mr5k8qj.pdf) (search-summary level)

**Volatility targeting**
- Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert (60+ assets, daily data from as early as 1926): vol targeting raises Sharpe only for "risk assets" (equity, credit; linked to the leverage effect), negligible for bonds/currencies/commodities; it consistently reduces tail risk and vol-of-vol across assets and reduces maximum drawdown in balanced and risk-parity portfolios (secondary summaries of the SSRN paper). — [Quantpedia](https://quantpedia.com/the-impact-of-volatility-targeting-on-equities-bonds-commodities-and-currencies); [Alpha Architect](https://alphaarchitect.com/volatility-targeting-improves-risk-adjusted-returns/)

**Stop-loss / take-profit**
- Kaminski & Lo ("When Do Stop-Loss Rules Stop Losses?", J. Financial Markets 2014): under a random walk, simple 0/1 stop-loss rules reduce expected return; with momentum (positive serial correlation) or regime-switching they can add value; on monthly 1950–2004 US equity data with bonds as the stop-out asset, certain rules added 50–100 bp per month during stop-out periods; on daily futures, certain rules increased return while reducing volatility at longer sampling frequencies (details differ across secondary summaries). — [MIT DSpace record](https://dspace.mit.edu/entities/publication/bb69ca4b-0cdc-487f-831d-63b2e84fafee); [summary](https://dailyspeculations.com/wordpress/?p=14247)
- Han, Zhou & Zhu ("Taming Momentum Crashes"): a 10% stop-loss on the momentum strategy cut worst monthly losses from −49.79% to within −11.34% (equal-weighted) and −65.34% to −23.69% (value-weighted), 1926–2011; results are gross, and stops add trading cost (commentary suggests 300–600 bp could be shaved). — [Alpha Architect](https://alphaarchitect.com/2016/08/taming-the-momentum-roller-coaster-fact-or-fiction/); [CXO Advisory](https://www.cxoadvisory.com/?p=25071)

**Drawdown control / circuit breakers**
- Grossman & Zhou (1993): optimal CRRA policy for controlling drawdown invests in risky assets in proportion to the "surplus" W_t − αM_t (wealth above a floor tied to the running maximum); assumes continuous rebalancing and no transaction costs; "decayed drawdown" forgives past losses at a set rate (higher decay -> higher growth, less absolute drawdown control). Klass & Nowicki (2005) show the strategy loses its optimal growth property in discrete time; with costs the policy is hard to implement near the limit. — [RePEc: Grossman & Zhou](https://ideas.repec.org/a/bla/mathfi/v3y1993i3p241-276.html); [Lund Univ. record](https://lup.lub.lu.se/record/223851)

### Inferences
- A usable per-asset sizing formula for a forecaster with confidence: size ∝ k · (forecast edge net of expected costs) / (forecast variance), with k ≈ 0.1–0.5 of Kelly and with the edge shrunk toward zero by forecast uncertainty/calibration error (because mean errors dominate); then cap by volatility targeting, per-name/sector/asset-class limits and portfolio-level gross/net/leverage limits. (Design synthesis from the Kelly and estimation-risk sources; k range is my judgement, not sourced.)
- ATR/trailing stops: I found *no* rigorous evidence specific to ATR-based stops; the Kaminski-Lo logic suggests stops are value-adding mainly in trending/momentum regimes, and for a mean-reverting short-horizon forecaster a hard stop can systematically exit just before the expected rebound. Prefer *risk-based* exits (position-level loss limit sized to risk budget, time stops tied to forecast horizon, "thesis-invalidated" exit when the updated forecast flips sign net of costs) plus portfolio-level drawdown breakers.
- Drawdown breaker in practice = scale exposure down as equity falls toward a floor (CPPI/Grossman-Zhou style), with hysteresis/cooldown to avoid whipsaw and cost churn.
- Black-Litterman in skfolio/Riskfolio is a natural way to inject model "views" with confidence (view uncertainty = forecast uncertainty); HRP/risk-parity are robust fallbacks when covariance estimates are noisy.

### Gaps
- No rigorous source on ATR-based or trailing stops specifically; no evidence on stop-loss behaviour for TSX/TSXV small caps or crypto.
- Harvey et al. and Chopra-Ziemba were read only through secondary summaries; Kelly formulas through Wikipedia (original Kelly 1956/Thorp papers not read).
- No source found for specific drawdown-breaker thresholds used in practice.

---

## 7. Multi-horizon decision-making (combining short- and long-horizon forecasts, net of costs)

### Takeaway
The principled framework is multi-period expected-utility maximization with transaction costs, solved in closed form for linear-quadratic cases (Gârleanu–Pedersen), numerically as receding-horizon convex optimization (Boyd et al. / cvxportfolio), or approximately by RL; the key insight is to trade partially toward an "aim" portfolio that blends the current and expected future optimal portfolios, weighting slowly-decaying signals more.

### Cited Findings
- Gârleanu & Pedersen: optimal policy = move partway from the current portfolio toward an "aim portfolio" that blends current and *future expected* Markowitz portfolios; predictors with slower mean-reversion (alpha decay) get more weight in the aim; tested on commodity futures with better net returns than naive benchmarks. — [NBER w15205](https://www.nber.org/papers/w15205); [CBS accepted manuscript](https://research-api.cbs.dk/ws/portalfiles/portal/60191441/lasse_heje_pedersen_et_al_dynamic_trading_acceptedmanuscript.pdf)
- Boyd et al. multi-period trading: the optimizer plans a sequence of future trades using forecasts of not-yet-known quantities, executes the first, and repeats (MPC); includes transaction and holding costs; cvxportfolio implements this with `MultiPeriodOptimization(planning_horizon=...)`. — [arXiv 1705.00109](https://arxiv.org/pdf/1705.00109); [cvxportfolio](https://github.com/cvxgrp/cvxportfolio)
- Kolm & Ritter: RL framed as multi-period optimal control for trading with costs, hedging and market making. — [Kolm & Ritter](https://cims.nyu.edu/~ritter/kolm2020modern.pdf)
- Moody & Saffell: under costs, optimal decisions depend on the current state (current position), the rationale for recurrent/RL training over one-shot forecast rules. — [NIPS paper](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading)

### Inferences
- Worked version of the user's example, as a decision rule: let μ(h) be the forecast net return at horizon h with uncertainty σ(h), and c the round-trip cost (spread + fees + slippage, for the current position size). "Sell now and re-enter later" beats "hold through the dip" only if the avoided dip (0.5% over 1 hour) minus c exceeds the opportunity risk that the +1.3% move arrives early/without the dip (probability-weighted using the uncertainty). For liquid US large caps c may be a few bp so tactical trimming can pay; for TSXV/small-cap or crypto altcoins with wide spreads c can exceed the dip, so the correct action is "hold" regardless of the forecast. (Reasoning; costs are asset-specific.)
- Implementation recipe: (1) convert each model's multi-horizon forecasts into a decay-weighted expected-return path for each asset (G-P: weight horizons by persistence); (2) solve a small receding-horizon problem in cvxpy/cvxportfolio over the next H steps with cost, risk, turnover and exposure constraints; (3) execute only the first step's trade, and only if |Δ| exceeds a no-trade band (trade partially toward the aim); (4) re-solve at each new forecast. This yields "hold today, sell tomorrow" automatically when the aim portfolio is higher later than now and costs make rotation unattractive.
- G-P's closed form depends on linear-quadratic assumptions (quadratic costs, linear mean-reverting return predictors; background knowledge, not shown in the fetched abstract); real costs (spread + concave impact + fixed fees) call for the numerical MPC version.

### Gaps
- I did not find an open-source Python package implementing the G-P closed form directly (cvxportfolio is the nearest) or a study comparing MPC vs RL for multi-horizon retail-scale trading.
- No source for how to calibrate horizon weights from a forecaster's reported confidence beyond G-P's decay-based weighting.

---

## 8. Production patterns: separation of concerns, kill switches, monitoring and drift, retraining cadence, paper-trading duration

### Takeaway
Separate signal / portfolio construction / risk / execution with a narrow interface (target weights/orders); enforce pre-trade limits and a hard kill switch independent of the model; monitor drift with open-source tools; there is no authoritative "required" paper-trading duration — practitioner guidance is 30–100 closed trades or ~3 months, with paper results treated as an upper bound.

### Cited Findings
- FinRL-X's pattern: weights are the sole interface between strategy and execution; a risk overlay scales exposure; broker-integrated execution with "guardrails"; Alpaca paper trading. — [FinRL-X paper](https://arxiv.org/html/2603.21330v1)
- Qlib's nested decision framework lets a portfolio-level strategy and an execution-level strategy/executor be optimized and run together at different time scales. — [microsoft/qlib](https://github.com/microsoft/qlib)
- Knight Capital (1 Aug 2012): SEC's first Rule 15c3-5 (market access) enforcement action, $12M settlement; failures cited: no pre-submission controls to catch erroneous orders (e.g., compare orders leaving the router with those entering), no controls blocking orders beyond capital thresholds because accounts were never linked to limits, and insufficient supervision of code updates and malfunctions; the firm also did not adequately review control effectiveness. — [SEC press release 2013-222](https://www.sec.gov/news/press-release/2013-222)
- JPMorgan constrained LOXM's behaviour during learning and kept it within the firm's electronic risk framework. — [MIT Technology Review](https://www.technologyreview.com/f/608467/jpmorgan-will-use-ai-to-carry-out-trades-across-the-world/)
- Drift/monitoring libraries (PyPI latest): river 0.26.1 (2026-08-21, online learning/drift detectors) — [PyPI](https://pypi.org/pypi/river/json); evidently 0.7.23 (2026-09-11) — [PyPI](https://pypi.org/pypi/evidently/json); alibi-detect 0.13.0 (2025-12-11) — [PyPI](https://pypi.org/pypi/alibi-detect/json); ruptures 1.1.10 (2025-09-10, change-point detection) — [PyPI](https://pypi.org/pypi/ruptures/json); hmmlearn 0.3.3 (2024-10-31, regime HMMs; slow cadence) — [PyPI](https://pypi.org/pypi/hmmlearn/json).
- Broker/exchange connectivity (PyPI latest): alpaca-py 0.44.0 (2026-08-11) — [PyPI](https://pypi.org/pypi/alpaca-py/json); ib-async 2.1.0 (2025-12-08) — [PyPI](https://pypi.org/pypi/ib-async/json); ccxt 4.5.90 (2026-10-23) — [PyPI](https://pypi.org/pypi/ccxt/json).
- Paper-trading duration guidance (practitioner/blog level): trade count matters more than calendar time — often 30–100 closed trades depending on variance; for slow strategies use small-size live incubation; a forum rule of thumb of at least 3 months; AlgoTest notes many traders use 2–4 weeks; demo fills ignore queue position, assume full fills at quote and never reject orders, so paper results are an upper bound and best for checking signal correctness, worst for estimating costs; Alpaca reports 75.2% of API users who paper-traded first placed their first live trade within 60 days (not a recommended benchmark). — [LuxAlgo](https://www.luxalgo.com/library/concept/forward-paper-trading-confirmation/); [ProRealCode thread](https://www.prorealcode.com/topic/for-how-long-do-you-test-an-algo-before-going-live/); [AlgoTest](https://algotest.in/blog/difference-between-backtesting-forward-testing-and-live-execution-in-ra-algos.md); [Alpaca](https://alpaca.markets/learn/paper-trading-vs-live-trading-a-data-backed-guide-on-when-to-start-trading-real-money)
- Systematic-review recommendations: monitor model drift and real-time performance, use walk-forward validation (see synthetic-data warning in sec. 1; qualitative only). — [arXiv 2512.10913](https://arxiv.org/html/2512.10913v1)
- Lopez de Prado recommends an "assembly line" organization for ML strategy research (summary). — [GARP page](https://www.garp.org/white-paper/the-10-reasons-most-machine-learning-funds-fail)

### Inferences
- Minimal production safeguards for a solo operator (mapping Knight's failures to a small system): (1) pre-trade checks in the execution module that the model cannot bypass — max order notional/shares, max position per name, max gross/net exposure, price-band check vs last quote, duplicate-order and orders-per-minute throttles; (2) account-level daily loss limit and max-drawdown breaker that flatten/halt; (3) a manual one-command kill switch plus a heartbeat watchdog (halt if data feed/forecaster stale); (4) idempotent order IDs and position reconciliation against the broker; (5) staged deploys (shadow -> paper -> tiny live), with config/code versioning.
- Drift monitoring: track forecast calibration (reliability/CRPS/pinball loss vs realized), PSI/KS on input features, realized vs predicted edge per horizon, and slippage vs modelled cost; trigger "reduce size" at moderate drift and "halt/retrain" at severe drift. Retraining cadence is unsourced; walk-forward re-fit schedules (e.g., the futures study re-fit in ~4-year blocks for daily data) are task-specific, so choose cadence by testing decay of OOS performance vs refit age.
- For a strategy that trades frequently, paper trading mostly validates plumbing and costs; the statistical evidence of edge should come from the purged walk-forward/DSR analysis, then a small-capital live phase measuring *realized slippage*.

### Gaps
- No authoritative source on retraining cadence for intraday/swing ML trading models, nor on recommended paper-trading duration (only blog/forum-level).
- No source found on kill-switch design specific to retail/solo systems (SEC 15c3-5 applies to broker-dealers); crypto exchange-side protections (cancel-on-disconnect, etc.) not researched.
- Canadian-specific regulatory/operational constraints (e.g., IIROC/CIRO rules, TSX order-protection) were not covered.

---

## 9. Candidate end-to-end reference architectures for a solo developer (ranked by realism of success)

### Takeaway
Ranked by realism: (1) forecast-driven, rule/optimizer-based decision layer with fractional-Kelly/vol-target sizing, receding-horizon cost-aware optimization, hard risk limits; (2) the same plus a supervised meta-labeling/gating model; (3) the same plus a constrained RL overlay (position-scaling/timing) validated in shadow mode, and RL used first where it has documented success — execution. A pure end-to-end deep-RL trader on bar data is not recommended.

### Cited Findings (supporting facts; architecture designs below are inferences)
- Weight-vector interface and risk overlay pattern: [FinRL-X](https://arxiv.org/html/2603.21330v1). Multi-period convex optimization with forecasts as input: [cvxportfolio](https://github.com/cvxgrp/cvxportfolio), [Boyd et al.](https://arxiv.org/pdf/1705.00109). Portfolio libs with forecast "views", costs, turnover, CVaR and purged CV: [skfolio](https://skfolio.org/). Fractional Kelly rationale: [Kelly criterion](https://en.wikipedia.org/wiki/Kelly_criterion), [estimation risk](https://arxiv.org/html/2508.18868v2). SB3-Contrib RecurrentPPO/MaskablePPO/TQC: [docs](https://sb3-contrib.readthedocs.io/en/master/). RL's documented industry niche is execution/market making: [Hambly et al.](https://www.arxiv.org/pdf/2112.04553), [LOXM](https://www.technologyreview.com/f/608467/jpmorgan-will-use-ai-to-carry-out-trades-across-the-world/). Recent careful test: learned policy ≈ simple rule net of costs: [arXiv 2607.00475](https://arxiv.org/html/2607.00475v1). Knight-style control failures: [SEC](https://www.sec.gov/news/press-release/2013-222).

### Inferences — the architectures

**Architecture A (most realistic): "Forecast -> expected-utility sizing -> receding-horizon optimizer -> risk gate -> execution"**
1. *Signal*: forecaster outputs per asset & horizon h: mean μ_h, quantiles / σ_h, calibration score, plus regime tag. Persist to a feature/forecast store with timestamps (no look-ahead).
2. *Edge model*: net edge per asset = Σ_h w_h·μ_h − expected cost (spread + fee + slippage ∝ size/ADV, vol); shrink toward 0 by forecast uncertainty; discard if net edge < threshold.
3. *Portfolio construction*: cvxpy or cvxportfolio MPC over the next H bars: maximize forecast return − risk (covariance from skfolio/Riskfolio shrinkage estimators or HRP fallback) − costs − turnover, with exposure/sector/liquidity constraints; fractional Kelly (0.1–0.5) as per-asset cap; vol-target overlay on the total book; no-trade band so you only trade partially toward the aim.
4. *Exit logic*: horizon-matched time stops, thesis-invalidation exits (forecast flips net of costs), position-level risk-budget stop; portfolio-level drawdown breaker (scale down toward floor with cooldown).
5. *Execution*: separate module taking target weights -> orders (limit/marketable-limit, TWAP slicing for size), pre-trade checks, kill switch, reconciliation; Alpaca/IBKR/ccxt adapters. Same code path in backtest, paper, live (the FinRL-X "weights as sole interface" idea).
6. *Validation*: purged/embargoed walk-forward + DSR/PBO; include 1/N and vol-target baselines; paper-trade several months / ≥100 closed trades while logging modelled vs realized slippage; then tiny live.
- Why most realistic: every component is a maintained library or classical result, behaviour is explainable, costs/risk are explicit, and it directly implements multi-horizon "hold today, sell tomorrow".

**Architecture B: A + supervised meta-labeling / bet-size gate**
- Add a LightGBM (or similar) secondary model trained with triple-barrier labels on the primary signal's trades, using inputs such as forecast quantiles, regime, volatility, spread, time-of-day, recent forecast error; calibrate probabilities (the calibration paper suggests this matters most for fixed sizing rules) and use p to scale or veto the Kelly-based size. Train/validate with purged CV (skfolio `CombinatorialPurgedCV` or custom).
- Moderately more complex; adds a second model that can itself overfit; evidence base is modest (see sec. 1), so require that it improves net DSR over A.

**Architecture C (least realistic for alpha, best for experimentation): A + constrained RL overlay (hybrid) and RL execution**
- Overlay: a MaskablePPO/RecurrentPPO or SAC policy (SB3/SB3-Contrib) whose state = forecast term structure + uncertainty, position, PnL, time-in-trade, cost/liquidity features; actions = scale target exposure in {0, 0.25, 0.5, 1}×(cap from A) or exit; reward = net-of-modelled-cost PnL minus drawdown/turnover penalty; trained on walk-forward historical replay with cost/latency/fill randomization and multiple seeds. Runs in **shadow mode** first (logs what it would do), promoted only if it beats A net of costs on purged OOS data and in forward shadow; hard risk limits from A always override.
- Execution RL (separate project, if HFT is a goal): mbt_gym / ABIDES-style simulator (ABIDES-Gym archived June 2025 — would need to be vendored) or Qlib RL order-execution for slicing/placement, where RL has documented industry use; this needs LOB data and an impact-aware simulator (other researchers' scope).
- Not recommended: a single end-to-end DRL agent mapping raw bars to buy/sell/hold (FinRL/gym-anytrading/TensorTrade style) as the primary trading engine — highest overfitting risk, weak evidence, cost sensitivity shown in TensorTrade's own README.

### Gaps
- No evidence base to estimate success probabilities; the ranking is judgement from the cited evidence and engineering risk.
- Asset-class specifics (TSX/TSXV liquidity and lot sizes, crypto 24/7 funding/fees, short-selling availability) were not researched here and materially affect cost models and which actions are even available.

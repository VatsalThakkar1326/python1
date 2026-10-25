# Simulators, Backtesting Engines and Paper-Trading Frameworks for Python (state as of 25 Oct 2026)

Method notes for the report writer: GitHub API access was blocked in this session, so star/fork counts come from GitHub repo pages as rendered when fetched (late Oct 2026; rounded by GitHub, e.g. "30.7k"). Most repo pages did not expose last-commit dates, so release dates come from PyPI and GitHub releases pages where I could get them. Pages were read through a summarising fetch tool, so a few details are second-hand; conflicts between sources are flagged inline. Anything I could not source is listed under Gaps instead of being asserted.

---

## 1. Framework-by-framework comparison: fidelity, speed, Python support, live parity, adapters, maintenance, license

### Takeaway
Only two open-source Python-callable engines in the set model the order book and queue position at a level that can approximate a real venue: NautilusTrader (event-driven, L1/L2/L3 books, queue-position tracking, fill and latency models, backtest/sandbox/live parity, Rust core) and hftbacktest (queue-position and latency models, L2/L3 replay, Numba/Rust, but live bots only in Rust and few releases since Dec 2025). Everything else (Freqtrade, Jesse, vectorbt, backtesting.py, bt, Zipline-reloaded, Backtrader) simulates fills at bar prices; LEAN sits in between and does not natively support TSX. Backtrader is effectively unmaintained.

### Cited Findings

#### Snapshot table (stars/forks as shown on repo page at fetch time; latest release from PyPI or GitHub releases)

| Framework | Stars (forks) | Latest release found | License | Sources |
|---|---|---|---|---|
| NautilusTrader | 30.7k (4.1k) | Stable 1.231.0 (2 Aug 2026, declared last v1/Cython release); v2 pre-release 2.0.0rc7 (15 Oct 2026) | LGPL-3.0 | [GitHub](https://github.com/nautechsystems/nautilus_trader), [releases](https://github.com/nautechsystems/nautilus_trader/releases), [PyPI JSON](https://pypi.org/pypi/nautilus_trader/json) |
| hftbacktest | 5.0k (943) | py 2.4.4 / rust 0.9.4 (10 Dec 2025); nothing newer on PyPI as of Oct 2026 | MIT | [GitHub](https://github.com/nkaz001/hftbacktest), [PyPI](https://pypi.org/project/hftbacktest/), [releases](https://github.com/nkaz001/hftbacktest/releases) |
| ABIDES (jpmorganchase/abides-jpmc-public) | 175 (95) | Repo archived 2 Jun 2025 | BSD-3-Clause | [GitHub](https://github.com/jpmorganchase/abides-jpmc-public) |
| LEAN / QuantConnect | 22.2k (5.3k) | Not verified (see Gaps) | Apache-2.0 | [GitHub](https://github.com/QuantConnect/Lean) |
| Backtrader | 23.6k (5.3k) | 1.9.78.123 (19 Apr 2023) | GPL-3.0 | [GitHub](https://github.com/mementum/backtrader), [PyPI](https://pypi.org/project/backtrader/) |
| zipline-reloaded | 2.0k (335) | Not retrieved | Apache-2.0 | [GitHub](https://github.com/stefan-jansen/zipline-reloaded) |
| vectorbt (open source) | not retrieved | 1.2.0 (19 Oct 2026) | Apache-2.0 + Commons Clause | [PyPI](https://pypi.org/project/vectorbt/) |
| backtesting.py | not retrieved | 0.6.6 (22 Jul 2026) | AGPL-3.0+ | [PyPI](https://pypi.org/project/backtesting/) |
| bt | 3.0k (507) | not retrieved | MIT | [GitHub](https://github.com/pmorissette/bt) |
| Freqtrade | not retrieved | 2026.9 (29 Sep 2026) | GPL-3.0 | [PyPI](https://pypi.org/project/freqtrade/) |
| Hummingbot | 20.7k (5.0k) | not shown | Apache-2.0 | [GitHub](https://github.com/hummingbot/hummingbot) |
| Jesse | 8.7k (1.3k) | not shown | MIT | [GitHub](https://github.com/jesse-ai/jesse) |
| OctoBot | 6.9k (1.3k) | not shown | GPL-3.0 | [GitHub](https://github.com/Drakkar-Software/OctoBot) |
| MarS (Microsoft) | ~1.9k (206) | v0.1.0 "first release" | MIT | [GitHub](https://github.com/microsoft/MarS) |

#### Maintenance flags
- Backtrader: last PyPI release 1.9.78.123 on 19 Apr 2023, with Python classifiers only up to 3.7 — [PyPI](https://pypi.org/project/backtrader/). The GitHub page shows 23.6k stars, 2,404 commits, no archived banner, and an empty Releases section, so it is not formally archived but has had no release in about 3.5 years — [GitHub](https://github.com/mementum/backtrader).
- ABIDES (JPMC public repo): "archived by the owner on Jun 2, 2025. It is now read-only"; README states "We do not do technical support, nor consulting" — [GitHub](https://github.com/jpmorganchase/abides-jpmc-public).
- zipline-reloaded: README says it is maintained for bug, security and compatibility fixes, with new feature work moved to "ML4T Backtest"; next release targets Python 3.11-3.14; pandas must be below 3.0 — [GitHub](https://github.com/stefan-jansen/zipline-reloaded). This is a maintenance-mode project.
- hftbacktest: release cadence in 2025 was 26 May, 3 Aug, 11 Aug, 24 Aug, 30 Sep, 10 Dec (gaps of 8 to 71 days); the latest PyPI release is still 2.4.4 dated 10 Dec 2025 — [releases](https://github.com/nkaz001/hftbacktest/releases), [PyPI](https://pypi.org/project/hftbacktest/). Recent release notes are mostly small fixes and connector/data-collector additions (Hyperliquid collector, Binance Spot connector, Bybit converter).
- NautilusTrader is very active: 22,004 commits on develop; releases 2.0.0rc3 (21 Aug 2026), rc4 (2 Sep), rc5 (15 Sep), rc6 (5 Oct), rc7 (15 Oct 2026) — [GitHub](https://github.com/nautechsystems/nautilus_trader), [releases](https://github.com/nautechsystems/nautilus_trader/releases).
- vectorbt: 1.2.0 (19 Oct 2026), 1.1.2 (9 Oct), 1.1.1 (26 Sep), 1.1.0 (5 Jul), 1.0.0 (22 Apr), 0.28.5 (26 Mar); single maintainer (polakowo) — [PyPI](https://pypi.org/project/vectorbt/).
- backtesting.py 0.6.6 on 22 Jul 2026 and Freqtrade 2026.9 on 29 Sep 2026 are both current — [PyPI backtesting](https://pypi.org/project/backtesting/), [PyPI freqtrade](https://pypi.org/project/freqtrade/).

#### Brief capability profile per framework
- **vectorbt / vectorbt PRO**: "vectorized backtesting that packs many configurations into NumPy arrays, with Numba and an optional Rust engine". The open-source package is described as the community edition of VectorBT PRO; PRO adds "parallelization, additional data integrations, portfolio optimization, pattern recognition, event projections, limit orders, leverage" and contract multipliers for futures. License is "fair-code": Apache 2.0 with Commons Clause (cannot sell products/services primarily based on it) — [PyPI](https://pypi.org/project/vectorbt/). Python >=3.11,<3.15.
- **backtesting.py**: single-asset-oriented framework with optimizer and interactive plots, AGPL-3.0+ (copyleft; matters if code is ever distributed or served) — [PyPI](https://pypi.org/project/backtesting/).
- **bt**: "flexible backtesting for Python"; page describes backtesting, stats and charts and does not mention live or paper trading — [GitHub](https://github.com/pmorissette/bt).
- **Backtrader**: event-driven; live support via Interactive Brokers (IbPy), Visual Chart and Oanda (REST only, "v20 did not support streaming when implemented") — [GitHub](https://github.com/mementum/backtrader).
- **zipline-reloaded**: "Pythonic, event-driven backtesting system" originally from Quantopian; the only bundle named in the README is Quandl, which needs a NASDAQ Data Link API key — [GitHub](https://github.com/stefan-jansen/zipline-reloaded).
- **Freqtrade**: crypto-only bot; spot exchanges Binance, BingX, Bitget, Bybit, Gate, HTX, Hyperliquid, Kraken, OKX, MyOKX and others; futures on Binance, Bitget, Bybit, Gate, Hyperliquid, Kraken, OKX; Python 3.11+; dry-run and live modes; backtesting and hyperopt; FreqAI ML module — [Freqtrade docs](https://www.freqtrade.io/en/stable/).
- **Hummingbot**: "high-frequency crypto trading bots"; CLOB CEX, CLOB DEX and AMM DEX connectors; V2 strategy framework (scripts, controllers, executors); V2 controller configs can be "backtested, deployed, and tuned live"; separate Hummingbot API and Gateway; Condor LLM harness — [GitHub](https://github.com/hummingbot/hummingbot).
- **Jesse**: crypto framework, MIT; paper trading, multi-symbol/multi-timeframe, spot and futures, shorting, "partial fills", Optuna/Ray optimisation, Monte Carlo and rule significance testing, scikit-learn ML pipeline, MCP server; reinforcement learning listed as "Coming Soon"; no speed benchmark figures given — [GitHub](https://github.com/jesse-ai/jesse).
- **OctoBot**: GPL-3.0 crypto bot, 15+ exchanges via CCXT; grid, DCA, TradingView, ChatGPT/Ollama modes; paper trading; built-in backtesting only in manual instances, not node mode; no stock/TradFi support mentioned — [GitHub](https://github.com/Drakkar-Software/OctoBot).

### Inferences
- For TSX/TSXV + US equities/options + crypto in one engine, the realistic shortlist is NautilusTrader (generic instruments, multi-venue, IB + crypto adapters) and, with caveats, LEAN. Freqtrade, Jesse, OctoBot and Hummingbot are crypto-only by their own descriptions and so cannot be the unified platform; they are useful only as reference designs for dry-run/paper workflows.
- Backtrader and (to a lesser degree) zipline-reloaded carry maintenance risk for a new build started in late 2026; vectorbt and backtesting.py are healthy but are research/screening tools that fill at bar prices.
- License matters: Nautilus is LGPL (fine for private use and linking), backtesting.py is AGPL, Freqtrade/Backtrader/OctoBot are GPL, vectorbt is Commons Clause. A solo developer trading privately is mostly unaffected, but anyone planning to distribute a product should check each.

### Gaps
- Last-commit dates and latest-release dates for Hummingbot, Jesse, OctoBot, zipline-reloaded, LEAN, bt, Backtrader (GitHub), and stars for Freqtrade, vectorbt, backtesting.py were not retrievable (GitHub API blocked; releases sections rendered empty).
- Whether the original `abides-sim/abides` repo is maintained was not checked.
- Minor source conflicts: Nautilus is shown as "LGPL-3.0" on the GitHub page (security section says "LGPL-3.0-only") but "LGPL-3.0-or-later" in PyPI metadata ([GitHub](https://github.com/nautechsystems/nautilus_trader) vs [PyPI JSON](https://pypi.org/pypi/nautilus_trader/json)).

---

## 2. NautilusTrader in depth (order book simulation, fill/latency models, adapters, backtest-live parity, v2 transition)

### Takeaway
NautilusTrader is the closest open-source match to "one engine, backtest to sandbox to live" with a realistic matching engine, but it is mid-migration: 1.231.0 (2 Aug 2026) is the last Cython-core release and the Rust-native v2 is at release candidate 7 (15 Oct 2026) with frequent breaking changes. Its queue-position and fill models are explicitly approximations; its latency and market-impact modelling is thinner than hftbacktest's latency model or any agent-based simulator.

### Cited Findings
- Description: "Production-grade Rust-native trading engine with deterministic event-driven architecture"; backtests multiple venues, instruments and strategies at once on quote tick, trade tick, bar, order book and custom data with nanosecond resolution; TIFs include IOC, FOK, GTC, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE; contingency orders OCO/OUO/OTO; post-only, reduce-only, iceberg — [GitHub](https://github.com/nautechsystems/nautilus_trader).
- Rust core with Python control plane via PyO3; v1 (Cython) now only gets critical security backports on `develop_v1`; `pip install nautilus_trader` without `--pre` installs the old v1 wheel with a different API; v2 needs `--pre`; Python 3.13-3.14 supported for v2; platforms Linux x86_64/ARM64, macOS ARM64, Windows x86_64 — [GitHub](https://github.com/nautechsystems/nautilus_trader).
- v1 to v2 timeline: 1.231.0 (2 Aug 2026) is "the planned final 1.x release", v1 gets about three months of security backports; 2.0.0rc3 (21 Aug) removed the legacy Cython v1 package and added Python v2 `LiveNode.run_async()` and Redis-backed cache/message bus; rc4 (2 Sep) changed backtest defaults (leverage 10x for margin accounts, 1x for cash), removed `settlement_prices` in favour of `InstrumentClose`, added custom Python fee models; rc5 (15 Sep) added inbound latency modelling to the sandbox execution client; rc6 (5 Oct) moved maker/taker fee rates from instruments to venue `fee_model` and removed the BitMEX adapter; rc7 (15 Oct) dropped Python 3.12, batched book deltas by default in `BacktestNode` so "fills can differ" (opt-out `batch_deltas`). v2 is still a release candidate, final 2.0.0 not shipped — [releases](https://github.com/nautechsystems/nautilus_trader/releases).
- PyPI stable line: 1.231.0, requires Python `>=3.12,<3.15`, LGPL-3.0-or-later — [PyPI JSON](https://pypi.org/pypi/nautilus_trader/json).
- Book types and data: venue `book_type` must match data granularity; Nautilus "cannot generate higher granularity data" from quotes/trades/bars. L1_MBP (default): quotes and bars update the book, trades trigger matching, deltas ignored. L2_MBP and L3_MBO: deltas/depth update the book, trades trigger matching, quotes and bars ignored. Data hierarchy most to least detailed: L3 MBO, L2 MBP, L1 quotes, trades, bars; bars "can't show intrabar order, spread, depth, or queue position". Sandbox execution also needs `book_type` L2/L3 to use deltas — [Data and venues](https://nautilustrader.io/docs/latest/concepts/backtesting/data-and-venues/).
- Fill models: shared parameters `prob_fill_on_limit` (default 1.0; applies when market touches but does not cross the limit price) and `prob_slippage` (default 0.0; L1 only, one-tick adverse move with given probability), `random_seed`. Built-in classes include DefaultFillModel, BestPriceFillModel (unlimited size at best), OneTickSlippageFillModel, ProbabilisticFillModel, TwoTierFillModel (10 at best, rest one tick worse), ThreeTierFillModel (50/30/20), LimitOrderPartialFillModel (5 at best), SizeAwareFillModel, CompetitionAwareFillModel (`liquidity_factor` default 0.3 of 1,000 units), VolumeSensitiveFillModel (25% of internal volume at best), MarketHoursFillModel (normal vs one-tick-wider spread). On L2/L3 the recorded book supplies prices/sizes and `prob_slippage` does not apply. Tier sizes are constants in instrument units; custom FillModel subclasses are low-level-engine only; `BacktestVenueConfig` accepts only built-in models — [Fill models](https://nautilustrader.io/docs/latest/concepts/backtesting/fill-models/).
- Queue position and trade-based execution: `queue_position` (needs `trade_execution=True`) tracks displayed quantity ahead of each LIMIT order; correct-side trades at the price reduce it; order becomes eligible at zero; book deletes/updates/snapshots adjust it; price change resets it. Stated limitations: LIMIT only; each order has an independent queue estimate; initial estimate uses only book state visible at acceptance; hidden orders and venue-specific priority rules not captured; `NO_AGGRESSOR` trades can clear queues early, which is optimistic. `liquidity_consumption=True` stops the same displayed liquidity (or trade size) filling multiple orders in one iteration; default is False. Trade-driven fills capped at `min(order.leaves_qty, trade.size)` — [Trade execution](https://nautilustrader.io/docs/latest/concepts/backtesting/trade-execution/).
- Venue config reference lists `book_type` (default L1_MBP), `queue_position` (default False), and a `latency_model` option; the Rust backtest crate describes configurable latency and fill models — [config reference (nightly)](https://nautilustrader.io/docs/nightly/api_reference/config), [docs.rs nautilus-backtest](https://docs.rs/nautilus-backtest). Note these pages are the nightly docs and may not match the stable release.
- Parity: "The same strategy and execution-algorithm code can run across backtest and live environments"; docs also warn live adds "venue rules, transport failures, timing, persistence, external order activity, and reconciliation" that a simulation may not reproduce; three environment contexts share one core: Backtest, Sandbox (live data + virtual execution), Live — [Live concepts](https://nautilustrader.io/docs/latest/concepts/live/), [Overview](https://nautilustrader.io/docs/latest/concepts/overview/).
- Live operational limits: run one concurrent `LiveNode` per process; multiple execution clients for one venue need explicit routing and do not give end-to-end account isolation; runner channels are unbounded with no backpressure; startup reconciliation aligns cached state with venue reports — [Live concepts](https://nautilustrader.io/docs/latest/concepts/live/).
- Adapters (21 listed on the docs integrations page): Stable: AX Exchange, Betfair, Binance, Blockchain (DeFi data), Bybit, Coinbase, Databento, Deribit, Derive, dYdX, Hyperliquid, Interactive Brokers (multi-venue brokerage), Kraken, Lighter, Lighter on Robinhood, OKX, Polymarket, Tardis; Beta: CoinAPI; Planned: FinFeed, Kalshi — [Integrations](https://nautilustrader.io/docs/latest/integrations/). The GitHub README labels FinFeed and Kalshi "Building" rather than "Planned" — [GitHub](https://github.com/nautechsystems/nautilus_trader).
- No native Gymnasium wrapper surfaced in searches; docs list "training AI trading agents with reinforcement learning (RL) or evolutionary strategies (ES)" as a use case of its high-throughput backtester, and a secondary source says it is single-node with built-in ML tooling out of scope per roadmap — [Overview](https://nautilustrader.io/docs/latest/concepts/overview/), [OpenSourceAlternatives (secondary)](https://www.opensourcealternatives.to/item/nautilus-trader).

### Inferences
- Nautilus has the right abstractions for the user's three asset classes (IB for US/Canadian equities and options, Binance/Bybit/OKX/Kraken/Coinbase for crypto, Databento/Tardis for data), but TSX/TSXV-specific behaviour (tick-size tables, board lots, halts, auctions) would have to be supplied through instrument definitions and custom data; I found no Nautilus doc stating TSX support.
- Starting on v1 (1.231.0) now means migrating again within months; starting on v2 RC means accepting breaking API changes every release. This is the main near-term risk of choosing Nautilus.
- For fill realism the best Nautilus configuration is L2/L3 deltas + trade ticks + `queue_position=True` + `liquidity_consumption=True` + latency model; with only bars/quotes the engine is a bar-level simulator like the others (the docs say so explicitly).

### Gaps
- Latency model semantics (insert/update/cancel latencies, whether they apply to market data too) were not available from the pages I could read; the `fill-prices-and-matching` page fetch failed (typo in my URL host), so auction/open-close matching and stop-trigger semantics in Nautilus are unverified beyond the TIF list.
- No published Nautilus throughput (events/sec) figure found in docs; CodSpeed benchmarks linked from the README were not read.
- Nautilus sandbox Python usage details (which venues' data feeds, how fees and latency are configured) were only confirmed through Rust API docs: [docs.rs nautilus-sandbox](https://docs.rs/nautilus-sandbox), [SandboxExecutionClient](https://nautilustrader.io/docs/core-nightly/nautilus_sandbox/execution/struct.SandboxExecutionClient.html).

---

## 3. hftbacktest in depth (queue models, latency, L2/L3, Rust/Python, live)

### Takeaway
hftbacktest is the most explicit about queue position and latency (pluggable queue-position models, feed latency and order-entry/response latency models using real measured latency), and it states plainly that it does not model market impact. It is HFT-oriented and crypto-focused (Binance, Bybit, Hyperliquid, Tardis data), live trading is Rust-only, and the last release was Dec 2025.

### Cited Findings
- Features: Numba-JIT Python and Rust; tick-by-tick simulation on a fixed interval or driven by feed/order-receipt events; reconstructs full book from L2 market-by-price and L3 market-by-order feeds; models feed and order latency; queue-position-aware fills; multi-asset, multi-exchange; live bots for Binance Futures and Bybit, Rust only; MIT; Python 3.11+; 5.0k stars — [GitHub](https://github.com/nkaz001/hftbacktest).
- Exchange (fill) models: `NoPartialFillExchange` (default) fills orders fully when price crosses best quote or a trade prints through, or when at front of queue at the trade price; liquidity-taking orders fill fully at best price regardless of available size ("can be unrealistic for large orders"). `PartialFillExchange` allows partial fills when at the front of the queue and fills takers against book quantity, but the docs warn this can still be unrealistic because replayed data cannot reflect your own execution's effect on the book — [Order fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html).
- Queue-position models: `RiskAverseQueueModel` (most conservative; cancellations/modifications only reduce quantity behind your order, position advances only on trades at your price); `ProbQueueModel` with probability functions (power variants PowerProbQueueFunc/2/3 and log variants LogProbQueueFunc/2; probability must be 0 at head of queue and 1 at tail); a Rust `L3QueueModel` trait exists for custom models — [Order fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html).
- Market impact: "your order cannot make any changes to the simulated market, no market impact is considered"; key assumption is orders are small enough to have no impact; docs recommend testing live and adjusting for differences — [Order fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html).
- Latency: three types (feed latency; order entry latency; order response latency). Models: `ConstantLatency`; `IntpOrderLatency` (interpolates recorded order latency records of request-local, exchange, response-local timestamps; "most accurate option when the data has fine time intervals"); `FeedLatency` (synthesises order latency from feed latency); custom `LatencyModel` trait in Rust. Docs suggest collecting latency data "by submitting unexecutable orders regularly" — [Latency models docs](https://hftbacktest.readthedocs.io/en/latest/latency_models.html).
- Data/collector additions in 2025 releases: Hyperliquid in the data collector, Binance Spot connector, Bybit converter, Tardis converter fix — [releases](https://github.com/nkaz001/hftbacktest/releases).
- Release recency: PyPI 2.4.4 dated 10 Dec 2025, Python >=3.11, MIT — [PyPI](https://pypi.org/project/hftbacktest/).

### Inferences
- hftbacktest is the right tool when the research question is "does this quoting/market-making/short-horizon alpha survive queue position and latency?" on crypto perps/spot with L2/L3 data. It is not a portfolio/multi-asset-class platform and has no equities broker adapters in the sources I read.
- Because live trading is Rust-only, "same code from backtest to live" for a Python user means rewriting the live leg in Rust or treating hftbacktest purely as a research simulator and executing via another stack.
- Zero market impact + latency/queue modelling means results are valid only for small order sizes relative to displayed liquidity.

### Gaps
- Docs conflict on L3: the README claims L3 market-by-order reconstruction while the order-fill page says "HftBacktest currently supports only Market-By-Price data" (page possibly outdated) — [GitHub](https://github.com/nkaz001/hftbacktest) vs [Order fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html). I could not resolve which is current.
- No throughput benchmark (events/sec) found; README gives none.
- The Dec 2025 release being the latest suggests a slowdown, but I could not read commit activity to confirm whether development continues on master.
- I did not verify any equities (TSX/US) data or order-routing support; the sources only mention crypto exchanges.

---

## 4. ABIDES / ABIDES-Gym and LEAN/QuantConnect (including TSX support)

### Takeaway
ABIDES is a NASDAQ-like multi-agent discrete-event simulator with a Gym wrapper, useful for training agents against reactive background agents, but the JPMC repo was archived on 2 Jun 2025 and the synthetic market has known unrealistic behaviours; it cannot replay real data. LEAN is a mature multi-asset engine with local (Docker/CLI) and cloud modes and broad brokerage support, but its market list contains no Canada/TSX entry, default slippage is zero, and prebuilt fill models do not produce partial fills.

### Cited Findings
**ABIDES**
- Repo: 175 stars, 95 forks, BSD-3-Clause, archived 2 Jun 2025 (read-only), "We do not do technical support, nor consulting" — [GitHub](https://github.com/jpmorganchase/abides-jpmc-public).
- Components: ABIDES-Core (general multi-agent discrete-event simulator), ABIDES-Markets (NASDAQ-mimicking exchange, stylised agents: market maker incl. POV maker, value, momentum, noise), ABIDES-Gym (OpenAI Gym wrapper with two environments); messaging supports latency models — [GitHub](https://github.com/jpmorganchase/abides-jpmc-public).
- ABIDES-Gym design: a "Gym agent" sends an interruption and raw state to the kernel at each wake-up; `reset()` and `step(a)` drive the simulation; environments: Daily Investor (BUY/HOLD/SELL fixed-size market orders, wakes every minute from 09:35, reward is change in mark-to-market value) and Algorithmic Execution (default parent order 20,000 shares over 4 hours, market/limit/no-op actions, wakes every 10 seconds, end-of-episode penalty for unexecuted shares). "An optional latency model is applied to the messaging system"; no limitations section; paper notes rare unrealistic behaviours in the synthetic market causing positive reward spikes; authors argue replaying history assumes the agent's trades don't affect prices — [ABIDES-Gym paper](https://arxiv.org/html/2110.14771).
- ABIDES-MARL extends ABIDES-Gym to multiple adaptive agents while staying compatible with standard RL libraries — [arXiv 2511.02016](https://arxiv.org/abs/2511.02016).

**LEAN / QuantConnect**
- Repo: 22.2k stars, 5.3k forks, Apache-2.0, 13,408 commits; Python and C#; `pip install lean` CLI with `backtest`, `optimize`, `research`, `live`; local runs via Docker plus a cloud hybrid — [GitHub](https://github.com/QuantConnect/Lean).
- Markets: `Common/Market.cs` defines USA, FXCM, Oanda, Dukascopy, Bitfinex, CME Globex, NYMEX, CBOT, ICE, CBOE, India, Kraken, Binance, Bitstamp, COMEX, CME, SGX, HKFE, NYSELIFFE, CFE, BinanceUS, Bybit, Coinbase, InteractiveBrokers, EUREX, OSE, DYDX, KRX and others, with no Canada, TSX or Toronto entry (markets can be added at runtime via `Market.Add()`) — [Market.cs](https://raw.githubusercontent.com/QuantConnect/Lean/master/Common/Market.cs).
- Staff forum answers (undated, "several years old"): the platform does not directly support trading non-US equity markets; international use requires local LEAN, creating your own market, putting your own data in the data folder and updating the symbol-properties and market-hours databases, "not a small task but it is possible"; one thread notes non-USD account currency "isn't officially supported yet" — [forum 1269](https://www.quantconnect.com/forum/discussion/1269/trading-and-back-testing-international-stocks-on-quantconnect/), [forum 9250](https://www.quantconnect.com/forum/discussion/9250/adding-a-complete-new-exchange-data/).
- US equity data in the cloud platform is the US SIP CTA/UTP feed — [QC US equity docs](https://www.quantconnect.com/docs/v2//writing-algorithms/securities/asset-classes/us-equity/requesting-data).
- Brokerage models (22 listed, including QuantConnect Paper Trading): Interactive Brokers, TradeStation, Tastytrade, Public, Alpaca, Charles Schwab, Clear Street, Webull, Binance (incl. BinanceUS), Bybit, Tradier, Kraken, Coinbase, Bitfinex, dYdX, Terminal Link, Bloomberg FIX, SSC Eze, Trading Technologies, Wolverine, Oanda — [QC brokerage models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/brokerages/supported-models).
- Slippage: `DefaultBrokerageModel` assigns `NullSlippageModel` (zero slippage) by default; `VolumeShareSlippageModel` scales slippage with order size relative to bar volume and "is more accurate with denser resolution" — [QC slippage](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts).
- Fills: brokerage models assign `EquityFillModel`, `FutureFillModel`, `FutureOptionFillModel`, else `ImmediateFillModel`; pre-built models assume orders "completely fill" in backtests, partial fills need a custom fill model (example caps fills at 30% of previous second bar volume); stale-price handling (threshold `stale_price_time_span`) fills only market orders on stale data; built-in models can fill US equity orders at official opening/closing auction prices — [QC trade fills](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts).
- Corporate actions: Split, Dividend, Delisting and SymbolChanged events are delivered to the algorithm; data normalization modes RAW, SCALED_RAW, ADJUSTED, TOTAL_RETURN; unfilled orders are adjusted on splits; delistings trigger warnings then liquidation — [QC corporate actions](https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/corporate-actions).

### Inferences
- LEAN is the strongest of the bar-level engines for US equities/options hygiene (corporate actions, delistings, auctions, brokerage models), but for TSX/TSXV the user would be building a custom market, custom data and calendar entries, so it is not a turnkey fit; Nautilus (generic instruments + IB adapter) looks lower-friction, though neither was verified for TSX.
- ABIDES is best used as a stress-test/training environment for execution or market-making agents against reactive agents, not for validating alpha on real historical data; its archived state means no upstream fixes.

### Gaps
- LEAN latest release/tag date and last commit date were not retrievable (the GitHub releases page rendered only old 2018-era releases, which I did not use).
- LEAN's paper-trading page returned 404, so details of paper brokerage behaviour (which fill model, cost, local vs cloud) are not confirmed beyond the "QuantConnect Paper Trading" brokerage-model listing.
- Whether QuantConnect's data library offers any Canadian equity dataset today was not confirmed; only older forum statements found.
- ABIDES-Gym throughput (steps/sec or wall-clock per simulated day) is not reported in the paper.

---

## 5. Newer/academic simulators (JAX-LOB, PyMarketSim, mbt_gym, MarS, LOB-Bench) and what each offers for market impact; replay vs agent-based vs generative

### Takeaway
Pure historical replay (hftbacktest, Nautilus L2/L3, bar engines) cannot show how other participants react to your orders; agent-based (ABIDES, PyMarketSim), model-based/analytic (mbt_gym), calibrated-feedback (Noble-Rosenbaum-Souilmi 2026) and generative (MarS) simulators each add an impact mechanism, but all are research-grade, and impact realism is still an open validation problem. LOB-Bench exists to measure generated-LOB realism including price-impact metrics.

### Cited Findings
- Replay limitation: Vyetrenko et al. argue historical replay cannot show the market's response to an agent's orders and many simulators "fail to reproduce statistics and stylized facts seen in real markets"; their paper catalogs realism metrics and finds "significant remaining discrepancies" — [Get Real, arXiv 1912.04941](https://arxiv.org/abs/1912.04941) (the replay-insufficiency statement is from a search summary of this paper: [search result via alphaxiv](https://alphaxiv.org/abs/1912.04941)).
- Mechanical vs reactive impact: reconstruction-based backtests (rebuild the book, insert hypothetical orders, simulate forward) capture "mechanical" impact but not reactive effects — [Kearns talk slides](https://www.cis.upenn.edu/~mkearns/talks/limstab.pdf) (older, 2006 per search summary).
- Calibrated interactive simulator (25 Mar 2026): large-tick assets, book state from spread and volume imbalance, event timing calibrated including a peak at exchange round-trip latency, feedback via signed trade flow with a power-law decay kernel giving concave impact during execution and partial reversion afterwards; reports profitability "highly sensitive to execution parameters"; no code link found — [arXiv 2603.24137](https://arxiv.org/abs/2603.24137).
- Queue-reactive deep-learning LOB simulation notes that execution-strategy backtests must replicate impact (concave rise, partial reversion) — [arXiv 2501.08822](https://arxiv.org/pdf/2501.08822) (via search summary). Survey: [Limit Order Book Simulations: A Review, arXiv 2402.17359](https://arxiv.org/pdf/2402.17359).
- JAX-LOB (Frey, Li, Nagy, Sapora, Lu, Zohren, Foerster, Calinescu; Aug 2023): "first GPU-enabled LOB simulator built to process thousands of books in parallel", "notably reduced per-message processing time", integrated with other JAX packages for end-to-end GPU RL training on an optimal-execution task — [arXiv 2308.13289](https://arxiv.org/abs/2308.13289). A search-engine summary of the arXiv versions quoted a "up to 75x" per-message speedup, but the abstract page I fetched gives no number, so treat 75x as unverified — [arXiv PDF](https://arxiv.org/pdf/2308.13289).
- mbt_gym (Jerome, Sanchez-Betancourt, Savani, Herdegen; ICAIF '23): Python gym environments for model-based LOB problems (market making, optimal execution), combinable across models, with vectorised implementations for faster RL; code at github.com/JJJerome/mbt_gym — [arXiv 2209.07823](https://arxiv.org/pdf/2209.07823), [Liverpool repository](https://livrepository.liverpool.ac.uk/3177015/).
- PyMarketSim: described in a WSC 2025 paper as an update to earlier work adding an RL-based background agent (Mascioli et al. 2024); predecessor JLMSim did not allow complex custom (e.g., ML) agents and its source was not released; another 2025 arXiv paper describes PyMarketSim as a modernised successor of MarketSim with a deep-RL agent — [INFORMS WSC25 paper](https://www.informs-sim.org/wsc25papers/con277.pdf), [arXiv 2502.16343](https://arxiv.org/pdf/2502.16343). I did not find its repo, license or stars.
- MarS (Microsoft Research Asia; ICLR 2025; arXiv 2409.07486): Large Market Model, an order-level generative foundation model that generates individual order events (price, volume, direction) so prices emerge from orders; stated uses include forecasting, detection, analysis and agent-training environments; "controlled generation with market impact" is a stated objective — [arXiv 2409.07486](https://arxiv.org/abs/2409.07486v2), [ICLR page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6261d3bc9e9326ddff055595aabd54e1-Abstract-Conference.html).
- MarS repo (~1.9k stars, MIT, 79 commits, v0.1.0 "first release"): ships a Market Impact Analysis tool that compares a baseline simulation with one including a TWAP agent; released order models 2M, 5M, 10M parameters on Hugging Face (larger ones pending legal approval; the README also says the pretrained model is "undergoing internal review", which conflicts with the release table); needs full order-level historical data for production use; Docker strongly recommended and non-Docker install unsupported; order model served via Ray Serve; Python 3.11/3.12 on Linux; "production deployment would need substantial inference optimization" — [GitHub](https://github.com/microsoft/MarS).
- LOB-Bench (Nagy, Frey et al.; arXiv 2502.09172, ICML 2025): Python tool evaluating generated message-by-order data in LOBSTER format using distributional differences in conditional/unconditional statistics (spread, volumes, order imbalance, inter-arrival times), a discriminator score, and "market impact metrics" (cross-correlations and price response functions); found autoregressive generative models beat a parametric LOB model and a (C)GAN; code at peernagy/lob_bench — [arXiv 2502.09172](https://arxiv.org/pdf/2502.09172).
- Taxonomy for the report: replay (hftbacktest, Nautilus, bar engines): historical data only, zero or heuristic impact; agent-based (ABIDES, PyMarketSim): impact via reactions of rule-based/RL background agents, realism depends on calibration; model-based/analytic (mbt_gym): stylised dynamics for market making/execution research; generative (MarS, deep queue-reactive models, LOB-Bench-evaluated models): learned order-flow models with impact emerging from generated reactions.

### Inferences
- None of the academic simulators supports TSX/US equity/options/crypto live parity, broker adapters, or fees/halts/corporate actions; they are training or stress-testing environments to be paired with an engine like Nautilus for realism and live parity.
- For a solo developer, the practical use is: (i) JAX-LOB or mbt_gym for fast RL prototyping of execution/market-making policies, (ii) ABIDES or MarS to examine how a strategy's footprint changes outcomes, (iii) LOB-Bench-style metrics to check whether any generative simulator is realistic enough to trust.
- Market impact for a small solo account is likely small relative to displayed liquidity in liquid names; it matters more for TSXV/illiquid names and size. (This is a judgement, not sourced.)

### Gaps
- PyMarketSim repo/license/maintenance, JAX-LOB repo URL and verified throughput numbers, mbt_gym stars/last commit, LOB-Bench stars were not retrieved.
- No independent evidence that MarS-generated impact matches real impact for equities beyond the authors' own claims; MarS data requirements (order-level history, China-market focus in the paper) were not checked against TSX/US data.
- I did not find a quantitative live-vs-backtest gap study; the closest is the 2026 reality-gap paper's sensitivity finding.

---

## 6. Sources of sim-to-real gap and which frameworks model which

### Takeaway
Realism is bounded by the data granularity you replay (bars, quotes, L2, L3), then by explicit models of queue position, partial fills, latency, fees and market impact. Only Nautilus (queue/liquidity consumption/fill models) and hftbacktest (queue/latency models) address microstructure; LEAN and bar engines are strongest on corporate actions and account mechanics; no source I found models short-sale borrow costs, halts or survivorship in a documented, built-in way (except LEAN's delisting handling).

### Cited Findings
- **Fill at bar prices / no order book**: Freqtrade backtests enter at the open (unless custom price logic), fill at the requested price with no slippage if within the candle's high/low, exit signals fill at next candle open, and "Order book depth and slippage are not modeled" — [Freqtrade backtesting](https://www.freqtrade.io/en/stable/backtesting/). LEAN defaults to zero slippage and complete fills — [QC slippage](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts), [QC fills](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts).
- **Intrabar ordering**: Freqtrade evaluates exit signal, stoploss, ROI, trailing stop in that order within a candle and assumes low occurs before high for stoploss; it warns backtests can show more stoploss exits than dry-run/live and recommends `--timeframe-detail` (e.g., 5m detail for a 1h strategy) — [Freqtrade backtesting](https://www.freqtrade.io/en/stable/backtesting/). Nautilus: bars "can't show intrabar order, spread, depth, or queue position" — [Nautilus data and venues](https://nautilustrader.io/docs/latest/concepts/backtesting/data-and-venues/).
- **Queue position**: modelled by Nautilus (`queue_position`, optimistic with NO_AGGRESSOR trades, per-order independent estimate, hidden orders not captured) and hftbacktest (RiskAverse and probabilistic queue models) — [Nautilus trade execution](https://nautilustrader.io/docs/latest/concepts/backtesting/trade-execution/), [hftbacktest order fill](https://hftbacktest.readthedocs.io/en/latest/order_fill.html). Nautilus `prob_fill_on_limit` is a touch probability, not a queue simulation — [Nautilus fill models](https://nautilustrader.io/docs/latest/concepts/backtesting/fill-models/).
- **Partial fills**: hftbacktest `PartialFillExchange`; Nautilus liquidity consumption and tiered/partial fill models, trade-fill cap at `min(leaves_qty, trade.size)`; LEAN needs custom fill model (prebuilt assume complete fills); Jesse lists "partial fills" — sources as above plus [Jesse](https://github.com/jesse-ai/jesse).
- **Latency (data + order round trip)**: hftbacktest feed latency, order entry and response latency with measured-latency interpolation — [hftbacktest latency](https://hftbacktest.readthedocs.io/en/latest/latency_models.html); Nautilus `latency_model` venue option and sandbox inbound latency (rc5) — [config](https://nautilustrader.io/docs/nightly/api_reference/config), [releases](https://github.com/nautechsystems/nautilus_trader/releases); ABIDES messaging latency model — [ABIDES-Gym paper](https://arxiv.org/html/2110.14771). The 2026 reality-gap paper calibrates a peak at exchange round-trip latency attributed to latency races — [arXiv 2603.24137](https://arxiv.org/abs/2603.24137).
- **Market impact**: not modelled in replay (hftbacktest says so explicitly); agent-based/generative/feedback simulators add it (Section 5).
- **Fees/rebates**: Freqtrade applies exchange default fee per pair, `--fee` applied twice (entry and exit) — [Freqtrade backtesting](https://www.freqtrade.io/en/stable/backtesting/); Nautilus moved maker/taker fee rates to a venue `fee_model` with per-instrument overrides and custom Python fee models (rc4/rc6) — [releases](https://github.com/nautechsystems/nautilus_trader/releases).
- **Historic exchange rules unavailable**: Freqtrade uses current exchange trading limits and precision because historic ones are unavailable — [Freqtrade backtesting](https://www.freqtrade.io/en/stable/backtesting/). (Relevant analogue for tick-size/lot-size drift on TSX.)
- **Corporate actions / delistings / survivorship**: LEAN delivers Split, Dividend, Delisting and SymbolChanged events, adjusts open orders on splits, offers RAW/ADJUSTED/TOTAL_RETURN normalization — [QC corporate actions](https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/corporate-actions). Nautilus rc4 replaced `settlement_prices` with `InstrumentClose` expiry data for derivatives — [releases](https://github.com/nautechsystems/nautilus_trader/releases).
- **Auctions / open-close**: LEAN built-in models can fill US equity orders at official opening/closing auction prices — [QC trade fills](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts). Nautilus supports AT_THE_OPEN / AT_THE_CLOSE time-in-force and a MarketHoursFillModel (wider synthetic spread) — [GitHub](https://github.com/nautechsystems/nautilus_trader), [fill models](https://nautilustrader.io/docs/latest/concepts/backtesting/fill-models/); how Nautilus matches against actual auction prints was not verified.
- **Stale data**: LEAN fills only market orders on price data one hour or more old (`stale_price_time_span`) — [QC trade fills](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts).
- **Look-ahead**: Freqtrade's docs point to a "Common Mistakes" section warning against using data not available in real market conditions; Jesse claims backtests are free of look-ahead bias (project claim, unverified) — [Freqtrade backtesting](https://www.freqtrade.io/en/stable/backtesting/), [Jesse](https://github.com/jesse-ai/jesse).
- **Batching artefacts**: Nautilus rc7 batches book deltas by default in `BacktestNode` so fills can differ from unbatched replay — [releases](https://github.com/nautechsystems/nautilus_trader/releases).
- **Parameter sensitivity**: simulated profitability found "highly sensitive to execution parameters" in a calibrated LOB simulator — [arXiv 2603.24137](https://arxiv.org/abs/2603.24137).

Coverage matrix (derived from the cited items above; "no" means the sources I read say it is absent or default-off, "?" means not verified):

| Effect | Nautilus | hftbacktest | LEAN | Freqtrade / bar engines | ABIDES |
|---|---|---|---|---|---|
| Book-based fills | yes (L2/L3) | yes | no (bars/ticks) | no | yes (sim book) |
| Queue position | yes, approximate | yes, several models | no | no | via sim exchange (not verified) |
| Partial fills | yes | yes (PartialFillExchange) | custom only | no/limited | yes (sim book) |
| Latency | latency_model + sandbox | feed + order latency | ? | no | messaging latency |
| Market impact | no (replay; synthetic liquidity models only) | no (explicit) | no (volume-share slippage only) | no | yes (agent reactions) |
| Fees | yes | ? | yes (brokerage models) | yes | ? |
| Corporate actions | ? | ? | yes (US) | n/a | n/a |
| Auctions | order types + hours fill model | ? | yes (US equities) | n/a | n/a |

### Inferences
- A defensible workflow is to treat the sim-to-real gap as a parameter set to be measured in shadow mode: queue-fill probability, latency, slippage and fee error should be estimated from live shadow/paper runs and fed back as fill/latency model parameters.
- For intraday/swing on bars, the dominant errors are fill price assumptions, intrabar ordering (stops) and missing slippage; those are cheap to approximate. For HFT, queue position and latency dominate; for both, data granularity (bars vs L2/L3) caps achievable realism regardless of engine.
- Borrow cost, locate/short-sale restrictions, halts, TSX board-lot/tick rules would need custom modelling in any framework discussed.

### Gaps
- No source found documenting built-in borrow-cost, short-locate, uptick/short-sale rule, or trading-halt/LULD modelling in Nautilus, hftbacktest, LEAN or others. Not asserted either way.
- Survivorship-bias-free universe handling in LEAN (coarse/fundamental universe, tradable-date history) was not covered by the page I fetched.
- Corporate-action handling in Nautilus/hftbacktest and fee modelling in hftbacktest were not verified.
- No empirical study quantifying live-vs-backtest slippage by framework was found.

---

## 7. Shadow / forward testing on live data (sandbox and paper execution)

### Takeaway
Nautilus (Sandbox environment context with a simulated execution client fed by real-time data), LEAN (QuantConnect Paper Trading brokerage model; local `lean live`), Freqtrade (dry-run), Hummingbot (`*_paper_trade` connectors), Jesse and OctoBot (paper trading) all support live data with simulated execution. Of these only Nautilus and LEAN are multi-asset and share the backtest code path; the crypto bots simulate fills with their own simpler logic.

### Cited Findings
- Nautilus: three environment contexts (Backtest, Sandbox = live data feeds and virtual execution, Live = real or paper accounts) on one core; `nautilus-sandbox` crate provides a `SandboxExecutionClient` that uses the `OrderMatchingEngine` to simulate execution against live market data (config: trader/account IDs, venue, starting balances, OMS type, account type); sandbox execution requires `book_type` L2/L3 to use order book deltas; rc5 added inbound latency modelling to the sandbox execution client — [Overview](https://nautilustrader.io/docs/latest/concepts/overview/), [docs.rs nautilus-sandbox](https://docs.rs/nautilus-sandbox), [Data and venues](https://nautilustrader.io/docs/latest/concepts/backtesting/data-and-venues/), [releases](https://github.com/nautechsystems/nautilus_trader/releases).
- LEAN: "QuantConnect Paper Trading" is listed as a brokerage model; the `lean live` CLI exists for local live runs — [QC brokerage models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/brokerages/supported-models), [GitHub](https://github.com/QuantConnect/Lean).
- Freqtrade: dry-run mode tests a strategy "with simulated money", docs say "Always start by running a trading bot in Dry-run" and "backtesting will never replace running a strategy in dry-run mode"; FreqAI retrains continuously in a background thread during dry/live runs — [Freqtrade docs](https://www.freqtrade.io/en/stable/), [Freqtrade backtesting](https://www.freqtrade.io/en/stable/backtesting/), [FreqAI](https://www.freqtrade.io/en/stable/freqai/).
- Hummingbot: paper-trade connectors; the `simple_pmm` script with `binance_paper_trade` "simulates trading against live Binance market data, so no API keys are required"; adding the `_paper_trade` suffix switches a connector (e.g., `bitfinex_paper_trade`); a 2019 post says dummy balances are set in `conf_global.yml` (possibly outdated) — [GitHub](https://github.com/hummingbot/hummingbot), [Hummingbot docs academy (search result)](https://docs.hummingbot.org/academy-content/how-to-use-advanced-market-making-features/).
- Jesse and OctoBot both advertise paper trading — [Jesse](https://github.com/jesse-ai/jesse), [OctoBot](https://github.com/Drakkar-Software/OctoBot).
- hftbacktest: live bots only for Binance Futures and Bybit, in Rust; docs recommend testing in a live market and adjusting for differences vs backtest — [GitHub](https://github.com/nkaz001/hftbacktest), [Order fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html).
- Nautilus live-vs-sim caveat: reconciliation, venue rules, transport failures etc. are live-only behaviours — [Live concepts](https://nautilustrader.io/docs/latest/concepts/live/).

### Inferences
- Shadow-mode fills are only as good as the paper engine: a sandbox that matches against the real live book (Nautilus L2/L3) tests latency, data and decision logic honestly but still cannot test queue priority or your own impact; small real-money live pilots are the only way to calibrate those.
- Running the identical Nautilus strategy in Backtest then Sandbox then Live, with a logged comparison of simulated vs real fills, is the cleanest research-to-live path found.

### Gaps
- How Freqtrade dry-run decides limit-order fills, and how Hummingbot paper-trade matches against the order book (and whether it models fees/slippage), were not found in current official docs.
- LEAN paper-trading behaviour (fill model used, local vs cloud, cost) not confirmed: the paper-trading page returned 404.
- Whether the Nautilus sandbox supports IB data feeds for TSX symbols was not checked.

---

## 8. Gymnasium-compatible wrappers and throughput vs fidelity

### Takeaway
Only a few simulators ship Gym interfaces (ABIDES-Gym, mbt_gym, JAX-LOB integrated with JAX RL; FreqAI has an RL module; MarS offers an agent-training environment). Nautilus and hftbacktest have no official Gymnasium wrapper in what I found. I found no published steps/sec figure for any of them, so the fidelity-vs-speed trade-off below is qualitative.

### Cited Findings
- ABIDES-Gym: Gym interface where the learner stays outside the simulator; the experimental agent interrupts the kernel at wake-ups; step granularity is set by environment wake-up times (1 minute for daily investor, 10 seconds for execution); paper reports no wall-clock/steps-per-second — [ABIDES-Gym paper](https://arxiv.org/html/2110.14771).
- mbt_gym: vectorised environments for faster RL training — [arXiv 2209.07823](https://arxiv.org/pdf/2209.07823).
- JAX-LOB: GPU simulator processing thousands of books in parallel with end-to-end GPU RL training on optimal execution — [arXiv 2308.13289](https://arxiv.org/abs/2308.13289).
- MarS: agent-training environment, but order model runs as a Ray Serve endpoint and production use "would need substantial inference optimization" — [GitHub](https://github.com/microsoft/MarS).
- Nautilus: docs cite RL/ES training as a use case of the high-throughput backtester; no built-in Gymnasium env found (an unrelated project, DeepTrade, uses Gymnasium) — [Overview](https://nautilustrader.io/docs/latest/concepts/overview/), search results on [NautilusTrader RL](https://www.pypi.org/project/nautilus_trader).
- FreqAI: any Python ML library can be plugged in, LightGBM used in the quick start, PyTorch via a Docker tag, a separate RL section and RL Docker tag exist (content of the RL page not read); Jesse lists reinforcement learning as "Coming Soon" — [FreqAI](https://www.freqtrade.io/en/stable/freqai/), [Jesse](https://github.com/jesse-ai/jesse).
- hftbacktest: Numba-JIT Python functions; no official RL wrapper mentioned on README — [GitHub](https://github.com/nkaz001/hftbacktest).

### Inferences
- Expected speed ordering (qualitative, not measured by sources): GPU-parallel JAX-LOB / vectorised mbt_gym / vectorbt are fastest for RL; Numba/Rust replay engines (hftbacktest, Nautilus) are next but are single-environment per process and need a custom Gym adapter; ABIDES-Gym (Python discrete-event, many agents) and MarS (neural order generation) are slowest per step but are the only ones with endogenous market impact.
- Nautilus's event-driven design (strategy callbacks) is not Gym-shaped; a Gymnasium wrapper would need to run the engine in a step-wise mode or in a separate thread/process, which I did not find documented.
- Recommended pattern: train RL in a fast, lower-fidelity environment, then fine-tune/validate in the high-fidelity engine (Nautilus/hftbacktest) and shadow mode. (Judgement.)

### Gaps
- No steps/sec or ticks/sec benchmarks found for Nautilus, hftbacktest, ABIDES-Gym, JAX-LOB (beyond the unverified "75x" claim), mbt_gym or MarS.
- Whether community Gym wrappers for Nautilus or hftbacktest exist and how fast they are was not established; FreqAI's RL environment details not read.

---

## 9. Practical recommendations (inferences from the evidence above)

### Takeaway
(a) High-frequency order-book work: hftbacktest for research (queue and latency models, L2/L3, measured latency) and/or Nautilus L2/L3 with queue_position for a single code path to live; (b) intraday/swing multi-asset portfolios: Nautilus (or LEAN if TSX is not required natively), with vectorbt for fast screening; (c) minimum rewrite from research to live: Nautilus v2 (same strategy code across Backtest/Sandbox/Live; IB plus crypto adapters). Use Freqtrade/Hummingbot only for crypto and as design references; avoid Backtrader.

### Cited Findings
(Every supporting fact is cited in Sections 1-8; the recommendation itself is inference.)
- Same-code claim and three environment contexts: [Live concepts](https://nautilustrader.io/docs/latest/concepts/live/), [Overview](https://nautilustrader.io/docs/latest/concepts/overview/).
- Nautilus adapters for IB, Databento, Binance, Bybit, OKX, Kraken, Coinbase, Tardis: [Integrations](https://nautilustrader.io/docs/latest/integrations/).
- LEAN has no Canada/TSX market constant: [Market.cs](https://raw.githubusercontent.com/QuantConnect/Lean/master/Common/Market.cs).
- hftbacktest live trading is Rust-only and market impact is not modelled: [GitHub](https://github.com/nkaz001/hftbacktest), [Order fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html).

### Inferences
- **(a) HFT / order book**: prototype and calibrate in hftbacktest (queue model choice, measured latency via unexecutable orders) for crypto L2/L3; if the live system must be Python and multi-venue, build on Nautilus L2/L3 + `queue_position` + `liquidity_consumption` and accept its coarser latency/queue approximations. Add a market-impact-aware layer (ABIDES/MarS-style) only to stress-test size.
- **(b) Intraday/swing multi-asset**: Nautilus bar/quote-driven backtests with IB adapter for US/TSX equities and options, crypto adapters for crypto; screen ideas with vectorbt first. If US-only equities were acceptable, LEAN's corporate-action, delisting and brokerage handling is more complete out of the box, but TSX would need a custom market.
- **(c) Minimum rewrite**: Nautilus v2 (Backtest, Sandbox, Live with unchanged strategy code); plan around RC churn (pin an rc version; watch for 2.0.0 final) and the ~3-month v1 security-backport window ending around Nov 2026.
- Sequence: (1) bar-level research in vectorbt or Nautilus; (2) high-fidelity Nautilus/hftbacktest replay with L2/L3 where data exists; (3) Sandbox shadow mode with logging of simulated vs observed fills; (4) small-size live pilot to calibrate queue/latency/impact parameters; (5) scale.
- RL training: use a fast surrogate (JAX-LOB / mbt_gym / vectorised bar env) for policy search, then validate in the high-fidelity engine; do not rely on MarS or ABIDES synthetic markets as proof of live profitability.

### Gaps
- Nothing here has been tested hands-on; recommendations rest on documentation and release notes only.
- TSX/TSXV specifics (data availability, IB Canada API restrictions, board lots, tick-size ladder, halts, short-sale rules) are out of this note's scope and unverified for every framework; one older QuantConnect forum thread says IB Canada accounts could not trade Canadian markets electronically via API, which is an old, unverified comment — [forum 7303 (search result)](https://www.quantconnect.com/forum/discussion/7303/canadians-not-allowed-to-use-interactive-brokers-api-anymore/).
- Options simulation fidelity (Greeks, assignment, expiry, option chains) for US options was not researched for any framework except that Nautilus rc4 added `InstrumentClose` expiry data and LEAN lists option position handling on splits.

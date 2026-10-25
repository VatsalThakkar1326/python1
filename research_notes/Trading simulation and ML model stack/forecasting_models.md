# Forecasting model stack for multi-horizon, cross-asset stock prediction with calibrated confidence (Python, as of 25 Oct 2026)

Scope notes: Version/date facts below were read from PyPI/GitHub pages on or about 25 Oct 2026 via a summarising fetch tool, so treat exact dates as "as reported by that page". Where a claim comes only from a search-engine summary (primary not retrieved) it is flagged "(secondary)". The GitHub MCP connector was restricted to the user's own repo, so commit-recency checks on third-party repos were not possible (listed under Gaps).

---

## 1. Model architectures for multi-horizon, multivariate forecasting (deep nets, GBDT, foundation models) and how they do on financial returns

### Takeaway
On financial returns, evidence favours gradient-boosted trees (and regularised linear / small NNs) on engineered, cross-sectionally normalised features as the strongest baseline; off-the-shelf time-series foundation models (Chronos, TimesFM, Moirai) are weak zero-shot on daily returns (negative out-of-sample R2 in the biggest study), and only finance-specific pre-training (e.g. Kronos, FinText) shows promise, mostly on author-run benchmarks without independent replication. Generic-benchmark rankings (GIFT-Eval, fev-bench) do not transfer to returns.

### Cited Findings

**Deep/hybrid architectures: what is available and what they support**
- NeuralForecast v3.3.0 was released 7 Oct 2026 (Python >=3.10, classifiers 3.10-3.13); release cadence is roughly monthly (3.1.4 on 15 Jan 2026 -> 3.2.0 on 10 Jul 2026 -> 3.3.0 on 7 Oct 2026) — [PyPI neuralforecast](https://pypi.org/project/neuralforecast/)
- NeuralForecast README lists 26 models including NHITS, NBEATS, TFT, PatchTST, iTransformer, TSMixer/TSMixerx, TiDE, TimesNet, DeepAR, StemGNN, TimeLLM; Mamba is not listed; Apache-2.0; the README claims "more than 30" models but names 26 — [GitHub Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast)
- NeuralForecast capability table: NBEATS (univariate, no exogenous); NHITS (univariate, future/historic/static exogenous); TFT (univariate, F/H/S exogenous); PatchTST (univariate, no exogenous); iTransformer (multivariate, future exogenous); TSMixer (multivariate, no exogenous); TSMixerx (multivariate, F/H/S); TiDE (univariate, F/H/S); TimesNet (univariate, future exogenous); DeepAR (univariate, F/H/S). Page notes multivariate models are "typically computationally expensive" and not necessarily more accurate than univariate ones — [Nixtla capabilities overview](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html)
- Darts (Apache-2.0, ~9.6k stars, Python 3.11+) offers TFT, Transformer, N-BEATS, N-HiTS, TCN, TiDE, TSMixer, DLinear/NLinear, RNNs; LightGBM/CatBoost/XGBoost wrappers; and zero-shot foundation-model wrappers `Chronos2Model`, `TimesFM2p5Model`, `TimesFM3Model`, `TiRexModel`, `TiRex2Model`, `PatchTSTFMModel` — [GitHub unit8co/darts](https://github.com/unit8co/darts)
- Generic evidence that simple models can beat fancy ones: a one-layer linear model family (LTSF-Linear/DLinear) outperformed Transformer-based long-horizon forecasters "in all cases" on nine real-world datasets (Zeng et al., May 2022) — [arXiv 2205.13504](https://arxiv.org/abs/2205.13504)
- Tree-based models "remain state-of-the-art on medium-sized data" (~10K samples) across 45 tabular datasets vs deep learning (Grinsztajn et al., Jul 2022) — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815)

**Qlib's own benchmark (CSI300 China A-shares; mean over 20 seeds; costs/dates not stated on the page; backtest changed in v0.8.0 so not comparable with older numbers)** — [Qlib benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)
- Alpha158 features: LightGBM IC 0.0448 / RankIC 0.0469 / ann. return 9.01% / IR 1.02; XGBoost IC 0.0498; CatBoost IC 0.0481; DoubleEnsemble (LightGBM base) IC 0.0521 / ann. ret 11.58% / IR 1.34 (best); Linear IC 0.0397; MLP IC 0.0376; TRA IC 0.0440; Transformer IC 0.0264; TabNet IC 0.0204; TFT (20 features) IC 0.0358 but RankIC only 0.0116.
- Alpha360 (raw-ish price/volume window) features: HIST IC 0.0522 / RankIC 0.0667 / ann. ret 9.87% / IR 1.37 (best IR); GATs IC 0.0476; GRU IC 0.0493; LightGBM IC 0.0400 / ann. ret 5.58%; Transformer IC 0.0114 with negative return (-2.7%); TabNet negative return; KRNN and Sandwich near/below zero.
- All ICs are in the 0.01-0.05 range — i.e. tiny — and the best deep model (HIST) barely exceeds DoubleEnsemble/LightGBM on Alpha158.

**Time-series foundation models (TSFMs): status and licences (Oct 2026)**
- Chronos-2 (Amazon; arXiv 2510.15821, submitted 17 Oct 2025): 120M params; group attention shares information across series in a group (related series, variates, or targets+covariates) for in-context learning; supports multivariate and past/future covariates zero-shot; trained on synthetic datasets with multivariate structure; reports SOTA on fev-bench, GIFT-Eval, Chronos Benchmark II — [arXiv abs](https://arxiv.org/abs/2510.15821), [PyPI chronos-forecasting](https://pypi.org/project/chronos-forecasting/)
- `chronos-forecasting` v2.3.2 (8 Sep 2026), Python >=3.10; checkpoints: Chronos-2 (`amazon/chronos-2` 120M, `autogluon/chronos-2-small` 28M, `autogluon/chronos-2-synth`), Chronos-Bolt (9M-205M), original T5 Chronos (8M-710M); `predict_df` supports `quantile_levels` — [PyPI chronos-forecasting](https://pypi.org/project/chronos-forecasting/). HF model card lists max context 8192 and Apache-2.0 (via search summary) — [HF amazon/chronos-2](https://huggingface.co/amazon/chronos-2)
- TimesFM (Google): README says TimesFM 3.0 released Aug 2026 (`google/timesfm-3.0-pytorch`; native multivariate + covariates; median + 9 quantiles; truncates context beyond 15,360). TimesFM 2.5 (15 Sep 2025): 200M params, 16k context, optional 30M quantile head; XReg covariates re-added 29 Oct 2025; LoRA fine-tuning example 9 Apr 2026. **Licence: 3.0 weights are under a separate non-commercial licence (commercial/production use of downloaded weights not permitted; allowed via Google Cloud services such as BigQuery ML); 2.5 and earlier weights Apache-2.0.** Repo is "not an officially supported Google product" — [GitHub google-research/timesfm](https://github.com/google-research/timesfm). (Read via a summarising fetch; verify the licence text before relying on it.)
- Moirai 2.0 (Salesforce): decoder-only, quantile forecasting, multi-token prediction (arXiv 2511.11698). Salesforce blog says #1 by MASE on GIFT-Eval among non-test-leaking models at launch; the arXiv paper itself reports 5th (MASE) and 6th (CRPS) among 30 foundation models — conflicting framing — [arXiv 2511.11698](https://arxiv.org/pdf/2511.11698), [Salesforce blog](https://www.salesforce.com/blog/moirai-2-0/)
- Kronos (Tsinghua; arXiv 2508.02739, Aug 2025; AAAI 2026): decoder-only Transformer over discretised K-line tokens (Binary Spherical Quantization tokenizer; 20-bit tokens split coarse/fine); sizes small 24.7M, base 102.3M, large 499.2M; pre-trained on >12B K-line records (abstract says 45 exchanges; appendix says "over 40 exchanges across >30 countries"; frequencies 1-minute to weekly); max context 512 tokens; test period starts July 2024, after the June 2024 pre-training cutoff — [arXiv HTML](https://arxiv.org/html/2508.02739v1), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39730)
- Kronos GitHub: MIT licence; Kronos-mini 4.1M (context 2048), small and base open (context 512), **large is not open-sourced**; fine-tuning pipeline uses Qlib on Chinese A-share daily data; README says the pipeline is "a demonstration", "not a production-ready quantitative trading system", and a realistic backtest would need costs, slippage and market impact; live demo is BTC/USDT 24h; no mention of US/TSX coverage; ~43.6k stars — [GitHub shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- Kronos claims (authors' own, zero-shot): +93% RankIC vs best TSFM and +87% vs best non-pretrained baseline for price forecasting, 9% lower volatility MAE, 22% better generative fidelity. Absolute numbers in the Kronos-small ablation are small: price RankIC 0.0254 / IC 0.0431; return RankIC 0.0622 / IC 0.0665; volatility R2 0.249. Investment simulation is on CSI300/CSI800 A-shares via Qlib with 0.15% cost per trade, 5-day minimum hold, 10-day horizon (exact returns only shown in charts). Paper says "25 baselines" but lists 18 — [arXiv HTML](https://arxiv.org/html/2508.02739v1)
- No independent (non-author) backtest of Kronos on US stocks, TSX or crypto was found in search — [search results](https://arxiv.org/pdf/2508.02739)

**TSFMs on financial returns specifically**
- Rahimikia, Ni, Wang (Nov 2025; daily firm-level excess returns, 94 countries, 1990-2023, expanding-window OOS 2001-2023, US test set >18M returns / ~10,000 securities): zero-shot Chronos-large (512-day context) OOS R2 = -1.37%, directional accuracy just above 51%; zero-shot TimesFM-500M R2 = -2.80%, directional accuracy just below 50%, annualised long-short return -1.47%. CatBoost R2 about -0.10% (avg) / -0.03% at 512 days, directional accuracy 51.16%; OLS about -0.47%. Models pre-trained from scratch on financial data improve (Chronos-small window 512: R2 -0.59% vs -1.27%, Sharpe 5.42, 36.84% ann. return), and with global data + synthetic augmentation reach directional accuracy 51.74% vs CatBoost 51.16%, Sharpe 6.78 vs 6.46. Conclusion: off-the-shelf TSFMs trail ensemble benchmarks; finance pre-training helps; **all headline portfolio results exclude transaction costs** — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1)
- Noguer i Alonso & Franklin (25 Jun 2026; TimeGPT, TimesFM-2.5, Moirai-2.0, Chronos, Chronos-2 vs NBEATS, NHITS, PatchTST, iTransformer, KAN on five liquid US equities AAPL/AMZN/GOOG/JPM/META, L=512, rolling origin 2024-2026): pretrained TSFMs win 8 of 10 task-level comparisons (Moirai-2.0 and TimesFM-2.5 best average ranks) but iTransformer wins both META tasks; gains over the zero-return random-walk benchmark are "small and sparse" — only 16 pretrained model/asset cases show positive skill, and a one-sided Diebold-Mariano test rejects only for Chronos on AMZN and Moirai-2.0 on GOOG. The paper evaluates forecast error, not trading utility, and notes realistic R2 for daily liquid-equity returns is typically below 1% — [arXiv 2606.27100](https://arxiv.org/pdf/2606.27100) (read from the PDF text)
- ISD 2025 study (secondary summary): zero-shot Chronos was competitive on accuracy for Nasdaq-100/S&P 500 daily forecasting but its derived trading performance lagged top benchmarks, and fine-tuned Chronos was statistically worse than zero-shot — [ISD 2025 paper](https://isd2025.fon.bg.ac.rs/wp-content/uploads/ISD_2025_camera_ready_265.pdf)
- Kronos authors themselves state existing TSFMs "often underperform non-pre-trained architectures" on candlestick data — [AAAI abstract](https://ojs.aaai.org/index.php/AAAI/article/view/39730)

**Gu-Kelly-Xiu model comparison (US stocks, monthly)**
- Out-of-sample monthly R2: OLS with all 920 features -3.46%; OLS-3 (size, B/M, momentum) 0.16%; elastic net 0.11%; PCR 0.26-0.27%; PLS 0.27%; NN1 0.33%; best NN3 0.40%; "four- and five-layer models fail to improve over NN3"; trees and NNs are best, gains attributed to nonlinear interactions — [RFS version of Gu, Kelly, Xiu (2020)](https://www.aqr.com:443/-/media/AQR/Documents/Journal-Articles/Empirical-Asset-Pricing-in-Machine-Learning.pdf?sc_lang=en) (numbers read from PDF text), [NBER w25398](https://www.nber.org/papers/w25398)
- Canada (secondary, blog write-up of Attig & Attig, "Bay Street Meets Machine Learning"): moderately flexible nonlinear models (XGBoost, mid-depth NNs) beat deeper NNs and linear benchmarks on Canadian stocks; sample period and cost treatment not stated — [CFA Institute Enterprising Investor 2026](https://rpc.cfainstitute.org/blogs/enterprising-investor/2026/machine-learning-improves-forecasting)

### Inferences
- The consistent pattern across Qlib (A-shares), GKX (US monthly), Rahimikia (global daily), and Canada (blog) is: GBDT or small NN on engineered features >= big sequence models. Transformers/TabNet/TFT are the models most likely to be worse than plain LightGBM (Qlib Alpha360 Transformer IC 0.011, negative return). Start with LightGBM/CatBoost; treat TFT/PatchTST/iTransformer as challengers that must beat it on a purged OOS test.
- Zero-shot TSFMs should be treated as (a) a cheap baseline and (b) possible feature generators/priors; they are not a plausible primary alpha source on raw returns. Finance-native pre-training (Kronos small/base, FinText) is the only TSFM branch with positive evidence, but it is author-reported, mostly Chinese A-shares / daily bars, and has a 512-token context.
- Licence matters for a live trader: TimesFM 3.0 weights are non-commercial; whether personal live trading counts as "commercial" is not settled by the text I read — check the licence or use TimesFM 2.5 (Apache-2.0), Chronos-2 (Apache-2.0 per model card), Kronos (MIT).
- "Multivariate" in NeuralForecast terms (iTransformer, TSMixer) is the only built-in way to let a deep model attend across assets; the "univariate" models are typically still trained globally across many series but do not mix information across series at input (background knowledge, not verified in this pass).
- With realistic R2 well under 1%, point forecasts like "-0.5% in 1 hour, +1.3% tomorrow noon" overstate what the data can deliver; the usable output is a conditional distribution whose mean is a small shift relative to its spread (IC of 0.03-0.05 in Qlib corresponds to correlations of that size).

### Gaps
- No financial-returns benchmark found for TimesNet, TiDE, TSMixer, DeepAR, N-HiTS under purged protocols other than the 5-stock NBEATS/NHITS/PatchTST/iTransformer comparison above.
- Mamba-based forecasters: found no financial-specific evidence and NeuralForecast does not ship one; unresolved.
- Lag-Llama: no current maintenance/release info found.
- Independent verification of Kronos zero-shot claims (US, TSX, crypto) not found; Kronos figures with numeric returns are only in charts I could not read.
- Chronos-2 / TimesFM-3.0 on financial returns: no peer-reviewed zero-shot R2 found (the June 2026 paper covers Chronos-2 but only forecast-error skill vs random walk, headline numbers not extracted).
- Exact TimesFM 3.0 licence wording and release facts come from a summarising fetch of the README; not cross-checked against the licence file.

---

## 2. Cross-stock relationships: GNNs, attention across assets, Qlib models, regimes, lead-lag

### Takeaway
Cross-asset structure is plausible and is exploited in the literature, but evidence that graph/cross-asset models add reliable out-of-sample alpha beyond strong GBDT baselines is thin: in Qlib's benchmark, graph models (GATs, HIST) give modest IC/return improvements at best, and I found no standardised benchmark comparing relation-graph choices (sector, correlation, supply chain). Regime detection works better with jump models than naive HMMs, mostly as risk control rather than return prediction.

### Cited Findings
- Qlib ships HIST, TRA, ADARNN, DoubleEnsemble, GATs, IGMTF, KRNN, TFT, TabNet, Localformer, TCN, ALSTM, SFM, CatBoost/XGBoost/LightGBM, plus DDG-DA/rolling retraining for market dynamics — [GitHub microsoft/qlib](https://github.com/microsoft/qlib)
- Qlib Alpha360 results (IC / RankIC / ann. return / IR): HIST 0.0522 / 0.0667 / 9.87% / 1.37; IGMTF 0.0480 / 0.0606 / 9.46% / 1.35; GATs 0.0476 / 0.0598 / 8.24% / 1.11; TRA 0.0485 / 0.0587 / 9.20% / 1.28; plain GRU 0.0493 / 0.0584 / 7.20% / 0.97; LSTM 0.0448 / 0.0549 / 6.47% / 0.90; LightGBM 0.0400 / 0.0499 / 5.58% / 0.76. On Alpha158 (20-feature versions): GATs IC 0.0349 vs GRU 0.0315; TRA IC 0.0404 (20 features), 0.0440 (full) vs LightGBM 0.0448 and DoubleEnsemble 0.0521 — [Qlib benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)
- GNN reviews: there is no off-the-shelf graph of inter-stock relations, so graph construction varies across studies (industry/subsidiary/people/product knowledge graphs, rolling correlations, fundamentals similarity, supply chain) — [arXiv 2111.15367](https://arxiv.org/pdf/2111.15367); a 2024 ACM Computing Surveys review (Patel, Jariwala, Chattopadhyay) covers GNN stock-forecasting methods — [BibBase entry](https://bibbase.org/network/publication/patel-jariwala-chattopadhyay-asystematicreviewongraphneuralnetworkbasedmethodsforstockmarketforecasting-2024)
- Chronos-2's group attention lets related series (including targets and covariates) exchange information in-context; TimesFM 3.0 adds native multivariate/covariate support; iTransformer is the multivariate "attention across variates" option in NeuralForecast — [Chronos-2 arXiv](https://arxiv.org/abs/2510.15821), [TimesFM GitHub](https://github.com/google-research/timesfm), [Nixtla capabilities](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html)
- Regime detection: HMMs are reported (via Nystrup et al., 2020, cited second-hand) to give unstable regime sequences with false alarms when regimes are imbalanced/persistent, signal-to-noise is low and data are limited; statistical jump models (jump penalty on switching) are the proposed fix. Shu, Yu, Mulvey (J. Asset Management 2024) test JM-guided strategies on US/German/Japanese equity indices 1990-2023 with transaction costs and trading delays, finding lower volatility and drawdown and better risk-adjusted returns than HMM-guided and buy-and-hold; the jump penalty is tuned on validation Sharpe, so partly tuned — [arXiv 2402.05272](https://arxiv.org/abs/2402.05272v3) (and search summary of related work)
- Nagel (RFS 2012): short-term reversal returns proxy for compensation for liquidity provision and are strongly time-varying and predictable by VIX; expected returns and conditional Sharpe ratios rise sharply in turmoil — [NBER w17653](https://nber.org/papers/w17653)
- Crypto lead-lag (secondary): a 2024 study is cited in a search summary as finding lagged returns of other cryptocurrencies predict focal-coin returns out of sample with sizeable long-short returns after costs; primary not retrieved and the link is only a result from the same search set (origin not confirmed) — [search result set](https://arxiv.org/pdf/2606.00071)
- Microstructure cross-section: Cont, Kukanov, Stoikov found order flow imbalance at the best bid/ask explains short-interval price changes linearly with slope inverse to depth (50 US stocks, TAQ) — [arXiv 1011.6402](https://arxiv.org/abs/1011.6402)

### Inferences
- The best-evidenced "cross-asset" information for a retail stack is cheap and tabular: cross-sectional ranks/z-scores of each stock's features vs the universe, sector/market returns, beta/idiosyncratic residuals, index/VIX/regime state, and (for crypto) BTC/ETH lagged returns — feed these to GBDT before building a GNN.
- Because graph models (GATs, HIST, IGMTF) beat plain GRU/LightGBM by margins comparable to seed noise in Qlib's table, and graph definitions are ad hoc, a GNN is a second-stage experiment with a pre-registered graph (e.g. sector + rolling correlation), not a first build.
- For regimes, prefer a jump-model/volatility-state approach with the regime as a feature or as a gate on position size, and use costs-and-delay-aware testing as Shu et al. do; do not let the regime model see future data (full-sample HMM fits leak).
- Nagel's result suggests short-term-reversal signals should be conditioned on a vol/VIX state; reversal edge is not constant.

### Gaps
- No study found that fairly benchmarks relation-graph types (industry vs correlation vs supply chain) under identical purged protocols.
- No TSX/TSXV-specific evidence for GNNs or lead-lag effects.
- Nystrup et al. HMM papers and the crypto lead-lag primary study were not retrieved.
- Qlib's Alpha158/Alpha360 composition (exact feature lists) was not retrieved in this pass.

---

## 3. Python libraries and platforms: capabilities, Canadian-data support, maintenance status (Oct 2026)

### Takeaway
Qlib is the most complete open-source quant research platform (data, alpha factors, models, backtest, online serving) but its releases have stalled on PyPI (0.9.7, 15 Aug 2025; Python 3.8-3.12 only) and its data support is built around CN and US; TSX/TSXV works only through user-supplied OHLCV in Qlib's binary format. The Nixtla stack, Darts, MAPIE, and skfolio are actively released in 2026.

### Cited Findings
- **Qlib**: PyPI pyqlib 0.9.7 released 15 Aug 2025 (previous: 0.9.6 on 23 Dec 2024); classifiers CPython 3.8-3.12 — [PyPI pyqlib](https://pypi.org/project/pyqlib/). GitHub README: MIT licence, ~49.7k stars, 314 open issues, 192 PRs, public roadmap, "in active and continuing development"; the Releases section shows v0.9.0 (9 Dec 2022) as the last GitHub release (stale vs PyPI); a breaking artifact-loading change (recorder loads restricted, `trusted=True`) is on main but unreleased; RD-Agent integration (LLM-driven factor mining, released 8 Aug 2024; paper arXiv 2505.15155) — [GitHub microsoft/qlib](https://github.com/microsoft/qlib)
- Qlib data: official dataset is "disabled temporarily" for data-security reasons, community source chenditc/investment_data recommended; default collector scrapes Yahoo Finance; README mentions Ibovespa (Brazil) index data — [GitHub microsoft/qlib](https://github.com/microsoft/qlib)
- Qlib docs: custom data via `scripts/dump_bin.py` from CSV/Parquet (one file per symbol with date column; "open, close, high, low, volume and factor at least"; prices adjusted; `factor` = adjustment factor); US data via `get_data.py qlib_data --region us`; region settings defined only for CN (trade unit 100, limit 0.099) and US (trade unit 1, no limit); daily default, 1-minute dataset shown for CN; no point-in-time DB mention; stock pools documented only for CSI indices — [Qlib data docs](https://qlib.readthedocs.io/en/latest/component/data.html)
- Yahoo Finance carries real-time intraday data for TSX, TSX Venture and TSX Alpha (TMX Datalinx agreement, 29 Sep 2020) — [TMX Datalinx press release](https://s21.q4cdn.com/671813756/files/doc_news/2020/09/1/TMX-Datalinx-to-Provide-Yahoo-Finance-with-Real-Time-Market-Data-for-Canadas-Premier-Equities-Markets-2020-9-29-2020.pdf)
- **NeuralForecast** 3.3.0 (7 Oct 2026), Python >=3.10; supports conformal intervals via `PredictionIntervals` passed to `fit(prediction_intervals=...)`, `n_windows` default 2, methods `conformal_distribution` (default) and `conformal_error`; also `DistributionLoss` (e.g. Normal) for parametric intervals — [PyPI](https://pypi.org/project/neuralforecast/), [Nixtla conformal tutorial](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/conformal_prediction.html)
- **mlforecast** 1.1.0 (10 Jul 2026), Python >=3.10: lag features, lag/target transforms, any sklearn-style regressor (examples use LightGBM), probabilistic forecasting with conformal prediction — [PyPI mlforecast](https://pypi.org/project/mlforecast/)
- **Darts**: Apache-2.0, ~9.6k stars, 1,618 commits, Python 3.11+, README says development ongoing; `ConformalNaiveModel` and `ConformalQRModel` for calibrated quantile intervals on pre-trained global models; `darts.ad` anomaly detection — [GitHub unit8co/darts](https://github.com/unit8co/darts)
- **PyTorch Forecasting** 1.8.0 (24 Jun 2026; prior 1.7.0 on 5 Apr 2026, 1.6.1 on 23 Jan 2026), Python >=3.10,<3.15 — [PyPI pytorch-forecasting](https://pypi.org/project/pytorch-forecasting/)
- **GluonTS** 0.17.0 (31 Jul 2026; 0.16.3 on 29 Jun 2026; 0.16.2 on 27 Jun 2025), Python >=3.10,<3.15 — [PyPI gluonts](https://pypi.org/project/gluonts/)
- **MAPIE** 1.6.0 (19 Oct 2026; 1.5.0 on 5 Aug 2026), Python >=3.10; prediction intervals/sets for regression, classification and time series — [PyPI MAPIE](https://pypi.org/project/MAPIE/)
- **skfolio** 1.11.0 (24 Oct 2026), Python >=3.10; includes `CombinatorialPurgedCV`, `WalkForward` — [PyPI skfolio](https://pypi.org/project/skfolio/)
- **tsfresh** 0.21.2 (31 May 2026; previous 0.21.1 on 30 Aug 2025), Python >=3.9 — [PyPI tsfresh](https://pypi.org/project/tsfresh/)
- **TA-Lib** (Python wrapper) 0.8.2 (12 Oct 2026), Python >=3.9, 150+ indicators, binary wheels bundle the C library, NumPy 2 supported on 0.8.x — [PyPI TA-Lib](https://pypi.org/project/TA-Lib/)
- **pandas-ta** latest listed 0.4.71b0 (14 Sep 2025, beta/pre-release), Python >=3.12, no archival notice on the page — [PyPI pandas-ta](https://pypi.org/project/pandas-ta/)
- Labeling libs: mlfinpy, finmlkit (`TBMLabel`, side and meta-label modes), fynance (`labels` module), and Hudson & Thames' original MLFinLab — [mlfinpy](https://mlfinpy.readthedocs.io/en/latest/_sources/Labelling.rst.txt), [finmlkit](https://finmlkit.readthedocs.io/en/latest/api/finmlkit.label.kit.html), [fynance](https://fynance.readthedocs.io/en/master/_modules/fynance/features/labels.html), [Hudson & Thames](https://hudsonthames.org/?p=3448)

### Inferences
- Qlib is best treated as a reference/benchmark framework and source of well-tested feature definitions and a rolling-retrain/backtest harness, rather than a dependency to anchor the project on: PyPI has had no release for ~14 months as of Oct 2026, Python 3.13 is not in the classifiers, and TSX would need a hand-built calendar/instrument/dump_bin pipeline with US-like region settings (inference: US region is the closest fit, 1-share trade unit, no price limits). Qlib's modelling-only pieces (DoubleEnsemble, HIST) can still be borrowed.
- A pragmatic Python stack by 2026 maintenance evidence: pandas/Polars + LightGBM/CatBoost (+ mlforecast for lag features) -> NeuralForecast or Darts for deep/foundation challengers -> MAPIE or Nixtla/Darts conformal for intervals -> skfolio (CPCV) or custom purged splits -> TA-Lib for indicators (pandas-ta looks less reliable: last listing is a 2025 beta requiring Python >=3.12).
- Darts' built-in wrappers for Chronos-2/TimesFM/TiRex make it the lowest-friction way to compare foundation models against GBDT under one API.

### Gaps
- Qlib's last commit date and whether a 0.9.8/1.0 release is imminent could not be checked (GitHub connector restricted); GitHub README does not show dates for recent commits.
- Darts and sktime and fracdiff PyPI release dates not retrieved (Darts PyPI and sktime PyPI pages failed to load; Darts GitHub page showed no release list).
- No source found confirming Qlib support for TSX specifically (no Canadian data collector mentioned); the above is based on documented custom-data path.
- tsfresh's feature count not stated on the page.
- yfinance reliability for .TO/.V symbols not verified (only the 2020 TMX-Yahoo agreement was found).

---

## 4. Uncertainty / confidence: quantile losses, distributions, conformal, ensembles, calibration, and turning distributions into signals

### Takeaway
Calibrated intervals are best obtained by combining a probabilistic output (quantile or distributional head, or a foundation model's quantiles) with a conformal layer that is adaptive under distribution shift (ACI); coverage guarantees are long-run only, and the single-digit-percent signal-to-noise of returns means well-calibrated forecasts will mostly be wide with a small mean shift.

### Cited Findings
- Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) wraps any black-box predictor, tracks one parameter to adapt to shift, and provably achieves the target coverage frequency over long-run intervals irrespective of the data-generating process (not a per-step guarantee) — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)
- Follow-up (Gibbs & Candès, JMLR 2024): original ACI over-weights history and its performance depends on a good step-size choice; the fix gives provably small regret over all local intervals — [arXiv 2208.08401](https://arxiv.org/pdf/2208.08401v2)
- MAPIE `TimeSeriesRegressor` (`mapie.regression`; older name `MapieTimeSeriesRegressor`) supports methods "enbpi" and "aci"; EnbPI needs a `BlockBootstrap` cv; the score-update method was renamed `partial_fit` -> `update` between v1.1.0 and v1.2.1; docs "latest" page still shows the old name — [MAPIE docs](https://mapie.readthedocs.io/en/stable/generated/mapie.regression.TimeSeriesRegressor.html); MAPIE 1.6.0 released 19 Oct 2026 — [PyPI MAPIE](https://pypi.org/project/MAPIE/)
- NeuralForecast: conformal from cross-validation windows on a point-loss model, or `DistributionLoss`; mlforecast and Darts also ship conformal options (see section 3) — [Nixtla tutorial](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/conformal_prediction.html), [mlforecast PyPI](https://pypi.org/project/mlforecast/), [Darts](https://github.com/unit8co/darts)
- Foundation-model native quantiles: Chronos-2 via `quantile_levels`; TimesFM 3.0 returns median + 9 quantiles (0.1-0.9) and 2.5 has an optional 30M quantile head; Moirai 2.0 is quantile-based — [Chronos PyPI](https://pypi.org/project/chronos-forecasting/), [TimesFM](https://github.com/google-research/timesfm), [Moirai 2.0](https://arxiv.org/pdf/2511.11698)
- Evaluation framework: Gneiting, Balabdaoui & Raftery (2007) frame the goal as maximising sharpness subject to calibration and propose PIT histograms, marginal calibration plots, sharpness diagrams and proper scoring rules; PIT alone is only a necessary condition for an ideal forecaster — [Gneiting et al. 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jrssb.pdf), [Pal arXiv 0902.0342](https://arxiv.org/pdf/0902.0342)
- Turning forecasts into trades (crypto, secondary): on hourly BTC-USDT 2018-2026, naive sign-based strategies fail once 10 bp costs are charged, while a cost-aware filter that trades only when forecast magnitude exceeds a cost-based threshold cuts turnover and restores profitability in selected configurations (single non-peer-reviewed preprint) — [search listing](https://www.emergentmind.com/papers/2606.00060), [related review](https://arxiv.org/pdf/2606.00071)
- ML forecast portfolios' performance "further deteriorates" with reasonable trading costs because of high turnover and extreme positions (Avramov, Cheng, Metzker, Management Science 2023) — [SSRN 3450322](https://papers.ssrn.com/abstract=3450322), [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.2022.4449)
- Regime/shift: Kronos pre-training cutoff vs test period separation is the kind of leakage control to copy for any pretrained forecaster (test starts July 2024 vs cutoff June 2024) — [arXiv 2508.02739](https://arxiv.org/html/2508.02739v1)

### Inferences
- Pipeline suggestion: train a quantile-loss GBDT or NN (or take TSFM quantiles) -> conformalise with ACI/EnbPI on a rolling basis per horizon -> monitor empirical coverage and PIT drift -> only then convert to decisions. Because ACI's guarantee is long-run and marginal, check coverage conditional on regime/volatility bucket and per asset class (TSX small caps vs US large caps vs crypto).
- Decision rule consistent with the cost-aware-filter and Avramov findings: trade only if expected return (distribution mean or chosen quantile) minus estimated round-trip cost (spread + fees + impact) exceeds a margin scaled by predictive spread; size by mean/variance or lower-quantile (e.g. 10th percentile must clear stop-loss distance). This is my synthesis, not a tested recipe from a cited source.
- Standard scores for judging the output (background knowledge, not sourced in this pass): pinball loss per quantile, CRPS, interval coverage and width, PIT uniformity, and per-bucket reliability; for the ranking signal, rank-IC with a t-statistic over time.
- Ensembles of seeds/models plus a conformal layer are a pragmatic uncertainty source; Qlib's DoubleEnsemble was top on Alpha158 (IC 0.0521), supporting ensembling as a performance as well as an uncertainty tool.

### Gaps
- No source found testing ACI/EnbPI coverage on equity-return forecasts specifically (a benchmark-aggregator page claiming ~90% coverage on stock prices was unverified and not used).
- No authoritative source retrieved on CRPS/pinball-loss definitions or on quantile-crossing handling; Nixtla loss pages (MQLoss, IQLoss) not retrieved.
- No source on volatility-scaled conformal intervals for returns.
- The hourly BTC paper's identity (arXiv ID) could not be confirmed from search results; treat the cost-filter result as an anecdote.

---

## 5. Targets and labels: returns vs prices vs direction, multi-horizon, triple-barrier, meta-labeling, volatility normalisation

### Takeaway
The literature predicts excess returns (not prices); direction accuracy is only a hair above 50% even for good models; triple-barrier labels and meta-labeling (AFML) are implemented in several open-source Python libraries but label overlap requires purged/embargoed validation.

### Cited Findings
- Gu-Kelly-Xiu predict stock-level excess returns at monthly (and annual) horizons; the annual-horizon OOS R2 is "nearly an order of magnitude larger" than monthly — [Gu, Kelly, Xiu RFS](https://www.aqr.com:443/-/media/AQR/Documents/Journal-Articles/Empirical-Asset-Pricing-in-Machine-Learning.pdf?sc_lang=en)
- Rahimikia et al. predict daily excess returns; best directional accuracies are 51.16% (CatBoost) to 51.74% (finance-pretrained Chronos-small) — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1)
- Noguer i Alonso & Franklin evaluate both linear and log returns, against a zero-return random walk (Bayes-optimal under L2 if returns are unpredictable); theory section bounds feasible daily R2 as typically below 1% — [arXiv 2606.27100](https://arxiv.org/pdf/2606.27100)
- Bitcoin: reviews report 50-60% accuracy classifying return direction (secondary) — [arXiv 2606.00071](https://arxiv.org/pdf/2606.00071)
- Kronos benchmarks include price series, returns (RankIC/IC), realized volatility (MAE/R2) and synthetic K-line generation — volatility is forecast with materially higher R2 (0.249 in the small ablation) than returns — [arXiv 2508.02739](https://arxiv.org/html/2508.02739v1)
- Triple-barrier method (AFML): upper/lower barriers set from per-observation volatility plus a vertical time barrier; label = first barrier touched. Meta-labeling: a primary model picks side, a secondary model decides whether/how much to bet. Libraries: mlfinpy, finmlkit (`TBMLabel`), fynance (also concurrency and uniqueness weights); fynance warns labels use future prices by construction so training must use purged/embargoed splits — [mlfinpy](https://mlfinpy.readthedocs.io/en/latest/_sources/Labelling.rst.txt), [finmlkit](https://finmlkit.readthedocs.io/en/latest/api/finmlkit.label.kit.html), [fynance](https://fynance.readthedocs.io/en/master/_modules/fynance/features/labels.html)
- Qlib's benchmark uses RankIC/IC (cross-sectional rank correlation with forward return) as the primary forecast metric alongside backtest return/IR — [Qlib benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)

### Inferences
- Recommended target design (synthesis): predict volatility-normalised forward log-returns (return / trailing realised vol, or excess vs market/sector) at several horizons as a multi-output regression or quantile set, plus a separate volatility forecast; use cross-sectional rank targets for the stock-selection model; reserve direction/triple-barrier labels for the exit/meta layer (does the signal clear the barrier before the stop). Volatility is the more predictable quantity (Kronos R2 0.25 vs returns IC ~0.04-0.07), which makes it the natural basis for sizing and stop distances and for scaling conformal intervals.
- Multi-horizon outputs should be trained jointly or as separate heads but evaluated per horizon with overlapping-label-aware tests (Newey-West/HAC or non-overlapping samples); the GKX note that annual R2 >> monthly R2 suggests longer horizons carry more predictable signal but with far fewer independent samples.

### Gaps
- Joubert's meta-labeling papers and the original AFML text were not retrieved; the triple-barrier/meta-label description comes from library docs.
- MLFinLab's current (open vs paid) status not confirmed.
- No source retrieved comparing volatility-normalised vs raw return targets empirically.

---

## 6. Feature engineering from pure price/volume (technical indicators, microstructure, cross-sectional ranks, fractional differentiation, calendar)

### Takeaway
Predictive power in the literature comes from a small set of families — momentum, short-term reversal, liquidity, volatility — transformed cross-sectionally; microstructure order-flow features carry real signal but only at very short horizons and need L2/tick data.

### Cited Findings
- GKX: all methods "agree on the same set of dominant predictive signals," variations on momentum, liquidity and volatility; short-term reversal is not named in the abstract — [NBER w25398](https://www.nber.org/papers/w25398)
- Qlib Alpha158 (engineered features) vs Alpha360: engineered Alpha158 features gave better IC for most models than the Alpha360 window (e.g. Transformer IC 0.0264 vs 0.0114; MLP 0.0376 vs 0.0273; XGBoost 0.0498 vs 0.0394), though LSTM/GRU/ALSTM/TCN and graph models did better on Alpha360 than their Alpha158 20-feature variants — [Qlib benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)
- Order flow imbalance (OFI): short-interval price changes are dominated by OFI at the best bid/ask, linear in OFI with slope inversely proportional to depth (TAQ, 50 US stocks) — [Cont, Kukanov, Stoikov, arXiv 1011.6402](https://arxiv.org/abs/1011.6402)
- Deep OFI (Kolm, Turiel, Westray, Mathematical Finance 2023): 115 Nasdaq stocks, order-flow inputs beat most raw order-book inputs with simpler networks; effective horizon of stock-specific forecasts is about two average price changes, with accuracy declining after — [IDEAS/Math. Finance](https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html)
- Canadian-stock study (secondary): anonymous-trading (broker-level order flow) variables added roughly 60-80 bp per month; gains concentrated in small-cap and value names; transaction costs not mentioned — [CFA Institute 2026](https://rpc.cfainstitute.org/blogs/enterprising-investor/2026/machine-learning-improves-forecasting)
- Nagel (2012): reversal returns time-vary strongly with VIX — a state-dependent feature interaction — [NBER w17653](https://nber.org/papers/w17653)
- Tools: TA-Lib 0.8.2 (150+ indicators, wheels, NumPy 2), tsfresh 0.21.2 (31 May 2026), pandas-ta (beta, Python >=3.12; last listed 14 Sep 2025) — [TA-Lib](https://pypi.org/project/TA-Lib/), [tsfresh](https://pypi.org/project/tsfresh/), [pandas-ta](https://pypi.org/project/pandas-ta/)

### Inferences
- Feature recipe supported by evidence: multi-window momentum (skipping most recent month/week), short-horizon reversal, realised/range-based volatility, volume/turnover/Amihud-style illiquidity, beta and idiosyncratic residual returns, overnight vs intraday split, distance from rolling highs/lows — each cross-sectionally ranked/z-scored per date within market (TSX, TSXV, US, crypto separately) and volatility-normalised. Add calendar/time-of-day features for intraday targets.
- Fractional differentiation (AFML) is a stationarity-preserving transform for price-level features; I retrieved no evidence on whether it improves out-of-sample results — treat as optional and test.
- Microstructure/OFI alpha requires L2/L3 or tick-by-tick data and fast execution; its useful horizon is a few price changes, so it is out of reach for a solo, retail-latency stack on TSX/US equities, though crypto order-book data from exchanges is cheaper to obtain (not verified).
- TSXV small caps carry the "predictable but hard to trade" pattern (GKX/Avramov/Canada blog): spreads and illiquidity may consume the signal.

### Gaps
- No source retrieved on fractional differentiation performance, on `fracdiff` library status, or on tsfresh's usefulness for financial series.
- Qlib Alpha158/Alpha360 feature definitions not retrieved; I only have benchmark results.
- No evidence found on calendar/time-of-day features' contribution for TSX/crypto.
- No retrieved quantification of TSX/TSXV spreads or costs.

---

## 7. Validation and overfitting: walk-forward, purged/embargoed CV, CPCV, deflated Sharpe, PBO, look-ahead and survivorship bias

### Takeaway
The multiple-testing literature says most published "alphas" are over-stated; the minimum bar is time-ordered OOS testing with purging/embargo, counting every trial run, and reporting Deflated Sharpe/PBO rather than best-of-N backtest Sharpe. Pretrained models and LLMs add a leakage channel (training data may overlap the test period).

### Cited Findings
- Deflated Sharpe Ratio (Bailey & López de Prado, JPM 2014): corrects for selection bias under multiple testing and non-normal returns; inputs are number of independent trials N, variance of trial Sharpe ratios, sample length T, skewness and kurtosis; the expected maximum Sharpe among N independent trials grows with N (Eq. 1, using the Euler-Mascheroni constant); worked example: a strategist reports annualised SR 2.5 over a 5-year daily sample (T=1250) after many tested variants and the investor rejects it as not a legitimate discovery at 95% confidence — [DSR paper PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) (read from PDF text; numeric N/variance in the example lost in extraction)
- Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu, J. Computational Finance 2015): combinatorially symmetric cross-validation (CSCV) over a matrix of N trial returns; argues hold-out is unreliable for backtests — [SSRN 2326253](https://papers.ssrn.com/abstract=2326253)
- skfolio `CombinatorialPurgedCV`: parameters `n_folds` (default 10), `n_test_folds` (default 8), `purged_size`, `embargo_size`; produces multiple recombined test "paths" instead of one; based on AFML (López de Prado 2018); skfolio also has `WalkForward` — [skfolio docs](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html), [PyPI skfolio](https://pypi.org/project/skfolio/)
- Label overlap: fynance warns triple-barrier labels use future prices by design, so purged/embargoed splits are required — [fynance](https://fynance.readthedocs.io/en/master/_modules/fynance/features/labels.html)
- Protocols used by the stronger empirical papers: GKX 30-year recursive out-of-sample analysis; Rahimikia expanding window 2001-2023 and rolling origin; Noguer i Alonso rolling-origin with Diebold-Mariano (HLN-corrected) vs random walk — [GKX](https://www.aqr.com:443/-/media/AQR/Documents/Journal-Articles/Empirical-Asset-Pricing-in-Machine-Learning.pdf?sc_lang=en), [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1), [arXiv 2606.27100](https://arxiv.org/pdf/2606.27100)
- Harvey, Liu, Zhu (RFS 2016): new factors should clear t > 3.0 instead of 2.0 given extensive data mining; secondary summaries (citing 296-316 factors) say roughly a third to a half of published factors may be false positives — [NBER w20592](https://www.nber.org/papers/w20592)
- Leakage in pretrained/LLM models: training leakage can make a model appear to predict past returns; benchmarks such as Look-Ahead-Bench and time-stamped models (ChronoBERT/ChronoGPT, DatedGPT) exist for LLMs; evidence on actual leakage impact is mixed — [arXiv 2412.07031](https://arxiv.org/pdf/2412.07031), [arXiv 2601.13770](https://arxiv.org/pdf/2601.13770), [arXiv 2502.21206](https://export.arxiv.org/pdf/2502.21206)
- Kronos's design (test starts after pre-training cutoff) and Rahimikia's headline "no transaction costs" status illustrate what to check in published claims — [arXiv 2508.02739](https://arxiv.org/html/2508.02739v1), [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1)
- Avramov et al.: ML profits drop sharply when excluding microcaps, distressed firms and high-limits-to-arbitrage periods (one summary: payoff down 62% excluding microcaps) — [SSRN 3450322](https://papers.ssrn.com/abstract=3450322), [summary](https://www.finews.asia/finance/32603-cuhk-ai-based-stock-picking-overrated)

### Inferences
- Concrete practice list (synthesis from the cited methods): (1) time-ordered splits only; per-horizon purge = label horizon + embargo >= max feature lookback; (2) nested design — tune on inner walk-forward, report on an untouched final period; (3) log every model/feature/hyperparameter trial and compute DSR with N = total trials; run PBO/CSCV on the trial matrix; (4) use CPCV to get a distribution of Sharpe across paths rather than a single path; (5) fit scalers/rank transforms and feature selection inside each training fold; (6) use point-in-time, delisting-inclusive universes (survivorship) — especially important for TSXV where many names delist; (7) report IC and its t-stat, DM test vs naive baselines, and net-of-cost metrics under realistic fill assumptions; (8) include a "no-trade/zero forecast" and a simple linear/GBDT baseline in every table; (9) for any pretrained model, confirm that pre-training cutoff precedes the test window, or keep a post-cutoff holdout.
- Rahimikia's headline Sharpe ratios (6.79 for CatBoost, long-short, explicitly before transaction costs, on a universe of ~10,000 US securities; portfolio-construction details not verified here) are very unlikely to be implementable, so a pre-cost Sharpe far above ~2 in a frequently rebalanced long-short on a broad universe should be read as a cost/capacity artifact until proven otherwise (my judgement, not a cited claim).

### Gaps
- Primary source for survivorship-bias evidence on TSX/TSXV not retrieved; no point-in-time TSX data vendor assessment done.
- DSR numeric example parameters (N, variance) not recovered from the PDF text; CPCV original text (AFML ch.12) not retrieved.
- No systematic replication study of deep-learning stock-prediction papers was found (see section 8).

---

## 8. Evidence check: how predictable are stock returns from historical data, and does "patterns repeat" hold?

### Takeaway
Some predictability exists and is statistically real (momentum, short-term reversal, liquidity, volatility signals; ML adds modestly via nonlinearity), but it is small (monthly R2 ~0.4%, daily directional accuracy ~51%, Qlib ICs 0.03-0.05), concentrated in small/illiquid names, decays after discovery (about 26% lower OOS and 58% lower post-publication per McLean-Pontiff) and shrinks sharply after costs. "Patterns repeat" is partly true but with time-varying strength, so a model must be expected to decay and be re-validated continuously.

### Cited Findings
**Magnitude of ML-based predictability**
- GKX (RFS 2020): OLS all features R2 -3.46%; best NN3 0.40% monthly; trees/NNs best; large-cap top-1,000 R2 0.52-0.70% for trees/NNs; S&P 500 timing with NN forecasts has annualised OOS Sharpe 0.77 vs 0.51 buy-and-hold (portfolio-level R2 for the bottom-up S&P 500 forecast is as high as 1.80% monthly for NN3, far above the stock-level 0.40%); stock-level NN long-short decile spread Sharpe 1.35 value-weighted and 2.45 equal-weighted vs OLS 0.61/0.83; the text notes value-weight portfolios are less sensitive to trading-cost considerations, and I did not find a net-of-cost Sharpe in the passages read — [Gu, Kelly, Xiu 2020](https://www.aqr.com:443/-/media/AQR/Documents/Journal-Articles/Empirical-Asset-Pricing-in-Machine-Learning.pdf?sc_lang=en)
- Kelly, Malamud, Zhou (J. Finance 2024): argue simple models understate predictability and that "complex" (parameters > observations) models predict better; I found no retrieved critique — [NBER w30217](https://nber.org/papers/w30217)
- Avramov, Cheng, Metzker (Mgmt Sci 2023): deep-learning profits come mainly from hard-to-arbitrage stocks and high limits-to-arbitrage states; value-weighted long-short returns 0.95-2.18%/month (FF6 alpha 0.62-1.87%) before costs (secondary); costs, microcap exclusion and high-volatility exclusions sharply reduce profits; authors still conclude the models identify real mispricing — [SSRN 3450322](https://papers.ssrn.com/abstract=3450322), [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.2022.4449)
- Daily horizon: CatBoost OOS R2 about -0.10%, directional accuracy 51.16%; OLS -0.47%; zero-shot TSFMs negative — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1)
- Canada (secondary blog): ML long-short monthly excess returns >1%, Sharpe >0.75 vs ~0.30 linear; concentrated in small-cap/value; no cost treatment stated — [CFA Institute 2026](https://rpc.cfainstitute.org/blogs/enterprising-investor/2026/machine-learning-improves-forecasting)
- Qlib CSI300 benchmark ICs 0.01-0.05 with best annualised returns ~9-12% (costs not stated) — [Qlib benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)

**Decay, replication and data mining**
- McLean & Pontiff (J. Finance 2016): across 97 predictors, returns are about 26% lower out of sample and 58% lower post-publication (the ~32-pt gap attributed to publication-informed trading); earlier drafts report different numbers (e.g. 82 characteristics: ~10% OOS decay, ~35% post-publication) — [Ivey copy of paper](https://ivey.uwo.ca/media/3775549/pontiff.pdf), [published copy](https://Www.Gwern.net/doc/economics/2016-mclean.pdf) (figures via search summary, version-dependent)
- Hou, Xue, Zhang (RFS 2020), 452 anomalies: 65% fail the |t| >= 1.96 hurdle once microcaps are controlled (NYSE breakpoints, value weights), 82% fail a 2.78 multiple-testing hurdle; secondary sources quote different counts — [IDEAS/RFS](https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2019-2133..html)
- Contrasting evidence: Chen & Zimmermann (Critical Finance Review 2022; 319 characteristics): for 161 "clearly significant" predictors, 98% replicate with t > 1.96; reproduced and original long-short t-stats closely aligned (slope ~0.88-0.90, R2 ~82-83%) — [Fed Board FEDS](https://www.federalreserve.gov/econres/feds/open-source-cross-sectional-asset-pricing.htm)
- Harvey-Liu-Zhu: t > 3.0 hurdle — [NBER w20592](https://www.nber.org/papers/w20592)
- Fischer & Krauss (LSTM on S&P 500 constituents 1992-2015): daily pre-cost return 0.46% and Sharpe 5.8, but excess returns appear arbitraged away from ~2010, with profitability fluctuating around zero after costs; a replication finds the original returns were driven by the 1990s (some years with Sharpe near 20) and that after 2008 average daily returns for all three models are consistently below 0.1% (breakpoint 2008-2010 sample-dependent; arXiv attribution of the replication to this specific ID is from a search summary) — [FAU working paper](https://iwf.rw.fau.de/files/2015/12/11-2017.pdf), [arXiv 2608.05755](https://arxiv.org/pdf/2608.05755)
- Nagel (2012): reversal profits behave like a liquidity-provision premium that spikes in high-VIX periods — [NBER w17653](https://nber.org/papers/w17653)

**Crypto**
- Secondary evidence: Bitcoin direction accuracy 50-60% in reviews; no OOS technical-rule predictability for Bitcoin in one study; Liu et al. (2023) ML on 3,703 coins (2013-2021) report OOS R2 4.855% with small coins more predictable; hourly BTC-USDT preprint (2018-2026): naive sign strategies fail at 10 bp costs; primaries not individually verified — [arXiv 2606.00071](https://arxiv.org/pdf/2606.00071), [emergentmind 2606.00060](https://www.emergentmind.com/papers/2606.00060), [Bentley listing](https://experts.bentley.edu/en/publications/forecasting-cryptocurrency-returns-with-machine-learning-5/)

**Replication of deep-learning stock-prediction papers**
- Direct systematic evidence was not found. Related: LLM-in-finance work flags training leakage and look-ahead as substantial risks — [arXiv 2412.07031](https://arxiv.org/pdf/2412.07031); TSFM papers above report negative zero-shot R2 and sparse significance vs random walk — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1), [arXiv 2606.27100](https://arxiv.org/pdf/2606.27100)

### Inferences
- Honest reading of "patterns always repeat": cross-sectional anomalies (momentum/reversal/liquidity/volatility) are repeatable on average (Chen-Zimmermann replication; GKX), but (a) their size is small relative to noise (R2 < 1%, direction ~51%), (b) they decay (about 26% OOS, up to 58% post-publication on average; Fischer-Krauss ~gone after ~2008-2010), (c) they depend on regime/liquidity (Nagel), and (d) a large share of the profit sits in small, illiquid, costly-to-trade stocks (Avramov, Canada blog, GKX equal-weight vs value-weight gap). A deep model "learning the patterns" does not escape these constraints; it mostly reproduces known anomalies with some nonlinearity.
- Expect realistic net Sharpe for a solo retail pipeline to be far below published headline numbers (GKX 1.35 VW / 2.45 EW, apparently gross; Rahimikia 6.8 pre-cost; Fischer-Krauss 5.8 pre-cost). I did not retrieve a net-of-cost Sharpe consensus (Novy-Marx & Velikov, Chen & Velikov were not fetched), so any specific net number would be a guess; a sensible planning stance is that a net Sharpe of 0.5-1.0 would be a good outcome and anything above ~2 after honest costs should trigger leakage audits. (Planning heuristic, not a cited statistic.)
- For horizon: the evidence suggests more predictability and lower cost-drag at weekly-to-monthly horizons in cross-section (GKX annual R2 ~10x monthly) than at 1-hour horizons, where only microstructure signals work (Kolm: ~2 price changes) and costs/latency dominate. The user's "1 hour vs tomorrow noon" example therefore sits in the hardest zone for a retail data setup; daily-to-weekly cross-sectional ranking plus volatility-based risk control has the best evidence per unit of cost.
- Crypto is the least efficient-looking of the three markets on the thin evidence (small coins, cross-coin lead-lag, 4.9% pooled R2 in one study), but costs, regime change and survivorship (delisted coins) are acute; all crypto claims here are secondary.

### Gaps
- Net-of-cost Sharpe consensus for ML signals (Novy-Marx & Velikov; Chen & Velikov; Gu-Kelly-Xiu's own cost analysis) not retrieved.
- No systematic estimate of how many published deep-learning stock prediction results fail to replicate was found; only anomaly-literature replication (Hou-Xue-Zhang vs Chen-Zimmermann) and individual case studies (Fischer-Krauss).
- McLean-Pontiff numbers vary by draft; the journal-version figures come from a search summary rather than the paper text.
- TSX/TSXV-specific out-of-sample evidence is limited to one unreviewed blog write-up (Attig & Attig); no peer-reviewed TSX ML result retrieved.
- Crypto primary papers (Liu et al. 2023; hourly BTC preprint; 2024 cross-coin lead-lag study) were not individually retrieved.
- Kelly-Malamud-Zhou critiques (e.g. from Nagel and others) not found.

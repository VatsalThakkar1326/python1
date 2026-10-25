# US Equities/Options and Crypto: Paper Trading, Testnets, Live Data Feeds, and Canadian-Resident Access (as of 25 Oct 2026)

Method note for the report writer: all pages were read through a fetch tool that returns a model-written summary of the page, not raw text. Treat exact prices and numbers as "as summarised on the page at fetch time (Oct 2026)" and re-check before purchase. Items marked ANECDOTAL come from blogs or forums. Items marked SECONDARY come from aggregators or third-party write-ups, not the vendor or regulator. Where I found nothing, it is listed under Gaps rather than guessed.

---

## 1. US equities/options broker APIs with paper trading: fill realism, data, Canadian eligibility

### Takeaway
Alpaca is the only US broker API that gives anyone (including Canadians) a free, email-only paper account with a documented fill model, but it fills at the NBBO with no slippage, no market impact, no queue position, and no size-vs-liquidity check. Free data is IEX-only, and the paid SIP/OPRA plan ($99/mo) is not currently sold to Canadian residents. IBKR's paper account is the most realistic broker-side simulator, but it requires an approved, funded live account. Several others (Schwab, tastytrade, Tradier) are either not usable for paper strategies or serve delayed or no data in their test environments.

### Cited Findings

**Alpaca (paper engine behaviour)**
- Paper orders are matched against the NBBO. A buy limit fills when its price reaches the best ask or higher. A sell limit fills when its price reaches the best bid or lower. Orders fill only once marketable. — [Alpaca paper trading docs](https://docs.alpaca.markets/docs/paper-trading)
- When an order is eligible to fill, 10% of the time it gets a partial fill of random size. The remainder is re-evaluated later if still marketable. — [Alpaca paper trading docs](https://docs.alpaca.markets/docs/paper-trading)
- Order size is not checked against NBBO quantities, so a paper order can fill far larger than the displayed liquidity. — [Alpaca paper trading docs](https://docs.alpaca.markets/docs/paper-trading)
- Not modelled: latency slippage, market impact, information leakage, queue position for non-marketable limits, price improvement, regulatory fees, dividends. Borrow fees are listed "Coming Soon". The docs say paper "may not match live results". — [Alpaca paper trading docs](https://docs.alpaca.markets/docs/paper-trading)
- Paper trading is free and open to all users. Anyone globally can create a Paper Only account with just an email address. Paper-only accounts receive only IEX market data. — [Alpaca paper trading docs](https://docs.alpaca.markets/docs/paper-trading)
- The same page says paper trading "simulates crypto trading". — [Alpaca paper trading docs](https://docs.alpaca.markets/docs/paper-trading)

**Alpaca (market data plans)**
- Basic (free): IEX-only real-time equities, 30 websocket symbols, historical limited to "latest 15 minutes" (i.e. SIP history is 15-min delayed), 200 historical calls/min, options on the "Indicative Pricing Feed", 200 options websocket quotes. — [Alpaca market data docs](https://docs.alpaca.markets/docs/about-market-data-api)
- Algo Trader Plus ($99/month): all US exchanges (SIP), unlimited websocket symbols, no historical delay limit, 10,000 historical calls/min, options on the full OPRA feed, 1,000 options websocket quotes. — [Alpaca market data docs](https://docs.alpaca.markets/docs/about-market-data-api)
- Equities and options historical data go back to 2016 on both plans. — [Alpaca market data docs](https://docs.alpaca.markets/docs/about-market-data-api)
- Alpaca crypto websocket offers trades, quotes, and an `orderbooks` channel (full snapshot first, then incremental updates, size 0 removes a level). Locations: `us` (Alpaca), `us-1` (Kraken US), `eu-1` (Kraken EU). Free-vs-paid status for crypto data was not stated. — [Alpaca real-time crypto data docs](https://docs.alpaca.markets/docs/real-time-crypto-pricing-data)

**Alpaca (options paper trading)**
- Options trading is enabled by default in the paper environment. Levels 0-3: L1 covered calls and cash-secured puts, L2 adds long calls/puts, L3 adds spreads. — [Alpaca options docs](https://docs.alpaca.markets/docs/options-trading)
- Order types: market, limit, stop, stop-limit (stop types single-leg only). Time in force day or GTC only. Whole-contract quantities only. No notional orders, no extended-hours orders. — [Alpaca options docs](https://docs.alpaca.markets/docs/options-trading)
- ITM contracts (by at least $0.01) are auto-exercised at expiry. If buying power is insufficient the position is sold within one hour before expiry. On paper, non-trade activities (exercise, assignment, expiry) appear in the Activities endpoint only the next day. Assignments are not pushed over websocket and must be polled via REST. — [Alpaca options docs](https://docs.alpaca.markets/docs/options-trading)

**Alpaca (Canadian-resident eligibility)**
- Alpaca staff on 14 Aug 2026: Alpaca does not currently offer market data subscriptions to Canadian residents because of Canadian regulatory issues. They are "actively working toward launching individual trading accounts for Canadian residents", with market data offered after those accounts launch. The stated target is regulatory approval around November 2026, after which launch would follow. — [Alpaca forum, staff reply dated 14 Aug 2026](https://forum.alpaca.markets/t/is-the-market-data-subscription-available-in-canada/19519)
- An earlier staff statement (undated in the search snippet) said Alpaca was "not accepting accounts from Canada" because the CSA requires firms doing business in Canada to be registered there. — [Search result set: Alpaca account eligibility / forum threads](https://alpaca.markets/learn/live-trading-account-non-us/) (SECONDARY: relayed via search summary, not read in full)
- The November 2026 date is a target in a forum post, not an official announcement. I found no official Alpaca announcement confirming it. — [Alpaca forum post, 14 Aug 2026](https://forum.alpaca.markets/t/is-the-market-data-subscription-available-in-canada/19519)

**Interactive Brokers (IBKR)**
- IBKR docs say a paper account runs on a simulator with no real execution or clearing and can differ from production. TWS API docs say the paper environment "relies on more simulated technologies than the Live trading environment", so order execution may vary. — [IBKR TWS API: paper trading limitations](https://www.interactivebrokers.com/docs/tws-api/doc/notes-limitations/limitations/paper-trading)
- Known gaps listed in IBKR's glossary/docs: VWAP, Auction, RFQ and Pegged-to-Market order types unsupported. Fills are simulated from the top of the book with no deep book access. Combo trading is limited. Stops and other complex orders are always simulated. No penny fills on US options in paper. If a market order partly executes, the simulator rejects the remainder. No mutual funds. — [IBKR glossary entry for paper trading](https://www.interactivebrokers.com/campus/wp-json/wp/v2/glossary-terms/179994)
- Access requires an approved and funded live account, from which you open the paper account in Account Management. — [IBKR glossary entry](https://www.interactivebrokers.com/campus/wp-json/wp/v2/glossary-terms/179994); [brokerlistings review](https://brokerlistings.com/reviews/interactive-brokers/does-interactive-brokers-have-paper-trading) (SECONDARY for the funded-account point)
- Third-party commentary says paper fills are often clean and near-instant, and paper limit orders can fill quickly where live ones would queue. — [PickMyTrade](https://pickmytrade.io/help/tws/tws-paper-fills-differ-live/) (SECONDARY/ANECDOTAL)
- ib_async is a third-party async Python library built on the TWS API and needs a locally running TWS or IB Gateway. The Client Portal Web API supports only IBKR Pro accounts. IBKR is consolidating to a single Web API with OAuth 2.0. — [brokerlistings: IBKR API overview](https://brokerlistings.com/reviews/interactive-brokers/does-interactive-brokers-have-an-api) (SECONDARY)
- Canada-specific: Interactive Brokers Canada does not allow clients to use their own trading application to electronically submit orders via API for products traded on a Canadian exchange or marketplace (a CIRO dealer-member rule), including third-party integrations. The wording scopes this to Canadian-exchange products. — [brokerlistings: IBKR API overview](https://brokerlistings.com/reviews/interactive-brokers/does-interactive-brokers-have-an-api); [TradersPost IBKR page](https://docs.traderspost.io/docs/all-supported-connections/interactive-brokers) (SECONDARY; the search summary did not say which of these two pages carried the statement)

**Tradier**
- The sandbox is a paper-trading account for testing integrations with delayed market data. Sandbox equities and options are delayed (15 minutes), while live brokerage data is real-time. Indices are unavailable in sandbox. — [Tradier market data docs](https://documentation.tradier.com/brokerage-api/overview/market-data)
- You must sign up for a Tradier Brokerage account and create a paper access token. Real-time data is only for Brokerage account holders. — [Tradier docs](https://docs.tradier.com/docs/endpoints)
- The 15-minute sandbox delay is fixed. — [TradersPost on Tradier](https://blog.traderspost.io/article/why-isnt-tradier-showing-as-a-paper-broker-option-in-traderspost) (SECONDARY)

**tastytrade**
- Sandbox (cert) host is `api.cert.tastyworks.com` versus production `api.tastyworks.com`. Separate OAuth credentials. Every request needs a `product/version` User-Agent. The sandbox does not serve market data. — [tastytrade developer docs](https://developer.tastytrade.com/docs/get-started/); [FAQ](https://developer.tastytrade.com/docs/faq/)

**Charles Schwab Trader API**
- No paper-trading support via the API. Evidence is a forum user relaying a Schwab support reply (secondhand), plus QuantConnect noting Schwab provides no paper-trading feature. paperMoney is a separate thinkorswim desktop feature. — [useThinkScript thread](https://usethinkscript.com/threads/schwab-api-paper-trading-for-automated-trading-systems.22197/post-161474); [TradersPost on Schwab paper](https://blog.traderspost.io/article/does-schwab-have-paper-trading) (SECONDARY/ANECDOTAL)

**TradeStation**
- A SIM API mirrors the live API: change the base URL from `https://api.tradestation.com/v3` to `https://sim-api.tradestation.com/v3`. SIM uses fake accounts and fake money. Orders are not executed, only simulated executions with instant "fills". — [TradeStation API: SIM vs Live](https://api.tradestation.com/docs/fundamentals/sim-vs-live)
- Simulated trading may only be done on symbols tied to your real-time data entitlements. — [TradeStation desktop help](https://help.tradestation.com/10_00/eng/TradeStationHelp/desktop/simulated_trading.htm)
- A third-party comparison lists TradeStation paper as having "delayed fills and market data sometimes", which conflicts with the official "instant fills" statement. — [TradersPost paper comparison](https://blog.traderspost.io/article/broker-paper-trading-comparison) (SECONDARY)

**Webull and moomoo**
- Webull announced an expanded paper trading experience on 21 Jul 2026 (crypto, futures, bonds, event contracts) that can connect via API. Whether it is available to Canadian accounts was not stated. — [Barchart/Webull release](https://www.barchart.com/story/news/3382513/webull-unveils-enhanced-paper-trading-experience-with-professional-grade-and-openapi-multi-asset-simulation)
- The unofficial `webull` PyPI package has a `paper_webull` class and warns Webull may change endpoints at any time. A third-party guide shows the official OpenAPI SDK with a region parameter of "us". — [PyPI webull](https://pypi.org/project/webull/); [Zuplo Webull API guide](https://zuplo.com/learning-center/webull-api) (SECONDARY)
- moomoo Canada exists (CIRO-regulated per one guide) and offers non-registered, RRSP, and TFSA accounts, US and Canadian stocks/ETFs, US options, and in-app paper trading. Margin accounts reportedly need a SIN, a Canadian address, and about CAD $7,000 in net assets. API access for Canadian accounts was not mentioned in any result. — [money.ca moomoo Canada review](https://money.ca/investing/reviews/moomoo-canada); [Finder CA](https://www.finder.com/ca/stock-trading/moomoo-review) (SECONDARY)

### Inferences
- For a "near-replica of a real platform", the fill-model quality ranking from the evidence is: IBKR paper (top-of-book simulator with documented gaps) > Alpaca paper (NBBO fills, random 10% partials, no size check) > TradeStation SIM (instant fills). For all of them, expect optimistic fills for limit orders and for any size larger than the touch.
- A Canadian resident can start today with Alpaca paper + free IEX data, but that IEX feed is a single-exchange feed. The 14 Aug 2026 staff post implies the SIP/OPRA subscription ($99/mo) is not purchasable by Canadians until the Canadian entity is approved (target Nov 2026, unconfirmed). Data realism for a SIP-equivalent signal must therefore come from elsewhere (Databento, Massive).
- IBKR is likely the best "real broker" route for a Canadian resident, but the IBKR Canada API restriction seems to apply to Canadian-exchange products only, so US stocks/options via API appear unaffected. This is an inference from secondary sources and should be verified with IBKR Canada.
- Alpaca options paper trading is usable but has structural limits that matter for options strategies: no extended hours, no notional, stop orders single-leg only, assignment not pushed.
- Background knowledge NOT verified in this session (flag when used): the `alpaca-py` SDK is the current official Alpaca Python SDK; `ib_async` is the maintained successor to `ib_insync`.

### Gaps
- E*TRADE API sandbox: not researched beyond the PDT notice. No findings.
- Canadian eligibility for Tradier, tastytrade, TradeStation, Schwab Trader API, Webull OpenAPI, and moomoo OpenAPI: no source found. Needs each broker's own eligibility pages or support.
- IBKR: no official statement found on exact market-data subscriptions needed for API use in paper mode.
- Alpaca's own paper docs do not describe order-rate limits or PDT behaviour on paper. The page did not cover them.

---

## 2. Crypto exchange testnets and demo trading: which mirror mainnet, which are isolated

### Takeaway
Binance's classic Spot Testnet has prices and order books independent of mainnet (thin, synthetic), while the newer Binance Demo Mode and the Bybit/OKX "Demo Trading" modes are designed to track real prices (Bybit's demo uses mainnet public websocket streams). Classic isolated testnets (Binance Testnet, Deribit, Hyperliquid) are best for API plumbing, not for realistic fills. Coinbase's Advanced Trade sandbox is a static mock and not useful for strategy simulation.

### Cited Findings

**Binance**
- Spot Testnet is separate from production (`testnet.binance.vision`). Funds are virtual and cannot be moved in or out. The testnet is "periodically reset to a blank state" about monthly without notice, which clears orders. API keys survive resets (since Aug 2020). Only `/api` endpoints exist, not `/sapi`. — [Binance Spot Testnet general info](https://developers.binance.com/docs/binance-spot-api-docs/testnet/general-info)
- Testnet endpoints: REST `https://testnet.binance.vision/api`; WS API `wss://ws-api.testnet.binance.vision/ws-api/v3`; streams `wss://stream.testnet.binance.vision/stream`; plus FIX order-entry, drop-copy, and market-data endpoints. IP and order rate limits and filters generally match the Spot API. — [Binance Spot Testnet general info](https://developers.binance.com/docs/binance-spot-api-docs/testnet/general-info)
- Binance's own comparison: "Testnet prices and order books are independent of the live exchange." Demo Mode's market data is "designed to be similar to live data". Testnet suits testing upcoming features, Demo Mode suits testing against realistic market conditions. — [Binance Spot Demo Mode general info](https://developers.binance.com/en/docs/products/spot/demo-mode/general-info)
- Demo Mode: balance can be reset any time in the UI; IP limits, unfilled-order counts, and exchange filters are "exactly the same" as live. Binance cautions that "market data is realistic but not real" and a strategy that works in Demo Mode may not work live. Endpoints: REST `https://demo-api.binance.com/api`, WS API `wss://demo-ws-api.binance.com/ws-api/v3`, streams `wss://demo-stream.binance.com`, FIX hosts `demo-fix-oe/dc/md.binance.com`. API keys are created while logged into a Binance account via Binance Demo Trading (`demo.binance.com`). — [Binance Spot Demo Mode general info](https://developers.binance.com/en/docs/products/spot/demo-mode/general-info)
- Binance Demo Trading launch coverage: users in markets Binance supports can access Demo after registering, even before KYC. Starting balances reported at $5,000 (spot) and $16,800 (futures), resettable. Futures demo reportedly covers USDⓈ-M and COIN-M. — [CXOToday press release coverage](https://cxotoday.com/press-release/new-binance-spot-and-futures-demo-trading-makes-crypto-learning-easy-and-lowers-barriers-of-entry-in-simulated-trading-experience/) (SECONDARY)
- Futures Testnet: Binance's developer quick-start says it can be used to practise futures trading and is currently API-only, with a separate "Futures Demo Trading" page that I could not retrieve. A third-party article gives the testnet host as `testnet.binancefuture.com` and WS `wss://stream.binancefuture.com`. — [Binance derivatives quick start](https://developers.binance.com/en/docs/products/derivatives-trading-options/quick-start); [Medium: Binance Futures client in Python](https://medium.com/@joaotx/building-a-binance-futures-client-in-python-part-i-3c3ac3b40639) (SECONDARY for hostnames)
- ANECDOTAL, 2021: a blogger using the Spot Testnet found prices dramatically different from mainnet, only about twenty pairs, and a market sell of 5,000 TRX that expired with zero execution because there were no buyers. Limit orders just above market sat unfilled. Current behaviour may differ. — [datawookie: Binance Spot Trading: Market Orders](https://www.r-bloggers.com/2021/10/binance-spot-trading-market-orders/); [Liquidity post](https://datawookie.dev/blog/2021/11/binance-spot-trading-liquidity/)

**Bybit**
- Demo Trading is an isolated module with its own account/UID, accessed by switching to "Demo Trading" on a mainnet login. REST host `https://api-demo.bybit.com`. Private WS `wss://stream-demo.bybit.com`; public streams use the mainnet `wss://stream.bybit.com`, and the docs say public WS data is "identical to that found on mainnet". WS Trade not supported. — [Bybit V5 Demo Trading docs](https://bybit-exchange.github.io/docs/v5/demo)
- Funds are requested via `POST /v5/account/demo-apply-money` (max per request: BTC 15, ETH 200, USDT 100,000, USDC 100,000; 1 request/min). Demo orders are kept for 7 days. Default rate limits apply and cannot be upgraded. "Not every API is available" but the unsupported ones are not listed. — [Bybit V5 Demo Trading docs](https://bybit-exchange.github.io/docs/v5/demo)
- Demo Trading is separate from the Bybit Testnet website, and the docs say using demo trading on testnet is "meaningless". Anyone with a mainnet account can use Demo Trading. — [Bybit V5 Demo Trading docs](https://bybit-exchange.github.io/docs/v5/demo)
- The docs do not explicitly say whether demo REST market data comes from mainnet. Only the public WS is stated as identical. — [Bybit V5 Demo Trading docs](https://bybit-exchange.github.io/docs/v5/demo)

**OKX**
- Demo Trading: log in, go to Trade > Demo Trading > Personal Center > Demo Trading API, create a demo key. Add header `x-simulated-trading: 1` to every request. REST host is the same as production (`https://openapi.okx.com`). WS hosts are `wss://wspap.okx.com/ws/v5/{public|private|business}`. Withdrawals, deposits, and purchase/redemption unavailable. Demo keys do not expire. — [OKX API v5 docs](https://www.okx.com/docs-v5/en/#overview-demo-trading-services)
- The section read does not say whether demo market data is live mainnet data, and gives no demo-specific rate limits. (The page was 406k characters and only the first 100k were read.) — [OKX API v5 docs](https://www.okx.com/docs-v5/en/#overview-demo-trading-services)

**Coinbase**
- The Advanced Trade sandbox is a static environment: no authentication, mocked responses in production format, base URL `https://api-sandbox.coinbase.com/api/v3/brokerage/`, only Accounts and Orders endpoints. An `X-Sandbox` header triggers predefined failure cases. — [Coinbase Advanced Trade sandbox docs](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/sandbox.md)
- Coinbase docs say international derivatives move from INTX to a Deribit-powered gateway on 9 Sep 2026 (hard cutover) with a new JSON-RPC 2.0 protocol. Spot and US futures stay on Advanced Trade. That date is now past as of this note, so INTX-era sandbox references may be stale. — [Coinbase Advanced Trade docs (search result)](https://docs.cdp.coinbase.com/advanced-trade/docs/changelog)

**Kraken**
- Futures demo is self-service at `demo-futures.kraken.com`, completely separate from production, with its own API key and no existing-account credentials needed. REST/WS structure is identical to production, only the host differs. Resets periodically. python-kraken-sdk has a sandbox flag for Futures. — [Kraken Futures introduction](https://docs.kraken.com/api/docs/guides/futures-introduction); [python-kraken-sdk docs](https://python-kraken-sdk.readthedocs.io/en/stable/05_futures/rest.html)
- Spot has a UAT environment, but only on request via an Account Manager, so it is not self-service for retail. — [Kraken quickstart](https://docs.kraken.com/home/guides/quickstart)

**Deribit**
- `test.deribit.com` is a completely separate platform from production and needs a separate account. Settings, credentials, balances, and API keys are not shared. Test funds come from a faucet. — [Deribit support: testnet funds](https://support.deribit.com/hc/en-us/articles/28685432379933-Depositing-test-funds-to-your-testnet-account); [Deribit support](https://support.deribit.com/hc/en-us/articles/28685393662365)
- A community write-up notes some testnet instruments have limited liquidity, and its author built a script that copies mainnet trades to testnet to make it more liquid. — [SilverBeavers deribit_testnet_copy_trader](https://github.com/SilverBeavers/deribit_testnet_copy_trader/blob/main/README.md) (ANECDOTAL)
- Deribit began routing selected spot pairs to Coinbase Exchange liquidity on 13 Aug 2026 (first pair BNB-USDC). — [Deribit support (search result)](https://support.deribit.com/hc/en-us/articles/25944687755293-Features)

**Hyperliquid**
- Official docs: to use the testnet faucet you must have deposited on mainnet with the same address, and you can claim 1,000 mock USDC. — [Hyperliquid docs: testnet faucet](https://hyperliquid.gitbook.io/Hyperliquid-docs/onboarding/testnet-faucet)
- Third-party guides say the testnet order book is much shallower than mainnet, so large market orders slip heavily, and market data is accurate but liquidity differs. Testnet API URL `https://api.hyperliquid-testnet.xyz` appears only in a community MCP config. — [supa.is Hyperliquid testnet guide](https://supa.is/article/hyperliquid-testnet-how-to-practice-trading-2026); [community .env example](https://glama.ai/mcp/servers/@caiovicentino/hyperliquid-mcp-server/blob/29c0786b3a4362bc75aed1d3ff5826a879dcaf26/.env.example) (SECONDARY)

### Inferences
- Classification by realism of the book the strategy trades against:
  - Real mainnet prices/book with simulated matching: Binance Demo Mode (designed similar, not identical), Bybit Demo (public WS identical to mainnet), OKX Demo (data source undocumented in the section read).
  - Isolated/synthetic liquidity: Binance Spot/Futures Testnet, Deribit testnet, Hyperliquid testnet, Kraken Futures demo (separate environment; liquidity not documented), Coinbase sandbox (static mock).
- None of these exchange simulators document queue-position modelling or market-impact against the real book. For HFT/order-book strategies the more realistic design is: subscribe to the real mainnet public L2/L3 websocket and run your own matching engine (a paper exchange) that consumes that live book, using exchange demo/testnet only to validate order-entry plumbing and message formats. This is a design inference, not a documented vendor feature.
- Binance Demo Mode's own caveat ("realistic but not real") means it should not be relied on as proof of live profitability.

### Gaps
- Official Binance Futures Demo Trading page (base URL, data realism): not retrievable.
- dYdX testnet: search returned nothing. No findings.
- Coinbase Exchange and Coinbase International sandboxes: no official documentation found in this pass.
- OKX demo: whether market data is live mainnet is not stated in the section I read (first 100k of 406k characters).
- Whether Canadian residents can create Binance Testnet, Bybit Testnet, Deribit testnet, or Hyperliquid testnet accounts: no source found. See section 7 for exchange-level Canada status.
- Official Hyperliquid testnet page returned "Page Not Found"; only the faucet page and third-party material were available.

---

## 3. CCXT: unified API, sandbox mode, CCXT Pro websockets

### Takeaway
CCXT is MIT-licensed, supports about 104 exchanges, includes a Python version, and exposes sandbox mode via `set_sandbox_mode(True)` where an exchange has a test environment. It is the practical way to write one execution layer across exchanges, but sandbox coverage is per exchange and demo-trading modes (OKX, Bybit, Binance Demo) may need exchange-specific handling.

### Cited Findings
- CCXT is MIT-licensed, currently lists 104 exchange markets/trading APIs, and supports JavaScript, TypeScript, Python, C#, PHP, Go, Java and Rust. — [CCXT GitHub README](https://github.com/ccxt/ccxt)
- "CCXT Certified" exchanges: binance, bybit, gate, okx, kucoin, bitget, hyperliquid, bingx, mexc, cryptocom, hashkey, woofipro. — [CCXT GitHub README](https://github.com/ccxt/ccxt)
- Sandbox mode: call `exchange.set_sandbox_mode(True)` (JS `setSandboxMode(true)`) right after constructing the exchange, before any other request. CCXT only switches to a sandbox where the exchange offers one, and unsupported exchanges raise an error. — [CCXT Manual](https://github-redirect.dependabot.com/ccxt/ccxt/wiki/Manual); [CCXT Rust sandbox example](https://docs.ccxt.com/docs/examples/rust/exchange_verbose_sandbox)
- Deribit gotcha: not passing an explicit boolean left requests pointed at production. — [ccxt issue summary on Deribit sandbox](https://lightrun.com/answers/ccxt-ccxt-deribit-error-with-sandbox-mode)
- The docs site lists CCXT Pro as the WebSocket streaming product (watch tickers, order books, trades, orders), with a separate Pro manual. — [CCXT docs](https://docs.ccxt.com/#/README?id=sandbox-mode); [CCXT GitHub README](https://github.com/ccxt/ccxt)

### Inferences
- Because CCXT's sandbox switch only repoints to a documented testnet host, it will not magically give Binance Demo Mode or OKX/Bybit "demo" behaviour unless the library has explicit support for those variants. Verify per exchange with a test call before relying on it. (An unverified hint from search: some exchanges expose an `enableDemoTrading` option; I could not confirm this.)
- CCXT's unified `fetch_order_book` / `watch_order_book` abstractions normalise L2 only. For L3/MBO or exact sequence handling you will likely use the exchange's native websocket directly.

### Gaps
- Whether CCXT Pro is free and bundled in the main package (vs a paid licence) was not confirmed from the pages read. Check the Pro manual and PyPI listing.
- A per-exchange table of which exchanges have `sandbox` URLs, and confirmation of `enableDemoTrading`: not found.

---

## 4. Live and historical market data for US equities and options: feeds, costs, order book depth

### Takeaway
Cheapest full-fidelity US equities/options path for an individual is Databento Standard ($199/mo, live data, 1 month of L2/L3 history, usage-based beyond) or Massive (formerly/also Polygon) Advanced ($199/mo real-time with trades and quotes). Alpaca's $99 SIP plan is cheaper but not currently available to Canadians. IEX Cloud no longer exists. Only Databento surfaced as offering MBO (L3) and MBP-10 (L2) for US equities.

### Cited Findings
- **Databento plans:** Usage-based (historical per-GB, no subscription; live usage-based not available for Databento US Equities and being phased out for other datasets). Standard $199/mo: live data, no license fees, 16+ years L0 history, 1 year L1 history, 1 month of L2 and L3 history, pay-as-you-go for more. Plus $1,750/mo license fees (annual contract). Unlimited $4,500/mo (annual contract). — [Databento pricing](https://databento.com/pricing)
- Databento schemas include MBO (L3) and MBP-10 (L2). Products listed: Nasdaq TotalView-ITCH, OPRA, CME Globex MDP 3.0, Databento US Equities. New users get $125 historical credits expiring after 6 months. Per-GB prices were not shown in the summary. — [Databento pricing](https://databento.com/pricing)
- EQUS.MINI is a derived top-of-book dataset built from ATS and Reg NMS venue feeds. Live data is included with Standard/Plus/Unlimited via the US Equities service. — [Databento EQUS.MINI docs](https://databento.com/docs/venues-and-datasets/equs-mini)
- OPRA: consolidated trades and NBBO across US options venues. Real-time data included with an OPRA subscription. Exchange license fees are passed through month-to-month after a subscriber-status questionnaire. Current dollar prices were not retrieved. — [Databento OPRA announcement](https://databento.com/blog/opra-data); [Databento OPRA catalog](https://databento.com/catalog/opra/OPRA.PILLAR)
- **Massive (pricing page; Polygon.io rebrand not stated on the page):** Stocks plans: Basic $0 (5 calls/min, 2 years, end-of-day), Starter $29 (unlimited calls, 5 years, 15-min delayed, websockets), Developer $79 (10 years, 15-min delayed, trades), Advanced $199 (20+ years, real-time, trades and quotes). Paid plans are "Individual use". 20% off for annual billing. NYSE Order Imbalances add-on $49/mo. Options/crypto plan details not retrieved. — [Massive pricing](https://massive.com/pricing)
- **Alpaca:** see section 1 (Basic free IEX; Algo Trader Plus $99/mo SIP and OPRA). — [Alpaca market data docs](https://docs.alpaca.markets/docs/about-market-data-api)
- **IEX Cloud:** IEX Group announced on 31 May 2024 that all IEX Cloud products would be retired by 31 Aug 2024. Former executives bid for the assets; final terms undisclosed. — [WatersTechnology / Integrity Research via search](https://www.integrity-research.com/iex-cloud-shuttered-though-former-execs-acquire-assets/); [Alpha Vantage migration guide](https://www.alphavantage.co/iexcloud_shutdown_analysis_and_migration/) (Alpha Vantage is a vendor, SECONDARY)
- **Tiingo:** Starter $0/mo (500 symbols/month, 50 requests/hour, 1,000/day); Power $30/mo or $300/yr individual (110,377 symbols, 10,000/hr, 100,000/day). Both "Internal Use Only". An "IEX Feed" and crypto are listed but tier mapping and websocket support were not legible. — [Tiingo pricing](https://www.tiingo.com/about/pricing)
- **Twelve Data:** plan cards show Basic free (8 credits/min), Grow $79 (55/min), Pro $229 (610/min, 500 WS credits), Ultra $999 (2,584/min, 2,500 WS). The page also shows lower "from" prices ($29/$99/$329) in a comparison table and structured data, unexplained. No Level 2 mentioned. — [Twelve Data pricing](https://twelvedata.com/pricing)
- **Intrinio:** one secondary source says IEX Cloud customers flagged Intrinio fundamentals at about $9.6k/year as a concern. — [WatersTechnology/Integrity via search](https://waterstechnology.com/node/7951882) (SECONDARY)

### Inferences
- Of the vendors fetched, only Databento lists MBO/MBP-10 at retail-reachable prices, and only 1 month of L2/L3 history is bundled in Standard, so training an order-book model on long L3 history means usage-based per-GB purchases (cost not retrieved). Plan to sample your own recorded live feed to build history.
- At $199/mo, Databento Standard and Massive Advanced cost the same. Databento gives depth (MBO/MBP) and CME futures; Massive's Advanced plan as shown gives trades and quotes (top-of-book) only.
- Free-tier equities data (Alpaca IEX, Tiingo Starter, Twelve Data Basic, Massive Basic) is fine for plumbing and swing-style development, but IEX-only quotes and end-of-day or limited-rate data do not reproduce a consolidated NBBO for intraday decisions.

### Gaps
- Per-GB historical rates for Databento (equities, OPRA, CME) and exact OPRA exchange-fee pass-through for an individual: not retrieved.
- Massive options/crypto/indices plan details, and confirmation of the Polygon.io to Massive rebrand: the pricing page uses only the name "Massive". No source on the rebrand found.
- Finnhub pricing: page returned only a tagline. No findings.
- Nasdaq TotalView direct pricing and Intrinio pricing: not retrieved (Databento does carry TotalView-ITCH).

---

## 5. Crypto data: full-depth websockets (free) and historical tick/L2/L3 vendors

### Takeaway
Real-time full-depth crypto data is free from the exchanges' own websockets (Binance diff-depth, Coinbase Exchange full/L3 channel, Alpaca orderbooks), so live L2 needs no paid vendor. Historical L2/tick replay costs real money: Tardis.dev lists $700-$1,200/mo for Solo-tier plans.

### Cited Findings
- **Binance spot websockets:** partial book depth streams `<symbol>@depth<5|10|20>[@100ms]`, and diff depth `<symbol>@depth[@100ms]` (default 1000ms) with `U`/`u` update IDs "used to locally manage an order book". Whether streams are free and the snapshot-sync procedure were not shown in the part of the page retrieved. — [Binance Spot WebSocket streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
- **Coinbase Exchange full channel (Level 3):** per a third-party mirror of the older docs it is the complete order book with received/open/match/done/change/activate messages, needs no auth, and is rate-limited. Official rate limits: 8 requests/sec per IP (burst 20), 100 client messages/sec per IP, and a cap of 10 websocket subscriptions per product per channel on non-paid accounts. — [cbpro full channel docs (mirror)](https://tessl.io/registry/tessl/pypi-cbpro/files/docs/websocket-client.md); [Coinbase Exchange WS rate limits](https://docs.cdp.coinbase.com/exchange/websocket-feed/rate-limits.md) (first is SECONDARY)
- **Tardis.dev historical subscriptions (monthly):** Perpetuals Academic $350 / Solo $700 / Professional $1,000 / Business $3,000; Options same; Spot $450 / $900 / $1,350 / $3,500; Derivatives same as spot; All Exchanges $650 / $1,200 / $2,200 / $6,000. Academic is quarterly/yearly billing only. Minimum order $300. Pricing is per plan, not per exchange. — [Tardis.dev pricing](https://tardis.dev/#pricing)
- Tardis coverage: 50+ spot and derivatives exchanges (Binance, OKX, Bybit, Hyperliquid, Deribit, Gate.io, Coinbase, Kraken, etc.). Data: tick-by-tick trades, incremental L2 updates, order book snapshots, quotes, options chains, derivative tickers, liquidations, and L3 books where available. Solo/Pro/Academic include 4 years of history on yearly billing. Prebuilt tick-level CSV datasets downloadable on all plans. Tardis Machine (local normalised replay server) appears to be only in Professional and Business. 1 API key on non-Business plans. — [Tardis.dev pricing](https://tardis.dev/#pricing)
- **Amberdata:** order book snapshots and events across spot, futures and options; REST access to historical order book data limited to the last 18 months since May 2025 (deeper via bulk delivery such as S3/Snowflake); subscription includes all market data for the selected exchange. Prices not found. — [Amberdata changelog](https://docs.amberdata.io/changelog/️-rest-access-to-historical-order-book-data-now-limited-to-18-months); [Amberdata ordering FAQ](https://www.amberdata.io/online-market-data-ordering-faq)
- **CoinAPI:** a vendor blog discusses "real" Level 3 data and which tiers include it. The pricing page URL I tried returned 404, so pricing was not retrieved. — [CoinAPI blog](https://www.coinapi.io/blog/tier-1-vs-tier-2-market-data-where-level-3-is-included)

### Inferences
- Because free exchange websockets provide full L2 depth in real time, a solo developer can record their own L2 history at near-zero data cost, and use paid replay (Tardis) only if they need back history from years before they started recording. At $700-$1,200/mo for Solo, Tardis is beyond a typical hobby budget unless a single month is bought for a one-off backfill (month-to-month purchase is an inference; I did not verify billing flexibility beyond the page's quarterly/yearly note for Academic).
- Tardis's price table shown on the page differs from older public descriptions I recall (a lower "Individual" tier); this may reflect a recent price/plan restructure. Treat the figures as current as of the fetch but double-check on the live page.
- L3 (per-order) crypto data exists only where exchanges publish it (Coinbase full channel is the clearest example found). Binance/Bybit/OKX offer L2 only. For queue-position modelling you need L3.

### Gaps
- Kaiko pricing and data tiers: no results. CoinAPI pricing: page not retrievable. Amberdata prices: not published in results.
- Binance depth snapshot limits, websocket connection limits, and whether public streams need no authentication: not shown in the retrieved portion.
- Tardis: whether single-month purchases and free-sample days exist was not confirmed (the page did not mention "first day of month free" but does offer sample file downloads).

---

## 6. Which market is easiest and cheapest for a realistic live simulation?

### Takeaway
Crypto is cheaper and simpler for a *realistic data* simulation (free real-time full-depth L2 from public websockets, 24/7, no PDT-type constraint), but access for a Canadian resident is constrained by exchange availability and Canadian regulation. US equities need paid consolidated data ($99-$199/mo) for fidelity, and Canadian access to some of those products is restricted or pending. A hybrid design is the cheapest realistic option for either market.

### Cited Findings
- Alpaca paper accounts are free and available worldwide but only get IEX data; SIP/OPRA requires $99/mo and is not offered to Canadians as of 14 Aug 2026. — [Alpaca paper docs](https://docs.alpaca.markets/docs/paper-trading); [Alpaca forum 14 Aug 2026](https://forum.alpaca.markets/t/is-the-market-data-subscription-available-in-canada/19519); [Alpaca market data docs](https://docs.alpaca.markets/docs/about-market-data-api)
- Databento Standard is $199/mo with live data; Massive Advanced $199/mo real-time. — [Databento pricing](https://databento.com/pricing); [Massive pricing](https://massive.com/pricing)
- Crypto exchange websockets for depth are public (Binance diff-depth, Alpaca crypto orderbooks, Coinbase full channel). — see section 5 sources.
- Binance Demo Mode is limited to "markets supported by Binance" at account registration. — [CXOToday coverage](https://cxotoday.com/press-release/new-binance-spot-and-futures-demo-trading-makes-crypto-learning-easy-and-lowers-barriers-of-entry-in-simulated-trading-experience/) (SECONDARY)
- The old US PDT rule no longer applies as a hard barrier after FINRA's June 2026 change (see section 7). — [FINRA Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10)

### Inferences (rough option ranking for a solo Canadian developer; cost estimates derived from cited prices)
1. **Crypto with self-built paper exchange on live public L2** (lowest cost, highest data realism for order-book models). Data cost $0 for live depth, plus optional Tardis for deep history. Execution testing via an exchange demo/testnet available to you. Key risk: which exchange you can legally and practically open an account at (see section 7). Registered-in-Canada exchanges (Coinbase, Kraken) have public data and, for Kraken, a self-service futures demo (Canadian access to Kraken derivatives not verified).
2. **US equities/options swing and intraday (minute-to-day horizon) via Alpaca paper + free IEX data**: $0 and workable for slow strategies, but not representative of NBBO for fast strategies. Upgrade with Databento Standard ($199/mo) for live consolidated data if needed.
3. **US equities high-frequency order book**: requires Databento (MBO/MBP-10) and your own matching engine. Cost of $199/mo plus per-GB history, and no broker paper engine models queue position, so realism comes from your own simulator.
- Because no vendor's paper engine models queue position or market impact (Alpaca explicit; IBKR top-of-book fills), the common best practice is a local replay-driven simulator fed by recorded L2/L3 data. Treat this as the core simulator design regardless of market, and use broker/exchange demos to validate order lifecycle and API behaviour.

### Gaps
- Actual cost of historical L2/L3 data volume per day for a typical symbol set (Databento per-GB rates not retrieved), so total cost of backfilling is unknown.
- Whether Alpaca's Canadian launch will arrive in Nov 2026 as targeted: unconfirmed.

---

## 7. Regulatory notes: PDT rule, Canadians trading US stocks, crypto regulation in Canada

### Takeaway
FINRA's pattern day trader designation and $25,000 minimum were eliminated by an SEC-approved Rule 4210 amendment effective 4 June 2026, replaced by an intraday margin-deficit standard, with broker phase-in allowed until 20 Oct 2027. In Canada, crypto platforms must be registered (restricted dealer regime). Coinbase, Kraken, and Bitbuy are registered. Binance and Bybit both announced exits from Canada in 2023, with no evidence of return.

### Cited Findings

**PDT**
- FINRA Regulatory Notice 26-10 (published 20 Apr 2026) amends Rule 4210, replacing the day-trading margin requirements with new intraday margin standards. Effective 4 Jun 2026. The day-trade-count rules for designating a pattern day trader and the $25,000 minimum equity are removed. — [FINRA Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10)
- SEC accelerated approval is cited as Release No. 105226, dated 14 Apr 2026. Members needing more time may phase in until 20 Oct 2027 (18 months). — [FINRA Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10)
- New paragraph (d)(2) requires members to calculate an "intraday margin deficit" on days with an IML-reducing transaction, and require deficits to be satisfied promptly. Repeat failures can lead to a 90-day restriction on increasing short positions or debit balances (exceptions for small deficits). The notice sets no new general minimum equity figure. — [FINRA Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10)
- E*TRADE says a $2,000 minimum equity under existing margin rules still applies, and buying power now reflects intraday margin excess rather than prior-day end-of-day figures. — [E*TRADE: PDT rule change](https://us.etrade.com/knowledge/library/margin/pattern-day-trading-rule-change) (broker source)

**Canadian crypto regulation and exchanges**
- OSC's registered-platform list (via search snippet): Coinbase Canada Inc. restricted dealer (dealer and marketplace), decision 3 Apr 2024. Payward Canada Inc. / Payward, Inc. (Kraken) restricted dealer, decision 1 Apr 2025. Bitbuy Technologies restricted dealer, 30 Nov 2023 (absorbed CoinSmart, whose registration shows expired). Other registered names in the broader list: Ndax, Shakepay, Netcoins, Newton, VirgoCX, Fidelity Digital Assets. — [OSC registered crypto asset trading platforms](https://osc.ca/en/industry/registration-and-compliance/registered-crypto-asset-trading-platforms) (direct fetch returned 403, so this is from the search-result excerpt; the page snapshot may date to April 2024 with later entries); [Savvy New Canadians](https://www.savvynewcanadians.com/regulated-crypto-exchanges-canada/) (SECONDARY)
- Bybit stopped opening accounts for Canadian residents from 31 May 2023. From 31 Jul 2023 existing Canadian users could only withdraw or reduce positions. Open margin/derivative positions were to be closed by 30 Sep 2023. Context: Canadian regulators required crypto firms to register and banned margin/leverage trading for Canadian clients. — [The Block](https://www.theblock.co/post/232602); [Decrypt](https://decrypt.co/142945/bybit-announces-exit-from-canada); [Informer Post](https://theinformerpost.beehiiv.com/p/crypto-exchange-bybit-leaves-canadian-market-weeks-following-same-move-by-binance)
- Binance: the OSC said in 2022 that no Binance entity held any form of securities registration in Ontario. Binance later announced it was leaving Canada, citing "new guidance related to stablecoins and investor limits" (exit date was not given in the excerpt I retrieved). — [BNN Bloomberg: OSC warns Binance](https://www.bnnbloomberg.ca/unacceptable-osc-warns-binance-remains-unregistered-in-ontario-1.1701606); [BetaKit: Binance exits Canada](https://betakit.com/binance-exits-canada-as-crypto-trading-platforms-feel-broader-impact-of-stricter-regulations/)
- I found no 2026 reporting that either Binance or Bybit has returned to Canada. — [Search results (Bybit)](https://www.financemagnates.com/cryptocurrency/bybit-joins-binance-and-others-to-exit-from-canada/); [Search results (Binance)](https://thelogic.co/news/binance-incorporates-in-alberta-hires-ex-securities-regulators-for-canadian-strategy/)

**Canadians and US stocks (partial)**
- moomoo Canada lets Canadians trade US and Canadian stocks/ETFs and US options in one account, with US trading 4 AM-8 PM ET. — [money.ca moomoo Canada review](https://money.ca/investing/reviews/moomoo-canada) (SECONDARY)
- Alpaca does not currently offer live accounts or data subscriptions to Canadian residents (target regulatory approval about Nov 2026). — [Alpaca forum 14 Aug 2026](https://forum.alpaca.markets/t/is-the-market-data-subscription-available-in-canada/19519)
- IBKR Canada restricts API order submission for Canadian-exchange products. — see section 1 (SECONDARY).

### Inferences
- Because Binance and Bybit exited Canada in 2023, a Canadian resident opening a Binance or Bybit account (needed for Bybit Demo and Binance Demo Mode, which require a main account login) would likely violate those platforms' terms. The Binance Spot Testnet and Bybit Testnet are separate sites, but I could not confirm whether they are accessible to Canadians. Using a VPN to get around this is a terms-of-service and regulatory risk and is not recommended.
- Public market data endpoints (websockets and REST) are generally separate from account registration, so building a data-recording and paper-matching stack on real mainnet streams does not depend on holding a trading account, though availability of public endpoints from a Canadian IP was not tested.
- For a Canadian wanting a legal, registered-platform path for eventual live crypto trading: Coinbase Canada, Kraken Canada, Bitbuy, Newton, Ndax, Shakepay and others appear on the OSC list, but leverage/margin products are restricted for Canadians (per The Block's context), which affects futures-style testnet relevance.
- The PDT removal helps a US-account day trader. But brokers have until Oct 2027 to implement, so some US brokers may still apply the old logic for now, and Canadian brokers fall under CIRO margin rules, which I did not research.

### Gaps
- CSA/OSC primary registry could not be fetched directly (403 on osc.ca). The registered-platform list is from search excerpts and a secondary site, and may be stale.
- OKX's Canadian status: no source found.
- Exact Binance exit date: not shown in the excerpt retrieved.
- CIRO margin and day-trading rules for Canadian brokers, and W-8BEN/US-tax issues for Canadians trading US securities: not researched in this pass.
- Ontario-specific treatment beyond the OSC statements above (e.g. whether any additional Ontario restrictions apply to Kraken or Coinbase derivatives): not researched.

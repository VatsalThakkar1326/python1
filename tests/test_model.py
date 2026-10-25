import numpy as np
import torch

from mark1.data.panel import synthetic_panel
from mark1.features.build import WARMUP, build_features
from mark1.model.dataset import Normalizer, WindowSampler, make_splits
from mark1.model.network import MarkOne, ModelConfig, pinball_loss

VIEWS = ((16, 1), (8, 4), (6, 16))


def _setup(n_times=WARMUP + 1500):
    fs = build_features(synthetic_panel(n_times=n_times, symbols=["AUSDT", "BUSDT", "CUSDT", "DUSDT"], seed=7))
    splits = make_splits(len(fs.times), VIEWS, gap=60, first_row=WARMUP)
    norm = Normalizer.fit(fs, splits["train"])
    cfg = ModelConfig(n_asset_features=fs.asset.shape[2], n_market_features=fs.market.shape[1], n_assets=4,
                      d_model=16, n_heads=2, n_layers=1, n_experts=2, tcn_dilations=(1, 2, 4), lag_minutes=(0, 5),
                      views=VIEWS, max_horizon=240)
    return fs, splits, norm, cfg


def test_forward_shapes_and_monotone_quantiles():
    fs, splits, norm, cfg = _setup()
    b = WindowSampler(fs, norm, VIEWS, splits["train"]).sample(batch_size=5, n_tau=3)
    model = MarkOne(cfg)
    tb = {k: ([torch.from_numpy(x) for x in v] if isinstance(v, list) else torch.from_numpy(np.asarray(v))) for k, v in b.items()}
    pred, diag = model(tb["asset_views"], tb["market_views"], tb["ready"], tb["tau"])
    assert pred.shape == (5, 3, 4, len(cfg.quantiles))
    assert torch.all(pred[..., 1:] >= pred[..., :-1])
    assert diag["regime_gates"][0].shape == (5, 2)
    loss = pinball_loss(pred, tb["y"], tb["mask"], torch.tensor(cfg.quantiles))
    loss.backward()
    assert torch.isfinite(loss)


def test_sampler_never_reads_rows_after_t():
    fs, splits, norm, _ = _setup()
    s = WindowSampler(fs, norm, VIEWS, splits["train"])
    t = s.rows[:4]
    a = s.batch(t, np.ones((4, 1), dtype=np.int64))
    fs.asset[t.max() + 1:] = 99.0   # poison the future
    fs.market[t.max() + 1:] = 99.0
    b = s.batch(t, np.ones((4, 1), dtype=np.int64))
    for va, vb in zip(a["asset_views"], b["asset_views"]):
        np.testing.assert_array_equal(va, vb)
    for va, vb in zip(a["market_views"], b["market_views"]):
        np.testing.assert_array_equal(va, vb)


def test_labels_stay_inside_split():
    fs, splits, norm, _ = _setup()
    s = WindowSampler(fs, norm, VIEWS, splits["train"])
    t = np.array([splits["train"].end - 10])
    b = s.batch(t, np.array([[5, 50]]))
    assert b["mask"][0, 0].any() and not b["mask"][0, 1].any()


def test_model_can_learn_a_planted_signal():
    """If the future is partly predictable from a feature, training should beat the baseline."""
    fs, splits, norm, cfg = _setup(n_times=WARMUP + 3000)
    # Plant: next 5-minute return follows the sign of feature ret_1 at t (strong, deterministic).
    lc = np.asarray(fs.log_close).copy()
    rng = np.random.default_rng(0)
    signal = np.sign(rng.normal(size=lc.shape))
    fs.asset[:, :, 0] = signal
    inc = np.zeros_like(lc)
    inc[1:] = 0.002 * signal[:-1] + rng.normal(0, 0.0005, size=lc[1:].shape)
    fs.log_close[:] = np.cumsum(inc, axis=0)
    norm = Normalizer.fit(fs, splits["train"])
    sampler = WindowSampler(fs, norm, VIEWS, splits["train"], seed=1)
    torch.manual_seed(0)
    model = MarkOne(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    q = torch.tensor(cfg.quantiles)
    losses = []
    for _ in range(150):
        b = sampler.sample(16, 2)
        b["tau"][:] = 1
        b = sampler.batch(b["t"], b["tau"])
        tb = {k: ([torch.from_numpy(x) for x in v] if isinstance(v, list) else torch.from_numpy(np.asarray(v))) for k, v in b.items()}
        pred, _ = model(tb["asset_views"], tb["market_views"], tb["ready"], tb["tau"])
        loss = pinball_loss(pred, tb["y"], tb["mask"], q)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-20:]) < 0.6 * np.mean(losses[:20])


def test_torch_feature_store_matches_numpy_inputs():
    from mark1.model.dataset import assemble_inputs
    from mark1.model.feature_store import TorchFeatureStore
    fs, splits, norm, _ = _setup()
    store = TorchFeatureStore(fs, norm, VIEWS, torch.device("cpu"), chunk=700)
    t = np.array([splits["train"].start, splits["train"].start + 37, splits["val"].start + 5])
    ref = assemble_inputs(fs, norm, VIEWS, t)
    av, mv, ready, sigma = store.inputs(torch.as_tensor(t))
    for a, b in zip(ref["asset_views"], av):
        np.testing.assert_allclose(a, b.numpy(), atol=5e-3)
    for a, b in zip(ref["market_views"], mv):
        np.testing.assert_allclose(a, b.numpy(), atol=5e-3)
    np.testing.assert_array_equal(ref["ready"], ready.numpy())
    np.testing.assert_allclose(ref["sigma"], sigma.numpy())


def test_encode_exposes_pathways_and_hides_masked_coins():
    fs, splits, norm, cfg = _setup()
    cfg.masked_coin_token = True
    model = MarkOne(cfg).eval()
    b = WindowSampler(fs, norm, VIEWS, splits["train"]).sample(batch_size=3, n_tau=1)
    av = [torch.from_numpy(x) for x in b["asset_views"]]
    mv = [torch.from_numpy(x) for x in b["market_views"]]
    ready = torch.from_numpy(b["ready"])
    coin, diag = model.encode(av, mv, ready)
    assert diag["own"].shape == coin.shape and diag["lead_lag"].shape == coin.shape and diag["market"].shape == (3, cfg.d_model)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[:, 1] = True
    _, d1 = model.encode(av, mv, ready, coin_mask=mask)
    av2 = [x.clone() for x in av]
    av2[0][:, 1] += 5.0                      # change only the hidden coin's inputs
    _, d2 = model.encode(av2, mv, ready, coin_mask=mask)
    torch.testing.assert_close(d1["own"], d2["own"])


def test_pathways_add_up_to_the_coin_token():
    fs, splits, norm, cfg = _setup()
    model = MarkOne(cfg).eval()
    b = WindowSampler(fs, norm, VIEWS, splits["train"]).sample(batch_size=2, n_tau=1)
    av = [torch.from_numpy(x) for x in b["asset_views"]]
    mv = [torch.from_numpy(x) for x in b["market_views"]]
    ready = torch.ones(2, 4, dtype=torch.bool)
    coin, d = model.encode(av, mv, ready)
    pre = d["own"] + d["cross"] + d["lead_lag"] + d["experts"]
    torch.testing.assert_close(model.final_norm(pre), coin, rtol=1e-4, atol=1e-5)

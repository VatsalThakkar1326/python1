"""Forward-return targets for any horizon, computed on the fly.

The target for asset i at row t and horizon tau minutes is the future log return scaled by
the volatility known at t:

    y = (log_close[t + tau] - log_close[t]) / (sigma[t] * sqrt(tau))

Scaling makes calm and wild periods, and BTC and small alts, comparable, so the model learns
shapes rather than raw sizes. Multiply back by sigma[t] * sqrt(tau) to get a return.
"""

from __future__ import annotations

import numpy as np

TARGET_CLIP = 20.0


def invalid_cumsum(valid: np.ndarray) -> np.ndarray:
    """Running count of invalid bars, used to check a whole [t, t + tau] span in O(1)."""
    c = np.zeros((valid.shape[0] + 1,) + valid.shape[1:], dtype=np.int64)
    c[1:] = np.cumsum(~valid, axis=0)
    return c


def forward_targets(
    log_close: np.ndarray,
    sigma: np.ndarray,
    inv_cum: np.ndarray,
    t: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Targets for rows t (shape [B]) and horizons tau (shape [B, K]) across all assets.

    Returns (y [B, K, N] float32, mask [B, K, N] bool). The mask is False when t + tau runs
    past the data or any bar in (t, t + tau] is invalid.
    """
    T = log_close.shape[0]
    t = t.astype(np.int64)
    end = t[:, None] + tau.astype(np.int64)                    # [B, K]
    inside = end < T
    end_c = np.minimum(end, T - 1)
    lc_now = log_close[t][:, None, :]                          # [B, 1, N]
    lc_end = log_close[end_c]                                  # [B, K, N]
    scale = sigma[t][:, None, :] * np.sqrt(tau)[:, :, None]   # [B, K, N]
    y = (lc_end - lc_now) / scale
    bad = inv_cum[end_c + 1] - inv_cum[t + 1][:, None, :]      # invalid bars in (t, end]
    mask = inside[:, :, None] & (bad == 0)
    y = np.clip(np.where(mask, y, 0.0), -TARGET_CLIP, TARGET_CLIP)
    return y.astype(np.float32), mask


def to_returns(y: np.ndarray, sigma_t: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Undo the scaling: log returns from scaled targets or predicted quantiles."""
    return y * sigma_t * np.sqrt(tau)

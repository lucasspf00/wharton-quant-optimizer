"""Single-portfolio combined Sharpe/Sortino optimization.

All input prices must be corporate-action-adjusted daily prices or a consistently
constructed total-return index. This file deliberately performs no data downloads.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


class PortfolioError(ValueError):
    """An invalid dataset, numerical condition, or infeasible portfolio."""


def validate_prices(prices: pd.DataFrame, min_observations: int = 126) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise PortfolioError("Provide a non-empty wide-format price table.")
    p = prices.copy()
    if len(p.columns) < 2:
        raise PortfolioError("At least two investment tickers are required.")
    p.columns = [str(s).strip().upper() for s in p.columns]
    if len(set(p.columns)) != len(p.columns):
        raise PortfolioError("Duplicate tickers are not allowed.")
    try:
        p.index = pd.to_datetime(p.index, errors="raise")
    except (ValueError, TypeError) as exc:
        raise PortfolioError("The first column must contain valid dates.") from exc
    if p.index.has_duplicates:
        raise PortfolioError("Dates must be unique; duplicate dates were found.")
    p = p.sort_index()
    try:
        p = p.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise PortfolioError("Prices must be numeric.") from exc
    if np.isinf(p.to_numpy(dtype=float)).any():
        raise PortfolioError("Infinite price values are not allowed.")
    # Do NOT invent daily prices via interpolation or forward filling.
    p = p.dropna(axis=0, how="any")
    if len(p) < min_observations + 1:
        raise PortfolioError(
            f"Only {len(p)} complete price rows. At least {min_observations + 1} are required. "
            "Check mismatched listing dates or use a shorter, documented common period."
        )
    if (p.to_numpy(dtype=float) <= 0).any():
        raise PortfolioError("All prices must be strictly positive.")
    if p.index.to_series().diff().dt.days.iloc[1:].max() > 90:
        raise PortfolioError("A data gap greater than 90 days was detected.")
    return p


def daily_returns(prices: pd.DataFrame, min_observations: int = 126) -> pd.DataFrame:
    p = validate_prices(prices, min_observations=min_observations)
    returns = p.pct_change(fill_method=None).iloc[1:]
    if (returns.to_numpy(dtype=float) <= -1).any():
        raise PortfolioError("A non-positive price or total loss was detected.")
    return returns


def annual_to_daily(rate: float) -> float:
    if rate <= -1:
        raise PortfolioError("Annual rate must be greater than -100%.")
    return (1.0 + rate) ** (1.0 / TRADING_DAYS) - 1.0


def bounded_simplex_projection(values: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {w: sum(w)=1, 0<=w<=cap}."""
    z = np.asarray(values, dtype=float)
    n = z.size
    if n < 2 or not 0 < cap <= 1 or n * cap < 1 - 1e-10:
        raise PortfolioError(f"Max position {cap:.1%} is infeasible for {n} assets.")
    lo, hi = np.min(z) - cap - 1, np.max(z) + 1
    for _ in range(100):
        mid = (lo + hi) / 2
        weights = np.clip(z - mid, 0, cap)
        if float(weights.sum()) > 1:
            lo = mid
        else:
            hi = mid
    result = np.clip(z - (lo + hi) / 2, 0, cap)
    if abs(result.sum() - 1) > 1e-8:
        raise PortfolioError("Projection numerical failure.")
    return result


@dataclass(frozen=True)
class Metrics:
    expected_return: float
    volatility: float
    downside_deviation: float
    sharpe: float
    sortino: float
    composite: float


@dataclass(frozen=True)
class OptimizationResult:
    tickers: tuple[str, ...]
    weights: np.ndarray
    metrics: Metrics
    correlation: pd.DataFrame
    mean_returns: np.ndarray
    covariance: np.ndarray
    benchmark_return: float
    benchmark_sharpe: float
    benchmark_sortino: float
    sharpe_scale: float
    sortino_scale: float
    downside_floor: float
    concentration_cap: float
    observations: int
    converged: bool
    optimizer_message: str


def portfolio_raw_metrics(
    weights: np.ndarray,
    daily: np.ndarray,
    mu: np.ndarray,
    covariance: np.ndarray,
    risk_free: float,
    mar: float,
    downside_floor: float,
) -> tuple[float, float, float, float, float]:
    w = np.asarray(weights, dtype=float)
    ann_return = float(mu @ w)
    ann_variance = float(w @ covariance @ w)
    vol = float(np.sqrt(max(0.0, ann_variance)))
    portfolio_daily = daily @ w
    mar_daily = annual_to_daily(mar)
    downside = float(np.sqrt(TRADING_DAYS * np.mean(np.square(np.minimum(portfolio_daily - mar_daily, 0.0)))))
    # Floors prevent a nearly zero measured risk from producing an unbounded ratio.
    sharpe = (ann_return - risk_free) / max(vol, 1e-6)
    sortino = (ann_return - mar) / max(downside, downside_floor)
    return ann_return, vol, downside, sharpe, sortino


def optimize_combined(
    returns: pd.DataFrame,
    *,
    sharpe_weight: float = 0.40,
    max_position: float = 0.40,
    risk_free: float = 0.03,
    mar: float | None = None,
    mean_shrinkage: float = 0.50,
    random_seed: int = 2026,
    n_starts: int = 9,
) -> OptimizationResult:
    """One allocation from a 40/60 scaled Sharpe/Sortino objective.

    Each ratio is divided by a robust 10th-to-90th percentile spread of ratios
    computed from *feasible* sampled portfolios. This aligns their observed
    variation without asserting the two raw ratios are interchangeable.
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2 or len(returns) < 126:
        raise PortfolioError("At least 126 joint daily returns and 2 assets are required.")
    if not 0 <= sharpe_weight <= 1:
        raise PortfolioError("Sharpe weight must be between 0 and 1.")
    if not 0 <= mean_shrinkage <= 1:
        raise PortfolioError("Mean shrinkage must be between 0 and 1.")
    if not -0.99 < risk_free < 1 or not -0.99 < (risk_free if mar is None else mar) < 1:
        raise PortfolioError("Risk-free rate and downside threshold must be valid decimal annual rates.")
    n = returns.shape[1]
    if max_position <= 0 or max_position > 1 or n * max_position < 1 - 1e-10:
        raise PortfolioError(f"The maximum position limit ({max_position:.0%}) is infeasible for {n} assets. "
                             f"It must be at least {1/n:.1%}.")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise PortfolioError("Returns have non-finite values.")
    data = returns.to_numpy(dtype=float)
    mar = risk_free if mar is None else mar
    empirical_mu = data.mean(axis=0) * TRADING_DAYS
    # Equal-weight cross-sectional prior, deliberately documented as a heuristic.
    pooled_mean = np.mean(empirical_mu)
    mu = (1 - mean_shrinkage) * empirical_mu + mean_shrinkage * pooled_mean
    cov = np.cov(data, rowvar=False, ddof=1) * TRADING_DAYS
    cov = (cov + cov.T) / 2
    if not np.isfinite(cov).all():
        raise PortfolioError("Covariance estimate is invalid.")

    equal_weights = np.full(n, 1.0 / n)
    # The adaptive floor limits extreme Sortino values on low-downside histories.
    eq_downside = portfolio_raw_metrics(equal_weights, data, mu, cov, risk_free, mar, 1e-6)[2]
    floor = max(0.015, 0.15 * eq_downside)
    rng = np.random.default_rng(random_seed)
    sample_weights = [equal_weights]
    # Diverse feasible reference portfolios define common-scale ranges; seeded for reproducibility.
    for i in range(256):
        concentration = (0.45, 1.0, 3.0)[i % 3]
        raw = rng.dirichlet(np.full(n, concentration))
        sample_weights.append(bounded_simplex_projection(raw, max_position))
    array = np.array([portfolio_raw_metrics(w, data, mu, cov, risk_free, mar, floor)[3:]
                      for w in sample_weights])
    sharpe_scale = max(0.10, float(np.quantile(array[:, 0], 0.90) - np.quantile(array[:, 0], 0.10)))
    sortino_scale = max(0.10, float(np.quantile(array[:, 1], 0.90) - np.quantile(array[:, 1], 0.10)))

    def composite(w: np.ndarray) -> float:
        *_, sh, so = portfolio_raw_metrics(w, data, mu, cov, risk_free, mar, floor)
        return float(sharpe_weight * sh / sharpe_scale + (1 - sharpe_weight) * so / sortino_scale)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, max_position)] * n
    candidates: list[tuple[float, np.ndarray, bool, str]] = []
    starts = [equal_weights] + sample_weights[1:n_starts]
    for start in starts:
        solution = minimize(lambda w: -composite(w), start, method="SLSQP",
                            bounds=bounds, constraints=constraints,
                            options={"maxiter": 800, "ftol": 1e-10})
        w = solution.x
        feasible = (np.isfinite(w).all() and abs(w.sum() - 1) < 1e-6 and
                    w.min() >= -1e-7 and w.max() <= max_position + 1e-7)
        if feasible:
            candidates.append((composite(w), w, bool(solution.success), str(solution.message)))
    if not candidates:
        raise PortfolioError("Optimization found no feasible allocation. Check price history and position limits.")
    score, weights, converged, message = max(candidates, key=lambda c: c[0])
    weights = np.clip(weights, 0, max_position)
    # Tiny numeric renormalization (without changing meaningful bound constraints).
    if abs(weights.sum() - 1) > 1e-7:
        raise PortfolioError("Numerical weight-sum error.")
    ann_ret, vol, downside, sharpe, sortino = portfolio_raw_metrics(
        weights, data, mu, cov, risk_free, mar, floor)
    eq_ret, _, _, eq_sharpe, eq_sortino = portfolio_raw_metrics(
        equal_weights, data, mu, cov, risk_free, mar, floor)
    return OptimizationResult(
        tickers=tuple(map(str, returns.columns)), weights=weights,
        metrics=Metrics(ann_ret, vol, downside, sharpe, sortino, score),
        correlation=returns.corr(), mean_returns=mu, covariance=cov,
        benchmark_return=eq_ret, benchmark_sharpe=eq_sharpe, benchmark_sortino=eq_sortino,
        sharpe_scale=sharpe_scale, sortino_scale=sortino_scale,
        downside_floor=floor, concentration_cap=max_position,
        observations=len(returns), converged=converged,
        optimizer_message=message)


def simulate_fixed_allocation(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Historical daily portfolio returns assuming daily constant-weight rebalancing.

    Not an out-of-sample backtest; only for descriptive diagnostics.
    """
    if len(weights) != returns.shape[1]:
        raise PortfolioError("Return columns and weights have different lengths.")
    return returns @ np.asarray(weights)


def max_drawdown(returns: pd.Series) -> float:
    growth = (1 + returns).cumprod()
    drawdowns = growth / np.maximum.accumulate(np.r_[1.0, growth.to_numpy()])[1:] - 1.0
    return float(drawdowns.min())

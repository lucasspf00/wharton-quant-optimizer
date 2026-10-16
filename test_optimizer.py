import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from optimizer import (
    PortfolioError, TRADING_DAYS, daily_returns, validate_prices, optimize_combined,
    bounded_simplex_projection, max_drawdown, simulate_fixed_allocation, annual_to_daily,
)
from market_data import parse_tickers, load_wide_csv
from projections import project_wharton, projection_issues


@pytest.fixture(scope="module")
def prices():
    return load_wide_csv(str(Path(__file__).resolve().parents[1] / "examples" / "SYNTHETIC_PRICES_NOT_REAL.csv"))


@pytest.fixture(scope="module")
def returns(prices):
    return daily_returns(prices)


@pytest.fixture(scope="module")
def result(returns):
    return optimize_combined(returns, sharpe_weight=.4, max_position=.4)


def test_wharton_first_year():
    df = project_wharton(.06)
    a = df.iloc[0]
    assert a['Year'] == 2027
    assert a['Opening balance'] == 0
    assert a['Beginning-year contribution'] == 300000
    assert a['Balance invested during year'] == 300000
    assert a['Annual investment earnings'] == pytest.approx(18000)
    assert a['End-of-year balance'] == pytest.approx(318000)


def test_second_year_contributes_before_earnings():
    df = project_wharton(.06)
    a = df[df.Year == 2028].iloc[0]
    assert a['Beginning-year contribution'] == 150000
    assert a['Balance invested during year'] == pytest.approx(468000)
    assert a['Annual investment earnings'] == pytest.approx(28080)
    assert a['End-of-year balance'] == pytest.approx(496080)


def test_beginning_year_payment_2033():
    df = project_wharton(.06)
    a = df[df.Year == 2033].iloc[0]
    assert a['Operating payment funded'] == 50000
    assert a['Balance invested during year'] == pytest.approx(a['Opening balance'] - 50000)
    assert a['Annual investment earnings'] == pytest.approx(a['Balance invested during year']*.06)


def test_ten_payments_exactly():
    df = project_wharton(.06)
    paid = df.loc[df['Required operating payment']>0]
    assert paid.Year.tolist() == list(range(2033, 2043))
    assert paid['Required operating payment'].sum() == 500000
    assert df.loc[df.Year < 2033, 'Required operating payment'].sum() == 0


def test_2026_not_initially_invested():
    df = project_wharton(.0)
    assert df.Year.min() == 2027 and df.Year.max() == 2042
    assert df.iloc[0]['Opening balance'] == 0


def test_zero_return_simple_projection():
    df = project_wharton(0)
    # 450k contributions and 500k liabilities creates a 50k unfunded payment.
    assert df['Operating payment funded'].sum() == 450000
    assert df['Operating payment shortfall'].sum() == 50000
    assert df.iloc[-1]['End-of-year balance'] == 0
    assert len(projection_issues(df)) == 1


def test_facility_payment_2033_only_and_never_double_counted():
    df = project_wharton(.06, facility_payment_2033=10_000)
    assert df['Facility payment funded'].sum() == 10000
    assert df['Facility payment requested'].sum() == 10000
    assert df.loc[df.Year == 2033, 'Balance invested during year'].iloc[0] == pytest.approx(
        df.loc[df.Year == 2033, 'Opening balance'].iloc[0] - 60000)


def test_protect_operations_over_facility():
    df = project_wharton(0, initial_contribution=10_000, second_contribution=0,
                         facility_payment_2033=100000)
    a = df[df.Year == 2033].iloc[0]
    assert a['Operating payment funded'] == 10000
    assert a['Facility payment funded'] == 0
    assert a['Operating payment shortfall'] == 40000
    assert a['Facility payment shortfall'] == 100000
    assert (df['End-of-year balance'] >= 0).all()


def test_negative_interest_avoids_negative_wealth():
    df = project_wharton(-.9)
    assert (df['End-of-year balance'] >= 0).all()
    assert (df['Operating payment shortfall'] >= 0).all()


def test_wrong_projection_inputs():
    with pytest.raises(ValueError):
        project_wharton(-1)
    with pytest.raises(ValueError):
        project_wharton(.03, facility_payment_2033=-1)


def test_prices_validated(prices):
    assert len(prices) > 800
    assert prices.notna().all().all()
    assert (prices>0).all().all()


def test_return_format(prices, returns):
    assert len(returns) == len(prices)-1
    assert returns.index[0] == prices.index[1]
    assert returns.columns.tolist() == prices.columns.tolist()


def test_nan_price_rows_removed(prices):
    p = prices.copy()
    p.iloc[60,0] = np.nan
    assert len(validate_prices(p)) == len(prices)-1


def test_negative_prices_rejected(prices):
    p = prices.copy()
    p.iloc[60,0] = -1
    with pytest.raises(PortfolioError):
        validate_prices(p)


def test_duplicate_tickers_rejected(prices):
    p = prices.copy()
    p.columns = ['A','A','C','D','E']
    with pytest.raises(PortfolioError):
        validate_prices(p)


def test_duplicate_dates_rejected(prices):
    p = prices.copy()
    idx = p.index.to_list()
    idx[1] = idx[0]
    p.index = pd.DatetimeIndex(idx)
    with pytest.raises(PortfolioError):
        validate_prices(p)


def test_two_asset_min_length():
    with pytest.raises(PortfolioError):
        optimize_combined(pd.DataFrame(np.ones((15,2))),max_position=.5)


def test_optimized_feasible_weights(result):
    assert result.weights.sum() == pytest.approx(1, abs=1e-7)
    assert result.weights.max() <= .4000001
    assert result.weights.min() >= -1e-8


def test_correlation_is_symmetric(result):
    assert np.allclose(result.correlation.values, result.correlation.values.T)
    assert np.allclose(np.diag(result.correlation.values), 1)


def test_all_metrics_are_finite(result):
    metrics = result.metrics
    assert all(np.isfinite([metrics.expected_return,metrics.volatility,metrics.downside_deviation,
                            metrics.sharpe,metrics.sortino,metrics.composite]))


def test_composite_correct(result):
    expected = .4*result.metrics.sharpe/result.sharpe_scale + .6*result.metrics.sortino/result.sortino_scale
    assert result.metrics.composite == pytest.approx(expected)


def test_one_allocation_does_both_metrics(result):
    assert len(result.weights) == len(result.tickers)
    assert isinstance(result.metrics.sharpe, float)
    assert isinstance(result.metrics.sortino, float)


def test_reproducible(returns, result):
    again = optimize_combined(returns, sharpe_weight=.4, max_position=.4)
    assert np.allclose(again.weights, result.weights, atol=1e-7)
    assert again.metrics.composite == pytest.approx(result.metrics.composite)


def test_changed_risk_weight_gives_valid_solution(returns):
    changed = optimize_combined(returns, sharpe_weight=.8, max_position=.4)
    assert changed.weights.sum() == pytest.approx(1)
    assert changed.weights.max() <= .4000001


def test_impossible_cap_rejected(returns):
    with pytest.raises(PortfolioError, match="infeasible"):
        optimize_combined(returns, max_position=.1)


def test_invalid_weight_rejected(returns):
    with pytest.raises(PortfolioError):
        optimize_combined(returns, sharpe_weight=1.4)


def test_projected_feasible_random_weights():
    w = bounded_simplex_projection(np.array([2., 3., -1., 5., 0.]), .30)
    assert w.sum() == pytest.approx(1, abs=1e-8)
    assert w.min() >= 0 and w.max() <= .3000001


def test_annual_to_daily():
    assert (1+annual_to_daily(.05))**252 == pytest.approx(1.05)


def test_max_drawdown():
    assert max_drawdown(pd.Series([-.10, .05, .1])) == pytest.approx(-.10)


def test_assets_order(returns, result):
    s = simulate_fixed_allocation(returns, result.weights)
    assert s.iloc[0] == pytest.approx(float(returns.iloc[0].values @ result.weights))


def test_yahoo_ticker_parse():
    assert parse_tickers("VTI, brk.b, vbr") == ["VTI", "BRK-B", "VBR"]
    with pytest.raises(PortfolioError):
        parse_tickers("VTI, vti")


def test_full_funding_6pct():
    df = project_wharton(.06)
    assert df['Operating payment shortfall'].sum() == 0
    assert df.iloc[-1]['End-of-year balance'] > 0

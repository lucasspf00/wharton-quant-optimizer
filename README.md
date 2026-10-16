# Laura Gao — Combined Sharpe + Sortino Optimizer

A deliberately **simple** portfolio tool for the 2026–2027 Wharton Global High School Investment Competition. It computes **one** long-only allocation that blends Sharpe and Sortino, shows daily-return correlations, and projects Laura Gao's portfolio **year by year**.

## Three features

1. **One combined allocation:** 40% scaled Sharpe + 60% scaled Sortino by default, adjustable weights, per-asset caps. Both ratios are calculated for the **same** optimized portfolio.
2. **Correlation analysis:** Pearson pairwise correlations on synchronized daily adjusted price returns, with a heatmap and CSV export.
3. **Annual projections:** Beginning-of-year $300,000 (2027), $150,000 (2028), ten $50,000 beginning-of-year operating payments (2033–2042). Annual earnings and year-end balances are displayed and downloadable.

## Install and launch

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.11 or 3.12 and select **Offline demo** to verify that the app launches without internet access. The demo is labeled **SYNTHETIC**. Change to **Yahoo Finance** for real prices or **Upload CSV** for your own data.

The price CSV is a wide table: `Date,VTI,VBR,VXUS,SGOV,BND`, with one row per date and positive **adjusted/total-return-consistent** closing prices. **Do not upload percentages in place of prices.** Dates with missing prices in any asset are dropped; no prices are fabricated or forward filled. At least 127 common prices are needed (126 daily returns).

## Deploy on Streamlit Community Cloud — prevents “file doesn't exist”

**This ZIP deliberately contains `app.py` at its ROOT, not in an extra nested folder.**

1. Create a new GitHub repository (for example `laura-gao-dual-optimizer`).
2. Extract the ZIP on your computer. Upload **all extracted files and the `examples` directory** into the repository root; do not upload the ZIP alone.
3. Your repository root must visibly contain `app.py`, `optimizer.py`, `projections.py`, `market_data.py`, `requirements.txt`, `examples/` and `tests/`.
4. In Streamlit Community Cloud choose the repository, `main` branch, and **Main file path = `app.py`**.
5. Deploy. For installation issues check the Streamlit build log. The app needs the `examples` CSV for default demo mode.

## Method details

- Daily returns: `P_t/P_{t−1} − 1` (adjusted closing price proxies for total returns).
- Expected annual arithmetic returns: `252 × average(daily returns)`. A configurable shrinkage blends asset expected returns with their equal-weight average to reduce the impact of noisy estimates. The 50% default is a modeling heuristic, not empirically certified.
- Annual covariance: `252 × sample daily covariance`.
- Portfolio annual volatility: `sqrt(wᵀΣw)`.
- Sharpe ratio: `(μ_p − r_f) / σ_p`.
- Sortino ratio: `(μ_p − MAR) / downside_deviation`, with downside deviation `sqrt(252 × mean(min(r_p,daily − MAR_daily, 0)²))` and a conservative *numerical floor* `max(1.5%, 15% × equal-weight downside deviation)` to prevent near-zero measured downside risk from creating an unbounded Sortino ratio. The floor is a modeling regularizer, not a financial guarantee.
- To compare both ratios on a common scale, sample 256 seeded, feasible portfolios. Each raw ratio is divided by `max(0.10, p90 − p10)` of sampled feasible ratio observations; optimize `0.4×scaled Sharpe + 0.6×scaled Sortino` using multiple starts, SLSQP, the simplex constraint `sum(weights)=1` and `0≤weights≤max_position`. This is **one** composite objective and returns **one** portfolio, rather than two alternatives. Normalization is based on the feasible portfolio universe and is not unique.
- Correlation: conventional Pearson correlations of synchronized daily returns.

**Cautions:** In-sample optimization is subject to overfitting, noisy expected returns and return-distribution uncertainty. The computed outcome is not the theoretically optimal investment in all future states. Sharpe/Sortino are not probabilities of meeting future financial obligations. This simple tool does **not** optimize operating reserve composition or give a mathematically reliable 2031 facility contribution interval.

## Wharton projection: fixed requirements vs assumptions

The official Laura Gao case study (2026–2027):

- 2026: Year 0; $0 invested yet.
- Beginning of 2027: $300,000 contribution.
- Beginning of 2028: another $150,000 contribution.
- Beginning of 2033 **before returns**: $50,000 operating payment, repeated at the beginning of every year through 2042 inclusive (10 payments).
- These payments are **nominal, not inflation-adjusted**.
- WInS's virtual $100,000 portfolio **must not be included** in long-term Laura projections.
- Teams also need a 2033 operating reserve, facility contribution and 2031 confidence range for the final competition strategy; those are outside this simplified app's analytical scope.

**Wharton does not prescribe an annual investment return.** The UI defaults to a **6% hypothetical return**; edit it or use a justified research assumption. You may also select **Use optimized historical expected return for projections** to copy the in-sample portfolio estimate into the annual table, but that choice is especially sensitive to historical performance and is not a reliable forward forecast. The model assumes a deterministic constant annual return, no taxes, and no external contributions beyond the two specified contributions.

For each year: `available = previous balance + beginning-year contribution`; fund operating payment first (up to available wealth), then optional 2033 hypothetical facility payment (if obligations are funded); `invested = remaining available`; `earnings = invested × assumed return`; `year-end wealth = invested + earnings`. Any unfunded operating payment is displayed explicitly; the simulation never invents negative wealth or borrowed money.

## Tests

```bash
python -m pytest -q
```

Tests cover cash-flow ordering, all ten payments, default illustrative outputs, one feasible allocation, normalized ratio score, impossible position caps, correlation, repeatability, CSV handling and missing data.

## Project layout

```
app.py                 Streamlit entrypoint (repository root)
optimizer.py           One combined Sharpe–Sortino portfolio, analytics
market_data.py         Yahoo Finance and CSV adapters
projections.py         Wharton cash-flow projections
requirements.txt       Python dependencies
examples/              Clearly labeled synthetic prices and sample annual projection
tests/                 Automated tests
README.md              This guide
```

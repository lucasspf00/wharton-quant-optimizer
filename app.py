"""Laura Gao — ONE combined Sharpe + Sortino Streamlit portfolio optimizer.

GitHub / Streamlit Community Cloud entry point: app.py (repository root).
"""
from __future__ import annotations

from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from market_data import fetch_yahoo_prices, load_wide_csv, parse_tickers
from optimizer import PortfolioError, daily_returns, optimize_combined, max_drawdown, simulate_fixed_allocation
from projections import project_wharton, projection_issues

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Laura Gao | Portfolio Optimizer", page_icon="📈", layout="wide")
st.title("Laura Gao · Portfolio Optimizer")
st.caption("One combined Sharpe–Sortino allocation • Correlations • Wharton 2027–2042 projections")

with st.sidebar:
    st.header("1. Investment data")
    mode = st.radio("Price source", ("Offline demo (synthetic)", "Yahoo Finance", "Upload CSV"))
    st.caption("Demo uses artificial prices only. Choose Yahoo Finance or upload your own historical adjusted prices for real analysis.")
    years = 5
    tickers = []
    uploaded = None
    if mode == "Yahoo Finance":
        symbols = st.text_input("Tickers (comma-separated)", "VTI, VBR, VXUS, BND, SGOV")
        years = st.slider("Years of price history", 2, 15, 5)
        try:
            tickers = parse_tickers(symbols)
        except PortfolioError as exc:
            st.error(str(exc))
            st.stop()
    elif mode == "Upload CSV":
        uploaded = st.file_uploader("Upload daily adjusted prices", type=["csv"])
        st.caption("Wide CSV: first column Date; remaining columns are tickers, prices must be positive. All dates are aligned without forward filling.")
    st.divider()
    st.header("2. Combined optimization")
    sharpe_share = st.slider("Sharpe importance", 0, 100, 40, 5)
    st.caption(f"Sharpe: {sharpe_share}%   ·   Sortino: {100-sharpe_share}%")
    max_position_pct = st.slider("Max weight per asset", 10, 100, 40, 5)
    rf_percent = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=25.0, value=3.0, step=0.25)
    target_pct = st.number_input("Sortino minimum acceptable return (%)", min_value=-30.0, max_value=30.0,
                                 value=3.0, step=0.25)
    shrink_pct = st.slider("Shrink historical mean forecasts (%)", 0, 100, 50, 5)
    st.caption("Mean shrinkage reduces extreme ticker-level return estimates; 50% is a modeling choice, not a competition rule.")
    st.divider()
    st.header("3. Future earnings")
    projection_pct = st.number_input("Assumed annual return (%)", min_value=-30.0, max_value=40.0,
                                     value=6.0, step=0.5)
    use_model_projection = st.checkbox("Use optimized historical expected return for projections", value=False)
    st.caption("6% is an illustration, NOT a Wharton-required rate. The historical estimate is backward-looking and can be unrealistic.")
    facility_dollars = st.number_input("Optional 2033 facility payment ($)", min_value=0, max_value=10_000_000,
                                       value=0, step=5_000)
    st.caption("An optional what-if only. This app does not design or certify a 2033 operating reserve.")
    virtual_size = st.number_input("WInS account size for weight illustration ($)", min_value=1_000,
                                   max_value=10_000_000, value=100_000, step=10_000)

@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(what: str, tickers_tuple: tuple[str, ...], length_years: int) -> pd.DataFrame:
    if what == "Offline demo (synthetic)":
        return load_wide_csv(str(HERE / "examples" / "SYNTHETIC_PRICES_NOT_REAL.csv"))
    return fetch_yahoo_prices(list(tickers_tuple), length_years)

try:
    if mode == "Upload CSV":
        if uploaded is None:
            st.info("Upload a CSV with a Date column and at least two price columns to start.")
            st.stop()
        prices = load_wide_csv(io.BytesIO(uploaded.getvalue()))
    else:
        with st.spinner("Loading price history..."):
            prices = get_prices(mode, tuple(tickers), years)
    returns = daily_returns(prices)
    with st.spinner("Optimizing one combined allocation..."):
        result = optimize_combined(returns, sharpe_weight=sharpe_share / 100,
                                   max_position=max_position_pct / 100,
                                   risk_free=rf_percent / 100,
                                   mar=target_pct / 100,
                                   mean_shrinkage=shrink_pct / 100)
except (PortfolioError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"An unexpected input/download error occurred: {exc}")
    st.stop()

if mode == "Offline demo (synthetic)":
    st.warning("DEMO MODE — these are synthetic historical prices, not investment recommendations. Select Yahoo Finance or upload CSV before making portfolio decisions.")

if not result.converged:
    st.warning("The best feasible numerical solution was found, but its solver did not declare convergence. Consider changing the position cap and rechecking the outcome.")

weights_df = pd.DataFrame({
    "Ticker": result.tickers,
    "Allocation (%)": np.round(result.weights * 100, 3),
    "WInS illustration ($)": np.round(result.weights * virtual_size, 2),
    "Estimated asset annual return (%)": np.round(result.mean_returns * 100, 2),
}).sort_values("Allocation (%)", ascending=False)
annual_projection_return = (result.metrics.expected_return if use_model_projection else projection_pct / 100)
if not -1 < annual_projection_return <= 2:
    st.error("The computed historical return is outside the projection model's valid range. "
             "Turn off the historical-return checkbox and enter a reasonable scenario assumption.")
    st.stop()
wealth_df = project_wharton(annual_projection_return, facility_payment_2033=float(facility_dollars))

allocation_tab, correlations_tab, financial_tab = st.tabs(
    ["Optimized portfolio", "Correlation", "Annual earnings · Laura Gao"]
)
with allocation_tab:
    st.subheader("Single combined portfolio")
    cols = st.columns(4)
    cols[0].metric("Expected annual return¹", f"{result.metrics.expected_return:.2%}")
    cols[1].metric("Annual volatility", f"{result.metrics.volatility:.2%}")
    cols[2].metric("Sharpe ratio", f"{result.metrics.sharpe:.3f}")
    cols[3].metric("Sortino ratio", f"{result.metrics.sortino:.3f}")
    st.caption("¹ Arithmetic historical mean with configurable shrinkage; not guaranteed future performance. Sortino uses the chosen minimum acceptable return. Annualization uses 252 trading days.")
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        st.download_button("Download optimized allocation CSV", weights_df.to_csv(index=False).encode(),
                           file_name="optimized_portfolio.csv", mime="text/csv")
    with c2:
        fig = px.pie(weights_df, names="Ticker", values="Allocation (%)", hole=0.55,
                     title="Optimized weights")
        fig.update_layout(legend_title_text="Assets", margin=dict(t=50, b=10, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    st.write("**How the two ratios are combined**")
    st.markdown(
        "One score is maximized: `Sharpe importance × scaled Sharpe + Sortino importance × scaled Sortino`. "
        "The scale factors are measured across reproducible, feasible candidate portfolios, so the raw "
        "ratio magnitudes do not automatically determine their importance. A downside-risk floor "
        "prevents a near-zero measured downside deviation from inflating the Sortino ratio. "
        "Only one allocation is returned, with long-only holdings and the selected position cap."
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Observed price-return days", f"{result.observations:,}")
    d2.metric("Estimated max drawdown²", f"{max_drawdown(simulate_fixed_allocation(returns, result.weights)):.2%}")
    d3.metric("Largest position", f"{max(result.weights):.1%}")
    st.caption("² In-sample constant-weight historical calculation, not an out-of-sample performance claim.")
    with st.expander("Methods, assumptions, and interpretation"):
        st.write("The two risk-adjusted ratios are computed on the same price-history window; "
                 "returns are annualized, weights must sum to 100%, and individual positions are capped. "
                 "Return forecasts shrink historical annual arithmetic returns toward the cross-sectional average.")
        st.write(f"Normalization scales: Sharpe {result.sharpe_scale:.3f}; Sortino {result.sortino_scale:.3f}. "
                 f"Downside deviation floor: {result.downside_floor:.2%}. "
                 "Ratios cannot guarantee future gains or funding reliability.")
        st.write("This is not a full Markowitz efficient-frontier analysis, multi-year stochastic optimizer, "
                 "or liability-matched reserve model. No fees, tax, trading restrictions, or "
                 "ticker eligibility for WInS are automatically verified.")

with correlations_tab:
    st.subheader("Correlation of historical daily returns")
    corr = result.correlation
    fig_corr = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                         text_auto=".2f", aspect="auto", title="Pearson correlation")
    fig_corr.update_layout(height=max(420, 65 * len(corr)), margin=dict(t=50, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.dataframe(corr.round(3), use_container_width=True)
    st.download_button("Download correlations CSV", corr.to_csv().encode(),
                       file_name="correlation_matrix.csv", mime="text/csv")
    st.info("Correlation describes how assets moved together historically; a low correlation alone does not guarantee better performance or lower tail risk.")

with financial_tab:
    st.subheader("Year-by-year earnings and cash flows · 2027–2042")
    a1, a2, a3 = st.columns(3)
    a1.metric("2027 initial contribution", "$300,000")
    a2.metric("2028 additional contribution", "$150,000")
    a3.metric("2033–2042 annual operating payment", "$50,000")
    st.markdown(
        "Wharton's case specifies all contributions and withdrawals at the **beginning** of each year. "
        "Earnings are applied to the remaining invested balance during that year. "
        f"The projection uses a {annual_projection_return:.2%} annual return "
        f"({'historical in-sample estimate' if use_model_projection else 'user-entered illustrative assumption'}); "
        "it is not prescribed by Wharton."
    )
    problems = projection_issues(wealth_df)
    if problems:
        for p in problems:
            st.error(p)
    else:
        st.success("All ten scheduled operating payments are covered in this deterministic illustration. This does not establish a high probability of funding success.")
    st.metric("Projected end-of-2042 balance", f"${wealth_df.iloc[-1]['End-of-year balance']:,.0f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wealth_df["Year"], y=wealth_df["End-of-year balance"],
                             name="End-of-year portfolio", mode="lines+markers"))
    fig.update_layout(title=f"Portfolio path at {annual_projection_return:.2%} annual return (illustration)",
                      xaxis_title="Year", yaxis_title="USD", margin=dict(t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)
    formatted = wealth_df.copy()
    for column in formatted.columns:
        if column != "Year":
            formatted[column] = formatted[column].map(lambda v: f"${v:,.0f}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)
    st.download_button("Download complete annual projection CSV", wealth_df.to_csv(index=False).encode(),
                       file_name="laura_gao_annual_2027_2042.csv", mime="text/csv")
    st.caption("A single constant-return path does not quantify uncertainty, establish the operating reserve, "
               "calculate a responsible facility contribution, or produce the 2031 co-sponsor range required for the Final Report. "
               "The optional facility payment above is a hypothetical scenario, not a recommendation.")

st.divider()
st.caption("WInS implementation dollars are calculated only from optimized weights × the virtual account size. "
           "They are never added to Laura's $300,000 and $150,000 cash flows.")

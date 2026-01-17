"""Nasdaq-specific dashboard built with Streamlit.

This dashboard focuses on the Nasdaq Composite index (symbol ``^IXIC``)
and provides:
- interactive price visualisations (candlestick, line + moving averages)
- key market metrics (daily change, year-to-date change, volatility proxy)
- comparison with a curated list of Nasdaq heavyweights
- sector overview for Nasdaq-100 ETF (``QQQ``) holdings

Run locally with ``streamlit run nasdaq_dashboard/app.py`` after installing
requirements.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Iterable, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

INDEX_SYMBOL = "^IXIC"
INDEX_NAME = "Nasdaq Composite"
ETF_SYMBOL = "QQQ"

DEFAULT_COMPARABLES: List[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "ADBE",
    "PEP",
]

PERIOD_OPTIONS = {
    "1 mois": 30,
    "3 mois": 90,
    "6 mois": 180,
    "1 an": 365,
    "3 ans": 365 * 3,
}

INTERVAL_OPTIONS = {
    "Journalier": "1d",
    "Hebdomadaire": "1wk",
    "Mensuel": "1mo",
}


@st.cache_data(show_spinner=False)
def load_history(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """Return historical OHLCV data for the provided symbol."""
    data = yf.download(
        symbol,
        start=start,
        end=end + timedelta(days=1),  # include end day
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"Aucune donnée disponible pour {symbol}.")
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data


@st.cache_data(show_spinner=False)
def load_qqq_holdings() -> pd.DataFrame:
    """Fetch Nasdaq-100 ETF (QQQ) holdings with sector information."""
    ticker = yf.Ticker(ETF_SYMBOL)
    holdings = ticker.funds_holdings
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    columns = {"holding": "Ticker", "holdingName": "Nom", "sector": "Secteur", "symbol": "Ticker"}
    holdings = holdings.rename(columns=columns)
    if "Ticker" not in holdings.columns:
        return pd.DataFrame()
    return holdings[["Ticker", "Nom", "Secteur", "holdingPercent"]].rename(
        columns={"holdingPercent": "Poids (%)"}
    )


def format_change(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:+.2%}"


def describe_market(data: pd.DataFrame) -> pd.DataFrame:
    close = data["Close"]
    latest = close.iloc[-1]
    change_1d = close.pct_change().iloc[-1]
    current_year_mask = close.index.year == close.index[-1].year
    if current_year_mask.any():
        first_of_year = close[current_year_mask].iloc[0]
        change_ytd = latest / first_of_year - 1
    else:
        change_ytd = float("nan")

    rolling = close.pct_change().dropna()
    daily_vol = rolling.std()
    annual_vol = daily_vol * math.sqrt(252)

    return pd.DataFrame(
        {
            "Dernière clôture": [f"{latest:,.2f}"],
            "Variation quotidienne": [format_change(change_1d)],
            "Performance YTD": [format_change(change_ytd) if pd.notna(change_ytd) else "N/A"],
            "Volatilité annualisée": [f"{annual_vol:.2%}" if pd.notna(annual_vol) else "N/A"],
        }
    )


def plot_price_chart(data: pd.DataFrame, interval: str) -> go.Figure:
    fig = go.Figure()

    if interval == "1d":
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Cours",
                increasing_line_color="#00CC96",
                decreasing_line_color="#EF553B",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Clôture",
                line=dict(color="#1f77b4", width=2),
            )
        )

    for window, color in [(20, "#636EFA"), (50, "#EF553B"), (100, "#00CC96")]:
        if len(data) >= window:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"].rolling(window).mean(),
                    mode="lines",
                    name=f"Moyenne mobile {window}",
                    line=dict(width=1.5, color=color),
                )
            )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Points",
    )
    return fig


def compute_performance(symbols: Iterable[str], start: date, end: date) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            hist = load_history(sym, start, end, "1d")
        except ValueError:
            continue
        close = hist["Close"]
        if len(close) < 2:
            continue
        daily_returns = close.pct_change().dropna()
        total_return = close.iloc[-1] / close.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(close)) - 1
        annual_vol = daily_returns.std() * math.sqrt(252)
        sharpe = (
            annual_return / annual_vol
            if pd.notna(annual_vol) and not math.isclose(annual_vol, 0.0)
            else float("nan")
        )
        rows.append(
            {
                "Ticker": sym,
                "Performance": total_return,
                "Perf. annualisée": annual_return,
                "Volatilité": annual_vol,
                "Sharpe": sharpe,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for column in ["Performance", "Perf. annualisée", "Volatilité", "Sharpe"]:
        df[column] = df[column].map(lambda x: float("nan") if pd.isna(x) else x)
    return df.sort_values("Performance", ascending=False)


def render_sector_breakdown() -> None:
    holdings = load_qqq_holdings()
    if holdings.empty:
        st.info("Les informations sectorielles de QQQ ne sont pas disponibles pour le moment.")
        return

    sector_weights = (
        holdings.groupby("Secteur")["Poids (%)"].sum().sort_values(ascending=False)
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=sector_weights.index,
                y=sector_weights.values,
                marker=dict(color="#636EFA"),
                name="Secteurs",
            )
        ]
    )
    fig.update_layout(
        title="Répartition sectorielle du Nasdaq-100 (QQQ)",
        xaxis_title="Secteur",
        yaxis_title="Poids (%)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Composants principaux"):
        st.dataframe(holdings.head(20).style.format({"Poids (%)": "{:.2f}"}))


def main() -> None:
    st.set_page_config(
        page_title="Dashboard Nasdaq",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Dashboard spécifique au Nasdaq")
    st.caption(
        "Suivi interactif de l'indice Nasdaq Composite, de ses principaux acteurs et de la répartition sectorielle du Nasdaq-100."
    )

    today = date.today()

    with st.sidebar:
        st.header("Paramètres")
        period_label = st.selectbox("Période", list(PERIOD_OPTIONS.keys()), index=3)
        suggested_start = today - timedelta(days=PERIOD_OPTIONS[period_label])
        start_date = st.date_input("Date de début", suggested_start)
        end_date = st.date_input("Date de fin", today)
        if start_date >= end_date:
            st.error("La date de début doit être antérieure à la date de fin.")
            st.stop()

        interval_label = st.selectbox("Granularité", list(INTERVAL_OPTIONS.keys()), index=0)
        interval_value = INTERVAL_OPTIONS[interval_label]

        st.subheader("Comparables")
        selected = st.multiselect(
            "Sélectionnez des valeurs Nasdaq",
            DEFAULT_COMPARABLES,
            default=DEFAULT_COMPARABLES[:5],
        )

    try:
        history = load_history(INDEX_SYMBOL, start_date, end_date, interval_value)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader(f"{INDEX_NAME}")
        st.plotly_chart(plot_price_chart(history, interval_value), use_container_width=True)

    with col_right:
        st.subheader("Indicateurs clés")
        st.table(describe_market(history))

    st.markdown("---")

    st.subheader("Comparaison des performances")
    if selected:
        performance = compute_performance(selected, start_date, end_date)
        if performance.empty:
            st.info("Impossible de récupérer les données de performance.")
        else:
            st.dataframe(
                performance.style.format(
                    {
                        "Performance": "{:+.2%}",
                        "Perf. annualisée": "{:+.2%}",
                        "Volatilité": "{:.2%}",
                        "Sharpe": "{:.2f}",
                    }
                )
            )
    else:
        st.info("Sélectionnez au moins une valeur pour afficher la comparaison.")

    st.markdown("---")

    st.subheader("Analyse sectorielle")
    render_sector_breakdown()


if __name__ == "__main__":
    main()

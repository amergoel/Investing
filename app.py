import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from signals import (
    fetch_prices,
    build_signals,
    build_decision_signals,
    build_master_signal,
    format_label
)

from backtests import (
    backtest_linear,
    backtest_convex,
    backtest_vol_target
)

# ============================================================
# Page setup
# ============================================================

st.set_page_config(layout="wide")
st.title("📊 Personal Market Regime Dashboard")

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Controls")

tickers_input = st.sidebar.text_input(
    "Tickers (Yahoo symbols, comma separated)",
    value="^GSPC,GLD,^VIX,^SPGSCI,USO,UNG"
)

strategy = st.sidebar.selectbox(
    "Backtest Strategy",
    ["Linear", "Convex", "Vol Target (15%)"]
)

refresh = st.sidebar.button("🔄 Refresh Data")

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# ============================================================
# Styling utilities
# ============================================================

def signal_to_rgb(x):
    if pd.isna(x):
        return (60, 60, 60)
    x = float(np.clip(x, -1, 1))
    r = int(255 * max(0, -x))
    g = int(255 * max(0, x))
    return (r, g, 0)

def signal_to_hex(x):
    r, g, b = signal_to_rgb(x)
    return f"#{r:02x}{g:02x}{b:02x}"

def row_style(row):
    rgb = signal_to_rgb(row["_signal"])
    return [
        f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); color: white"
    ] * len(row)

# ============================================================
# Data loading (CACHED)
# ============================================================

@st.cache_data(show_spinner=True)
def load_all(tickers):
    prices = fetch_prices(tickers)
    masters = {}

    for t in tickers:
        sig = build_signals(prices[t])
        dec = build_decision_signals(sig)
        masters[t] = build_master_signal(dec)

    return prices, masters

if refresh:
    st.cache_data.clear()

prices, masters = load_all(tickers)

# ============================================================
# Current Signals Table
# ============================================================

st.subheader("📍 Current Signals")

rows = []

for t in tickers:
    df = masters[t].dropna(subset=["master_norm"])
    if len(df) < 2:
        continue

    act = df.iloc[-2]["master_norm"]
    cur = df.iloc[-1]["master_norm"]

    rows.append({
        "Asset": t,
        "Act On (Yesterday)": round(act, 2),
        "Current Computed": round(cur, 2),
        "Interpretation": format_label(act),
        "_signal": act
    })

summary_df = pd.DataFrame(rows)

if not summary_df.empty:
    styled = (
        summary_df
        .style
        .apply(row_style, axis=1)
        .hide(axis="columns", subset=["_signal"])
    )
    st.dataframe(styled, use_container_width=True)

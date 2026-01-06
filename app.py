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

from signals import get_close


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

if not isinstance(prices, dict) or not isinstance(masters, dict):
    st.error(f"BUG: prices is {type(prices)}, masters is {type(masters)}. They must both be dicts.")
    st.stop()


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
    







# ============================================================
# Portfolio Construction Table (Today)
# (Separate from the backtesting section — does NOT touch plots)
# ============================================================

st.divider()
st.subheader("🧮 Portfolio Construction (Today)")

def _to_tz_naive_series(x: pd.Series) -> pd.Series:
    if getattr(x.index, "tz", None) is not None:
        x = x.copy()
        x.index = x.index.tz_localize(None)
    return x

def _make_position_input(price_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact inputs we need, aligned on dates:
      - close
      - master_norm
    """
    p = get_close(price_df)
    s = master_df["master_norm"]

    if isinstance(p, pd.DataFrame):
        p = p.squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()

    p = _to_tz_naive_series(p)
    s = _to_tz_naive_series(s)

    out = pd.DataFrame({"close": p, "master_norm": s}).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out

def _position_series(df_in: pd.DataFrame, selected_strategy: str) -> pd.Series:
    """
    Matches the position logic inside backtests.py:
      Linear:     pos = master_norm.shift(1).clip(lower=0)
      Convex:     pos = (master_norm**2).shift(1).clip(lower=0)
      Vol target: pos = master_norm.shift(1).clip(lower=0) * scale
                 scale = (target / realized_vol).clip(0,2)
    """
    price = df_in["close"]
    returns = price.pct_change().fillna(0)

    base = df_in["master_norm"].shift(1).fillna(0).clip(lower=0)

    if selected_strategy == "Linear":
        return base

    if selected_strategy == "Convex":
        return (df_in["master_norm"] ** 2).shift(1).fillna(0).clip(lower=0)

    # Vol Target (15%)
    realized_vol = returns.rolling(20).std() * np.sqrt(252)
    scale = (0.15 / realized_vol).clip(0, 2).fillna(0)
    return base * scale

def _bullish_label(x: float) -> str:
    if pd.isna(x):
        return ""
    if x ==0:
        return "Neutral"
    elif x < 0.33:
        return "Mildly bullish"
    elif x < 0.66:
        return "Bullish"
    else:
        return "Very bullish"

# Build per-ticker inputs + find latest common date
inputs = {}
index_sets = []
for t in tickers:
    try:
        df_in = _make_position_input(prices[t], masters[t])
        if not df_in.empty:
            inputs[t] = df_in
            index_sets.append(set(df_in.index))
    except Exception:
        pass

if not inputs:
    st.warning("No data available to compute positions.")
else:
    common = set.intersection(*index_sets) if index_sets else set()
    as_of = max(common) if common else None

    positions = {}
    for t, df_in in inputs.items():
        pos = _position_series(df_in, strategy).dropna()
        if pos.empty:
            positions[t] = np.nan
            continue

        if as_of is not None and as_of in pos.index:
            positions[t] = float(pos.loc[as_of])
        else:
            positions[t] = float(pos.iloc[-1])

    pos_df = pd.DataFrame.from_dict(positions, orient="index", columns=["Current Position"])
    pos_df.index.name = "Asset"

    # ✅ New interpretation column
    pos_df["Interpretation"] = pos_df["Current Position"].apply(_bullish_label)

    if as_of is not None:
        st.caption(f"As of {pd.to_datetime(as_of).date()} (latest common date across assets)")
    else:
        st.caption("As of each asset’s latest available date (no single common date across all assets)")

    st.dataframe(
        pos_df.sort_index().style.format({"Current Position": "{:.4f}"}),
        use_container_width=True
    )
















# ============================================================
# Backtesting Section (Notebook-style plots)
# ============================================================

st.divider()
st.header("📈 Strategy Backtest (Notebook plots)")

def _to_tz_naive_index(x: pd.Series) -> pd.Series:
    # Handles tz-aware indices without breaking tz-naive ones
    if getattr(x.index, "tz", None) is not None:
        x = x.copy()
        x.index = x.index.tz_localize(None)
    return x

def make_bt_input(price_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean, aligned df with exactly the columns your backtests expect:
      - close
      - master_norm
    """
    p = get_close(price_df)
    s = master_df["master_norm"]

    # Force 1D series
    if isinstance(p, pd.DataFrame):
        p = p.squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()

    p = _to_tz_naive_index(p)
    s = _to_tz_naive_index(s)

    out = pd.DataFrame({"close": p, "master_norm": s}).dropna()

    # Defensive cleanup
    out = out[~out.index.duplicated(keep="last")].sort_index()

    return out

def run_one_backtest(ticker: str, selected_strategy: str) -> pd.DataFrame:
    df_in = make_bt_input(prices[ticker], masters[ticker])
    if df_in.empty:
        raise ValueError("No overlapping dates between price + signal series.")

    if selected_strategy == "Linear":
        strat, buyhold = backtest_linear(df_in, df_in)
    elif selected_strategy == "Convex":
        strat, buyhold = backtest_convex(df_in, df_in)
    else:
        strat, buyhold = backtest_vol_target(df_in, df_in, target_vol=0.15)

    # ✅ Wrap tuple output into the exact structure the plot code expects
    return pd.DataFrame({"cum_strategy": strat, "cum_buyhold": buyhold})

# UI: pick ONE asset (simple + reliable)
bt_ticker = st.selectbox("Asset", tickers, index=0)
run_bt = st.button("▶️ Run Backtest", type="primary")

if run_bt:
    try:
        res = run_one_backtest(bt_ticker, strategy)

        # ---- EXACT notebook-style plot ----
        # (two lines, title, legend, figsize)
        fig = plt.figure(figsize=(10, 5))

        if strategy == "Linear":
            strat_label = "Strategy"
            title = f"{bt_ticker} — Backtest"
        elif strategy == "Convex":
            strat_label = "Strategy (convex)"
            title = f"{bt_ticker} — Convex Scaling Backtest"
        else:
            strat_label = "Strategy (vol-target)"
            title = f"{bt_ticker} — Vol-Target 15% Backtest"

        plt.plot(res["cum_strategy"], label=strat_label)
        plt.plot(res["cum_buyhold"], label="Buy & Hold")
        plt.title(title)
        plt.legend()

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    except Exception as e:
        st.error(f"{bt_ticker}: {e}")

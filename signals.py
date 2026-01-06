import numpy as np
import pandas as pd
import yfinance as yf

# ===============================
# DATA FETCHING
# ===============================

def fetch_prices(tickers, start="2000-01-01", interval="1d"):
    prices = {}

    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start,
            interval=interval,
            progress=False
        )

        df = df[["Close"]].rename(columns={"Close": "close"})
        prices[ticker] = df

    return prices


def get_close(df):
    if "close" in df.columns:
        return df["close"]
    if "Close" in df.columns:
        return df["Close"]
    raise ValueError("No close price column found")


# ===============================
# CORE SIGNALS
# ===============================

def momentum(prices, lookback=252):
    return prices / prices.shift(lookback) - 1


def rolling_vol(prices, window=20):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std()


def sanity_distance(prices, ma_window=50):
    ma = prices.rolling(ma_window).mean()
    return (prices - ma) / ma


def build_signals(price_df):
    prices = get_close(price_df)

    signals = pd.DataFrame(index=price_df.index)
    signals["momentum_12m"] = momentum(prices)
    signals["vol_20d"] = rolling_vol(prices)
    signals["sanity_dist_50d"] = sanity_distance(prices)

    return signals


# ===============================
# DECISION SIGNALS
# ===============================

def momentum_signal(series):
    sig = pd.Series(index=series.index, dtype=float)
    sig[series > 0] = 1
    sig[series < 0] = -1
    sig[series == 0] = 0
    return sig


def volatility_regime(vol, band_k=1.5, window=20):
    mean = vol.rolling(window).mean()
    std = vol.rolling(window).std()

    upper = mean + band_k * std
    lower = mean - band_k * std

    reg = pd.Series(index=vol.index, dtype=float)
    reg[vol > upper] = -1      # panic
    reg[vol < lower] = 1       # calm
    reg[(vol <= upper) & (vol >= lower)] = 0

    return reg


def sanity_signal(dist, window=50):
    std = dist.rolling(window).std()
    sig = pd.Series(0, index=dist.index, dtype=float)

    sig[dist > std] = -1
    sig[dist < -std] = 1

    return sig


def build_decision_signals(signals):
    out = pd.DataFrame(index=signals.index)

    out["momentum_sig"] = momentum_signal(signals["momentum_12m"])
    out["vol_regime"] = volatility_regime(signals["vol_20d"])
    out["sanity_sig"] = sanity_signal(signals["sanity_dist_50d"])

    return out


# ===============================
# MASTER SIGNAL
# ===============================

def build_master_signal(decisions):
    master = (
        decisions["momentum_sig"]
        + decisions["vol_regime"]
        + decisions["sanity_sig"]
    )

    out = decisions.copy()
    out["master_raw"] = master
    out["master_norm"] = master / 3.0

    return out


# ===============================
# LABELING
# ===============================

LABEL_BINS = [-np.inf, -0.66, -0.33, 0.0, 0.33, 0.66, np.inf]
LABELS = [
    "Preservation Mode",
    "Defensive",
    "Caution",
    "Neutral / Slow",
    "Mild Positive",
    "Strong Risk-On",
]

def format_label(x):
    return pd.cut(
        [x],
        bins=LABEL_BINS,
        labels=LABELS,
        include_lowest=True
    )[0]

import numpy as np
from signals import get_close

def backtest_linear(price_df, master_df):
    price = get_close(price_df)
    returns = price.pct_change()

    position = master_df["master_norm"].shift(1).clip(lower=0)

    strat = (1 + position * returns).cumprod()
    buyhold = (1 + returns).cumprod()

    return strat, buyhold


def backtest_convex(price_df, master_df):
    price = get_close(price_df)
    returns = price.pct_change()

    position = (master_df["master_norm"] ** 2).shift(1).clip(lower=0)

    strat = (1 + position * returns).cumprod()
    buyhold = (1 + returns).cumprod()

    return strat, buyhold


def backtest_vol_target(price_df, master_df, target_vol=0.15):
    price = get_close(price_df)
    returns = price.pct_change()

    realized_vol = returns.rolling(20).std() * np.sqrt(252)
    scale = target_vol / realized_vol
    scale = scale.clip(0, 2)

    position = master_df["master_norm"].shift(1).clip(lower=0) * scale

    strat = (1 + position * returns).cumprod()
    buyhold = (1 + returns).cumprod()

    return strat, buyhold

# features.py
# Builds the EXACT 8 features your LSTM was trained on, from raw BTC-USD OHLCV.
# Use the same file in training and in the app so they never diverge.
# يبني الميزات الثماني نفسها التي تدرّب عليها نموذجك، من بيانات السعر الخام.

import numpy as np
import pandas as pd

# must match the order used in training / يجب أن يطابق ترتيب التدريب
FEATURE_ORDER = ["ret_1", "ret_5", "rsi", "close_ema12",
                 "close_ema26", "ema_cross", "vol_chg", "volatility"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input: DataFrame with 'Close' and 'Volume'. Output: the 8 model features, NaNs dropped."""
    import ta
    df = df.copy()
    close, vol = df["Close"], df["Volume"]
    rsi14 = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    ema12 = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    ema26 = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    df["ret_1"]       = close.pct_change(1)
    df["ret_5"]       = close.pct_change(5)
    df["rsi"]         = rsi14 / 100.0
    df["close_ema12"] = close / ema12 - 1
    df["close_ema26"] = close / ema26 - 1
    df["ema_cross"]   = ema12 / ema26 - 1
    df["vol_chg"]     = np.log(vol / vol.rolling(20).mean())
    df["volatility"]  = df["ret_1"].rolling(10).std()

    return df.dropna(subset=FEATURE_ORDER).reset_index(drop=True)


def last_window(df: pd.DataFrame, feature_cols, look_back: int) -> np.ndarray:
    """Return the most recent window shaped (1, look_back, n_features) for prediction."""
    data = df[feature_cols].values.astype("float32")
    if len(data) < look_back:
        raise ValueError(f"Need at least {look_back} rows, got {len(data)}.")
    win = data[-look_back:]
    return win.reshape(1, look_back, len(feature_cols))

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter


# ---------- Reused helpers from DonchianATRTrendV2 (unchanged) ----------

def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()


def _atr_rma(df: DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def compute_supertrend(df: DataFrame, atr_period: int, multiplier: float) -> DataFrame:
    """
    TradingView-style Supertrend:
    - ATR via RMA (Wilder)
    - Basic bands: hl2 ± multiplier * ATR
    - Final bands persist using previous-close rules
    - Flip when close ≥ upper or close ≤ lower (touch-inclusive)
    Returns DataFrame with ['st_trend','st_line','atr'] aligned to df.index
    """
    out = pd.DataFrame(index=df.index)

    # Core components
    hl2 = (df["high"] + df["low"]) / 2.0
    atr = _atr_rma(df, atr_period)
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Final bands (non-crossing persistence)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    for i in range(1, len(df)):
        # Upper band persistence
        if df["close"].iat[i - 1] > final_upper.iat[i - 1]:
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = min(basic_upper.iat[i], final_upper.iat[i - 1])
        # Lower band persistence
        if df["close"].iat[i - 1] < final_lower.iat[i - 1]:
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = max(basic_lower.iat[i], final_lower.iat[i - 1])

    # Trend and active line
    st_trend = pd.Series(index=df.index, dtype=int)
    st_line = pd.Series(index=df.index, dtype=float)
    st_trend.iloc[0] = 1
    st_line.iloc[0] = final_lower.iloc[0]

    for i in range(1, len(df)):
        if st_trend.iloc[i - 1] == 1:
            # Uptrend → check flip down
            if df["close"].iloc[i] <= final_lower.iloc[i]:
                st_trend.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]
            else:
                st_trend.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]
        else:
            # Downtrend → check flip up
            if df["close"].iloc[i] >= final_upper.iloc[i]:
                st_trend.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]
            else:
                st_trend.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]

    out["st_trend"] = st_trend
    out["st_line"] = st_line
    out["atr"] = atr
    return out


# ---------- Strategy ----------

class SuperEMA(IStrategy):
    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 320

    minimal_roi = {"0": 0.03}

    max_open_trades = 3

    # Minimal risk scaffolding (tune to taste)
    stoploss = -0.03

    # Supertrend parameters (same semantics as your file)
    st_period = IntParameter(7, 21, default=10, space="buy", optimize=True)
    st_mult = DecimalParameter(1.5, 4.0, decimals=2, default=3.0, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # EMAs
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

        # Supertrend (reuse your TradingView-style implementation)
        stp = int(self.st_period.value)
        stm = float(self.st_mult.value)
        st = compute_supertrend(df, stp, stm)
        df["st_trend"] = st["st_trend"]
        df["st_line"] = st["st_line"]
        df["atr"] = st["atr"]

        # EMA state and ST flips
        df["ema_up"] = df["ema9"] > df["ema26"]
        df["ema_down"] = df["ema9"] < df["ema26"]
        df["st_flip_up"] = (df["st_trend"] == 1) & (df["st_trend"].shift(1) == -1)
        df["st_flip_down"] = (df["st_trend"] == -1) & (df["st_trend"].shift(1) == 1)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # Enter long only when EMA9>EMA26 is already true AND Supertrend flips up now
        long_cond = (df["ema_up"] & df["st_flip_up"] & (df["volume"] > 0))
        df.loc[long_cond, "enter_long"] = 1
        df.loc[long_cond, "enter_tag"] = "ema_aligned_and_ST_flip_up"

        # Enter short only when EMA9<EMA26 is already true AND Supertrend flips down now
        short_cond = (df["ema_down"] & df["st_flip_down"] & (df["volume"] > 0))
        df.loc[short_cond, "enter_short"] = 1
        df.loc[short_cond, "enter_tag"] = "ema_aligned_and_ST_flip_down"

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        # Optional: exit on opposite ST flip
        # df.loc[df["st_flip_down"], "exit_long"] = 1
        # df.loc[df["st_flip_up"], "exit_short"] = 1
        return df

from datetime import datetime
from functools import reduce
from typing import Any

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter


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
        # Upper: if prev close > prev final_upper ⇒ reset to basic_upper; else keep min(basic, prev final)
        if df["close"].iat[i - 1] > final_upper.iat[i - 1]:
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = min(basic_upper.iat[i], final_upper.iat[i - 1])

        # Lower: if prev close < prev final_lower ⇒ reset to basic_lower; else keep max(basic, prev final)
        if df["close"].iat[i - 1] < final_lower.iat[i - 1]:
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = max(basic_lower.iat[i], final_lower.iat[i - 1])

    # Trend and active line (initialize to uptrend with lower band)
    st_trend = pd.Series(index=df.index, dtype=int)
    st_line = pd.Series(index=df.index, dtype=float)

    st_trend.iloc[0] = 1
    st_line.iloc[0] = final_lower.iloc[0]

    for i in range(1, len(df)):
        if st_trend.iloc[i - 1] == 1:
            # Uptrend: flip down if close ≤ final_lower; otherwise stay up on lower band
            if df["close"].iloc[i] <= final_lower.iloc[i]:
                st_trend.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]
            else:
                st_trend.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]
        else:
            # Downtrend: flip up if close ≥ final_upper; otherwise stay down on upper band
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


class DonchianATRTrend(IStrategy):
    # Core settings
    timeframe = "1h"
    can_short = True
    startup_candle_count = 320
    process_only_new_candles = True

    stoploss = -0.03
    minimal_roi = {"0": 0.03}
    max_open_trades = 3
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_only_offset_is_reached = True
    trailing_stop_positive_offset = 0.015

    # Hyperoptable signal parameters
    buy_dc_period = IntParameter(10, 60, default=20, space="buy", optimize=True)
    buy_confirm_mid = BooleanParameter(default=False, space="buy", optimize=True)

    st_period = IntParameter(7, 21, default=10, space="buy", optimize=True)
    st_mult = DecimalParameter(1.5, 4.0, decimals=2, default=3.0, space="buy", optimize=True)

    slip_off = DecimalParameter(0.0, 0.5, decimals=2, default=0.10, space="buy", optimize=True)
    # Risk & size
    risk_per_trade = DecimalParameter(0.01, 0.018, decimals=3, default=0.015, space="buy")
    min_stake_usd = DecimalParameter(20, 200, decimals=0, default=60, space="buy")
    stake_cap_pct = DecimalParameter(0.10, 0.30, decimals=2, default=0.30, space="buy", optimize=False)
    initial_atr_mult = DecimalParameter(1.4, 3.8, decimals=2, default=2.3, space="sell")

    custom_exit_flag = BooleanParameter(default=False, space="sell", optimize=True)

    buy_params = {
        "buy_confirm_mid": False,
        "buy_dc_period": 12,
        "min_stake_usd": 39.0,
        "risk_per_trade": 0.015,
        "slip_off": 0.01,
        "st_mult": 1.97,
        "st_period": 10
    }

    sell_params = {
        "custom_exit_flag": False,
        "initial_atr_mult": 3.71
    }

    @property
    def protections(self):
        return [
            # Short post-exit pause to avoid immediate re-entries into churn
            {"method": "CooldownPeriod", "stop_duration_candles": 3},

            # Clustered stoploss protector with per-side locking for futures
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,  # 24h on 1h timeframe
                "trade_limit": 4,  # block if >=4 SLs in window
                "stop_duration_candles": 6,  # pause 6h
                "required_profit": 0.0,  # consider any losing SL
                "only_per_pair": False,  # evaluate across all pairs
                "only_per_side": True  # lock only the losing side
            },

            # Short-term equity drawdown guard (daily)
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,  # last 2 days
                "trade_limit": 12,  # require sufficient activity
                "stop_duration_candles": 6,  # pause 6h
                "max_allowed_drawdown": 0.10  # 10% max DD threshold
            },

            # Medium-term equity drawdown guard (3-day horizon)
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 72,  # last 3 days
                "trade_limit": 20,  # stricter evidence threshold
                "stop_duration_candles": 12,  # pause 12h
                "max_allowed_drawdown": 0.15  # 15% max DD threshold
            },
        ]

    plot_config = {
        "main_plot": {
            "dc_upper": {"color": "orange"},
            "dc_lower": {"color": "orange"},
            "dc_mid": {"color": "gray"},
            "st_line": {"color": "green"},
        },
        "subplots": {
            "ATR": {"atr": {"color": "purple"}},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        n = int(self.buy_dc_period.value)
        df["dc_upper"] = df["high"].rolling(n, min_periods=n).max()
        df["dc_lower"] = df["low"].rolling(n, min_periods=n).min()
        df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2.0

        # Use previous candle’s channel for signals
        df["dc_upper_prev"] = df["dc_upper"].shift(1)
        df["dc_lower_prev"] = df["dc_lower"].shift(1)
        df["dc_mid_prev"] = df["dc_mid"].shift(1)

        stp = int(self.st_period.value)
        stm = float(self.st_mult.value)
        st = compute_supertrend(df, stp, stm)
        df["st_trend"] = st["st_trend"]
        df["st_line"] = st["st_line"]
        df["atr"] = st["atr"]

        # Basic hygiene
        df["volume_mean_slow"] = df["volume"].rolling(20).mean()
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        long_cond = []
        short_cond = []

        # Long: breakout above previous upper + supertrend up
        long_cond.append(df["close"] > df["dc_upper_prev"])
        long_cond.append(df["st_trend"] == 1)
        if bool(self.buy_confirm_mid.value):
            long_cond.append(df["close"] > df["dc_mid_prev"])
        long_cond.append(df["volume"] > 0)

        # Short: breakdown below previous lower + supertrend down
        short_cond.append(df["close"] < df["dc_lower_prev"])
        short_cond.append(df["st_trend"] == -1)
        if bool(self.buy_confirm_mid.value):
            short_cond.append(df["close"] < df["dc_mid_prev"])
        short_cond.append(df["volume"] > 0)

        if long_cond:
            df.loc[reduce(lambda a, b: a & b, long_cond), "enter_long"] = 1
            df.loc[df["enter_long"] == 1, "enter_tag"] = "DC_ST_breakout_long"

        if short_cond:
            df.loc[reduce(lambda a, b: a & b, short_cond), "enter_short"] = 1
            df.loc[df["enter_short"] == 1, "enter_tag"] = "DC_ST_breakout_short"

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        if self.custom_exit_flag.value:
            # Exit when Supertrend flips against the position, or revert into channel mid
            df.loc[(df["st_trend"] == -1) | (df["close"] < df["dc_mid"]), "exit_long"] = 1
            df.loc[(df["st_trend"] == 1) | (df["close"] > df["dc_mid"]), "exit_short"] = 1

        return df

    def custom_entry_price(self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs) -> float:
        # Offset entry by a fraction of ATR to avoid chasing marginal breaks
        # Positive offset for long (enter slightly above), negative for short
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        atr = float(df.iloc[-1]["atr"]) if not df.empty else 0.0
        off = float(self.slip_off.value) * atr
        return float(proposed_rate + (off if side == "long" else -off))

    def custom_stake_amount(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_stake: float,
            min_stake: float | None,
            max_stake: float | None,
            leverage: float,
            entry_tag: str | None,
            side: str,
            **kwargs: Any,
    ) -> float:
        # Base ATR/stop based position sizing
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        sl_dist = self._stop_dist_pct(df)

        equity = float(self.wallets.get_total(self.stake_currency))
        risk = float(self.risk_per_trade.value)
        lev = float(leverage or 1.0)

        stake = (equity * risk) / max(1e-6, (sl_dist * lev))

        # Respect exchange limits
        if min_stake:
            stake = max(stake, float(min_stake))
        if max_stake:
            stake = min(stake, float(max_stake))

        stake_cap = equity * float(self.stake_cap_pct.value)
        stake = min(stake, stake_cap)

        # Enforce minimal absolute stake
        if stake < float(self.min_stake_usd.value):
            return 0.0

        return float(stake)

    # -----------------------
    # Helpers
    # -----------------------

    def _atr_pct(self, df: DataFrame) -> float:
        return float(df["atr"].iat[-1]) / max(1e-9, float(df["close"].iat[-1]))

    def _stop_dist_pct(self, df: pd.DataFrame) -> float:
        atr_pct = self._atr_pct(df)
        sld = float(self.initial_atr_mult.value) * atr_pct
        return min(sld, 0.08)

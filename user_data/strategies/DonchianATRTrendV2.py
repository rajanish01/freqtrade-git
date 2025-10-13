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


class DonchianATRTrendV2(IStrategy):
    # Core settings
    timeframe = "1h"
    can_short = True
    startup_candle_count = 320
    process_only_new_candles = True

    stoploss = -0.15
    max_open_trades = 3

    # Use custom ROI/SL logic
    use_custom_roi = True
    use_custom_stoploss = True

    # Hyperoptable signal parameters
    buy_dc_period = IntParameter(10, 60, default=20, space="buy", optimize=True)
    buy_confirm_mid = BooleanParameter(default=False, space="buy", optimize=True)

    st_period = IntParameter(7, 21, default=10, space="buy", optimize=True)
    st_mult = DecimalParameter(1.5, 4.0, decimals=2, default=3.0, space="buy", optimize=True)

    slip_off = DecimalParameter(0.0, 0.5, decimals=2, default=0.10, space="buy", optimize=True)
    # Risk & size
    risk_per_trade = DecimalParameter(0.01, 0.018, decimals=3, default=0.015, space="buy")
    min_stake_usd = DecimalParameter(20, 200, decimals=0, default=60, space="buy")
    stake_cap_pct = DecimalParameter(0.10, 0.30, decimals=2, default=0.30, space="buy",
                                     optimize=False)
    initial_atr_mult = DecimalParameter(1.4, 3.8, decimals=2, default=2.3, space="sell")

    custom_exit_flag = BooleanParameter(default=False, space="sell", optimize=True)

    # Exit / ROI shaping
    roi_start = DecimalParameter(0.015, 0.035, decimals=3, default=0.022, space="sell", optimize=True)  # initial target
    roi_floor = DecimalParameter(0.003, 0.012, decimals=3, default=0.006, space="sell", optimize=True)  # asymptotic min
    roi_decay_minutes = IntParameter(20, 120, default=60, space="sell", optimize=True)

    sl_floor = DecimalParameter(0.025, 0.035, decimals=3, default=0.03, space="sell", optimize=True)

    # Trailing / activation and locks (retained but with more protective defaults)
    profit_lock_trigger = DecimalParameter(0.010, 0.030, decimals=3, default=0.018, space="sell", optimize=True)
    activate_after_atr_mult = DecimalParameter(2.0, 4.0, decimals=1, default=2.8, space="sell", optimize=True)
    arm_max_minutes = IntParameter(10, 120, default=45, space="sell", optimize=True)
    trail_chandelier_mult = DecimalParameter(3.0, 8.5, decimals=2, default=4.5, space="sell", optimize=True)

    # Global/hard limits (unchanged)
    hard_sl_pct = DecimalParameter(-0.12, -0.04, decimals=3, default=-0.08, space="sell", optimize=True)
    max_sl_cap = DecimalParameter(0.03, 0.050, decimals=3, default=0.040, space="sell", optimize=True)

    # === Leverage shaping ===
    lev_target_c = DecimalParameter(0.06, 0.20, decimals=3, default=0.10, space="buy", optimize=True)
    signal_boost_max = DecimalParameter(1.00, 1.50, decimals=2, default=1.20, space="buy", optimize=True)
    vol_weak_cut = DecimalParameter(0.60, 1.00, decimals=2, default=0.85, space="buy", optimize=True)

    user_max_leverage = IntParameter(1, 4, default=4, space="buy", optimize=True)  # hard cap

    # Liquidation safety (requires futures/margin)
    mmr_est = DecimalParameter(0.003, 0.020, decimals=3, default=0.006, space="sell", optimize=True)
    liq_buffer_mult = DecimalParameter(2.5, 7.0, decimals=1, default=4.0, space="sell", optimize=True)

    buy_params = {
        "buy_confirm_mid": False,
        "buy_dc_period": 12,
        "lev_target_c": 0.189,
        "min_stake_usd": 39.0,
        "risk_per_trade": 0.013,
        "signal_boost_max": 1.47,
        "slip_off": 0.01,
        "st_mult": 1.97,
        "st_period": 10,
        "user_max_leverage": 1,
        "vol_weak_cut": 0.72
    }

    sell_params = {
        "activate_after_atr_mult": 4.0,
        "arm_max_minutes": 60,
        "custom_exit_flag": False,
        "hard_sl_pct": -0.057,
        "initial_atr_mult": 3.71,
        "liq_buffer_mult": 4.8,
        "max_sl_cap": 0.048,
        "mmr_est": 0.014,
        "profit_lock_trigger": 0.024,
        "roi_decay_minutes": 60,
        "roi_floor": 0.01,
        "roi_start": 0.015,
        "sl_floor": 0.030,
        "trail_chandelier_mult": 7.4
    }

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
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

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Current ATR% (unlevered)
        price = float(current_rate)
        atr_now = float(df["atr"].iat[-1]) if not df.empty else 0.0
        atr_pct_now = atr_now / max(1e-9, price)

        # Entry and unlevered move since entry
        try:
            entry = float(trade.open_rate)
            if trade.is_long:
                px_move = (price / entry) - 1.0
                be_dist = max(0.0, (price / entry) - 1.0)
            else:
                px_move = (entry / price) - 1.0
                be_dist = max(0.0, (entry / price) - 1.0)
        except Exception:
            entry, px_move, be_dist = price, 0.0, 0.0

        # Activation: unlevered move ≥ k * ATR% OR after N minutes
        atr_trigger = float(self.activate_after_atr_mult.value) * max(1e-9, atr_pct_now)
        trade_duration_mins = self._trade_duration_in_mins(trade, current_time)
        activated = (px_move >= atr_trigger) or (0 < int(self.arm_max_minutes.value) <= trade_duration_mins)

        # Floors and dynamic stop distance
        lev = float(getattr(trade, "leverage", 1.0) or 1.0)
        dynamic_dist = float(self._dynamic_stop_pct(df, float(current_rate), lev))
        min_floor = float(self.sl_floor.value)

        # Break-even lock trigger (entry-based)
        lock_trig = float(self.profit_lock_trigger.value)
        lock_to_be = (be_dist >= lock_trig)

        if not activated:
            early = max(min_floor, dynamic_dist)
            if lock_to_be:
                early = max(min_floor, min(early, be_dist + 1e-6))
            return -float(early)

        # After activation: chandelier trail versus ATR
        ce_dist = float("inf")
        try:
            k = float(self.trail_chandelier_mult.value)
            since_entry = df.loc[df.index >= trade.date_entry_fill_utc]
            if k > 0 and atr_now > 0 and not since_entry.empty:
                if trade.is_long:
                    hh = float(since_entry["high"].max())
                    stop_px = hh - k * atr_now
                    ce_dist = max(1e-6, (price - stop_px) / max(1e-9, price))
                else:
                    ll = float(since_entry["low"].min())
                    stop_px = ll + k * atr_now
                    ce_dist = max(1e-6, (stop_px - price) / max(1e-9, price))
        except Exception:
            pass

        final_dist = max(min_floor, min(dynamic_dist, ce_dist))
        if lock_to_be:
            final_dist = max(min_floor, min(final_dist, be_dist + 1e-6))
        return -float(final_dist)

    def custom_roi(self, pair, trade, current_time, trade_duration, entry_tag, side, **kwargs) -> float | None:
        # Decay ROI from roi_start -> roi_floor over roi_decay_minutes, then hold floor
        t = self._trade_duration_in_mins(trade, current_time)
        t_decay = int(self.roi_decay_minutes.value)
        start = float(self.roi_start.value)
        floor = float(self.roi_floor.value)

        if t_decay <= 0:
            return floor

        progress = min(1.0, max(0.0, t / float(t_decay)))
        target = (1.0 - progress) * start + progress * floor
        # Return required minimal ROI (fractional), Freqtrade expects a positive value for profit target
        return float(max(floor, target))

    def leverage(
            self,
            pair: str,
            current_time,
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            entry_tag: str,
            side: str,
            **kwargs,
    ) -> int:
        """
        Returns integer leverage (1–N).
        Based on stop-distance, signal strength, and liquidity — similar to V11.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return 1

        # --- 1. Base from stop distance ---
        sl_dist = float(self._stop_dist_pct(df))  # ~initial_atr_mult * ATR%
        c = float(self.lev_target_c.value)
        lev = c / max(1e-6, sl_dist)  # smaller stops -> higher leverage

        # --- 2. Shape by signal and liquidity ---
        sig_boost_max = float(self.signal_boost_max.value)
        lev *= (1.0 + self._signal_strength(df, side) * (sig_boost_max - 1.0))
        lev *= self._liquidity_factor(df)

        # --- 3. Liquidation safety cap ---
        k = float(self.liq_buffer_mult.value)
        mmr = float(self.mmr_est.value)
        max_safe_by_liq = 1.0 / max(1e-6, (mmr + k * sl_dist))
        lev = min(lev, max_safe_by_liq)

        # --- 4. Respect user/exchange caps ---
        user_cap = float(self.user_max_leverage.value)
        exch_cap = float(max_leverage or user_cap)
        lev = min(lev, user_cap, exch_cap)

        # --- 5. Round to nearest integer (at least 1) ---
        lev = int(max(1, round(lev)))

        return lev

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

    def _signal_strength(self, df: DataFrame, side: str) -> float:
        """
        Approximate edge using ADX (if present) + distance vs Donchian edge.
        V2 already has dc_* columns and ATR/ST — we reuse those.
        """
        # ADX might not exist in V2; be defensive
        adx_now = float(df["adx"].iat[-1]) if "adx" in df.columns else 20.0
        adx_nrm = min(1.0, max(0.0, (adx_now - 10.0) / 30.0))  # ~[0..1]

        # Distance beyond prior Donchian edge
        if side == "long":
            hi = float(df["high"].iat[-1])
            edge = max(0.0, (hi - float(df["dc_upper_prev"].iat[-1])) / max(1e-9, hi))
        else:
            lo = float(df["low"].iat[-1])
            edge = max(0.0, (float(df["dc_lower_prev"].iat[-1]) - lo) / max(1e-9, lo))

        return float(min(1.0, 0.6 * adx_nrm + 0.4 * edge))

    def _liquidity_factor(self, df: DataFrame) -> float:
        """
        Derate leverage when current volume is weak vs its MA (already in V2 as volume_mean_slow).
        """
        v = float(df["volume"].iat[-1])
        vm = float(df.get("vol_ma", df.get("volume_mean_slow", df["volume"].rolling(20).mean())).iat[-1])
        vm = max(1e-9, vm)
        ratio = v / vm
        if ratio >= 1.0:
            return 1.0
        cut = float(self.vol_weak_cut.value)
        if ratio <= cut:
            return 0.60
        return 0.60 + 0.40 * (ratio - cut) / max(1e-9, (1.0 - cut))

    def _round_to_step(self, x: float, step: float) -> float:
        return max(step, round(x / step) * step)

    def _dynamic_stop_pct(self, df: DataFrame, current_rate: float, leverage: float) -> float:
        try:
            price = float(current_rate)
            a = float(df["atr"].iat[-1])
        except Exception:
            price, a = max(float(current_rate) or 1.0, 1.0), 0.0

        atr_ratio = (a / price) if price > 0 else 0.0
        atr_stop = float(self.initial_atr_mult.value) * atr_ratio

        mmr = float(self.mmr_est.value)
        k = float(self.liq_buffer_mult.value)
        lev = max(1.0, float(leverage or 1.0))
        liq_dist = max(1e-6, 1.0 / lev - mmr)  # distance to liq
        liq_safe_stop = max(1e-6, liq_dist / max(k, 1e-6))

        sl_cap = float(self.max_sl_cap.value)
        hard_cap = abs(float(self.hard_sl_pct.value))
        min_floor = float(self.sl_floor.value)

        final_stop = max(min_floor, min(max(1e-6, atr_stop), liq_safe_stop, sl_cap, hard_cap))
        if not (final_stop == final_stop) or final_stop <= 0:  # NaN/invalid
            final_stop = max(min_floor, min(hard_cap, sl_cap))
        return float(final_stop)

    def _trade_duration_in_mins(self, trade, current_time: datetime) -> int:
        try:
            return int((current_time - trade.date_entry_fill_utc).total_seconds() // 60)
        except Exception:
            return 0

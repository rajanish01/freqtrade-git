from datetime import datetime
from functools import reduce
from typing import Any

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter


def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()


def _atr_rma(df: DataFrame, period: int) -> pd.Series:
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _rma(tr, period)


def compute_supertrend(df: DataFrame, atr_period: int, multiplier: float) -> DataFrame:
    """
    TradingView-style Supertrend:
    - ATR via RMA (Wilder)
    - Bands: hl2 ± multiplier * ATR
    - Persistence rules; flip when close crosses
    Returns DataFrame with ['st_trend','st_line','atr']
    """
    out = pd.DataFrame(index=df.index)

    hl2 = (df["high"] + df["low"]) / 2.0
    atr = _atr_rma(df, atr_period)

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):
        if df["close"].iat[i - 1] > final_upper.iat[i - 1]:
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = min(basic_upper.iat[i], final_upper.iat[i - 1])

        if df["close"].iat[i - 1] < final_lower.iat[i - 1]:
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = max(basic_lower.iat[i], final_lower.iat[i - 1])

    st_trend = pd.Series(index=df.index, dtype=int)
    st_line = pd.Series(index=df.index, dtype=float)
    st_trend.iloc[0] = 1
    st_line.iloc[0] = final_lower.iloc[0]

    for i in range(1, len(df)):
        if st_trend.iloc[i - 1] == 1:
            if df["close"].iloc[i] <= final_lower.iloc[i]:
                st_trend.iloc[i] = -1
                st_line.iloc[i] = final_upper.iloc[i]
            else:
                st_trend.iloc[i] = 1
                st_line.iloc[i] = final_lower.iloc[i]
        else:
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


class DonchianATRTrendV21(IStrategy):
    INTERFACE_VERSION = 3
    _chan_state: dict = {}

    timeframe = "1h"
    can_short = True
    startup_candle_count = 320
    process_only_new_candles = True

    stoploss = -0.15
    max_open_trades = 3

    use_custom_roi = True
    use_custom_stoploss = True

    # ---- Signal params (unchanged entries/exits) ----
    buy_dc_period = IntParameter(10, 60, default=20, space="buy", optimize=True)
    buy_confirm_mid = BooleanParameter(default=False, space="buy", optimize=True)

    st_period = IntParameter(7, 21, default=10, space="buy", optimize=True)
    st_mult = DecimalParameter(1.5, 4.0, decimals=2, default=3.0, space="buy", optimize=True)

    slip_off = DecimalParameter(0.0, 0.5, decimals=2, default=0.10, space="buy", optimize=True)

    # Risk & size
    risk_per_trade = DecimalParameter(0.01, 0.018, decimals=3, default=0.015, space="buy")
    min_stake_usd = DecimalParameter(20, 200, decimals=0, default=60, space="buy")
    stake_cap_pct = DecimalParameter(0.10, 0.30, decimals=2, default=0.30, space="buy", optimize=True)

    initial_atr_mult = DecimalParameter(1.4, 3.8, decimals=2, default=2.3, space="sell")

    custom_exit_flag = BooleanParameter(default=False, space="sell", optimize=True)

    # Exit / ROI shaping
    roi_start = DecimalParameter(0.015, 0.03, decimals=3, default=0.022, space="sell", optimize=True)
    roi_floor = DecimalParameter(0.003, 0.012, decimals=3, default=0.006, space="sell", optimize=True)
    roi_decay_minutes = IntParameter(20, 120, default=60, space="sell", optimize=True)

    # Global/hard limits
    hard_sl_pct = DecimalParameter(-0.12, -0.04, decimals=3, default=-0.08, space="sell", optimize=True)
    max_sl_cap = DecimalParameter(0.02, 0.025, decimals=3, default=0.020, space="sell", optimize=True)
    sl_floor = DecimalParameter(0.015, 0.020, decimals=3, default=0.15, space="sell", optimize=True)

    # ROI/SL Channel params (hyperoptable)
    chan_init_pct = DecimalParameter(0.01, 0.04, decimals=3, default=0.02, space="sell", optimize=True)
    chan_step_atr_mult = DecimalParameter(0.2, 1.2, decimals=2, default=0.5, space="sell", optimize=True)
    chan_min_sl = DecimalParameter(0.003, 0.02, decimals=3, default=0.006, space="sell", optimize=True)
    chan_min_roi = DecimalParameter(0.003, 0.02, decimals=3, default=0.006, space="sell", optimize=True)
    chan_max_widen = DecimalParameter(1.5, 4.0, decimals=2, default=3.0, space="sell", optimize=True)
    chan_lock_be = DecimalParameter(0.005, 0.03, decimals=3, default=0.01, space="sell", optimize=True)
    chan_stale_minutes = IntParameter(20, 120, default=60, space="sell", optimize=True)

    # Trailing/locks (kept)
    profit_lock_trigger = DecimalParameter(0.010, 0.030, decimals=3, default=0.018, space="sell", optimize=True)
    activate_after_atr_mult = DecimalParameter(2.0, 4.0, decimals=1, default=2.8, space="sell", optimize=True)
    arm_max_minutes = IntParameter(10, 120, default=45, space="sell", optimize=True)
    trail_chandelier_mult = DecimalParameter(3.0, 8.5, decimals=2, default=4.5, space="sell", optimize=True)

    # Buy hyperspace params:
    buy_params = {
        "buy_confirm_mid": True,
        "buy_dc_period": 18,
        "min_stake_usd": 48.0,
        "risk_per_trade": 0.013,
        "slip_off": 0.03,
        "st_mult": 1.82,
        "st_period": 15,
        "stake_cap_pct": 0.3,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "activate_after_atr_mult": 3.1,
        "arm_max_minutes": 35,
        "chan_init_pct": 0.01,
        "chan_lock_be": 0.026,
        "chan_max_widen": 1.77,
        "chan_min_roi": 0.015,
        "chan_min_sl": 0.011,
        "chan_stale_minutes": 44,
        "chan_step_atr_mult": 1.16,
        "custom_exit_flag": False,
        "hard_sl_pct": -0.094,
        "initial_atr_mult": 3.29,
        "max_sl_cap": 0.025,
        "profit_lock_trigger": 0.028,
        "roi_decay_minutes": 96,
        "roi_floor": 0.004,
        "roi_start": 0.02,
        "sl_floor": 0.02,
        "trail_chandelier_mult": 3.79,
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
        "subplots": {"ATR": {"atr": {"color": "purple"}}},
    }

    # ---- Indicators & Signals (unchanged logic) ----
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        n = int(self.buy_dc_period.value)
        df["dc_upper"] = df["high"].rolling(n, min_periods=n).max()
        df["dc_lower"] = df["low"].rolling(n, min_periods=n).min()
        df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2.0
        df["dc_upper_prev"] = df["dc_upper"].shift(1)
        df["dc_lower_prev"] = df["dc_lower"].shift(1)
        df["dc_mid_prev"] = df["dc_mid"].shift(1)

        stp = int(self.st_period.value)
        stm = float(self.st_mult.value)
        st = compute_supertrend(df, stp, stm)
        df["st_trend"] = st["st_trend"]
        df["st_line"] = st["st_line"]
        df["atr"] = st["atr"]
        df["volume_mean_slow"] = df["volume"].rolling(20).mean()
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        long_cond = [df["close"] > df["dc_upper_prev"], df["st_trend"] == 1]
        if bool(self.buy_confirm_mid.value):
            long_cond.append(df["close"] > df["dc_mid_prev"])
        long_cond.append(df["volume"] > 0)

        short_cond = [df["close"] < df["dc_lower_prev"], df["st_trend"] == -1]
        if bool(self.buy_confirm_mid.value):
            short_cond.append(df["close"] < df["dc_mid_prev"])
        short_cond.append(df["volume"] > 0)

        df.loc[reduce(lambda a, b: a & b, long_cond), "enter_long"] = 1
        df.loc[df["enter_long"] == 1, "enter_tag"] = "DC_ST_breakout_long"
        df.loc[reduce(lambda a, b: a & b, short_cond), "enter_short"] = 1
        df.loc[df["enter_short"] == 1, "enter_tag"] = "DC_ST_breakout_short"
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        if self.custom_exit_flag.value:
            df.loc[(df["st_trend"] == -1) | (df["close"] < df["dc_mid"]), "exit_long"] = 1
            df.loc[(df["st_trend"] == 1) | (df["close"] > df["dc_mid"]), "exit_short"] = 1
        return df

    # ---- Order mgmt ----
    def custom_entry_price(self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        atr = float(df.iloc[-1]["atr"]) if not df.empty else 0.0
        off = float(self.slip_off.value) * atr
        return float(proposed_rate + (off if side == "long" else -off))

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        # Initialize channel state on fill
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            price = float(trade.open_rate)
            atr = float(df.iloc[-1]["atr"]) if not df.empty else 0.0
            _ = atr / max(1e-9, price)  # A (not used directly here)
        except Exception:
            pass
        W0 = float(self.chan_init_pct.value)
        init = max(W0, float(self.chan_min_sl.value))
        self._chan_state[trade.id] = {
            "sl_dist": float(init),
            "tp_dist": float(init),
            "last_ref_pnl": 0.0,
            "max_w": float(W0) * float(self.chan_max_widen.value),
        }

    # ---- Channel engine ----
    def _update_channel(self, trade: Trade, price: float, atr_pct: float, be_dist: float, t_minutes: int):
        """Returns (sl_dist, tp_dist) or None. Stores state in self._chan_state[trade.id]."""
        ud = self._chan_state.get(trade.id)
        if not ud:
            return None
        step = float(self.chan_step_atr_mult.value) * float(atr_pct)

        # Current unlevered PnL fraction
        try:
            if not trade.is_short:
                pnl = (price / float(trade.open_rate)) - 1.0
            else:
                pnl = (float(trade.open_rate) / price) - 1.0
        except Exception:
            pnl = 0.0

        moved_green = (pnl - ud.get("last_ref_pnl", 0.0)) >= (step + 1e-12)
        moved_red = (ud.get("last_ref_pnl", 0.0) - pnl) >= (step + 1e-12)

        sl_d = float(ud.get("sl_dist", 0.0))
        tp_d = float(ud.get("tp_dist", 0.0))
        max_w = float(ud.get("max_w", float(self.chan_init_pct.value) * float(self.chan_max_widen.value)))

        if moved_green and step > 0.0:
            sl_d = max(float(self.chan_min_sl.value), sl_d - step)
            tp_d = min(max_w, tp_d + step)
            ud["last_ref_pnl"] = pnl
        elif moved_red and step > 0.0:
            tp_d = max(float(self.chan_min_roi.value), tp_d - step)
            ud["last_ref_pnl"] = pnl

        if be_dist >= float(self.chan_lock_be.value):
            sl_d = min(sl_d, be_dist + 1e-6)

        if t_minutes >= int(self.chan_stale_minutes.value):
            tp_d = max(float(self.chan_min_roi.value), min(tp_d, float(self.chan_min_roi.value)))

        ud["sl_dist"], ud["tp_dist"] = float(sl_d), float(tp_d)
        self._chan_state[trade.id] = ud
        return float(sl_d), float(tp_d)

    # ---- Custom SL/ROI ----
    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        price = float(current_rate)
        atr_now = float(df["atr"].iat[-1]) if not df.empty else 0.0
        atr_pct_now = atr_now / max(1e-9, price)

        lev = float(getattr(trade, "leverage", 1.0) or 1.0)
        dyn_sl = float(self._dynamic_stop_pct(df, price, lev))
        min_floor = float(self.sl_floor.value)

        try:
            if trade.is_long:
                be_dist = max(0.0, (price / float(trade.open_rate)) - 1.0)
            else:
                be_dist = max(0.0, (float(trade.open_rate) / price) - 1.0)
        except Exception:
            be_dist = 0.0

        tmin = self._trade_duration_in_mins(trade, current_time)
        ch = self._update_channel(trade, price, atr_pct_now, be_dist, tmin)
        if ch is None:
            final = max(min_floor, dyn_sl)
            return -float(final)

        sl_d, _ = ch
        final = max(min_floor, min(dyn_sl, sl_d))
        return -float(final)

    def custom_roi(self, pair, trade: Trade, current_time, trade_duration, entry_tag, side, **kwargs) -> float | None:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None
        price = float(df["close"].iat[-1])
        atr_now = float(df["atr"].iat[-1])
        atr_pct_now = atr_now / max(1e-9, price)

        tmin = self._trade_duration_in_mins(trade, current_time)
        try:
            if not trade.is_short:
                be_dist = max(0.0, (price / float(trade.open_rate)) - 1.0)
            else:
                be_dist = max(0.0, (float(trade.open_rate) / price) - 1.0)
        except Exception:
            be_dist = 0.0

        ch = self._update_channel(trade, price, atr_pct_now, be_dist, tmin)
        if ch is None:
            # fallback to time-decay curve
            t = tmin
            t_decay = int(self.roi_decay_minutes.value)
            start = float(self.roi_start.value)
            floor = float(self.roi_floor.value)
            if t_decay <= 0:
                return floor
            progress = min(1.0, max(0.0, t / float(t_decay)))
            target = (1.0 - progress) * start + progress * floor
            return float(max(floor, target))

        _, tp_d = ch
        floor = float(self.roi_floor.value)
        return float(max(floor, tp_d))

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float | None,
                            leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        sl_dist = self._stop_dist_pct(df)
        equity = float(self.wallets.get_total(self.stake_currency))
        risk = float(self.risk_per_trade.value)
        lev = float(leverage or 1.0)
        stake = (equity * risk) / max(1e-6, (sl_dist * lev))
        if min_stake:
            stake = max(stake, float(min_stake))
        if max_stake:
            stake = min(stake, float(max_stake))
        stake_cap = equity * float(self.stake_cap_pct.value)
        stake = min(stake, stake_cap)
        if stake < float(self.min_stake_usd.value):
            return 0.0
        return float(stake)

    # ---- Helpers ----
    def _dynamic_stop_pct(self, df: DataFrame, price: float, leverage: float) -> float:
        """Volatility- and leverage-aware dynamic stop (pct of price)."""
        try:
            atr_pct = float(df["atr"].iat[-1]) / max(1e-9, float(price))
        except Exception:
            atr_pct = 0.0
        base = float(self.initial_atr_mult.value) * atr_pct
        mmr = 0.005  # tune per exchange
        lev = max(1.0, float(leverage or 1.0))
        liq_dist = max(1e-6, 1.0 / lev - mmr)
        liq_safe = liq_dist / 2.0
        sl = max(
            float(self.sl_floor.value),
            min(max(1e-6, base), float(self.max_sl_cap.value), abs(float(self.hard_sl_pct.value)), liq_safe),
        )
        if not (sl == sl) or sl <= 0:
            sl = max(float(self.sl_floor.value), min(abs(float(self.hard_sl_pct.value)), float(self.max_sl_cap.value)))
        return float(sl)

    def _atr_pct(self, df: DataFrame) -> float:
        return float(df["atr"].iat[-1]) / max(1e-9, float(df["close"].iat[-1]))

    def _stop_dist_pct(self, df: pd.DataFrame) -> float:
        atr_stop = (float(self.initial_atr_mult.value) * float(df["atr"].iat[-1])) / max(1e-9,
                                                                                         float(df["close"].iat[-1]))
        mmr = 0.005
        lev = 1.0
        liq_dist = max(1e-6, 1.0 / lev - mmr)
        liq_safe_stop = max(1e-6, liq_dist / 2.0)
        sl_cap = float(self.max_sl_cap.value)
        hard_cap = abs(float(self.hard_sl_pct.value))
        min_floor = float(self.sl_floor.value)
        final_stop = max(min_floor, min(max(1e-6, atr_stop), liq_safe_stop, sl_cap, hard_cap))
        if not (final_stop == final_stop) or final_stop <= 0:
            final_stop = max(min_floor, min(hard_cap, sl_cap))
        return float(final_stop)

    def _trade_duration_in_mins(self, trade, current_time: datetime) -> int:
        try:
            return int((current_time - trade.date_entry_fill_utc).total_seconds() // 60)
        except Exception:
            return 0


########## NOTES ##############
"""
- Major issue found, SL sits at ~3.5% and roi 1.5%, causing losses in log run. Trying to fix this issue in V22.
"""

# tv_sr_1h_strategy.py
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, merge_informative_pair, \
    BooleanParameter


class PivotFlipped(IStrategy):
    INTERFACE_VERSION = 3

    # Core settings
    timeframe = "1h"
    informative_timeframe = "1d"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 400  # warmup for pivots/ATR/EMA

    # Order config
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC",
    }

    # ROI/Stoploss framework
    minimal_roi = {}  # use custom ROI below
    use_custom_roi = True
    stoploss = -0.1  # safety net; dynamic stop via custom_stoploss
    use_custom_stoploss = True

    # Exit signaling
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    max_open_trades = 3

    # Hyperoptable parameters
    # Pivot detection windows
    pivot_left = IntParameter(2, 7, default=3, space="buy")
    pivot_right = IntParameter(2, 7, default=3, space="buy")

    # Trend filter
    ema_period = IntParameter(50, 300, default=200, space="buy")

    # Daily PP filter
    use_pp_filter = CategoricalParameter([True, False], default=True, space="buy")

    custom_exit_flag = BooleanParameter(default=False, space="sell", optimize=True)

    # Entry modes
    use_bounce = CategoricalParameter([True, False], default=True, space="buy")
    use_breakout = CategoricalParameter([True, False], default=True, space="buy")

    # Touch and breakout buffers (relative)
    sr_touch_tolerance = DecimalParameter(0.000, 0.010, decimals=3, default=0.002, space="buy")
    brk_buffer_up = DecimalParameter(0.000, 0.015, decimals=3, default=0.003, space="buy")
    brk_buffer_dn = DecimalParameter(0.000, 0.015, decimals=3, default=0.003, space="buy")

    # ATR-based dynamic stop
    atr_period = IntParameter(7, 30, default=14, space="sell")
    atr_mult_long = DecimalParameter(1.0, 4.0, decimals=1, default=2.0, space="sell")
    atr_mult_short = DecimalParameter(1.0, 4.0, decimals=1, default=2.0, space="sell")
    sr_stop_buffer = DecimalParameter(0.000, 0.010, decimals=3, default=0.002, space="sell")

    # ROI staging (minutes and thresholds)
    roi_t1 = IntParameter(10, 360, default=90, space="sell")
    roi_t2 = IntParameter(60, 1440, default=360, space="sell")
    roi_t3 = IntParameter(240, 2880, default=1440, space="sell")
    roi_p1 = DecimalParameter(0.002, 0.050, decimals=3, default=0.012, space="sell")
    roi_p2 = DecimalParameter(0.001, 0.030, decimals=3, default=0.008, space="sell")
    roi_p3 = DecimalParameter(0.000, 0.020, decimals=3, default=0.004, space="sell")

    # Optional leverage (futures)
    leverage_opt = IntParameter(1, 5, default=1, space="buy")

    # Buy hyperspace params:
    buy_params = {
        "brk_buffer_dn": 0.008,
        "brk_buffer_up": 0.009,
        "ema_period": 175,
        "leverage_opt": 3,
        "pivot_left": 3,
        "pivot_right": 4,
        "sr_touch_tolerance": 0.001,
        "use_bounce": True,
        "use_breakout": True,
        "use_pp_filter": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "atr_mult_long": 4.0,
        "atr_mult_short": 3.9,
        "atr_period": 27,
        "custom_exit_flag": False,
        "roi_p1": 0.01,
        "roi_p2": 0.001,
        "roi_p3": 0.011,
        "roi_t1": 33,
        "roi_t2": 1284,
        "roi_t3": 1138,
        "sr_stop_buffer": 0.004,
    }

    plot_config = {
        "main_plot": {
            "ema_trend": {"color": "orange"},
            "last_sup": {"color": "green"},
            "last_res": {"color": "red"},
            "pp": {"color": "blue"},
            "r1": {"color": "purple"},
            "s1": {"color": "purple"},
        },
        "subplots": {
            "ATR": {"atr": {"color": "gray"}},
        },
    }

    def version(self) -> str:
        return "1.0.0-1h"

    def informative_pairs(self):
        # daily informative for each whitelist pair
        pairs = [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]
        return list(set(pairs))

    # ---------- Helpers ----------
    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _atr(df: DataFrame, period: int) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    def _compute_pivots(self, df: DataFrame, lb_left: int, lb_right: int) -> DataFrame:
        # Confirmed pivot highs/lows using rolling max/min shifted by right window for confirmation
        win = lb_left + lb_right + 1
        high_rol = df["high"].rolling(window=win, min_periods=win).max().shift(lb_right)
        low_rol = df["low"].rolling(window=win, min_periods=win).min().shift(lb_right)
        df["pivot_high"] = (df["high"] >= high_rol) & high_rol.notna()
        df["pivot_low"] = (df["low"] <= low_rol) & low_rol.notna()
        df["last_sup"] = df["low"].where(df["pivot_low"]).ffill()
        df["last_res"] = df["high"].where(df["pivot_high"]).ffill()
        return df

    def _compute_daily_pivots(self, inf: DataFrame) -> DataFrame:
        d = inf.copy()
        pp = (d["high"] + d["low"] + d["close"]) / 3.0
        r1 = 2 * pp - d["low"]
        s1 = 2 * pp - d["high"]
        r2 = pp + (d["high"] - d["low"])
        s2 = pp - (d["high"] - d["low"])
        d["pp"] = pp.shift(1)
        d["r1"] = r1.shift(1)
        d["s1"] = s1.shift(1)
        d["r2"] = r2.shift(1)
        d["s2"] = s2.shift(1)
        return d[["date", "pp", "r1", "s1", "r2", "s2"]]

    # ---------- Indicators ----------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # Trend / volatility
        ema_p = int(self.ema_period.value)
        df["ema_trend"] = self._ema(df["close"], ema_p)

        atr_p = int(self.atr_period.value)
        df["atr"] = self._atr(df, atr_p)

        # Confirmed S/R pivots on strategy timeframe (1h)
        df = self._compute_pivots(df, int(self.pivot_left.value), int(self.pivot_right.value))

        # Daily floor pivots via informative 1d merged with suffixes (_1d)
        if self.dp:
            inf_tf = self.informative_timeframe  # "1d"
            informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=inf_tf)

            # Compute prior-day pivots and shift(1) so today's candles compare to yesterday's levels
            d = informative.copy()
            pp = (d["high"] + d["low"] + d["close"]) / 3.0
            r1 = 2 * pp - d["low"]
            s1 = 2 * pp - d["high"]
            r2 = pp + (d["high"] - d["low"])
            s2 = pp - (d["high"] - d["low"])
            d["pp"] = pp.shift(1)
            d["r1"] = r1.shift(1)
            d["s1"] = s1.shift(1)
            d["r2"] = r2.shift(1)
            d["s2"] = s2.shift(1)
            d = d[["date", "pp", "r1", "s1", "r2", "s2"]]

            # Safe merge with suffixes; enable ffill to have 1d values available intraday
            df = merge_informative_pair(df, d, self.timeframe, inf_tf, ffill=True)

        # Ensure suffixed informative cols exist even if informative missing
        for c in [f"pp_{self.informative_timeframe}",
                  f"r1_{self.informative_timeframe}",
                  f"s1_{self.informative_timeframe}",
                  f"r2_{self.informative_timeframe}",
                  f"s2_{self.informative_timeframe}"]:
            if c not in df.columns:
                df[c] = np.nan

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # init signals
        df["enter_long"] = 0
        df["enter_short"] = 0
        if "enter_tag" not in df.columns:
            df["enter_tag"] = ""

        has_sr = all(c in df.columns for c in ["last_sup", "last_res"])
        vol_ok = df["volume"] > 0

        # Column names for informative pivots with suffix (_1d)
        pp_col = f"pp_{self.informative_timeframe}"

        if has_sr:
            tol = float(self.sr_touch_tolerance.value)
            buf_up = float(self.brk_buffer_up.value)
            buf_dn = float(self.brk_buffer_dn.value)

            # Trend filters with NaN-guarded PP
            trend_long = df["close"] > df["last_sup"]
            trend_short = df["close"] < df["last_res"]

            if bool(self.use_pp_filter.value):
                pp_ok = df[pp_col].notna()
                trend_long = trend_long & (~pp_ok | (df["close"] > df[pp_col]))
                trend_short = trend_short & (~pp_ok | (df["close"] < df[pp_col]))

            # Levels must exist
            lvl_ok = df["last_sup"].notna() & df["last_res"].notna()

            # Bounce logic
            long_bounce = lvl_ok & (df["low"] <= df["last_sup"] * (1 + tol)) & (df["close"] > df["last_sup"])
            short_bounce = lvl_ok & (df["high"] >= df["last_res"] * (1 - tol)) & (df["close"] < df["last_res"])

            # Breakout logic with buffers
            long_breakout = lvl_ok & (df["close"] > df["last_res"] * (1 + buf_up)) & \
                            (df["close"].shift(1) <= df["last_res"] * (1 + buf_up))
            short_breakdown = lvl_ok & (df["close"] < df["last_sup"] * (1 - buf_dn)) & \
                              (df["close"].shift(1) >= df["last_sup"] * (1 - buf_dn))

            long_mode = (bool(self.use_bounce.value) & long_bounce) | (bool(self.use_breakout.value) & long_breakout)
            short_mode = (bool(self.use_bounce.value) & short_bounce) | (
                    bool(self.use_breakout.value) & short_breakdown)

            long_cond = vol_ok & trend_long & long_mode
            short_cond = vol_ok & trend_short & short_mode

            df.loc[long_cond, "enter_short"] = 1
            df.loc[long_cond & long_bounce, "enter_tag"] = "S/R bounce short"
            df.loc[long_cond & long_breakout, "enter_tag"] = "S/R breakout short"

            df.loc[short_cond, "enter_long"] = 1
            df.loc[short_cond & short_bounce, "enter_tag"] = "S/R bounce long"
            df.loc[short_cond & short_breakdown, "enter_tag"] = "S/R breakdown long"

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        if self.custom_exit_flag.value:
            df["exit_long"] = 0
            df["exit_short"] = 0
            if "exit_tag" not in df.columns:
                df["exit_tag"] = ""

            has_sr = all(c in df.columns for c in ["last_sup", "last_res"])
            vol_ok = df["volume"] > 0

            # Column names for informative pivots with suffix (_1d)
            r1_col = f"r1_{self.informative_timeframe}"
            s1_col = f"s1_{self.informative_timeframe}"

            if has_sr:
                # Long exits: hit resistance, lose support, or reach R1 if available
                hit_res = df["last_res"].notna() & (df["high"] >= df["last_res"])
                lose_sup = df["last_sup"].notna() & (df["close"] < df["last_sup"]) & (
                        df["close"].shift(1) >= df["last_sup"])
                exit_to_r1 = df[r1_col].notna() & (df["close"] >= df[r1_col])

                long_exit = vol_ok & (hit_res | lose_sup | exit_to_r1)
                df.loc[vol_ok & hit_res, "exit_tag"] = "hit_resistance"
                df.loc[vol_ok & lose_sup, "exit_tag"] = "lost_support"
                df.loc[vol_ok & exit_to_r1, "exit_tag"] = "to_R1"
                df.loc[long_exit, "exit_long"] = 1

                # Short exits: hit support, reclaim resistance, or reach S1 if available
                hit_sup = df["last_sup"].notna() & (df["low"] <= df["last_sup"])
                reclaim_res = df["last_res"].notna() & (df["close"] > df["last_res"]) & (
                        df["close"].shift(1) <= df["last_res"])
                exit_to_s1 = df[s1_col].notna() & (df["close"] <= df[s1_col])

                short_exit = vol_ok & (hit_sup | reclaim_res | exit_to_s1)
                df.loc[vol_ok & hit_sup, "exit_tag"] = "hit_support"
                df.loc[vol_ok & reclaim_res, "exit_tag"] = "reclaimed_resistance"
                df.loc[vol_ok & exit_to_s1, "exit_tag"] = "to_S1"
                df.loc[short_exit, "exit_short"] = 1

        return df

    # ---------- Custom Stoploss ----------
    def custom_stoploss(
            self,
            pair: str,
            trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            after_fill: bool,
            **kwargs,
    ) -> Optional[float]:
        """
        Dynamic stop: min(ATR multiple, pivot-buffer stop).
        Returns a negative ratio (e.g. -0.03) or None to keep current SL.
        """
        try:
            # Access latest 1h row
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if df is None or df.empty:
                return None
            row = df.iloc[-1]

            atr = float(row.get("atr", np.nan))
            last_sup = float(row.get("last_sup", np.nan))
            last_res = float(row.get("last_res", np.nan))
            close = float(row.get("close", current_rate))

            if np.isnan(atr) or np.isnan(last_sup) or np.isnan(last_res):
                return None

            # ATR stops
            atr_mult_l = float(self.atr_mult_long.value)
            atr_mult_s = float(self.atr_mult_short.value)
            sr_buf = float(self.sr_stop_buffer.value)

            if not trade.is_short:
                # Long: stop below support or ATR multiple
                sl_price_pivot = last_sup * (1 - sr_buf)
                sl_price_atr = close - atr_mult_l * atr
                sl_price = max(sl_price_pivot, sl_price_atr)  # tighter of the two on the upside
                sl_ratio = (sl_price / current_rate) - 1.0
                return max(sl_ratio, self.stoploss)  # cap by hard SL
            else:
                # Short: stop above resistance or ATR multiple
                sl_price_pivot = last_res * (1 + sr_buf)
                sl_price_atr = close + atr_mult_s * atr
                sl_price = min(sl_price_pivot, sl_price_atr)
                sl_ratio = 1.0 - (sl_price / current_rate)
                return max(-sl_ratio, self.stoploss)
        except Exception:
            return None

    # ---------- Custom ROI ----------
    def custom_roi(
            self,
            pair: str,
            trade,
            current_time: datetime,
            trade_duration: int,
            entry_tag: str,
            side,
            **kwargs,
    ) -> Optional[float]:
        """
        Time-based ROI ladder:
          0..t1 -> p1
          t1..t2 -> p2
          t2..t3 -> p3
          >t3 -> 0 (let trailing/stop manage)
        """
        t1 = int(self.roi_t1.value)
        t2 = int(self.roi_t2.value)
        t3 = int(self.roi_t3.value)
        p1 = float(self.roi_p1.value)
        p2 = float(self.roi_p2.value)
        p3 = float(self.roi_p3.value)

        if trade_duration <= t1:
            return p1
        elif trade_duration <= t2:
            return p2
        elif trade_duration <= t3:
            return p3
        else:
            return 0.0

    # ---------- Leverage (futures) ----------
    def leverage(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            entry_tag: str | None,
            side: str,
            **kwargs,
    ) -> float:
        return float(int(self.leverage_opt.value))

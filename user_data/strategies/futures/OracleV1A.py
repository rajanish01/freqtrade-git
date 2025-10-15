# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter


class OracleV1A(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True
    startup_candle_count = 360

    use_custom_stoploss = True
    use_custom_roi = True
    use_exit_signal = True
    position_adjustment_enable = True

    # A wide emergency exchange stop – actual risk is controlled by custom_stoploss
    stoploss = -0.30

    max_open_trades = 3

    # -----------------------------
    # Hyperopt / Params
    # -----------------------------

    # Entry – RSI quantile percentage
    buy_rsi_pct = IntParameter(22, 36, default=28, space="buy")
    bb_shift = DecimalParameter(1.03, 1.10, decimals=3, default=1.06, space="buy")
    vol_k = DecimalParameter(1.0, 1.5, decimals=2, default=1.15, space="buy")
    min_atr_pct = DecimalParameter(0.006, 0.020, decimals=3, default=0.010, space="buy")
    band_overshoot_min = DecimalParameter(0.008, 0.020, decimals=3, default=0.012, space="buy")

    # Adaptive RSI margin = rsi_margin_base + rsi_margin_k * atr_pct (atr_pct ~ 0.005..0.03)
    # NOTE: No "*100" here – that was the over-scaling bug in V52
    rsi_margin_base = IntParameter(2, 10, default=6, space="buy")
    rsi_margin_k = DecimalParameter(80.0, 300.0, decimals=1, default=160.0, space="buy")

    # Higher timeframe filter
    use_d1_filter = BooleanParameter(default=True, space="buy")
    use_d1_slope = BooleanParameter(default=True, space="buy")
    slope_window = IntParameter(5, 12, default=8, space="buy")

    # Dynamic dip lookback based on ATR% regime
    dip_lookback = IntParameter(2, 8, default=4, space="buy")  # base
    dyn_dip_atr_q_low = DecimalParameter(0.2, 0.4, decimals=2, default=0.25, space="buy")
    dyn_dip_atr_q_high = DecimalParameter(0.6, 0.8, decimals=2, default=0.75, space="buy")
    dip_lookback_lo = IntParameter(4, 10, default=6, space="buy")
    dip_lookback_hi = IntParameter(1, 4, default=2, space="buy")

    # Side-specific cooldowns
    entry_cooldown_bars_long = IntParameter(4, 12, default=6, space="buy")
    entry_cooldown_bars_short = IntParameter(4, 12, default=6, space="buy")

    # Reclaim entries & pullback targeting
    enable_reclaim = BooleanParameter(default=False, space="buy")
    pullback_pct = DecimalParameter(0.000, 0.010, decimals=3, default=0.003, space="buy")
    pullback_use_low = IntParameter(0, 1, default=1, space="buy")
    entry_pullback_max = DecimalParameter(0.003, 0.012, decimals=3, default=0.005, space="buy")
    entry_atr_cap = DecimalParameter(0.50, 1.20, decimals=2, default=0.80, space="buy")

    # Min notional
    min_stake_usd = DecimalParameter(20, 200, decimals=0, default=60, space="buy")

    # Leverage/positioning
    risk_per_trade = DecimalParameter(0.005, 0.02, decimals=3, default=0.012, space="buy")
    lev_target_c = DecimalParameter(0.06, 0.20, decimals=3, default=0.10, space="buy")
    user_max_leverage = IntParameter(2, 4, default=2, space="buy")
    signal_boost_max = DecimalParameter(1.00, 1.50, decimals=2, default=1.20, space="buy")
    vol_weak_cut = DecimalParameter(0.60, 1.00, decimals=2, default=0.85, space="buy")
    lev_round_step = DecimalParameter(0.05, 0.25, decimals=2, default=0.10, space="buy")

    # Exit / Trailing / ROI
    hard_sl_pct = DecimalParameter(-0.12, -0.04, decimals=3, default=-0.08, space="sell")
    initial_atr_mult = DecimalParameter(1.4, 3.8, decimals=2, default=2.3, space="sell")
    trail_chandelier_mult = DecimalParameter(3.0, 8.5, decimals=2, default=4.5, space="sell")

    min_roi_floor = DecimalParameter(0.010, 0.060, decimals=3, default=0.025, space="sell")
    roi_atr_mult = DecimalParameter(0.90, 2.20, decimals=2, default=1.30, space="sell")
    trail_roi_mult = DecimalParameter(2.5, 8.0, decimals=2, default=4.0, space="sell")
    max_roi_cap = DecimalParameter(0.06, 0.20, decimals=2, default=0.10, space="sell")

    min_sl_floor = DecimalParameter(0.010, 0.050, decimals=3, default=0.030, space="sell")
    max_sl_cap = DecimalParameter(0.035, 0.090, decimals=3, default=0.040, space="sell")
    mmr_est = DecimalParameter(0.003, 0.020, decimals=3, default=0.006, space="sell")
    liq_buffer_mult = DecimalParameter(2.5, 7.0, decimals=1, default=4.0, space="sell")
    activate_after_atr_mult = DecimalParameter(2.2, 5.5, decimals=1, default=3.5, space="sell")
    arm_max_minutes = IntParameter(15, 180, default=60, space="sell")

    # Exit signals (optional helpers)
    exit_ema_cross = BooleanParameter(default=True, space="sell")
    exit_atr_exhaust = BooleanParameter(default=True, space="sell")
    atr_exhaust_mult = DecimalParameter(0.40, 0.90, decimals=2, default=0.55, space="sell")

    # Profit lock – arm breakeven lock once profit >= trigger
    profit_lock_trigger = DecimalParameter(0.010, 0.050, decimals=3, default=0.020, space="sell")

    # Time-decay for ROI
    time_decay_minutes = IntParameter(120, 720, default=300, space="sell")
    time_decay_add = DecimalParameter(0.005, 0.030, decimals=3, default=0.012, space="sell")

    # Price chasing controls (new)
    chase_enable = BooleanParameter(default=True, space="buy", optimize=False)
    chase_max_bps = IntParameter(10, 60, default=35, space="buy")  # absolute cap, e.g., 35 bps = 0.35%
    chase_rate_bps_per_min = IntParameter(2, 12, default=6, space="buy")  # linear chase per minute since signal

    # Buy hyperspace params:
    buy_params = {
        "band_overshoot_min": 0.013,
        "bb_shift": 1.089,
        "buy_rsi_pct": 33,
        "chase_enable": False,
        "chase_max_bps": 41,
        "chase_rate_bps_per_min": 12,
        "dip_lookback": 6,
        "dip_lookback_hi": 1,
        "dip_lookback_lo": 9,
        "dyn_dip_atr_q_high": 0.63,
        "dyn_dip_atr_q_low": 0.22,
        "enable_reclaim": True,
        "entry_atr_cap": 1.12,
        "entry_cooldown_bars_long": 9,
        "entry_cooldown_bars_short": 11,
        "entry_pullback_max": 0.009,
        "lev_round_step": 0.05,
        "lev_target_c": 0.197,
        "min_atr_pct": 0.014,
        "min_stake_usd": 163.0,
        "pullback_pct": 0.007,
        "pullback_use_low": 1,
        "risk_per_trade": 0.02,
        "rsi_margin_base": 8,
        "rsi_margin_k": 256.4,
        "signal_boost_max": 1.26,
        "slope_window": 8,
        "use_d1_filter": True,
        "use_d1_slope": True,
        "user_max_leverage": 4,
        "vol_k": 1.25,
        "vol_weak_cut": 0.88,
    }

    # Sell hyperspace params:
    sell_params = {
        "activate_after_atr_mult": 2.7,
        "arm_max_minutes": 62,
        "atr_exhaust_mult": 0.45,
        "exit_atr_exhaust": True,
        "exit_ema_cross": False,
        "hard_sl_pct": -0.072,
        "initial_atr_mult": 1.45,
        "liq_buffer_mult": 3.5,
        "max_roi_cap": 0.13,
        "max_sl_cap": 0.078,
        "min_roi_floor": 0.017,
        "min_sl_floor": 0.012,
        "mmr_est": 0.007,
        "profit_lock_trigger": 0.022,
        "roi_atr_mult": 1.35,
        "time_decay_add": 0.029,
        "time_decay_minutes": 302,
        "trail_chandelier_mult": 7.33,
        "trail_roi_mult": 7.57,
    }

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 24,
                "trade_limit": 8,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.025
            },
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
        ]

    # -----------------------------
    # Indicators
    # -----------------------------
    def populate_indicators(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["rsi"] = ta.RSI(df, timeperiod=14)

        bb_u, bb_m, bb_l = ta.BBANDS(df["close"], timeperiod=20)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb_u, bb_m, bb_l

        df["atr"] = ta.ATR(df, timeperiod=14)
        df["vol_mean"] = df["volume"].rolling(30).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # RSI quantiles
        pct = self.buy_rsi_pct.value / 100.0
        df["rsi_floor"] = df["rsi"].rolling(100, min_periods=25).quantile(pct)
        df["rsi_ceiling"] = df["rsi"].rolling(100, min_periods=25).quantile(1 - pct)

        # Daily EMA200 (informative)
        try:
            d1, _ = self.dp.get_analyzed_dataframe(metadata["pair"], "1d")
            if d1 is not None and not d1.empty and "close" in d1.columns:
                d1_ema200 = ta.EMA(d1["close"], timeperiod=200)
                df["d1_ema200"] = d1_ema200.reindex(df.index, method="ffill")
            else:
                df["d1_ema200"] = df["ema200"]
        except Exception:
            df["d1_ema200"] = df["ema200"]

        # ATR% quantiles for dynamic lookback selection
        df["atr_pct_q_low"] = df["atr_pct"].rolling(200, min_periods=50).quantile(float(self.dyn_dip_atr_q_low.value))
        df["atr_pct_q_high"] = df["atr_pct"].rolling(200, min_periods=50).quantile(float(self.dyn_dip_atr_q_high.value))

        return df

    # -----------------------------
    # Entry
    # -----------------------------
    def populate_entry_trend(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        lb_adj = df["bb_lower"] * float(self.bb_shift.value)
        ub_adj = df["bb_upper"] / float(self.bb_shift.value)

        vol_ok = df["volume"] > (df["vol_mean"] * float(self.vol_k.value))

        # Overshoot requirements
        overshoot_long = (lb_adj - df["close"]) / df["close"] >= float(self.band_overshoot_min.value)
        overshoot_short = (df["close"] - ub_adj) / df["close"] >= float(self.band_overshoot_min.value)

        # Adaptive RSI margin (vol-aware) – fixed scaling
        rsi_margin = float(self.rsi_margin_base.value) + float(self.rsi_margin_k.value) * df["atr_pct"].fillna(0.0)
        rsi_long_ok = df["rsi"] <= (df["rsi_floor"] - rsi_margin)
        rsi_short_ok = df["rsi"] >= (df["rsi_ceiling"] + rsi_margin)

        # Volatility / non-micro floors
        atr_ok = df["atr_pct"] >= float(self.min_atr_pct.value)
        exp_roi_ok = (df["atr_pct"] * float(self.roi_atr_mult.value)).fillna(0.0) >= float(self.min_roi_floor.value)
        exp_sl_ok = (df["atr_pct"] * float(self.initial_atr_mult.value)).fillna(0.0) >= float(self.min_sl_floor.value)
        base_filters = vol_ok & atr_ok & exp_roi_ok & exp_sl_ok

        # Higher timeframe regime (optional)
        if bool(self.use_d1_filter.value):
            long_regime = (df["ema50"] > df["ema200"]) & (df["close"] > df["d1_ema200"])
            short_regime = (df["ema50"] < df["ema200"]) & (df["close"] < df["d1_ema200"])
            if bool(self.use_d1_slope.value):
                w = int(self.slope_window.value)
                long_regime &= df["d1_ema200"] > df["d1_ema200"].shift(w)
                short_regime &= df["d1_ema200"] < df["d1_ema200"].shift(w)
        else:
            long_regime = (df["ema50"] > df["ema200"])
            short_regime = (df["ema50"] < df["ema200"])

        # Dynamic dip lookback regime flags
        base_lb = max(1, int(self.dip_lookback.value))
        lb_lo = max(1, int(self.dip_lookback_lo.value))
        lb_hi = max(1, int(self.dip_lookback_hi.value))

        low_thr = df["atr_pct_q_low"]
        high_thr = df["atr_pct_q_high"]
        is_low = (df["atr_pct"] <= low_thr).fillna(False)
        is_high = (df["atr_pct"] >= high_thr).fillna(False)

        # Raw signals
        dip_long = long_regime & rsi_long_ok & (df["close"] < lb_adj) & overshoot_long & base_filters
        rally_short = short_regime & rsi_short_ok & (df["close"] > ub_adj) & overshoot_short & base_filters

        # Reclaim signals (optional)
        reclaim_long = False
        reclaim_short = False
        if bool(self.enable_reclaim.value):
            lb_touch = (df["low"] < lb_adj)
            ub_touch = (df["high"] > ub_adj)

            lb_roll_lo = lb_touch.rolling(lb_lo, min_periods=1).max().astype(bool)
            lb_roll_mid = lb_touch.rolling(base_lb, min_periods=1).max().astype(bool)
            lb_roll_hi = lb_touch.rolling(lb_hi, min_periods=1).max().astype(bool)

            ub_roll_lo = ub_touch.rolling(lb_lo, min_periods=1).max().astype(bool)
            ub_roll_mid = ub_touch.rolling(base_lb, min_periods=1).max().astype(bool)
            ub_roll_hi = ub_touch.rolling(lb_hi, min_periods=1).max().astype(bool)

            lb_touch_roll = pd.Series(
                np.where(is_low, lb_roll_lo, np.where(is_high, lb_roll_hi, lb_roll_mid)),
                index=df.index
            ).astype(bool)
            ub_touch_roll = pd.Series(
                np.where(is_low, ub_roll_lo, np.where(is_high, ub_roll_hi, ub_roll_mid)),
                index=df.index
            ).astype(bool)

            reclaim_long = long_regime & lb_touch_roll & (
                    qtpylib.crossed_above(df["close"], df["ema50"]) | qtpylib.crossed_above(df["rsi"], df["rsi_floor"])
            ) & base_filters

            reclaim_short = short_regime & ub_touch_roll & (
                    qtpylib.crossed_below(df["close"], df["ema50"]) | qtpylib.crossed_below(df["rsi"],
                                                                                            df["rsi_ceiling"])
            ) & base_filters

        # Side-specific cooldowns
        def cooldown(sig: pd.Series, bars: int) -> pd.Series:
            recent = sig.shift(1).rolling(bars).max().fillna(False).astype(bool)
            return sig & ~recent

        long_raw = (dip_long | reclaim_long).astype(bool) & exp_roi_ok & exp_sl_ok
        short_raw = (rally_short | reclaim_short).astype(bool) & exp_roi_ok & exp_sl_ok

        long_ok = cooldown(long_raw, int(self.entry_cooldown_bars_long.value))
        short_ok = cooldown(short_raw, int(self.entry_cooldown_bars_short.value))

        dip_long_ok = dip_long & long_ok
        reclaim_long_ok = reclaim_long & long_ok & ~dip_long_ok

        rally_short_ok = rally_short & short_ok
        reclaim_short_ok = reclaim_short & short_ok & ~rally_short_ok

        df.loc[dip_long_ok, ["enter_long", "enter_tag"]] = (1, "bullish_dip")
        df.loc[reclaim_long_ok, ["enter_long", "enter_tag"]] = (1, "reclaim_dip")
        df.loc[rally_short_ok, ["enter_short", "enter_tag"]] = (1, "bearish_rally")
        df.loc[reclaim_short_ok, ["enter_short", "enter_tag"]] = (1, "reclaim_rally")

        return df

    # -----------------------------
    # Exit signals (augmented)
    # -----------------------------
    def populate_exit_trend(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # Classic RSI exits (kept)
        df.loc[(df["rsi"] > 70), ["exit_long", "exit_tag"]] = (1, "exit_rsi")
        df.loc[(df["rsi"] < 30), ["exit_short", "exit_tag"]] = (1, "exit_rsi")

        # EMA crossback: price crosses back through ema20/ema50 (optional)
        if bool(self.exit_ema_cross.value):
            cross_dn = qtpylib.crossed_below(df["close"], df["ema20"]) | qtpylib.crossed_below(df["close"], df["ema50"])
            cross_up = qtpylib.crossed_above(df["close"], df["ema20"]) | qtpylib.crossed_above(df["close"], df["ema50"])
            df.loc[cross_dn, ["exit_long", "exit_tag"]] = (1, "exit_ema_cross")
            df.loc[cross_up, ["exit_short", "exit_tag"]] = (1, "exit_ema_cross")

        # ATR exhaustion: vol collapse relative to median – helps bank when trend stalls (optional)
        if bool(self.exit_atr_exhaust.value):
            med = df["atr_pct"].rolling(100, min_periods=25).median()
            thr = float(self.atr_exhaust_mult.value)# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter


class OracleV1A(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True
    startup_candle_count = 360

    use_custom_stoploss = True
    use_custom_roi = True
    use_exit_signal = True
    position_adjustment_enable = True

    # A wide emergency exchange stop – actual risk is controlled by custom_stoploss
    stoploss = -0.30

    max_open_trades = 3

    # -----------------------------
    # Hyperopt / Params
    # -----------------------------

    # Entry – RSI quantile percentage
    buy_rsi_pct = IntParameter(22, 36, default=28, space="buy")
    bb_shift = DecimalParameter(1.03, 1.10, decimals=3, default=1.06, space="buy")
    vol_k = DecimalParameter(1.0, 1.5, decimals=2, default=1.15, space="buy")
    min_atr_pct = DecimalParameter(0.006, 0.020, decimals=3, default=0.010, space="buy")
    band_overshoot_min = DecimalParameter(0.008, 0.020, decimals=3, default=0.012, space="buy")

    # Adaptive RSI margin = rsi_margin_base + rsi_margin_k * atr_pct (atr_pct ~ 0.005..0.03)
    # NOTE: No "*100" here – that was the over-scaling bug in V52
    rsi_margin_base = IntParameter(2, 10, default=6, space="buy")
    rsi_margin_k = DecimalParameter(80.0, 300.0, decimals=1, default=160.0, space="buy")

    # Higher timeframe filter
    use_d1_filter = BooleanParameter(default=True, space="buy")
    use_d1_slope = BooleanParameter(default=True, space="buy")
    slope_window = IntParameter(5, 12, default=8, space="buy")

    # Dynamic dip lookback based on ATR% regime
    dip_lookback = IntParameter(2, 8, default=4, space="buy")  # base
    dyn_dip_atr_q_low = DecimalParameter(0.2, 0.4, decimals=2, default=0.25, space="buy")
    dyn_dip_atr_q_high = DecimalParameter(0.6, 0.8, decimals=2, default=0.75, space="buy")
    dip_lookback_lo = IntParameter(4, 10, default=6, space="buy")
    dip_lookback_hi = IntParameter(1, 4, default=2, space="buy")

    # Side-specific cooldowns
    entry_cooldown_bars_long = IntParameter(4, 12, default=6, space="buy")
    entry_cooldown_bars_short = IntParameter(4, 12, default=6, space="buy")

    # Reclaim entries & pullback targeting
    enable_reclaim = BooleanParameter(default=False, space="buy")
    pullback_pct = DecimalParameter(0.000, 0.010, decimals=3, default=0.003, space="buy")
    pullback_use_low = IntParameter(0, 1, default=1, space="buy")
    entry_pullback_max = DecimalParameter(0.003, 0.012, decimals=3, default=0.005, space="buy")
    entry_atr_cap = DecimalParameter(0.50, 1.20, decimals=2, default=0.80, space="buy")

    # Min notional
    min_stake_usd = DecimalParameter(20, 200, decimals=0, default=60, space="buy")

    # Leverage/positioning
    risk_per_trade = DecimalParameter(0.005, 0.02, decimals=3, default=0.012, space="buy")
    lev_target_c = DecimalParameter(0.06, 0.20, decimals=3, default=0.10, space="buy")
    user_max_leverage = IntParameter(2, 4, default=2, space="buy")
    signal_boost_max = DecimalParameter(1.00, 1.50, decimals=2, default=1.20, space="buy")
    vol_weak_cut = DecimalParameter(0.60, 1.00, decimals=2, default=0.85, space="buy")
    lev_round_step = DecimalParameter(0.05, 0.25, decimals=2, default=0.10, space="buy")

    # Exit / Trailing / ROI
    hard_sl_pct = DecimalParameter(-0.12, -0.04, decimals=3, default=-0.08, space="sell")
    initial_atr_mult = DecimalParameter(1.4, 3.8, decimals=2, default=2.3, space="sell")
    trail_chandelier_mult = DecimalParameter(3.0, 8.5, decimals=2, default=4.5, space="sell")

    min_roi_floor = DecimalParameter(0.010, 0.060, decimals=3, default=0.025, space="sell")
    roi_atr_mult = DecimalParameter(0.90, 2.20, decimals=2, default=1.30, space="sell")
    trail_roi_mult = DecimalParameter(2.5, 8.0, decimals=2, default=4.0, space="sell")
    max_roi_cap = DecimalParameter(0.06, 0.20, decimals=2, default=0.10, space="sell")

    min_sl_floor = DecimalParameter(0.010, 0.050, decimals=3, default=0.030, space="sell")
    max_sl_cap = DecimalParameter(0.035, 0.090, decimals=3, default=0.040, space="sell")
    mmr_est = DecimalParameter(0.003, 0.020, decimals=3, default=0.006, space="sell")
    liq_buffer_mult = DecimalParameter(2.5, 7.0, decimals=1, default=4.0, space="sell")
    activate_after_atr_mult = DecimalParameter(2.2, 5.5, decimals=1, default=3.5, space="sell")
    arm_max_minutes = IntParameter(15, 180, default=60, space="sell")

    # Exit signals (optional helpers)
    exit_ema_cross = BooleanParameter(default=True, space="sell")
    exit_atr_exhaust = BooleanParameter(default=True, space="sell")
    atr_exhaust_mult = DecimalParameter(0.40, 0.90, decimals=2, default=0.55, space="sell")

    # Profit lock – arm breakeven lock once profit >= trigger
    profit_lock_trigger = DecimalParameter(0.010, 0.050, decimals=3, default=0.020, space="sell")

    # Time-decay for ROI
    time_decay_minutes = IntParameter(120, 720, default=300, space="sell")
    time_decay_add = DecimalParameter(0.005, 0.030, decimals=3, default=0.012, space="sell")

    # Price chasing controls (new)
    chase_enable = BooleanParameter(default=True, space="buy", optimize=False)
    chase_max_bps = IntParameter(10, 60, default=35, space="buy")  # absolute cap, e.g., 35 bps = 0.35%
    chase_rate_bps_per_min = IntParameter(2, 12, default=6, space="buy")  # linear chase per minute since signal

    # Buy hyperspace params:
    buy_params = {
        "band_overshoot_min": 0.013,
        "bb_shift": 1.089,
        "buy_rsi_pct": 33,
        "chase_enable": False,
        "chase_max_bps": 41,
        "chase_rate_bps_per_min": 12,
        "dip_lookback": 6,
        "dip_lookback_hi": 1,
        "dip_lookback_lo": 9,
        "dyn_dip_atr_q_high": 0.63,
        "dyn_dip_atr_q_low": 0.22,
        "enable_reclaim": True,
        "entry_atr_cap": 1.12,
        "entry_cooldown_bars_long": 9,
        "entry_cooldown_bars_short": 11,
        "entry_pullback_max": 0.009,
        "lev_round_step": 0.05,
        "lev_target_c": 0.197,
        "min_atr_pct": 0.014,
        "min_stake_usd": 163.0,
        "pullback_pct": 0.007,
        "pullback_use_low": 1,
        "risk_per_trade": 0.02,
        "rsi_margin_base": 8,
        "rsi_margin_k": 256.4,
        "signal_boost_max": 1.26,
        "slope_window": 8,
        "use_d1_filter": True,
        "use_d1_slope": True,
        "user_max_leverage": 4,
        "vol_k": 1.25,
        "vol_weak_cut": 0.88,
    }

    # Sell hyperspace params:
    sell_params = {
        "activate_after_atr_mult": 2.7,
        "arm_max_minutes": 62,
        "atr_exhaust_mult": 0.45,
        "exit_atr_exhaust": True,
        "exit_ema_cross": False,
        "hard_sl_pct": -0.072,
        "initial_atr_mult": 1.45,
        "liq_buffer_mult": 3.5,
        "max_roi_cap": 0.13,
        "max_sl_cap": 0.078,
        "min_roi_floor": 0.017,
        "min_sl_floor": 0.012,
        "mmr_est": 0.007,
        "profit_lock_trigger": 0.022,
        "roi_atr_mult": 1.35,
        "time_decay_add": 0.029,
        "time_decay_minutes": 302,
        "trail_chandelier_mult": 7.33,
        "trail_roi_mult": 7.57,
    }

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 24,
                "trade_limit": 8,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.025
            },
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
        ]

    # -----------------------------
    # Indicators
    # -----------------------------
    def populate_indicators(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["rsi"] = ta.RSI(df, timeperiod=14)

        bb_u, bb_m, bb_l = ta.BBANDS(df["close"], timeperiod=20)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb_u, bb_m, bb_l

        df["atr"] = ta.ATR(df, timeperiod=14)
        df["vol_mean"] = df["volume"].rolling(30).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # RSI quantiles
        pct = self.buy_rsi_pct.value / 100.0
        df["rsi_floor"] = df["rsi"].rolling(100, min_periods=25).quantile(pct)
        df["rsi_ceiling"] = df["rsi"].rolling(100, min_periods=25).quantile(1 - pct)

        # Daily EMA200 (informative)
        try:
            d1, _ = self.dp.get_analyzed_dataframe(metadata["pair"], "1d")
            if d1 is not None and not d1.empty and "close" in d1.columns:
                d1_ema200 = ta.EMA(d1["close"], timeperiod=200)
                df["d1_ema200"] = d1_ema200.reindex(df.index, method="ffill")
            else:
                df["d1_ema200"] = df["ema200"]
        except Exception:
            df["d1_ema200"] = df["ema200"]

        # ATR% quantiles for dynamic lookback selection
        df["atr_pct_q_low"] = df["atr_pct"].rolling(200, min_periods=50).quantile(float(self.dyn_dip_atr_q_low.value))
        df["atr_pct_q_high"] = df["atr_pct"].rolling(200, min_periods=50).quantile(float(self.dyn_dip_atr_q_high.value))

        return df

    # -----------------------------
    # Entry
    # -----------------------------
    def populate_entry_trend(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        lb_adj = df["bb_lower"] * float(self.bb_shift.value)
        ub_adj = df["bb_upper"] / float(self.bb_shift.value)

        vol_ok = df["volume"] > (df["vol_mean"] * float(self.vol_k.value))

        # Overshoot requirements
        overshoot_long = (lb_adj - df["close"]) / df["close"] >= float(self.band_overshoot_min.value)
        overshoot_short = (df["close"] - ub_adj) / df["close"] >= float(self.band_overshoot_min.value)

        # Adaptive RSI margin (vol-aware) – fixed scaling
        rsi_margin = float(self.rsi_margin_base.value) + float(self.rsi_margin_k.value) * df["atr_pct"].fillna(0.0)
        rsi_long_ok = df["rsi"] <= (df["rsi_floor"] - rsi_margin)
        rsi_short_ok = df["rsi"] >= (df["rsi_ceiling"] + rsi_margin)

        # Volatility / non-micro floors
        atr_ok = df["atr_pct"] >= float(self.min_atr_pct.value)
        exp_roi_ok = (df["atr_pct"] * float(self.roi_atr_mult.value)).fillna(0.0) >= float(self.min_roi_floor.value)
        exp_sl_ok = (df["atr_pct"] * float(self.initial_atr_mult.value)).fillna(0.0) >= float(self.min_sl_floor.value)
        base_filters = vol_ok & atr_ok & exp_roi_ok & exp_sl_ok

        # Higher timeframe regime (optional)
        if bool(self.use_d1_filter.value):
            long_regime = (df["ema50"] > df["ema200"]) & (df["close"] > df["d1_ema200"])
            short_regime = (df["ema50"] < df["ema200"]) & (df["close"] < df["d1_ema200"])
            if bool(self.use_d1_slope.value):
                w = int(self.slope_window.value)
                long_regime &= df["d1_ema200"] > df["d1_ema200"].shift(w)
                short_regime &= df["d1_ema200"] < df["d1_ema200"].shift(w)
        else:
            long_regime = (df["ema50"] > df["ema200"])
            short_regime = (df["ema50"] < df["ema200"])

        # Dynamic dip lookback regime flags
        base_lb = max(1, int(self.dip_lookback.value))
        lb_lo = max(1, int(self.dip_lookback_lo.value))
        lb_hi = max(1, int(self.dip_lookback_hi.value))

        low_thr = df["atr_pct_q_low"]
        high_thr = df["atr_pct_q_high"]
        is_low = (df["atr_pct"] <= low_thr).fillna(False)
        is_high = (df["atr_pct"] >= high_thr).fillna(False)

        # Raw signals
        dip_long = long_regime & rsi_long_ok & (df["close"] < lb_adj) & overshoot_long & base_filters
        rally_short = short_regime & rsi_short_ok & (df["close"] > ub_adj) & overshoot_short & base_filters

        # Reclaim signals (optional)
        reclaim_long = False
        reclaim_short = False
        if bool(self.enable_reclaim.value):
            lb_touch = (df["low"] < lb_adj)
            ub_touch = (df["high"] > ub_adj)

            lb_roll_lo = lb_touch.rolling(lb_lo, min_periods=1).max().astype(bool)
            lb_roll_mid = lb_touch.rolling(base_lb, min_periods=1).max().astype(bool)
            lb_roll_hi = lb_touch.rolling(lb_hi, min_periods=1).max().astype(bool)

            ub_roll_lo = ub_touch.rolling(lb_lo, min_periods=1).max().astype(bool)
            ub_roll_mid = ub_touch.rolling(base_lb, min_periods=1).max().astype(bool)
            ub_roll_hi = ub_touch.rolling(lb_hi, min_periods=1).max().astype(bool)

            lb_touch_roll = pd.Series(
                np.where(is_low, lb_roll_lo, np.where(is_high, lb_roll_hi, lb_roll_mid)),
                index=df.index
            ).astype(bool)
            ub_touch_roll = pd.Series(
                np.where(is_low, ub_roll_lo, np.where(is_high, ub_roll_hi, ub_roll_mid)),
                index=df.index
            ).astype(bool)

            reclaim_long = long_regime & lb_touch_roll & (
                    qtpylib.crossed_above(df["close"], df["ema50"]) | qtpylib.crossed_above(df["rsi"], df["rsi_floor"])
            ) & base_filters

            reclaim_short = short_regime & ub_touch_roll & (
                    qtpylib.crossed_below(df["close"], df["ema50"]) | qtpylib.crossed_below(df["rsi"],
                                                                                            df["rsi_ceiling"])
            ) & base_filters

        # Side-specific cooldowns
        def cooldown(sig: pd.Series, bars: int) -> pd.Series:
            recent = sig.shift(1).rolling(bars).max().fillna(False).astype(bool)
            return sig & ~recent

        long_raw = (dip_long | reclaim_long).astype(bool) & exp_roi_ok & exp_sl_ok
        short_raw = (rally_short | reclaim_short).astype(bool) & exp_roi_ok & exp_sl_ok

        long_ok = cooldown(long_raw, int(self.entry_cooldown_bars_long.value))
        short_ok = cooldown(short_raw, int(self.entry_cooldown_bars_short.value))

        dip_long_ok = dip_long & long_ok
        reclaim_long_ok = reclaim_long & long_ok & ~dip_long_ok

        rally_short_ok = rally_short & short_ok
        reclaim_short_ok = reclaim_short & short_ok & ~rally_short_ok

        df.loc[dip_long_ok, ["enter_long", "enter_tag"]] = (1, "bullish_dip")
        df.loc[reclaim_long_ok, ["enter_long", "enter_tag"]] = (1, "reclaim_dip")
        df.loc[rally_short_ok, ["enter_short", "enter_tag"]] = (1, "bearish_rally")
        df.loc[reclaim_short_ok, ["enter_short", "enter_tag"]] = (1, "reclaim_rally")

        return df

    # -----------------------------
    # Exit signals (augmented)
    # -----------------------------
    def populate_exit_trend(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # Classic RSI exits (kept)
        df.loc[(df["rsi"] > 70), ["exit_long", "exit_tag"]] = (1, "exit_rsi")
        df.loc[(df["rsi"] < 30), ["exit_short", "exit_tag"]] = (1, "exit_rsi")

        # EMA crossback: price crosses back through ema20/ema50 (optional)
        if bool(self.exit_ema_cross.value):
            cross_dn = qtpylib.crossed_below(df["close"], df["ema20"]) | qtpylib.crossed_below(df["close"], df["ema50"])
            cross_up = qtpylib.crossed_above(df["close"], df["ema20"]) | qtpylib.crossed_above(df["close"], df["ema50"])
            df.loc[cross_dn, ["exit_long", "exit_tag"]] = (1, "exit_ema_cross")
            df.loc[cross_up, ["exit_short", "exit_tag"]] = (1, "exit_ema_cross")

        # ATR exhaustion: vol collapse relative to median – helps bank when trend stalls (optional)
        if bool(self.exit_atr_exhaust.value):
            med = df["atr_pct"].rolling(100, min_periods=25).median()
            thr = float(self.atr_exhaust_mult.value)
            atr_collapse = df["atr_pct"] < (thr * med)
            df.loc[atr_collapse, ["exit_long", "exit_tag"]] = (1, "exit_atr_exhaust")
            df.loc[atr_collapse, ["exit_short", "exit_tag"]] = (1, "exit_atr_exhaust")

        return df

    # -----------------------------
    # Entry price with chasing
    # -----------------------------
    def custom_entry_price(
            self,
            pair: str,
            trade: Optional[Any],
            current_time: datetime,
            proposed_rate: float,
            entry_tag: str,
            side: str,
            **kwargs: Any
    ) -> Optional[float]:
        """
        Baseline: pullback target with ATR cap.
        Chasing: linearly move target toward current price as minutes since last signal elapse,
                 up to chase_max_bps. Requires little unfilledtimeout to re-place orders.
        """
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = df.iloc[-1].squeeze()
            atr = float(last["atr"])
            close = float(last["close"])
            low = float(last.get("low", close))
            high = float(last.get("high", close))
        except Exception:
            return None

        use_extreme = int(self.pullback_use_low.value) == 1
        pb = float(self.pullback_pct.value)
        max_pct = float(self.entry_pullback_max.value)
        atr_cap = float(self.entry_atr_cap.value)

        # Baseline pullback target (as in V52)
        if side == "long":
            ref = low if use_extreme else close
            raw_target = ref * (1.0 - pb)
            floor_pct = proposed_rate * (1.0 - max_pct)
            floor_atr = ref - atr_cap * atr
            baseline = max(raw_target, floor_pct, floor_atr)
            # Respect "no worse than proposed"
            base_limited = min(baseline, proposed_rate)
        elif side == "short":
            ref = high if use_extreme else close
            raw_target = ref * (1.0 + pb)
            cap_pct = proposed_rate * (1.0 + max_pct)
            cap_atr = ref + atr_cap * atr
            baseline = min(raw_target, cap_pct, cap_atr)
            base_limited = max(baseline, proposed_rate)
        else:
            return None

        # Chasing logic
        if bool(self.chase_enable.value):
            try:
                sig_col = "enter_long" if side == "long" else "enter_short"
                # Find last signal bar time
                sig_idx = df.index[df[sig_col] == 1]
                if len(sig_idx) > 0:
                    last_sig_time = sig_idx[-1]
                    mins = max(0, int((current_time - last_sig_time).total_seconds() // 60))
                else:
                    mins = 0
            except Exception:
                mins = 0

            max_bps = int(self.chase_max_bps.value)
            rate_bps = int(self.chase_rate_bps_per_min.value)
            allowed_bps = min(max_bps, rate_bps * mins)
            allowed_frac = allowed_bps / 10000.0

            if side == "long":
                # Do not place more than 'allowed_bps' below current price
                chase_floor = close * (1.0 - allowed_frac)
                target = max(base_limited, chase_floor)
                return float(target)
            else:
                # Do not place more than 'allowed_bps' above current price
                chase_cap = close * (1.0 + allowed_frac)
                target = min(base_limited, chase_cap)
                return float(target)

        return float(base_limited)

    # -----------------------------
    # Custom stoploss with true breakeven clamp
    # -----------------------------
    def custom_stoploss(
            self,
            pair: str,
            trade: Any,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Duration / leverage
        trade_duration_mins = self._trade_duration_in_mins(trade, current_time)
        lev = float(getattr(trade, "leverage", 1.0) or 1.0)

        # Baseline dynamic ATR/liquidation stop (%)
        dynamic_dist = float(self._dynamic_stop_pct(df, float(current_rate), lev))
        min_floor = float(self.min_sl_floor.value)

        # Activation conditions
        k_atr = float(self.activate_after_atr_mult.value)
        arm_cap_min = int(self.arm_max_minutes.value)

        atr_now = float(df["atr"].iat[-1]) if not df.empty else 0.0
        atr_pct_now = atr_now / max(1e-9, float(current_rate))
        atr_trigger = k_atr * max(1e-9, atr_pct_now)

        activated = False
        if current_profit is not None and current_profit >= atr_trigger:
            activated = True
        elif 0 < arm_cap_min <= trade_duration_mins:
            activated = True

        # Profit lock – true breakeven clamp
        lock_trig = float(self.profit_lock_trigger.value)
        lock_to_breakeven = (current_profit is not None) and (current_profit >= lock_trig)

        # Compute true breakeven distance (fraction of current price)
        try:
            entry = float(trade.open_rate)
            price = float(current_rate)
            if getattr(trade, "is_long", True):
                be_dist = max(0.0, (price - entry) / max(1e-9, price))
            else:
                be_dist = max(0.0, (entry - price) / max(1e-9, price))
        except Exception:
            be_dist = 0.0

        if not activated:
            early_dist = max(min_floor, dynamic_dist)
            if lock_to_breakeven:
                early_dist = max(min_floor, min(early_dist, be_dist + 1e-6))
            return -float(early_dist)

        # After activation – also compute chandelier
        ce_dist = float("inf")
        try:
            k = float(self.trail_chandelier_mult.value)
            atr = float(df["atr"].iat[-1])
            if k > 0 and atr > 0:
                since_entry = df.loc[df.index >= trade.date_entry_fill_utc]
                if not since_entry.empty:
                    price = float(current_rate)
                    if getattr(trade, "is_long", True):
                        hh = float(since_entry["high"].max())
                        stop_price = hh - k * atr
                        ce_dist = max(1e-6, (price - stop_price) / max(1e-9, price))
                    else:
                        ll = float(since_entry["low"].min())
                        stop_price = ll + k * atr
                        ce_dist = max(1e-6, (stop_price - price) / max(1e-9, price))
        except Exception:
            pass

        # Blend: tighter of dynamic vs CE, respect floor and breakeven lock
        final_dist = max(min_floor, min(dynamic_dist, ce_dist))
        if lock_to_breakeven:
            final_dist = max(min_floor, min(final_dist, be_dist + 1e-6))

        return -float(final_dist)

    # -----------------------------
    # Custom ROI (nearest anchor + time-decay)
    # -----------------------------
    def custom_roi(
            self,
            pair: str,
            trade: Any,
            current_time: datetime,
            trade_duration: int,
            entry_tag: str,
            side: str,
            **kwargs,
    ) -> float:
        min_roi_floor = float(self.min_roi_floor.value)
        roi_atr_mult_entry = float(self.roi_atr_mult.value)
        roi_atr_mult_trail = float(self.trail_roi_mult.value)
        max_roi_cap = float(self.max_roi_cap.value)

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Base floor from entry ATR% (nearest index to entry time)
        try:
            idx = df.index.get_indexer([trade.date_entry_fill_utc], method="nearest")[0]
            entry_atr = float(df["atr"].iloc[idx])
            entry_price = float(trade.open_rate)
            entry_atr_pct = entry_atr / max(1e-9, entry_price)
        except Exception:
            entry_atr_pct = float(df["atr"].iat[-1]) / max(1e-9, float(df["close"].iat[-1]))

        roi_floor = max(min_roi_floor, roi_atr_mult_entry * entry_atr_pct)

        # Trailing ROI based on excursion – ATR-scaled
        try:
            since_entry = df.loc[df.index >= trade.date_entry_fill_utc]
            atr = float(since_entry["atr"].iloc[-1])
            if side == "long":
                highest = float(since_entry["high"].max())
                trailing_roi = max(
                    min_roi_floor,
                    (highest - trade.open_rate) / trade.open_rate - roi_atr_mult_trail * atr / max(1e-9, highest)
                )
            else:
                lowest = float(since_entry["low"].min())
                trailing_roi = max(
                    min_roi_floor,
                    (trade.open_rate - lowest) / trade.open_rate - roi_atr_mult_trail * atr / max(1e-9, lowest)
                )
        except Exception:
            trailing_roi = min_roi_floor

        roi_target = max(roi_floor, trailing_roi)

        # Time decay: after N minutes, demand a bit more ROI
        decay_min = int(self.time_decay_minutes.value)
        decay_add = float(self.time_decay_add.value)
        if trade_duration is not None and trade_duration >= decay_min:
            roi_target += decay_add

        final_roi = min(max_roi_cap, roi_target)
        return float(final_roi)

    # -----------------------------
    # Position sizing
    # -----------------------------
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

        if stake < float(self.min_stake_usd.value):
            return 0.0

        return float(stake)

    # -----------------------------
    # Leverage (stepped float, no int cast)
    # -----------------------------
    def leverage(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            entry_tag: str,
            side: str,
            **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        sl_dist = self._stop_dist_pct(df)
        c = float(self.lev_target_c.value)
        L = c / max(1e-6, sl_dist)

        # Signal boosting and liquidity derating
        L *= (1.0 + (self._signal_strength(df, side) * (float(self.signal_boost_max.value) - 1.0)))
        L *= self._liquidity_factor(df)

        # Liquidation safety cap
        k = float(self.liq_buffer_mult.value)
        mmr = float(self.mmr_est.value)
        max_safe_by_liq = 1.0 / max(1e-6, (mmr + k * sl_dist))
        L = min(L, max_safe_by_liq)

        # User/exchange caps + step rounding
        user_cap = float(self.user_max_leverage.value)
        exch_cap = float(max_leverage or user_cap)
        step = float(self.lev_round_step.value)

        L = self._round_to_step(min(L, user_cap, exch_cap), step)
        L = max(step, int(L))

        return float(L)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _atr_pct(self, df: pd.DataFrame) -> float:
        return float(df["atr"].iat[-1]) / max(1e-9, float(df["close"].iat[-1]))

    def _stop_dist_pct(self, df: pd.DataFrame) -> float:
        atr_pct = self._atr_pct(df)
        sld = float(self.initial_atr_mult.value) * atr_pct
        return min(sld, 0.08)

    def _signal_strength(self, df: pd.DataFrame, side: str) -> float:
        rsi = float(df["rsi"].iat[-1])
        rsi_floor = float(df["rsi_floor"].iat[-1])
        rsi_ceil = float(df.get("rsi_ceiling", df["rsi"]).iat[-1])

        close = float(df["close"].iat[-1])
        lb = float(df["bb_lower"].iat[-1]) * float(self.bb_shift.value)
        ub = float(df["bb_upper"].iat[-1]) / float(self.bb_shift.value)

        if side == "long":
            rsi_edge = max(0.0, (rsi_floor - rsi) / max(1.0, rsi_floor))
            bb_edge = max(0.0, (lb - close) / max(1e-9, lb))
        else:
            rsi_edge = max(0.0, (rsi - rsi_ceil) / max(1.0, 100.0 - rsi_ceil))
            bb_edge = max(0.0, (close - ub) / max(1e-9, ub))
        return min(1.0, 0.6 * rsi_edge + 0.4 * bb_edge)

    def _liquidity_factor(self, df: pd.DataFrame) -> float:
        v = float(df["volume"].iat[-1])
        vm = max(1e-9, float(df["vol_mean"].iat[-1]))
        ratio = v / vm
        if ratio >= 1.0:
            return 1.0
        cut = float(self.vol_weak_cut.value)
        if ratio <= cut:
            return 0.60
        return 0.60 + 0.40 * (ratio - cut) / max(1e-9, (1.0 - cut))

    def _round_to_step(self, x: float, step: float) -> float:
        return max(step, round(x / step) * step)

    def _dynamic_stop_pct(self, df: pd.DataFrame, current_rate: float, leverage: float) -> float:
        try:
            price = float(current_rate)
            atr = float(df["atr"].iat[-1])
        except Exception:
            price, atr = max(float(current_rate) or 1.0, 1.0), 0.0

        if not price or price <= 0 or price != price:
            price = 1.0

        atr_ratio = (atr / price) if price > 0 else 0.0
        if atr_ratio != atr_ratio or atr_ratio < 0:
            atr_ratio = 0.0

        try:
            atr_stop = float(self.initial_atr_mult.value) * atr_ratio
        except Exception:
            atr_stop = 0.02

        try:
            mmr = float(self.mmr_est.value)
        except Exception:
            mmr = 0.005

        try:
            k = float(self.liq_buffer_mult.value)
        except Exception:
            k = 3.0

        lev = float(leverage or 1.0)
        if lev <= 0:
            lev = 1.0

        liq_dist = max(1e-6, 1.0 / lev - mmr)
        liq_safe_stop = max(1e-6, liq_dist / max(k, 1e-6))

        sl_cap = float(self.max_sl_cap.value)
        hard_cap = abs(float(self.hard_sl_pct.value))
        min_floor = float(self.min_sl_floor.value)

        final_stop = max(min_floor, min(max(1e-6, atr_stop), liq_safe_stop, sl_cap, hard_cap))
        if not (final_stop == final_stop) or final_stop <= 0:
            final_stop = max(min_floor, min(hard_cap, sl_cap))
        return float(final_stop)

    def _trade_duration_in_mins(self, trade, current_time: datetime) -> int:
        try:
            elapsed = current_time - trade.date_entry_fill_utc
            return int(float(elapsed.total_seconds()) / 60)
        except Exception:
            return 0

            atr_collapse = df["atr_pct"] < (thr * med)
            df.loc[atr_collapse, ["exit_long", "exit_tag"]] = (1, "exit_atr_exhaust")
            df.loc[atr_collapse, ["exit_short", "exit_tag"]] = (1, "exit_atr_exhaust")

        return df

    # -----------------------------
    # Entry price with chasing
    # -----------------------------
    def custom_entry_price(
            self,
            pair: str,
            trade: Optional[Any],
            current_time: datetime,
            proposed_rate: float,
            entry_tag: str,
            side: str,
            **kwargs: Any
    ) -> Optional[float]:
        """
        Baseline: pullback target with ATR cap.
        Chasing: linearly move target toward current price as minutes since last signal elapse,
                 up to chase_max_bps. Requires little unfilledtimeout to re-place orders.
        """
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = df.iloc[-1].squeeze()
            atr = float(last["atr"])
            close = float(last["close"])
            low = float(last.get("low", close))
            high = float(last.get("high", close))
        except Exception:
            return None

        use_extreme = int(self.pullback_use_low.value) == 1
        pb = float(self.pullback_pct.value)
        max_pct = float(self.entry_pullback_max.value)
        atr_cap = float(self.entry_atr_cap.value)

        # Baseline pullback target (as in V52)
        if side == "long":
            ref = low if use_extreme else close
            raw_target = ref * (1.0 - pb)
            floor_pct = proposed_rate * (1.0 - max_pct)
            floor_atr = ref - atr_cap * atr
            baseline = max(raw_target, floor_pct, floor_atr)
            # Respect "no worse than proposed"
            base_limited = min(baseline, proposed_rate)
        elif side == "short":
            ref = high if use_extreme else close
            raw_target = ref * (1.0 + pb)
            cap_pct = proposed_rate * (1.0 + max_pct)
            cap_atr = ref + atr_cap * atr
            baseline = min(raw_target, cap_pct, cap_atr)
            base_limited = max(baseline, proposed_rate)
        else:
            return None

        # Chasing logic
        if bool(self.chase_enable.value):
            try:
                sig_col = "enter_long" if side == "long" else "enter_short"
                # Find last signal bar time
                sig_idx = df.index[df[sig_col] == 1]
                if len(sig_idx) > 0:
                    last_sig_time = sig_idx[-1]
                    mins = max(0, int((current_time - last_sig_time).total_seconds() // 60))
                else:
                    mins = 0
            except Exception:
                mins = 0

            max_bps = int(self.chase_max_bps.value)
            rate_bps = int(self.chase_rate_bps_per_min.value)
            allowed_bps = min(max_bps, rate_bps * mins)
            allowed_frac = allowed_bps / 10000.0

            if side == "long":
                # Do not place more than 'allowed_bps' below current price
                chase_floor = close * (1.0 - allowed_frac)
                target = max(base_limited, chase_floor)
                return float(target)
            else:
                # Do not place more than 'allowed_bps' above current price
                chase_cap = close * (1.0 + allowed_frac)
                target = min(base_limited, chase_cap)
                return float(target)

        return float(base_limited)

    # -----------------------------
    # Custom stoploss with true breakeven clamp
    # -----------------------------
    def custom_stoploss(
            self,
            pair: str,
            trade: Any,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Duration / leverage
        trade_duration_mins = self._trade_duration_in_mins(trade, current_time)
        lev = float(getattr(trade, "leverage", 1.0) or 1.0)

        # Baseline dynamic ATR/liquidation stop (%)
        dynamic_dist = float(self._dynamic_stop_pct(df, float(current_rate), lev))
        min_floor = float(self.min_sl_floor.value)

        # Activation conditions
        k_atr = float(self.activate_after_atr_mult.value)
        arm_cap_min = int(self.arm_max_minutes.value)

        atr_now = float(df["atr"].iat[-1]) if not df.empty else 0.0
        atr_pct_now = atr_now / max(1e-9, float(current_rate))
        atr_trigger = k_atr * max(1e-9, atr_pct_now)

        activated = False
        if current_profit is not None and current_profit >= atr_trigger:
            activated = True
        elif 0 < arm_cap_min <= trade_duration_mins:
            activated = True

        # Profit lock – true breakeven clamp
        lock_trig = float(self.profit_lock_trigger.value)
        lock_to_breakeven = (current_profit is not None) and (current_profit >= lock_trig)

        # Compute true breakeven distance (fraction of current price)
        try:
            entry = float(trade.open_rate)
            price = float(current_rate)
            if getattr(trade, "is_long", True):
                be_dist = max(0.0, (price - entry) / max(1e-9, price))
            else:
                be_dist = max(0.0, (entry - price) / max(1e-9, price))
        except Exception:
            be_dist = 0.0

        if not activated:
            early_dist = max(min_floor, dynamic_dist)
            if lock_to_breakeven:
                early_dist = max(min_floor, min(early_dist, be_dist + 1e-6))
            return -float(early_dist)

        # After activation – also compute chandelier
        ce_dist = float("inf")
        try:
            k = float(self.trail_chandelier_mult.value)
            atr = float(df["atr"].iat[-1])
            if k > 0 and atr > 0:
                since_entry = df.loc[df.index >= trade.date_entry_fill_utc]
                if not since_entry.empty:
                    price = float(current_rate)
                    if getattr(trade, "is_long", True):
                        hh = float(since_entry["high"].max())
                        stop_price = hh - k * atr
                        ce_dist = max(1e-6, (price - stop_price) / max(1e-9, price))
                    else:
                        ll = float(since_entry["low"].min())
                        stop_price = ll + k * atr
                        ce_dist = max(1e-6, (stop_price - price) / max(1e-9, price))
        except Exception:
            pass

        # Blend: tighter of dynamic vs CE, respect floor and breakeven lock
        final_dist = max(min_floor, min(dynamic_dist, ce_dist))
        if lock_to_breakeven:
            final_dist = max(min_floor, min(final_dist, be_dist + 1e-6))

        return -float(final_dist)

    # -----------------------------
    # Custom ROI (nearest anchor + time-decay)
    # -----------------------------
    def custom_roi(
            self,
            pair: str,
            trade: Any,
            current_time: datetime,
            trade_duration: int,
            entry_tag: str,
            side: str,
            **kwargs,
    ) -> float:
        min_roi_floor = float(self.min_roi_floor.value)
        roi_atr_mult_entry = float(self.roi_atr_mult.value)
        roi_atr_mult_trail = float(self.trail_roi_mult.value)
        max_roi_cap = float(self.max_roi_cap.value)

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Base floor from entry ATR% (nearest index to entry time)
        try:
            idx = df.index.get_indexer([trade.date_entry_fill_utc], method="nearest")[0]
            entry_atr = float(df["atr"].iloc[idx])
            entry_price = float(trade.open_rate)
            entry_atr_pct = entry_atr / max(1e-9, entry_price)
        except Exception:
            entry_atr_pct = float(df["atr"].iat[-1]) / max(1e-9, float(df["close"].iat[-1]))

        roi_floor = max(min_roi_floor, roi_atr_mult_entry * entry_atr_pct)

        # Trailing ROI based on excursion – ATR-scaled
        try:
            since_entry = df.loc[df.index >= trade.date_entry_fill_utc]
            atr = float(since_entry["atr"].iloc[-1])
            if side == "long":
                highest = float(since_entry["high"].max())
                trailing_roi = max(
                    min_roi_floor,
                    (highest - trade.open_rate) / trade.open_rate - roi_atr_mult_trail * atr / max(1e-9, highest)
                )
            else:
                lowest = float(since_entry["low"].min())
                trailing_roi = max(
                    min_roi_floor,
                    (trade.open_rate - lowest) / trade.open_rate - roi_atr_mult_trail * atr / max(1e-9, lowest)
                )
        except Exception:
            trailing_roi = min_roi_floor

        roi_target = max(roi_floor, trailing_roi)

        # Time decay: after N minutes, demand a bit more ROI
        decay_min = int(self.time_decay_minutes.value)
        decay_add = float(self.time_decay_add.value)
        if trade_duration is not None and trade_duration >= decay_min:
            roi_target += decay_add

        final_roi = min(max_roi_cap, roi_target)
        return float(final_roi)

    # -----------------------------
    # Position sizing
    # -----------------------------
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

        if stake < float(self.min_stake_usd.value):
            return 0.0

        return float(stake)

    # -----------------------------
    # Leverage (stepped float, no int cast)
    # -----------------------------
    def leverage(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            entry_tag: str,
            side: str,
            **kwargs,
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        sl_dist = self._stop_dist_pct(df)
        c = float(self.lev_target_c.value)
        L = c / max(1e-6, sl_dist)

        # Signal boosting and liquidity derating
        L *= (1.0 + (self._signal_strength(df, side) * (float(self.signal_boost_max.value) - 1.0)))
        L *= self._liquidity_factor(df)

        # Liquidation safety cap
        k = float(self.liq_buffer_mult.value)
        mmr = float(self.mmr_est.value)
        max_safe_by_liq = 1.0 / max(1e-6, (mmr + k * sl_dist))
        L = min(L, max_safe_by_liq)

        # User/exchange caps + step rounding
        user_cap = float(self.user_max_leverage.value)
        exch_cap = float(max_leverage or user_cap)
        step = float(self.lev_round_step.value)

        L = self._round_to_step(min(L, user_cap, exch_cap), step)
        L = max(step, int(L))

        return float(L)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _atr_pct(self, df: pd.DataFrame) -> float:
        return float(df["atr"].iat[-1]) / max(1e-9, float(df["close"].iat[-1]))

    def _stop_dist_pct(self, df: pd.DataFrame) -> float:
        atr_pct = self._atr_pct(df)
        sld = float(self.initial_atr_mult.value) * atr_pct
        return min(sld, 0.08)

    def _signal_strength(self, df: pd.DataFrame, side: str) -> float:
        rsi = float(df["rsi"].iat[-1])
        rsi_floor = float(df["rsi_floor"].iat[-1])
        rsi_ceil = float(df.get("rsi_ceiling", df["rsi"]).iat[-1])

        close = float(df["close"].iat[-1])
        lb = float(df["bb_lower"].iat[-1]) * float(self.bb_shift.value)
        ub = float(df["bb_upper"].iat[-1]) / float(self.bb_shift.value)

        if side == "long":
            rsi_edge = max(0.0, (rsi_floor - rsi) / max(1.0, rsi_floor))
            bb_edge = max(0.0, (lb - close) / max(1e-9, lb))
        else:
            rsi_edge = max(0.0, (rsi - rsi_ceil) / max(1.0, 100.0 - rsi_ceil))
            bb_edge = max(0.0, (close - ub) / max(1e-9, ub))
        return min(1.0, 0.6 * rsi_edge + 0.4 * bb_edge)

    def _liquidity_factor(self, df: pd.DataFrame) -> float:
        v = float(df["volume"].iat[-1])
        vm = max(1e-9, float(df["vol_mean"].iat[-1]))
        ratio = v / vm
        if ratio >= 1.0:
            return 1.0
        cut = float(self.vol_weak_cut.value)
        if ratio <= cut:
            return 0.60
        return 0.60 + 0.40 * (ratio - cut) / max(1e-9, (1.0 - cut))

    def _round_to_step(self, x: float, step: float) -> float:
        return max(step, round(x / step) * step)

    def _dynamic_stop_pct(self, df: pd.DataFrame, current_rate: float, leverage: float) -> float:
        try:
            price = float(current_rate)
            atr = float(df["atr"].iat[-1])
        except Exception:
            price, atr = max(float(current_rate) or 1.0, 1.0), 0.0

        if not price or price <= 0 or price != price:
            price = 1.0

        atr_ratio = (atr / price) if price > 0 else 0.0
        if atr_ratio != atr_ratio or atr_ratio < 0:
            atr_ratio = 0.0

        try:
            atr_stop = float(self.initial_atr_mult.value) * atr_ratio
        except Exception:
            atr_stop = 0.02

        try:
            mmr = float(self.mmr_est.value)
        except Exception:
            mmr = 0.005

        try:
            k = float(self.liq_buffer_mult.value)
        except Exception:
            k = 3.0

        lev = float(leverage or 1.0)
        if lev <= 0:
            lev = 1.0

        liq_dist = max(1e-6, 1.0 / lev - mmr)
        liq_safe_stop = max(1e-6, liq_dist / max(k, 1e-6))

        sl_cap = float(self.max_sl_cap.value)
        hard_cap = abs(float(self.hard_sl_pct.value))
        min_floor = float(self.min_sl_floor.value)

        final_stop = max(min_floor, min(max(1e-6, atr_stop), liq_safe_stop, sl_cap, hard_cap))
        if not (final_stop == final_stop) or final_stop <= 0:
            final_stop = max(min_floor, min(hard_cap, sl_cap))
        return float(final_stop)

    def _trade_duration_in_mins(self, trade, current_time: datetime) -> int:
        try:
            elapsed = current_time - trade.date_entry_fill_utc
            return int(float(elapsed.total_seconds()) / 60)
        except Exception:
            return 0

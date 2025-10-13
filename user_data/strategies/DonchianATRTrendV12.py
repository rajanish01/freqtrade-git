# --- DonchianATRTrendV3.py ------------------------------------------------
# Entry: Donchian/ATR breakout with Supertrend & 4h regime gating (unchanged)
# Exit/SL/ROI/Stake/Leverage: Oracle-style (ATR-anchored ROI & SL, time-decay,
#  chandelier trail, breakeven clamp, equity×risk stake, liquidity/signal aware lev)
#
# Requires: futures mode for leverage hooks. Timeframe 1h; informative 4h.

from __future__ import annotations

from datetime import datetime
from functools import reduce
from typing import Tuple, Any, Dict

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, BooleanParameter


# -----------------------
# Utility / indicators
# -----------------------

def rma(series: Series, length: int) -> Series:
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()


def true_range(df: DataFrame) -> Series:
    pc = df["close"].shift(1)
    return np.maximum(df["high"] - df["low"], np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))


def atr(df: DataFrame, length: int = 14) -> Series:
    return rma(true_range(df), length)


def di_plus(df: DataFrame, length: int = 14) -> Series:
    up = df["high"] - df["high"].shift(1)
    dn = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    return (100.0 * rma(pd.Series(plus_dm, index=df.index), length) / atr(df, length).replace(0.0, np.nan)).fillna(0.0)


def di_minus(df: DataFrame, length: int = 14) -> Series:
    up = df["high"] - df["high"].shift(1)
    dn = df["low"].shift(1) - df["low"]
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    return (100.0 * rma(pd.Series(minus_dm, index=df.index), length) / atr(df, length).replace(0.0, np.nan)).fillna(0.0)


def adx(df: DataFrame, length: int = 14) -> Series:
    pdi = di_plus(df, length);
    mdi = di_minus(df, length)
    dx = (100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)).fillna(0.0)
    return rma(dx, length)


def ema(series: Series, length: int) -> Series:
    return series.ewm(span=length, adjust=False).mean()


def donchian(df: DataFrame, length: int = 20) -> Tuple[Series, Series]:
    upper = df["high"].rolling(length, min_periods=1).max()
    lower = df["low"].rolling(length, min_periods=1).min()
    return upper, lower


def supertrend(df: DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> Tuple[Series, Series]:
    atrv = atr(df, atr_period)
    hl2 = (df["high"] + df["low"]) / 2.0
    ub = hl2 + multiplier * atrv
    lb = hl2 - multiplier * atrv

    st = pd.Series(index=df.index, dtype=float)
    sd = pd.Series(index=df.index, dtype=float)
    st.iloc[0] = ub.iloc[0];
    sd.iloc[0] = 1.0

    for i in range(1, len(df)):
        prev_st = st.iloc[i - 1]
        prev_dir = sd.iloc[i - 1]
        cu = ub.iloc[i]
        cl = lb.iloc[i]
        c = df["close"].iloc[i]
        if prev_dir == 1:
            cu = min(cu, prev_st)
        else:
            cl = max(cl, prev_st)
        if c > cu:
            sd.iloc[i] = 1.0; st.iloc[i] = cl
        elif c < cl:
            sd.iloc[i] = -1.0; st.iloc[i] = cu
        else:
            sd.iloc[i] = prev_dir; st.iloc[i] = cl if prev_dir == 1 else cu
    return sd.astype(int), st


# -----------------------
# Strategy
# -----------------------

class DonchianATRTrendV12(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = True
    timeframe: str = "1h"
    informative_timeframe: str = "4h"
    startup_candle_count: int = 320

    # We now use full custom stack inspired by OracleV1A
    use_custom_stoploss: bool = True
    use_custom_roi: bool = True
    use_exit_signal: bool = True
    position_adjustment_enable: bool = True

    # Wide exchange/emergency SL – actual risk via custom_stoploss
    stoploss = -0.30
    minimal_roi = {"0": 0.99}  # unused; custom_roi governs
    trailing_stop = False
    max_open_trades = 3

    # =======================
    # ENTRY (UNCHANGED)
    # =======================

    # Donchian & breakout buffer
    dc_len = IntParameter(10, 60, default=20, space="buy")
    dc_atr_buf = DecimalParameter(0.0, 1.0, decimals=2, default=0.25, space="buy")
    dc_atr_buf_short = DecimalParameter(0.0, 1.0, decimals=2, default=0.10, space="buy")

    # Trend / regime gates
    adx_min = IntParameter(10, 35, default=18, space="buy")
    ema200_slope_min = DecimalParameter(0.0, 0.005, decimals=4, default=0.0005, space="buy")
    atrp_min = DecimalParameter(0.001, 0.02, decimals=4, default=0.0030, space="buy")
    atrp_max = DecimalParameter(0.02, 0.15, decimals=3, default=0.060, space="buy")
    adx_min_short = IntParameter(10, 35, default=18, space="buy")
    ema200_slope_min_short = DecimalParameter(0.0, 0.005, decimals=4, default=0.0003, space="buy")

    # Volume filter (relative to MA)
    vol_ma_len = IntParameter(10, 80, default=30, space="buy")
    vol_k = DecimalParameter(0.5, 1.5, decimals=2, default=0.8, space="buy")
    vol_ma_len_short = IntParameter(10, 80, default=30, space="buy")
    vol_k_short = DecimalParameter(0.5, 1.5, decimals=2, default=0.8, space="buy")

    # Supertrend settings (LTF & HTF)
    st_atr_period = IntParameter(7, 21, default=10, space="buy")
    st_multiplier = DecimalParameter(1.5, 4.0, decimals=1, default=3.0, space="buy")

    # HTF gates
    use_htf_ema = BooleanParameter(default=True, space="buy")
    use_htf_st = BooleanParameter(default=True, space="buy")
    atrp_min_short = DecimalParameter(0.001, 0.02, decimals=4, default=0.0030, space="buy")
    atrp_max_short = DecimalParameter(0.02, 0.15, decimals=3, default=0.060, space="buy")
    short_break_mode = CategoricalParameter(["donchian", "st_or_dc"], default="st_or_dc", space="buy")
    short_price_gate = CategoricalParameter(["ema_only", "ema_or_slope"], default="ema_or_slope", space="buy")
    st_break_eps_atr = DecimalParameter(0.0, 0.6, decimals=2, default=0.10,
                                        space="buy")  # how far below ST line (as ATR multiple)

    # =======================
    # ORACLE-STYLE EXIT / SL / ROI / RISK / LEVERAGE
    # =======================

    # Risk & size
    risk_per_trade = DecimalParameter(0.01, 0.018, decimals=3, default=0.006, space="buy")  # equity fraction

    # Min notional guard
    min_stake_usd = DecimalParameter(20, 200, decimals=0, default=60, space="buy")
    htf_gate_mode = CategoricalParameter(["ALL", "ANY"], default="ANY", space="buy")

    # Leverage (INT 1–4x); signal/liquidity modifiers retained
    lev_target_c = DecimalParameter(0.06, 0.20, decimals=3, default=0.10, space="buy")
    signal_boost_max = DecimalParameter(1.00, 1.50, decimals=2, default=1.20, space="buy")
    vol_weak_cut = DecimalParameter(0.60, 1.00, decimals=2, default=0.85, space="buy")

    user_max_leverage = IntParameter(1, 4, default=4, space="buy")  # hard cap 1–4x
    lev_round_step = DecimalParameter(0.05, 0.25, decimals=2, default=0.10, space="buy")

    # Stop model (ATR-anchored + liquidation safety)
    hard_sl_pct = DecimalParameter(-0.12, -0.04, decimals=3, default=-0.08, space="sell")
    initial_atr_mult = DecimalParameter(1.4, 3.8, decimals=2, default=2.3, space="sell")
    min_sl_floor = DecimalParameter(0.030, 0.050, decimals=3, default=0.030, space="sell")
    max_sl_cap = DecimalParameter(0.05, 0.080, decimals=3, default=0.050, space="sell")
    mmr_est = DecimalParameter(0.003, 0.020, decimals=3, default=0.006, space="sell")
    liq_buffer_mult = DecimalParameter(2.5, 7.0, decimals=1, default=4.0, space="sell")

    # Activation & trailing (breakeven + chandelier)
    profit_lock_trigger = DecimalParameter(0.010, 0.050, decimals=3, default=0.020, space="sell")
    activate_after_atr_mult = DecimalParameter(2.2, 5.5, decimals=1, default=3.5, space="sell")
    arm_max_minutes = IntParameter(15, 180, default=60, space="sell")
    trail_chandelier_mult = DecimalParameter(3.0, 8.5, decimals=2, default=4.5, space="sell")

    # Custom ROI: ATR-anchored floor + excursion trail + time-decay (then capped)
    min_roi_floor = DecimalParameter(0.03, 0.050, decimals=3, default=0.025, space="sell")
    roi_atr_mult = DecimalParameter(0.90, 2.20, decimals=2, default=1.30, space="sell")
    trail_roi_mult = DecimalParameter(2.5, 8.0, decimals=2, default=4.0, space="sell")
    max_roi_cap = DecimalParameter(0.06, 0.10, decimals=2, default=0.10, space="sell")
    time_decay_minutes = IntParameter(60, 720, default=300, space="sell")
    time_decay_add = DecimalParameter(0.005, 0.030, decimals=3, default=0.012, space="sell")

    # Optional exit helpers
    exit_ema_cross = BooleanParameter(default=True, space="sell")
    exit_atr_exhaust = BooleanParameter(default=True, space="sell")
    atr_exhaust_mult = DecimalParameter(0.40, 0.90, decimals=2, default=0.55, space="sell")

    buy_params = {
        "adx_min": 35,
        "adx_min_short": 25,
        "atrp_max": 0.138,
        "atrp_max_short": 0.031,
        "atrp_min": 0.0052,
        "atrp_min_short": 0.0094,
        "dc_atr_buf": 0.06,
        "dc_atr_buf_short": 0.09,
        "dc_len": 32,
        "ema200_slope_min": 0.002,
        "ema200_slope_min_short": 0.0001,
        "htf_gate_mode": "ANY",
        "lev_round_step": 0.08,
        "lev_target_c": 0.074,
        "min_stake_usd": 134.0,
        "risk_per_trade": 0.013,
        "short_break_mode": "st_or_dc",
        "short_price_gate": "ema_only",
        "signal_boost_max": 1.42,
        "st_atr_period": 9,
        "st_break_eps_atr": 0.45,
        "st_multiplier": 1.7,
        "use_htf_ema": True,
        "use_htf_st": False,
        "user_max_leverage": 3,
        "vol_k": 0.65,
        "vol_k_short": 0.86,
        "vol_ma_len": 73,
        "vol_ma_len_short": 47,
        "vol_weak_cut": 0.68,
    }

    # Sell hyperspace params:
    sell_params = {
        "activate_after_atr_mult": 2.9,
        "arm_max_minutes": 179,
        "atr_exhaust_mult": 0.47,
        "exit_atr_exhaust": False,
        "exit_ema_cross": False,
        "hard_sl_pct": -0.099,
        "initial_atr_mult": 3.4,
        "liq_buffer_mult": 4.7,
        "max_roi_cap": 0.08,
        "max_sl_cap": 0.076,
        "min_roi_floor": 0.03,
        "min_sl_floor": 0.05,
        "mmr_est": 0.013,
        "profit_lock_trigger": 0.041,
        "roi_atr_mult": 1.22,
        "time_decay_add": 0.005,
        "time_decay_minutes": 546,
        "trail_chandelier_mult": 7.08,
        "trail_roi_mult": 2.78,
    }

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 24,
                "trade_limit": 6,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.05
            },
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
        ]

    # -----------------------
    # Informative pairs
    # -----------------------
    def informative_pairs(self):
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    # -----------------------
    # Indicators
    # -----------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # LTF (1h)
        df["ema200"] = ema(df["close"], 200)
        df["ema200_prev"] = df["ema200"].shift(3)
        df["ema200_slope"] = (df["ema200"] / df["ema200_prev"]) - 1.0

        df["atr"] = atr(df, 14)
        df["atrp"] = (df["atr"] / df["close"]).abs().clip(lower=0)

        df["adx"] = adx(df, 14)

        dc_len = int(self.dc_len.value)
        dc_u, dc_l = donchian(df, dc_len)
        df["dc_upper_prev"] = dc_u.shift(1)
        df["dc_lower_prev"] = dc_l.shift(1)

        stp = int(self.st_atr_period.value)
        stm = float(self.st_multiplier.value)
        st_trend, st_line = supertrend(df, stp, stm)
        df["st_trend"] = st_trend
        df["st_line"] = st_line

        vlen = int(self.vol_ma_len.value)
        df["vol_ma"] = df["volume"].rolling(vlen, min_periods=1).mean()

        # HTF regime (4h)
        dfi = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        dfi["ema200"] = ema(dfi["close"], 200)
        dfi["ema200_prev"] = dfi["ema200"].shift(3)
        dfi["ema200_slope"] = (dfi["ema200"] / dfi["ema200_prev"]) - 1.0
        st_trend_h, _ = supertrend(dfi, stp, stm)
        dfi["st_trend_htf"] = st_trend_h

        df = df.merge(
            dfi[["ema200", "ema200_slope", "st_trend_htf"]].rename(columns={
                "ema200": "ema200_htf", "ema200_slope": "ema200_slope_htf"
            }),
            left_index=True, right_index=True, how="left"
        ).ffill()

        # Forward-fill with new pandas style (.ffill())
        for c in ["dc_upper_prev", "dc_lower_prev", "ema200_slope", "ema200_slope_htf", "st_line"]:
            if c in df:
                df[c] = df[c].ffill()

        # Extra columns for exit helpers
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["vol_mean"] = df["volume"].rolling(30).mean()
        df["atr_pct"] = df["atr"] / df["close"]
        return df

    # -----------------------
    # Entry logic (UNCHANGED)
    # -----------------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        LONG: unchanged (strict), requires Donchian + ST alignment + EMA200 regime + ADX + HTF ALL
        SHORT: mirrored but more permissive, with extra knobs to increase valid short frequency:
               - smaller Donchian ATR buffer for shorts (dc_atr_buf_short)
               - break mode "donchian" or "st_or_dc"
               - price gate "ema_only" or "ema_or_slope"
               - wider/looser ATR% window for shorts (via params)
               - HTF gates ANY/ALL via htf_gate_mode (default ANY)
               - relaxed ST epsilon for shorts
               - short-specific volume filter (vol_ma_len_short / vol_k_short)
        """
        # init
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""

        # -------- shared / long thresholds --------
        buf_long = float(self.dc_atr_buf.value)
        adx_min_l = int(self.adx_min.value)
        slope_min_l = float(self.ema200_slope_min.value)
        atrp_min_l = float(self.atrp_min.value)
        atrp_max_l = float(self.atrp_max.value)
        v_k = float(self.vol_k.value)

        use_htf_ema = bool(self.use_htf_ema.value)
        use_htf_st = bool(self.use_htf_st.value)

        # -------- short-specific thresholds (more permissive) --------
        # If short-specific params aren't defined, gracefully fall back to long/shared ones.
        buf_short = float(getattr(self, "dc_atr_buf_short", self.dc_atr_buf).value)
        adx_min_s = int(self.adx_min_short.value) if hasattr(self, "adx_min_short") else adx_min_l
        slope_min_s = float(self.ema200_slope_min_short.value) if hasattr(self,
                                                                          "ema200_slope_min_short") else slope_min_l
        atrp_min_s = float(self.atrp_min_short.value) if hasattr(self, "atrp_min_short") else atrp_min_l
        atrp_max_s = float(self.atrp_max_short.value) if hasattr(self, "atrp_max_short") else atrp_max_l

        gate_mode = str(self.htf_gate_mode.value).upper() if hasattr(self, "htf_gate_mode") else "ANY"  # "ANY"/"ALL"
        sb_mode = str(self.short_break_mode.value) if hasattr(self, "short_break_mode") else "st_or_dc"
        sp_gate = str(self.short_price_gate.value) if hasattr(self, "short_price_gate") else "ema_or_slope"

        # Relax the ST epsilon default for shorts (friendlier than 0.10/0.30)
        st_eps_k = float(self.st_break_eps_atr.value) if hasattr(self, "st_break_eps_atr") else 0.05

        # Donchian ATR buffers (absolute price distances)
        dc_buf_abs_long = buf_long * df["atr"]
        dc_buf_abs_short = buf_short * df["atr"]

        # =================
        # LONG conditions (unchanged)
        # =================
        long_cond = [
            df["dc_upper_prev"].notna(),
            df["high"] > (df["dc_upper_prev"] + dc_buf_abs_long),
            ((df["st_trend"] == 1) | (df["close"] > df["st_line"])),  # LTF ST alignment
            (df["close"] > df["ema200"]),
            (df["ema200_slope"] > slope_min_l),
            (df["adx"] >= adx_min_l),
            df["atrp"].between(atrp_min_l, atrp_max_l),
            df["volume"] >= (v_k * df["vol_ma"]),
        ]

        # HTF gates for LONG: strict ALL
        htf_long = []
        if use_htf_ema:
            htf_long.append(df["ema200_slope_htf"] > 0)
        if use_htf_st:
            htf_long.append(df["st_trend_htf"] == 1)
        if htf_long:
            long_cond.append(reduce(np.logical_and, htf_long))

        mask_l = reduce(np.logical_and, long_cond)
        df.loc[mask_l, "enter_long"] = 1
        df.loc[mask_l, "enter_tag"] = "DC_ST_breakout_long"

        # =================
        # SHORT conditions (loosened)
        # =================
        # Short-side volume MA & factor (more permissive in liquidity fades)
        vlen_s = int(getattr(self, "vol_ma_len_short", self.vol_ma_len).value) if hasattr(self,
                                                                                          "vol_ma_len_short") else int(
            self.vol_ma_len.value)
        df["vol_ma_s"] = df["volume"].rolling(vlen_s, min_periods=1).mean()
        v_k_s = float(getattr(self, "vol_k_short", self.vol_k).value) if hasattr(self, "vol_k_short") else v_k

        # Break condition for shorts:
        # "donchian": classic lower-band break with buffer
        # "st_or_dc": allow either Donchian break OR an ST line break with a small ATR epsilon below it
        st_epsilon = st_eps_k * df["atr"]
        dc_break_s = df["dc_lower_prev"].notna() & (df["low"] < (df["dc_lower_prev"] - dc_buf_abs_short))
        st_break_s = ((df["st_trend"] == -1) | (df["close"] < (df["st_line"] - st_epsilon)))
        short_break = dc_break_s if (sb_mode == "donchian") else (dc_break_s | st_break_s)

        # Price regime gate for shorts:
        # "ema_only": require price < EMA200 (classic)
        # "ema_or_slope": allow if price < EMA200 OR EMA200 slope negative (helps during shallow bounces)
        if sp_gate == "ema_or_slope":
            price_regime_s = (df["close"] < df["ema200"]) | (df["ema200_slope"] < 0)
        else:
            price_regime_s = (df["close"] < df["ema200"])

        short_cond = [
            short_break,
            price_regime_s,
            (df["ema200_slope"] < -slope_min_s),
            (df["adx"] >= adx_min_s),
            df["atrp"].between(atrp_min_s, atrp_max_s),
            df["volume"] >= (v_k_s * df["vol_ma_s"]),
        ]

        # HTF gates for SHORT: ANY (default) or ALL
        htf_short = []
        if use_htf_ema:
            htf_short.append(df["ema200_slope_htf"] < 0)
        if use_htf_st:
            htf_short.append(df["st_trend_htf"] == -1)
        if htf_short:
            comb = np.logical_or if gate_mode == "ANY" else np.logical_and
            short_cond.append(reduce(comb, htf_short))

        mask_s = reduce(np.logical_and, short_cond)
        df.loc[mask_s, "enter_short"] = 1
        df.loc[mask_s, "enter_tag"] = "DC_ST_breakout_short"

        return df

    # -----------------------
    # Exit helpers (optional)
    # -----------------------
    def populate_exit_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_short"] = 0
        if bool(self.exit_ema_cross.value):
            cross_dn = ((df["close"] < df["ema20"]) | (df["close"] < df["ema50"]))
            cross_up = ((df["close"] > df["ema20"]) | (df["close"] > df["ema50"]))
            df.loc[cross_dn, ["exit_long", "exit_tag"]] = (1, "exit_ema_cross")
            df.loc[cross_up, ["exit_short", "exit_tag"]] = (1, "exit_ema_cross")

        if bool(self.exit_atr_exhaust.value):
            med = df["atr_pct"].rolling(100, min_periods=25).median()
            thr = float(self.atr_exhaust_mult.value)
            atr_collapse = df["atr_pct"] < (thr * med)
            df.loc[atr_collapse, ["exit_long", "exit_tag"]] = (1, "exit_atr_exhaust")
            df.loc[atr_collapse, ["exit_short", "exit_tag"]] = (1, "exit_atr_exhaust")
        return df

    # -----------------------
    # Custom Stoploss (ATR-based + breakeven clamp + chandelier Oracle-style)
    # -----------------------
    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        lev = float(getattr(trade, "leverage", 1.0) or 1.0)
        dynamic_dist = float(self._dynamic_stop_pct(df, float(current_rate), lev))
        min_floor = float(self.min_sl_floor.value)

        # Activation: profit >= k * ATR%  OR after N minutes
        trade_duration_mins = self._trade_duration_in_mins(trade, current_time)
        atr_now = float(df["atr"].iat[-1]) if not df.empty else 0.0
        atr_pct_now = atr_now / max(1e-9, float(current_rate))
        atr_trigger = float(self.activate_after_atr_mult.value) * max(1e-9, atr_pct_now)
        activated = (current_profit is not None and current_profit >= atr_trigger) or \
                    (0 < int(self.arm_max_minutes.value) <= trade_duration_mins)

        # Breakeven lock once profit clears trigger
        lock_trig = float(self.profit_lock_trigger.value)
        lock_to_be = (current_profit is not None) and (current_profit >= lock_trig)
        try:
            entry = float(trade.open_rate)
            price = float(current_rate)
            be_dist = ((price - entry) / max(1e-9, price)) if trade.is_long else ((entry - price) / max(1e-9, price))
            be_dist = max(0.0, be_dist)
        except Exception:
            be_dist = 0.0

        if not activated:
            early = max(min_floor, dynamic_dist)
            if lock_to_be: early = max(min_floor, min(early, be_dist + 1e-6))
            return -float(early)

        # After activation: chandelier trail
        ce_dist = float("inf")
        try:
            k = float(self.trail_chandelier_mult.value)
            atr = float(df["atr"].iat[-1])
            since_entry = df.loc[df.index >= trade.date_entry_fill_utc]
            price = float(current_rate)
            if k > 0 and atr > 0 and not since_entry.empty:
                if trade.is_long:
                    hh = float(since_entry["high"].max())
                    stop_px = hh - k * atr
                    ce_dist = max(1e-6, (price - stop_px) / max(1e-9, price))
                else:
                    ll = float(since_entry["low"].min())
                    stop_px = ll + k * atr
                    ce_dist = max(1e-6, (stop_px - price) / max(1e-9, price))
        except Exception:
            pass

        final_dist = max(min_floor, min(dynamic_dist, ce_dist))
        if lock_to_be:
            final_dist = max(min_floor, min(final_dist, be_dist + 1e-6))
        return -float(final_dist)

    # -----------------------
    # Custom ROI (ATR-based floor + excursion trail + time-decay)
    # -----------------------
    def custom_roi(self, pair: str, trade: Any, current_time: datetime,
                   trade_duration: int, entry_tag: str, side: str, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        min_floor = float(self.min_roi_floor.value)
        roi_k_entry = float(self.roi_atr_mult.value)
        roi_k_trail = float(self.trail_roi_mult.value)
        cap = float(self.max_roi_cap.value)

        # Entry ATR% (nearest to fill)
        try:
            idx = df.index.get_indexer([trade.date_entry_fill_utc], method="nearest")[0]
            entry_atr = float(df["atr"].iloc[idx])
            entry_price = float(trade.open_rate)
            entry_atr_pct = entry_atr / max(1e-9, entry_price)
        except Exception:
            entry_atr_pct = float(df["atr"].iat[-1]) / max(1e-9, float(df["close"].iat[-1]))

        roi_floor = max(min_floor, roi_k_entry * entry_atr_pct)

        # Trailing ROI based on favorable excursion minus ATR cushion
        try:
            since_entry = df.loc[df.index >= trade.date_entry_fill_utc]
            atr = float(since_entry["atr"].iloc[-1])
            if side == "long":
                highest = float(since_entry["high"].max())
                trail = max(min_floor, (highest - trade.open_rate) / trade.open_rate
                            - roi_k_trail * atr / max(1e-9, highest))
            else:
                lowest = float(since_entry["low"].min())
                trail = max(min_floor, (trade.open_rate - lowest) / trade.open_rate
                            - roi_k_trail * atr / max(1e-9, lowest))
        except Exception:
            trail = min_floor

        roi_target = max(roi_floor, trail)

        # Time-decay asks for a little more after long holds
        if trade_duration is not None and trade_duration >= int(self.time_decay_minutes.value):
            roi_target += float(self.time_decay_add.value)

        return float(min(cap, roi_target))

    # -----------------------
    # INT leverage (1–4x), Oracle-style signal/liquidity shaping
    # -----------------------
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

    # -----------------------
    # Position sizing (stake) – equity×risk, SL & leverage aware, cap at 10% wallet
    # -----------------------
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
        # Use distance vs Supertrend/Donchian edges and ADX to approximate "edge"
        adx_now = float(df["adx"].iat[-1])
        adx_nrm = min(1.0, max(0.0, (adx_now - 10.0) / 30.0))
        if side == "long":
            edge = float(df["high"].iat[-1] - df["dc_upper_prev"].iat[-1]) / max(1e-9, float(df["high"].iat[-1]))
        else:
            edge = float(df["dc_lower_prev"].iat[-1] - df["low"].iat[-1]) / max(1e-9, float(df["low"].iat[-1]))
        edge = max(0.0, edge)
        return float(min(1.0, 0.6 * adx_nrm + 0.4 * edge))

    def _liquidity_factor(self, df: DataFrame) -> float:
        v = float(df["volume"].iat[-1])
        vm = max(1e-9, float(df["vol_ma"].iat[-1]))
        ratio = v / vm
        if ratio >= 1.0: return 1.0
        cut = float(self.vol_weak_cut.value)
        if ratio <= cut: return 0.60
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
        min_floor = float(self.min_sl_floor.value)

        final_stop = max(min_floor, min(max(1e-6, atr_stop), liq_safe_stop, sl_cap, hard_cap))
        if not (final_stop == final_stop) or final_stop <= 0:
            final_stop = max(min_floor, min(hard_cap, sl_cap))
        return float(final_stop)

    def _trade_duration_in_mins(self, trade, current_time: datetime) -> int:
        try:
            return int((current_time - trade.date_entry_fill_utc).total_seconds() // 60)
        except Exception:
            return 0

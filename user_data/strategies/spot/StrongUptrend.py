# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime
from functools import reduce
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import (
    IStrategy,
    Trade,
    IntParameter,
    BooleanParameter
)


class StrongUptrend(IStrategy):
    """
    Strong Uptrend Strategy - 4H Timeframe (Swing Trading)
    Optimized to reduce noise and false signals on a 4-hour timeframe.
    """

    INTERFACE_VERSION = 3
    timeframe = '4h'
    minimal_roi = {
        "0": 0.12
    }

    stoploss = -0.34

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True


    exit_profit_only = True
    max_open_trades = 5

    startup_candle_count = 100

    # Hyperopt parameters for tuning
    ema_fast_period = IntParameter(8, 20, default=12, space='buy')
    ema_medium_period = IntParameter(21, 35, default=26, space='buy')
    ema_long_period = IntParameter(40, 70, default=50, space='buy')

    adx_period = IntParameter(14, 21, default=14, space='buy')
    adx_threshold = IntParameter(25, 40, default=30, space='buy')

    rsi_period = IntParameter(14, 21, default=14, space='buy')
    rsi_buy_threshold = IntParameter(50, 65, default=55, space='buy')

    macd_fast = IntParameter(12, 18, default=12, space='buy')
    macd_slow = IntParameter(26, 35, default=26, space='buy')
    macd_signal = IntParameter(9, 12, default=9, space='buy')

    volume_check = BooleanParameter(default=True, space='buy')
    confirmation_candles = IntParameter(1, 3, default=2, space='buy')

    # Buy hyperspace params:
    buy_params = {
        "adx_period": 15,
        "adx_threshold": 39,
        "confirmation_candles": 3,
        "ema_fast_period": 12,
        "ema_long_period": 40,
        "ema_medium_period": 30,
        "macd_fast": 15,
        "macd_signal": 11,
        "macd_slow": 34,
        "rsi_buy_threshold": 50,
        "rsi_period": 21,
        "volume_check": False,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs
        periods = set()
        periods.update(range(self.ema_fast_period.low, self.ema_fast_period.high + 1))
        periods.update(range(self.ema_medium_period.low, self.ema_medium_period.high + 1))
        periods.update(range(self.ema_long_period.low, self.ema_long_period.high + 1))
        periods.update([100, 200])
        for period in periods:
            dataframe[f'ema_{period}'] = ta.EMA(dataframe, timeperiod=period)

        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period.value)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        macd = ta.MACD(dataframe,
                       fastperiod=self.macd_fast.value,
                       slowperiod=self.macd_slow.value,
                       signalperiod=self.macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['volume_mean_20'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_mean_50'] = dataframe['volume'].rolling(window=50).mean()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        dataframe['close_delta_4'] = dataframe['close'].pct_change(4)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (dataframe[f'ema_{self.ema_fast_period.value}'] > dataframe[f'ema_{self.ema_medium_period.value}']) &
            (dataframe[f'ema_{self.ema_medium_period.value}'] > dataframe[f'ema_{self.ema_long_period.value}'])
        )

        conditions.append(dataframe['close'] > dataframe['ema_200'])

        conditions.append(
            (dataframe[f'ema_{self.ema_fast_period.value}'] > dataframe[f'ema_{self.ema_fast_period.value}'].shift(2)) &
            (dataframe[f'ema_{self.ema_medium_period.value}'] > dataframe[f'ema_{self.ema_medium_period.value}'].shift(
                2)) &
            (dataframe[f'ema_{self.ema_long_period.value}'] > dataframe[f'ema_{self.ema_long_period.value}'].shift(1))
        )

        conditions.append(dataframe['adx'] > self.adx_threshold.value)
        conditions.append(dataframe['adx'] > dataframe['adx'].shift(1))
        conditions.append(dataframe['plus_di'] > dataframe['minus_di'] * 1.2)
        conditions.append(
            (dataframe['rsi'] > self.rsi_buy_threshold.value) & (dataframe['rsi'] < 75)
        )
        conditions.append(dataframe['rsi'] > dataframe['rsi'].shift(1))
        conditions.append(
            (dataframe['macd'] > dataframe['macdsignal']) & (dataframe['macd'] > 0)
        )
        conditions.append(
            (dataframe['macdhist'] > 0) & (dataframe['macdhist'] > dataframe['macdhist'].shift(1))
        )
        conditions.append(dataframe['close'] > dataframe['bb_middleband'])
        conditions.append(dataframe['bb_width'] > dataframe['bb_width'].shift(1))

        if self.volume_check.value:
            conditions.append(dataframe['volume'] > dataframe['volume_mean_20'])
        conditions.append(dataframe['close_delta_4'] > 0.02)
        conditions.append(dataframe['close'] > dataframe['open'])

        if self.confirmation_candles.value > 1:
            for i in range(1, self.confirmation_candles.value):
                conditions.append(
                    dataframe[f'ema_{self.ema_fast_period.value}'].shift(i) >
                    dataframe[f'ema_{self.ema_medium_period.value}'].shift(i)
                )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_tag'
            ] = 'strong_4h_uptrend'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (dataframe[f'ema_{self.ema_fast_period.value}'] < dataframe[f'ema_{self.ema_medium_period.value}']) &
            (dataframe[f'ema_{self.ema_fast_period.value}'].shift(1) >= dataframe[
                f'ema_{self.ema_medium_period.value}'].shift(1))
        )
        conditions.append(
            (dataframe['adx'] < self.adx_threshold.value - 10) &
            (dataframe['adx'] < dataframe['adx'].shift(1)) &
            (dataframe['adx'].shift(1) < dataframe['adx'].shift(2))
        )
        conditions.append(
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macdhist'] < 0) &
            (dataframe['macdhist'] < dataframe['macdhist'].shift(1))
        )
        conditions.append(
            (dataframe['rsi'] > 75) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
            (dataframe['rsi'].shift(1) < dataframe['rsi'].shift(2))
        )
        conditions.append(
            (dataframe['minus_di'] > dataframe['plus_di']) &
            (dataframe['minus_di'].shift(1) <= dataframe['plus_di'].shift(1))
        )
        conditions.append(
            (dataframe['close'] < dataframe['ema_50']) &
            (dataframe['close'].shift(1) >= dataframe['ema_50'].shift(1))
        )
        conditions.append(
            (dataframe['close'] < dataframe['open']) &
            ((dataframe['open'] - dataframe['close']) / dataframe['open'] > 0.03)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_tag'
            ] = '4h_exit_signal'

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return False
        last_candle = dataframe.iloc[-1]
        if last_candle['close'] < last_candle['open']:
            candle_drop = (last_candle['open'] - last_candle['close']) / last_candle['open']
            if candle_drop > 0.05:
                return False
        return True

    def confirm_trade_exit(
            self,
            pair: str,
            trade: Trade,
            order_type: str,
            amount: float,
            rate: float,
            time_in_force: str,
            exit_reason: str,
            current_time: datetime,
            **kwargs,
    ) -> bool:
        """
        Called before a trade exit order is placed.
        Return False to abort exit.
        """
        # Block force exit if trade is at loss
        if exit_reason == "force_sell" and trade.calc_profit_ratio(rate) < 0:
            return False  # Prevent exit order placement

        # Allow all other exits
        return True

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
from functools import reduce
from datetime import datetime
import talib.abstract as ta

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.exchange import timeframe_to_prev_date

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

"""
Keltner Channel and RSI Trend Trading System

Buy/Long Entry: 
    - The RSI crosses above 50 ...
    - Enter a long trade after the first candle that closes above top band of the Keltner channel
    - Set a stop loss under the low of the first candle that close above the KC  
    - You can also enter a long trade when price touches or tests back into the Keltner channel if the RSI remains above 50

Hanson, David. Bitcoin Trading Strategies: Algorithmic Trading Strategies For Bitcoin And Cryptocurrency That Work (p. 52). Kindle Edition. 

退出统一中轨(避免当前规则与第三点冲突) n 不需要
前一根close低于前三根最低点卖出 x 回测效果差
close在轨道外 且 low大于中规 且 RSI符合条件 -->继续买入 y
大于4小时收益低于3%离场 y
dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe) x 加入后回测结果略降
使用ema_kc y
使用自定义跟随止损 x 无法确定跟随止损值

base on KCRSI_m3_1h
加入macd辅助排除假信号
限制RSI范围 避免假突破 x 差别不大
"""

class KCRSIMACD_1h(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = True

    # Buy hyperspace params:
    buy_params = {
     'rsi_low': 51, 'rsi_high': 59
    }

    # Sell hyperspace params:
    sell_params = {
     'rsi_low': 41, 'rsi_high': 49
    }

    # ROI table:
    minimal_roi = {
        "0":10
    }

    # Stoploss:
    stoploss = -0.15

    use_custom_stoploss=  False

    # Trailing stop:
    trailing_stop =  True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.15
    trailing_only_offset_is_reached = True
    
    timeframe='1h'
    startup_candle_count = 50

    def numpy_rolling_window(self,data, window):
        shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
        strides = data.strides + (data.strides[-1],)
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


    def numpy_rolling_series(func):
        def func_wrapper(self,data, window, as_source=False):
            series = data.values if isinstance(data, pd.Series) else data

            new_series = np.empty(len(series)) * np.nan
            calculated = func(self,series, window)
            new_series[-len(calculated):] = calculated

            if as_source and isinstance(data, pd.Series):
                return pd.Series(index=data.index, data=new_series)

            return new_series

        return func_wrapper


    @numpy_rolling_series
    def numpy_rolling_mean(self,data, window, as_source=False):
        return np.mean(self.numpy_rolling_window(data, window), axis=-1)

    def rolling_mean(self,series, window=200, min_periods=None):
        min_periods = window if min_periods is None else min_periods
        if min_periods == window and len(series) > window:
            return self.numpy_rolling_mean(series, window,True)
        else:
            try:
                return series.rolling(window=window, min_periods=min_periods).mean()
            except Exception as e:  # noqa: F841
                return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()
    
    def true_range(self,bars):
        return pd.DataFrame({
            "hl": bars['high'] - bars['low'],
            "hc": abs(bars['high'] - bars['close'].shift(1)),
            "lc": abs(bars['low'] - bars['close'].shift(1))
        }).max(axis=1)
    
    def atr(self,bars, window=14):
        tr = self.true_range(bars)

        res = self.rolling_mean(tr, window)

        return pd.Series(res)

    def ema_kc(self,bars,ema_window,window=14, atrs=2):
        typical_mean = self.rolling_mean(bars[f'ema{ema_window}'], window)
        atrval = self.atr(bars, window) * atrs

        upper = typical_mean + atrval
        lower = typical_mean - atrval

        return pd.DataFrame(index=bars.index, data={
            'upper': upper.values,
            'mid': typical_mean.values,
            'lower': lower.values
        })

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=35)
        dataframe['ema24'] = ta.EMA(dataframe, timeperiod=24)
        kc = self.ema_kc(dataframe,ema_window=24,window=40, atrs=0.5)
        # kc = qtpylib.keltner_channel(dataframe, window=35, atrs=0.5)
        dataframe['kc_upper'] = kc['upper']
        dataframe['kc_lower'] = kc['lower']
        dataframe['kc_mid'] = kc['mid']

        macd = ta.MACD(dataframe,fast = 12,slow = 24, smooth=12)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # LONG
        dataframe.loc[(
                (dataframe['macd'] > 0) &
                (dataframe['rsi'] > self.buy_params['rsi_low']) &
                (qtpylib.crossed_above(dataframe['close'], dataframe['kc_upper']))
            ), 
            ['enter_long','enter_tag']] = (1,'closs cross')
        
        dataframe.loc[(
                (dataframe['macd'] > 0) &
                (dataframe['rsi'] > self.buy_params['rsi_low']) &
                (dataframe['close'] > dataframe['kc_upper']) &
                (qtpylib.crossed_below(dataframe['low'], dataframe['kc_upper']))
            ), 
            ['enter_long','enter_tag']] = (1,'touch kc')
        
        # SHORT
        dataframe.loc[(
                (dataframe['macd'] < 0) &
                (dataframe['rsi'] < self.sell_params['rsi_high']) &
                (qtpylib.crossed_below(dataframe['close'], dataframe['kc_lower']))
            ), 
            ['enter_short','enter_tag']] = (1,'closs cross')
        
        dataframe.loc[(
                (dataframe['macd'] < 0) &
                (dataframe['rsi'] < self.sell_params['rsi_high']) &
                (dataframe['close'] < dataframe['kc_lower']) &
                (qtpylib.crossed_above(dataframe['high'], dataframe['kc_lower']))
            ), 
            ['enter_short','enter_tag']] = (1,'touch kc')
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        return dataframe

    def leverage(self, pair: str, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 5.0
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if (current_time - trade.open_date_utc).seconds/3600 > 4 and current_profit < 0.03:
            return 'below 3% 4h'
        if (current_time - trade.open_date_utc).seconds/3600 > 8 and current_profit < 0.1:
            return 'below 10% 8h'
        if (current_time - trade.open_date_utc).seconds/3600 > 12 and current_profit < 0.15:
            return 'below 15% 12h'

        # if trade.enter_tag == 'pattern' and current_profit < -0.01:
        #     return 'pattern invalid'
        
        if trade.is_short:
            if (current_time - trade.open_date_utc).seconds/3600 < 3 and  trade.enter_tag == 'closs cross':
                if current_rate>last_candle['kc_mid']:
                    return 'hit kc mid'
            else:
                if current_rate>last_candle['kc_lower']:
                    return 'hit kc lower'
        else:
            if (current_time - trade.open_date_utc).seconds/3600 < 3 and  trade.enter_tag == 'closs cross':
                if current_rate<last_candle['kc_mid']:
                    return 'hit kc mid'
            else:
                if current_rate<last_candle['kc_upper']:
                    return 'hit kc upper'


    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
        
    #             # evaluate highest to lowest, so that highest possible stop is used
    #     if 0.2 > current_profit >= 0.15:
    #         return 0.01
    #     elif 0.25> current_profit >= 0.2:
    #         return 0.05
    #     elif current_profit > 0.25:
    #         return 0.08

    #     # return maximum stoploss value, keeping current stoploss price unchanged
    #     return 0.25

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1
            }
        ]
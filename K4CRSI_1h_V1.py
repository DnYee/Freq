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
                                IStrategy, IntParameter,stoploss_from_absolute)
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

base on KCRSIMACD_1h
删除macd辅助
限制RSI范围 避免假突破 x 差别不大
使用4个KC判断止损止盈
删除触碰外轨买入
统一使用普通kc
入场信号穿越第二kc以第一kc外轨止损

20230416:
close超过2kc不进场
止损只设定为mid，超过2kc设定为1kc
"""

class K4CRSI_1h_V1(IStrategy):
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
        "0":0.99
    }

    # Stoploss:
    stoploss = -0.99

    use_custom_stoploss = True

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    
    timeframe='1h'
    startup_candle_count = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=12)

        kc_0 = qtpylib.keltner_channel(dataframe, window=36, atrs=0.5)
        dataframe['kc_0_upper'] = kc_0['upper']
        dataframe['kc_0_lower'] = kc_0['lower']
        dataframe['kc_0_mid'] = kc_0['mid']

        kc_2 = qtpylib.keltner_channel(dataframe, window=36, atrs=2.36)
        dataframe['kc_2_upper'] = kc_2['upper']
        dataframe['kc_2_lower'] = kc_2['lower']

        kc_3 = qtpylib.keltner_channel(dataframe, window=36, atrs=3.28)
        dataframe['kc_3_upper'] = kc_3['upper']
        dataframe['kc_3_lower'] = kc_3['lower']

        kc_5 = qtpylib.keltner_channel(dataframe, window=36, atrs=5)
        dataframe['kc_5_upper'] = kc_5['upper']
        dataframe['kc_5_lower'] = kc_5['lower']

        kc_6 = qtpylib.keltner_channel(dataframe, window=36, atrs=6.18)
        dataframe['kc_6_upper'] = kc_6['upper']
        dataframe['kc_6_lower'] = kc_6['lower']

        kc_10 = qtpylib.keltner_channel(dataframe, window=36, atrs=10)
        dataframe['kc_10_upper'] = kc_10['upper']
        dataframe['kc_10_lower'] = kc_10['lower']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # LONG
        dataframe.loc[(
                (dataframe['rsi'] > self.buy_params['rsi_low']) &
                (dataframe['close'] < dataframe['kc_2_upper']) &
                (qtpylib.crossed_above(dataframe['close'], dataframe['kc_0_upper']))
            ), 
            ['enter_long','enter_tag']] = (1,'closs cross')
        
        
        # SHORT
        dataframe.loc[(
                (dataframe['rsi'] < self.sell_params['rsi_high']) &
                (dataframe['close'] > dataframe['kc_2_lower']) &
                (qtpylib.crossed_below(dataframe['close'], dataframe['kc_0_lower']))
            ), 
            ['enter_short','enter_tag']] = (1,'closs cross')
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(
                (qtpylib.crossed_above(dataframe['high'], dataframe['kc_10_upper']))
            ), 
            ['exit_long','exit_tag']] = (1,'cross kc_10 upper')
        
        dataframe.loc[(
                (qtpylib.crossed_below(dataframe['low'], dataframe['kc_10_lower']))
            ), 
            ['exit_short','exit_tag']] = (1,'cross kc_10 lower')

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
        return 1.0
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        
        if trade.is_short and current_rate>last_candle['kc_0_mid']:
            return 'hit kc_0 mid'
        elif current_rate<last_candle['kc_0_mid']:
            return 'hit kc_0 mid'


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        
        # evaluate highest to lowest, so that highest possible stop is used
        if trade.is_short:
            if current_rate <= last_candle['kc_6_lower']:
                return stoploss_from_absolute(last_candle['kc_5_lower'], current_rate, is_short=trade.is_short)
            elif current_rate <= last_candle['kc_3_lower']:
                return stoploss_from_absolute(last_candle['kc_2_lower'], current_rate, is_short=trade.is_short)
            elif current_rate <= last_candle['kc_2_lower']:
                return stoploss_from_absolute(last_candle['kc_0_lower'], current_rate, is_short=trade.is_short)
            
        else:
            if current_rate >= last_candle['kc_6_upper']:
                return stoploss_from_absolute(last_candle['kc_5_upper'], current_rate, is_short=trade.is_short)
            elif current_rate >= last_candle['kc_3_upper']:
                return stoploss_from_absolute(last_candle['kc_2_upper'], current_rate, is_short=trade.is_short)
            elif current_rate >= last_candle['kc_2_upper']:
                return stoploss_from_absolute(last_candle['kc_0_upper'], current_rate, is_short=trade.is_short)

        # return maximum stoploss value, keeping current stoploss price unchanged
        return 0.99

    # @property
    # def protections(self):
    #     return [
    #         {
    #             "method": "CooldownPeriod",
    #             "stop_duration_candles": 1
    #         }
    #     ]
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import logging
from typing import Dict

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
from functools import reduce
from datetime import datetime
import talib.abstract as ta

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter,stoploss_from_absolute,informative)
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
止损只设定为mid,超过2kc设定为1kc

20230423:
加入模型判断信号可信度
加入趋势判断
加入BTC作为辅助指标
加入前10根K线穿越方向
"""

class AI_K4CRSI_V1(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = True

    timeframe = '15m'
    inf_timeframe = '1h'

    # Stoploss:
    stoploss = -0.99
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True

    startup_candle_count = 50
    process_only_new_candles = True

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

    def target_num(self,x):

      lo_idx = [99999]
      up_idx = [99999]
      for i,x_ in enumerate(x):
        if x_ > 0:
          up_idx.append(i)
        elif x_ < 0:
          lo_idx.append(i)
    
      if min(lo_idx)<min(up_idx):
        return -1
      elif min(lo_idx)>min(up_idx):
        return 1
      else :
        return 0

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-ema"] = ta.EMA(dataframe, timeperiod=period)

        dataframe["%-pct-change"] = dataframe["close"].pct_change(periods=period)
        dataframe["%-raw_volume"] = dataframe["volume"].pct_change(periods=period)

        dataframe['%-close_max'] = dataframe['close'].rolling(period).max()
        dataframe['%-close_min'] = dataframe['close'].rolling(period).min()

        dataframe['%-pct_close_max'] = (dataframe['%-close_max'] - dataframe['close']) / dataframe['close']
        dataframe['%-pct_close_min'] = (dataframe['%-close_min'] - dataframe['close']) / dataframe['close']

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        kc_0 = qtpylib.keltner_channel(dataframe, window=36, atrs=0.5)
        dataframe['%-kc_0_mid'] = kc_0['mid']
        dataframe['%-kc_0_upper'] = kc_0['upper']
        dataframe['%-kc_0_lower'] = kc_0['lower']

        dataframe['%-crs_ab_kc_up1'] = np.where((dataframe['close'] > dataframe['%-kc_0_upper']) & (dataframe['close'].shift(1) < dataframe['%-kc_0_upper'].shift(1)),1,0)
        dataframe['%-crs_ab_kc_up2'] = np.where((dataframe['close'].shift(1) > dataframe['%-kc_0_upper'].shift(1)) & (dataframe['close'].shift(2) < dataframe['%-kc_0_upper'].shift(2)),1,0)
        dataframe['%-crs_ab_kc_up3'] = np.where((dataframe['close'].shift(2) > dataframe['%-kc_0_upper'].shift(2)) & (dataframe['close'].shift(3) < dataframe['%-kc_0_upper'].shift(3)),1,0)
        dataframe['%-crs_ab_kc_up4'] = np.where((dataframe['close'].shift(3) > dataframe['%-kc_0_upper'].shift(3)) & (dataframe['close'].shift(4) < dataframe['%-kc_0_upper'].shift(4)),1,0)
        dataframe['%-crs_ab_kc_up5'] = np.where((dataframe['close'].shift(4) > dataframe['%-kc_0_upper'].shift(4)) & (dataframe['close'].shift(5) < dataframe['%-kc_0_upper'].shift(5)),1,0)

        dataframe['%-crs_bl_kc_lo1'] = np.where((dataframe['close'] < dataframe['%-kc_0_lower']) & (dataframe['close'].shift(1) > dataframe['%-kc_0_lower'].shift(1)),1,0)
        dataframe['%-crs_bl_kc_lo2'] = np.where((dataframe['close'].shift(1) < dataframe['%-kc_0_lower'].shift(1)) & (dataframe['close'].shift(2) > dataframe['%-kc_0_lower'].shift(2)),1,0)
        dataframe['%-crs_bl_kc_lo3'] = np.where((dataframe['close'].shift(2) < dataframe['%-kc_0_lower'].shift(2)) & (dataframe['close'].shift(3) > dataframe['%-kc_0_lower'].shift(3)),1,0)
        dataframe['%-crs_bl_kc_lo4'] = np.where((dataframe['close'].shift(3) < dataframe['%-kc_0_lower'].shift(3)) & (dataframe['close'].shift(4) > dataframe['%-kc_0_lower'].shift(4)),1,0)
        dataframe['%-crs_bl_kc_lo5'] = np.where((dataframe['close'].shift(4) < dataframe['%-kc_0_lower'].shift(4)) & (dataframe['close'].shift(5) > dataframe['%-kc_0_lower'].shift(5)),1,0)

        dataframe['%-crs_ab_kc_md1'] = np.where((dataframe['close'] > dataframe['%-kc_0_mid']) & (dataframe['close'].shift(1) < dataframe['%-kc_0_mid'].shift(1)),1,0)
        dataframe['%-crs_ab_kc_md2'] = np.where((dataframe['close'].shift(1) > dataframe['%-kc_0_mid'].shift(1)) & (dataframe['close'].shift(2) < dataframe['%-kc_0_mid'].shift(2)),1,0)
        dataframe['%-crs_ab_kc_md3'] = np.where((dataframe['close'].shift(2) > dataframe['%-kc_0_mid'].shift(2)) & (dataframe['close'].shift(3) < dataframe['%-kc_0_mid'].shift(3)),1,0)
        dataframe['%-crs_ab_kc_md4'] = np.where((dataframe['close'].shift(3) > dataframe['%-kc_0_mid'].shift(3)) & (dataframe['close'].shift(4) < dataframe['%-kc_0_mid'].shift(4)),1,0)
        dataframe['%-crs_ab_kc_md5'] = np.where((dataframe['close'].shift(4) > dataframe['%-kc_0_mid'].shift(4)) & (dataframe['close'].shift(5) < dataframe['%-kc_0_mid'].shift(5)),1,0)

        dataframe['%-crs_bl_kc_md1'] = np.where((dataframe['close'] < dataframe['%-kc_0_mid']) & (dataframe['close'].shift(1) > dataframe['%-kc_0_mid'].shift(1)),1,0)
        dataframe['%-crs_bl_kc_md2'] = np.where((dataframe['close'].shift(1) < dataframe['%-kc_0_mid'].shift(1)) & (dataframe['close'].shift(2) > dataframe['%-kc_0_mid'].shift(2)),1,0)
        dataframe['%-crs_bl_kc_md3'] = np.where((dataframe['close'].shift(2) < dataframe['%-kc_0_mid'].shift(2)) & (dataframe['close'].shift(3) > dataframe['%-kc_0_mid'].shift(3)),1,0)
        dataframe['%-crs_bl_kc_md4'] = np.where((dataframe['close'].shift(3) < dataframe['%-kc_0_mid'].shift(3)) & (dataframe['close'].shift(4) > dataframe['%-kc_0_mid'].shift(4)),1,0)
        dataframe['%-crs_bl_kc_md5'] = np.where((dataframe['close'].shift(4) < dataframe['%-kc_0_mid'].shift(4)) & (dataframe['close'].shift(5) > dataframe['%-kc_0_mid'].shift(5)),1,0)

        # Downtrend checks
        dataframe['%-is_downtrend_3'] = np.where((dataframe['close'] < dataframe['open']) & (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & (dataframe['close'].shift(2) < dataframe['open'].shift(2)),1,0)
        dataframe['%-is_downtrend_5'] = np.where((dataframe['close'] < dataframe['open']) & (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & (dataframe['close'].shift(2) < dataframe['open'].shift(2)) & (dataframe['close'].shift(3) < dataframe['open'].shift(3)) & (dataframe['close'].shift(4) < dataframe['open'].shift(4)),1,0)
        # Uptrend checks
        dataframe['%-is_uptrend_3'] = np.where((dataframe['close'] > dataframe['open']) & (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & (dataframe['close'].shift(2) > dataframe['open'].shift(2)),1,0)
        dataframe['%-is_uptrend_5'] = np.where((dataframe['close'] > dataframe['open']) & (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & (dataframe['close'].shift(2) > dataframe['open'].shift(2)) & (dataframe['close'].shift(3) > dataframe['open'].shift(3)) & (dataframe['close'].shift(4) > dataframe['open'].shift(4)),1,0)

        # Volume
        dataframe['%-vol_mean_factor_6'] = dataframe['volume'] / dataframe['volume'].rolling(6).mean()

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        # dataframe['&s-up_or_down'] = np.where(dataframe["close"].shift(-50) >
        #                                       dataframe["close"], 'up', 'down')

        # type 1 
        # cross_upper = ((dataframe["close"] < dataframe["kc_2_upper_1h"]) &
        #                     (dataframe["close"] >= dataframe["kc_0_mid_1h"]) & 
        #                     (((dataframe["close"] > dataframe["kc_2_upper_1h"]).rolling(40).max()==1) & 
        #                     ((dataframe["close"] < dataframe["kc_0_mid_1h"]).rolling(40).max()==0)).shift(-40))

        # cross_lower = ((dataframe["close"] > dataframe["kc_2_lower_1h"]) & 
        #                     (dataframe["close"] <= dataframe["kc_0_mid_1h"]) & 
        #                     (((dataframe["close"] < dataframe["kc_2_lower_1h"]).rolling(40).max()==1) & 
        #                     ((dataframe["close"] > dataframe["kc_0_mid_1h"]).rolling(40).max()==0)).shift(-40))


        # dataframe['&s-cross_or_stay'] = np.where((cross_upper | cross_lower ), 'cross', 'stay')

        # type 2
        # cross_upper = ((dataframe["close"] < dataframe["kc_2_upper_1h"]) &
        #                     (((dataframe["close"] > dataframe["kc_2_upper_1h"]).rolling(40).max()==1) & 
        #                     ((dataframe["close"] < dataframe["kc_0_lower_1h"]).rolling(40).max()==0)).shift(-40))

        # cross_lower = ((dataframe["close"] > dataframe["kc_2_lower_1h"]) & 
        #                     (((dataframe["close"] < dataframe["kc_2_lower_1h"]).rolling(40).max()==1) & 
        #                     ((dataframe["close"] > dataframe["kc_0_upper_1h"]).rolling(40).max()==0)).shift(-40))

        # dataframe['&s-cross_or_stay'] = np.where(cross_upper, 'up',np.where(cross_lower, 'down', 'stay'))

        # type 3 
        dataframe['up_or_lo'] = np.where((dataframe["close"] < dataframe["kc_2_lower_1h"]),-1,np.where((dataframe["close"] > dataframe["kc_2_upper_1h"]),1,0))
        dataframe['traget_num'] = dataframe['up_or_lo'].rolling(window=40).apply(self.target_num, raw=True).shift(-40)
        dataframe['&s-cross_or_stay'] = np.where((dataframe["close"] < dataframe["kc_2_upper_1h"]) & (dataframe['traget_num']==1), 'up',
                                                 np.where((dataframe["close"] > dataframe["kc_2_lower_1h"]) & (dataframe['traget_num']==-1), 'down', 'stay'))
        
        # print("ai:",dataframe)
        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        kc_0 = qtpylib.keltner_channel(dataframe, window=36, atrs=0.5)
        dataframe['kc_0_mid'] = kc_0['mid']
        dataframe['kc_0_upper'] = kc_0['upper']
        dataframe['kc_0_lower'] = kc_0['lower']

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

        dataframe["rsi_12"] = ta.RSI(dataframe, timeperiod=12)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # print("final 1 :",dataframe.columns.values)
        dataframe = self.freqai.start(dataframe, metadata, self)
        # print("final:",dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # pair = metadata["pair"].replace(":", "")

        # # LONG
        # dataframe.loc[((dataframe['&s-cross_or_stay'] == 'cross') &
        #         (dataframe['rsi_12_1h'] > self.buy_params['rsi_low']) &
        #         (dataframe['close'] < dataframe['kc_2_upper_1h']) &
        #         (qtpylib.crossed_above(dataframe['close'], dataframe['kc_0_upper_1h']))
        #     ), 
        #     ['enter_long','enter_tag']] = (1,'closs cross')
        
        
        # # SHORT
        # dataframe.loc[((dataframe['&s-cross_or_stay'] == 'cross') &
        #         (dataframe['rsi_12_1h'] < self.sell_params['rsi_high']) &
        #         (dataframe['close'] > dataframe['kc_2_lower_1h']) &
        #         (qtpylib.crossed_below(dataframe['close'], dataframe['kc_0_lower_1h']))
        #     ), 
        #     ['enter_short','enter_tag']] = (1,'closs cross')

        # LONG
        dataframe.loc[((dataframe['&s-cross_or_stay'] == 'up') &
                (dataframe['rsi_12_1h'] > self.buy_params['rsi_low']) &
                (qtpylib.crossed_above(dataframe['close'], dataframe['kc_0_upper_1h']))
            ), 
            ['enter_long','enter_tag']] = (1,'closs cross')
        
        
        # SHORT
        dataframe.loc[((dataframe['&s-cross_or_stay'] == 'down') &
                (dataframe['rsi_12_1h'] < self.sell_params['rsi_high']) &
                (qtpylib.crossed_below(dataframe['close'], dataframe['kc_0_lower_1h']))
            ), 
            ['enter_short','enter_tag']] = (1,'closs cross')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # pair = metadata["pair"].replace(":", "")

        dataframe.loc[(
                (qtpylib.crossed_above(dataframe['high'], dataframe['kc_10_upper_1h']))
            ), 
            ['exit_long','exit_tag']] = (1,'cross kc_10 upper')
        
        dataframe.loc[(
                (qtpylib.crossed_below(dataframe['low'], dataframe[f'kc_10_lower_1h']))
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
        
        # if trade.is_short and current_rate>last_candle['kc_0_mid_1h']:
        #     return 'hit kc_0 mid'
        # elif current_rate<last_candle['kc_0_mid_1h']:
        #     return 'hit kc_0 mid'

        # if trade.is_short and current_rate>last_candle['kc_0_upper_1h']:
        #     return 'hit kc_0 mid'
        # elif (not trade.is_short) and current_rate<last_candle['kc_0_lower_1h']:
        #     return 'hit kc_0 mid'

        if trade.is_short:
            if current_rate>last_candle['kc_0_upper_1h']:
                return 'hit kc_0 upper'
        else:
            if current_rate<last_candle['kc_0_lower_1h']:
                return 'hit kc_0 lower'


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        
        # evaluate highest to lowest, so that highest possible stop is used
        if trade.is_short:
            if current_rate <= last_candle['kc_6_lower_1h']:
                return stoploss_from_absolute(last_candle['kc_5_lower_1h'], current_rate, is_short=trade.is_short)
            elif current_rate <= last_candle['kc_3_lower_1h']:
                return stoploss_from_absolute(last_candle['kc_2_lower_1h'], current_rate, is_short=trade.is_short)
            elif current_rate <= last_candle['kc_2_lower_1h']:
                return stoploss_from_absolute(last_candle['kc_0_lower_1h'], current_rate, is_short=trade.is_short)
            
        else:
            if current_rate >= last_candle['kc_6_upper_1h']:
                return stoploss_from_absolute(last_candle['kc_5_upper_1h'], current_rate, is_short=trade.is_short)
            elif current_rate >= last_candle['kc_3_upper_1h']:
                return stoploss_from_absolute(last_candle['kc_2_upper_1h'], current_rate, is_short=trade.is_short)
            elif current_rate >= last_candle['kc_2_upper_1h']:
                return stoploss_from_absolute(last_candle['kc_0_upper_1h'], current_rate, is_short=trade.is_short)

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
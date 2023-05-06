# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import logging
from typing import Dict

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame,Series
from typing import Optional, Union
from functools import reduce
from datetime import datetime
import pandas_ta as pta

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

base on AI_K4CRSI_V1
20230504:
结合NFIX2新增特征

base on AI_K4CRSI_V3
20230505
修改模型目标为收益率
"""

class AI_NFIX2_V1(IStrategy):
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
    use_custom_stoploss = False

    startup_candle_count = 800
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

      lose_idx = [99999]
      prof_idx = [99999]

      org_x = x[0]
      for i,x_ in enumerate(x):
        if (x_ - org_x) / org_x > 0.01:
          prof_idx.append(i)
        elif (x_ - org_x) / org_x < -0.01:
          lose_idx.append(i)
    
      if min(lose_idx)<min(prof_idx):
        return -1
      elif min(lose_idx)>min(prof_idx):
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
        # dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)
        # dataframe["%-ema"] = ta.EMA(dataframe, timeperiod=period)
        if metadata["tf"] in ("15m","1h"):
            dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)

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
        # print("final 1 :",dataframe.columns.values)
        # dataframe["%-pct-change"] = dataframe["close"].pct_change()
        # dataframe["%-raw_volume"] = dataframe["volume"]
        # dataframe["%-raw_price"] = dataframe["close"]
        if metadata["tf"] in ("15m","1h"):
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


        if metadata["tf"] == "4h":
            # RSI
            dataframe['%-rsi_14'] = ta.RSI(dataframe, timeperiod=14, fillna=True)
    
            dataframe['%-rsi_14_max_6'] = dataframe['%-rsi_14'].rolling(6).max()
    
            # EMA
            dataframe['%-ema_12'] = ta.EMA(dataframe, timeperiod=12)
            dataframe['%-ema_26'] = ta.EMA(dataframe, timeperiod=26)
            dataframe['%-ema_50'] = ta.EMA(dataframe, timeperiod=50)
            
            dataframe["%-ema_12-pct"] = dataframe["%-ema_12"].pct_change(periods=6)
            dataframe["%-ema_26-pct"] = dataframe["%-ema_26"].pct_change(periods=6)
            dataframe["%-ema_50-pct"] = dataframe["%-ema_50"].pct_change(periods=6)
            # SMA
            dataframe['%-sma_12'] = ta.SMA(dataframe, timeperiod=12)
            dataframe['%-sma_26'] = ta.SMA(dataframe, timeperiod=26)
            dataframe['%-sma_50'] = ta.SMA(dataframe, timeperiod=50)
    
            # Williams %R
            dataframe['%-r_14'] = williams_r(dataframe, period=14)
    
            # CTI
            dataframe['%-cti_20'] = pta.cti(dataframe["close"], length=20)
    
            # S/R
            res_series = dataframe['high'].rolling(window = 5, center=True).apply(lambda row: is_resistance(row), raw=True).shift(2)
            sup_series = dataframe['low'].rolling(window = 5, center=True).apply(lambda row: is_support(row), raw=True).shift(2)
            dataframe['%-res_level'] = Series(np.where(res_series, np.where(dataframe['close'] > dataframe['open'], dataframe['close'], dataframe['open']), float('NaN'))).ffill()
            dataframe['%-res_hlevel'] = Series(np.where(res_series, dataframe['high'], float('NaN'))).ffill()
            dataframe['%-sup_level'] = Series(np.where(sup_series, np.where(dataframe['close'] < dataframe['open'], dataframe['close'], dataframe['open']), float('NaN'))).ffill()
    
            # Downtrend checks
            dataframe['%-not_downtrend'] = np.where(((dataframe['close'] > dataframe['close'].shift(2)) | (dataframe['%-rsi_14'] > 50.0)),1,0)
    
            dataframe['%-is_downtrend_3'] = np.where(((dataframe['close'] < dataframe['open']) & (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & (dataframe['close'].shift(2) < dataframe['open'].shift(2))),1,0)

            # Uptrend checks
            dataframe['%-not_uptrend'] = np.where(((dataframe['close'] < dataframe['close'].shift(2)) | (dataframe['%-rsi_14'] < 50.0)),1,0)
    
            dataframe['%-is_uptrend_3'] = np.where(((dataframe['close'] > dataframe['open']) & (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & (dataframe['close'].shift(2) > dataframe['open'].shift(2))),1,0)
            
            # Wicks
            dataframe['%-top_wick_pct'] = ((dataframe['high'] - np.maximum(dataframe['open'], dataframe['close'])) / np.maximum(dataframe['open'], dataframe['close']))
            dataframe['%-low_wick_pct'] = ((dataframe['low'] - np.minimum(dataframe['open'], dataframe['close'])) / np.minimum(dataframe['open'], dataframe['close']))
            
            # Candle change
            dataframe['%-change_pct'] = (dataframe['close'] - dataframe['open']) / dataframe['open']
    
            # Max highs
            dataframe['high_max_3'] = dataframe['high'].rolling(3).max()
            dataframe['high_max_12'] = dataframe['high'].rolling(12).max()
            dataframe['high_max_24'] = dataframe['high'].rolling(24).max()
            dataframe['high_max_36'] = dataframe['high'].rolling(36).max()
            dataframe['high_max_48'] = dataframe['high'].rolling(48).max()

            # Min lows
            dataframe['low_min_3'] = dataframe['low'].rolling(3).min()
            dataframe['low_min_12'] = dataframe['low'].rolling(12).min()
            dataframe['low_min_24'] = dataframe['low'].rolling(24).min()
            dataframe['low_min_36'] = dataframe['low'].rolling(36).min()
            dataframe['low_min_48'] = dataframe['low'].rolling(48).min()
    
            dataframe['%-pct_change_high_max_1_12'] = (dataframe['high'] - dataframe['high_max_12']) / dataframe['high_max_12']
            dataframe['%-pct_change_high_max_3_12'] = (dataframe['high_max_3'] - dataframe['high_max_12']) / dataframe['high_max_12']
            dataframe['%-pct_change_high_max_3_24'] = (dataframe['high_max_3'] - dataframe['high_max_24']) / dataframe['high_max_24']
            dataframe['%-pct_change_high_max_3_36'] = (dataframe['high_max_3'] - dataframe['high_max_36']) / dataframe['high_max_36']
            dataframe['%-pct_change_high_max_3_48'] = (dataframe['high_max_3'] - dataframe['high_max_48']) / dataframe['high_max_48']
    
            dataframe['%-pct_change_low_min_1_12'] = (dataframe['low'] - dataframe['low_min_12']) / dataframe['low_min_12']
            dataframe['%-pct_change_low_min_3_12'] = (dataframe['low_min_3'] - dataframe['low_min_12']) / dataframe['low_min_12']
            dataframe['%-pct_change_low_min_3_24'] = (dataframe['low_min_3'] - dataframe['low_min_24']) / dataframe['low_min_24']
            dataframe['%-pct_change_low_min_3_36'] = (dataframe['low_min_3'] - dataframe['low_min_36']) / dataframe['low_min_36']
            dataframe['%-pct_change_low_min_3_48'] = (dataframe['low_min_3'] - dataframe['low_min_48']) / dataframe['low_min_48']
            # Volume
            dataframe['%-volume_mean_factor_6'] = dataframe['volume'] / dataframe['volume'].rolling(6).mean()


        if metadata["tf"] == "1h":
    
            # EMA
            dataframe['%-ema_12'] = ta.EMA(dataframe, timeperiod=12)
            dataframe['%-ema_26'] = ta.EMA(dataframe, timeperiod=26)
            dataframe['%-ema_50'] = ta.EMA(dataframe, timeperiod=50)

            dataframe["%-ema_12-pct"] = dataframe["%-ema_12"].pct_change(periods=12)
            dataframe["%-ema_26-pct"] = dataframe["%-ema_26"].pct_change(periods=12)
            dataframe["%-ema_50-pct"] = dataframe["%-ema_50"].pct_change(periods=12)
    
            # SMA
            dataframe['%-sma_12'] = ta.SMA(dataframe, timeperiod=12)
            dataframe['%-sma_26'] = ta.SMA(dataframe, timeperiod=26)
            dataframe['%-sma_50'] = ta.SMA(dataframe, timeperiod=50)
    
            # BB
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
            dataframe['%-bb20_2_low'] = bollinger['lower']
            dataframe['%-bb20_2_mid'] = bollinger['mid']
            dataframe['%-bb20_2_upp'] = bollinger['upper']
    
            dataframe['%-bb20_2_width'] = ((dataframe['%-bb20_2_upp'] - dataframe['%-bb20_2_low']) / dataframe['%-bb20_2_mid'])
    
            # Williams %R
            dataframe['%-r_14'] = williams_r(dataframe, period=14)
            dataframe['%-r_96'] = williams_r(dataframe, period=96)
    
            # CTI
            dataframe['%-cti_20'] = pta.cti(dataframe["close"], length=20)
            dataframe['%-cti_40'] = pta.cti(dataframe["close"], length=40)
    
            # S/R
            res_series = dataframe['high'].rolling(window = 5, center=True).apply(lambda row: is_resistance(row), raw=True).shift(2)
            sup_series = dataframe['low'].rolling(window = 5, center=True).apply(lambda row: is_support(row), raw=True).shift(2)
            dataframe['%-res_level'] = Series(np.where(res_series, np.where(dataframe['close'] > dataframe['open'], dataframe['close'], dataframe['open']), float('NaN'))).ffill()
            dataframe['%-res_hlevel'] = Series(np.where(res_series, dataframe['high'], float('NaN'))).ffill()
            dataframe['%-sup_level'] = Series(np.where(sup_series, np.where(dataframe['close'] < dataframe['open'], dataframe['close'], dataframe['open']), float('NaN'))).ffill()
    
            # Pump protections
            dataframe['%-oc_pct_change_48'] = range_percent_change(self, dataframe, 'OC', 48)
            dataframe['%-oc_pct_change_36'] = range_percent_change(self, dataframe, 'OC', 36)
            dataframe['%-oc_pct_change_24'] = range_percent_change(self, dataframe, 'OC', 24)
            dataframe['%-oc_pct_change_12'] = range_percent_change(self, dataframe, 'OC', 12)
            dataframe['%-oc_pct_change_6'] = range_percent_change(self, dataframe, 'OC', 6)
    
            # Downtrend checks
    
            dataframe['%-is_downtrend_3'] = np.where(((dataframe['close'] < dataframe['open']) & (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & (dataframe['close'].shift(2) < dataframe['open'].shift(2))),1,0)
    
            dataframe['%-is_downtrend_5'] = np.where(((dataframe['close'] < dataframe['open']) & (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & (dataframe['close'].shift(2) < dataframe['open'].shift(2)) & (dataframe['close'].shift(3) < dataframe['open'].shift(3)) & (dataframe['close'].shift(4) < dataframe['open'].shift(4))),1,0)
            
            # Downtrend checks
    
            dataframe['%-is_uptrend_3'] = np.where(((dataframe['close'] > dataframe['open']) & (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & (dataframe['close'].shift(2) > dataframe['open'].shift(2))),1,0)
    
            dataframe['%-is_uptrend_5'] = np.where(((dataframe['close'] > dataframe['open']) & (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & (dataframe['close'].shift(2) > dataframe['open'].shift(2)) & (dataframe['close'].shift(3) > dataframe['open'].shift(3)) & (dataframe['close'].shift(4) > dataframe['open'].shift(4))),1,0)
            
            # Wicks
            dataframe['%-top_wick_pct'] = ((dataframe['high'] - np.maximum(dataframe['open'], dataframe['close'])) / np.maximum(dataframe['open'], dataframe['close']))
            dataframe['%-low_wick_pct'] = ((dataframe['low'] - np.minimum(dataframe['open'], dataframe['close'])) / np.minimum(dataframe['open'], dataframe['close']))
            # Candle change
            dataframe['%-change_pct'] = (dataframe['close'] - dataframe['open']) / dataframe['open']
    
            # Max highs
            dataframe['high_max_3'] = dataframe['high'].rolling(3).max()
            dataframe['high_max_6'] = dataframe['high'].rolling(6).max()
            dataframe['high_max_12'] = dataframe['high'].rolling(12).max()
            dataframe['high_max_24'] = dataframe['high'].rolling(24).max()

            dataframe['%-pct_change_high_max_3_12'] = (dataframe['high_max_3'] - dataframe['high_max_12']) / dataframe['high_max_12']
            dataframe['%-pct_change_high_max_6_12'] = (dataframe['high_max_6'] - dataframe['high_max_12']) / dataframe['high_max_12']
            dataframe['%-pct_change_high_max_6_24'] = (dataframe['high_max_6'] - dataframe['high_max_24']) / dataframe['high_max_24']

            # Min lows
            dataframe['low_min_3'] = dataframe['low'].rolling(3).min()
            dataframe['low_min_6'] = dataframe['low'].rolling(6).min()
            dataframe['low_min_12'] = dataframe['low'].rolling(12).min()
            dataframe['low_min_24'] = dataframe['low'].rolling(24).min()

            dataframe['%-pct_change_low_min_3_12'] = (dataframe['low_min_3'] - dataframe['low_min_12']) / dataframe['low_min_12']
            dataframe['%-pct_change_low_min_6_12'] = (dataframe['low_min_6'] - dataframe['low_min_12']) / dataframe['low_min_12']
            dataframe['%-pct_change_low_min_6_24'] = (dataframe['low_min_6'] - dataframe['low_min_24']) / dataframe['low_min_24']
    
            # Volume
            dataframe['%-volume_mean_factor_12'] = dataframe['volume'] / dataframe['volume'].rolling(12).mean()

        if metadata["tf"] == "15m":

            # EMA
            dataframe['%-ema_12'] = ta.EMA(dataframe, timeperiod=12)
            dataframe['%-ema_26'] = ta.EMA(dataframe, timeperiod=26)

            dataframe["%-ema_12-pct"] = dataframe["%-ema_12"].pct_change(periods=16)
            dataframe["%-ema_26-pct"] = dataframe["%-ema_26"].pct_change(periods=16)

            # SMA
            dataframe['%-sma_200'] = ta.SMA(dataframe, timeperiod=200)

            # CTI
            dataframe['%-cti_20'] = pta.cti(dataframe["close"], length=20)

            # Downtrend checks
    
            dataframe['%-is_downtrend_3'] = np.where(((dataframe['close'] < dataframe['open']) & (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & (dataframe['close'].shift(2) < dataframe['open'].shift(2))),1,0)
    
            dataframe['%-is_downtrend_5'] = np.where(((dataframe['close'] < dataframe['open']) & (dataframe['close'].shift(1) < dataframe['open'].shift(1)) & (dataframe['close'].shift(2) < dataframe['open'].shift(2)) & (dataframe['close'].shift(3) < dataframe['open'].shift(3)) & (dataframe['close'].shift(4) < dataframe['open'].shift(4))),1,0)
            
            # Downtrend checks

            dataframe['%-is_uptrend_3'] = np.where(((dataframe['close'] > dataframe['open']) & (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & (dataframe['close'].shift(2) > dataframe['open'].shift(2))),1,0)
    
            dataframe['%-is_uptrend_5'] = np.where(((dataframe['close'] > dataframe['open']) & (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & (dataframe['close'].shift(2) > dataframe['open'].shift(2)) & (dataframe['close'].shift(3) > dataframe['open'].shift(3)) & (dataframe['close'].shift(4) > dataframe['open'].shift(4))),1,0)
            
            # Volume
            dataframe['%-volume_mean_factor_12'] = dataframe['volume'] / dataframe['volume'].rolling(12).mean()
        
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
        # Pivots change
        dataframe['%-pct_pivot_close'] = (dataframe['close'] - dataframe['%-pivot_1d']) / dataframe['%-pivot_1d']

        dataframe['%-pct_sup1_close'] = (dataframe['close'] - dataframe['%-sup1_1d']) / dataframe['%-sup1_1d']
        dataframe['%-pct_sup2_close'] = (dataframe['close'] - dataframe['%-sup2_1d']) / dataframe['%-sup2_1d']
        dataframe['%-pct_sup3_close'] = (dataframe['close'] - dataframe['%-sup3_1d']) / dataframe['%-sup3_1d']

        dataframe['%-pct_res1_close'] = (dataframe['close'] - dataframe['%-res1_1d']) / dataframe['%-res1_1d']
        dataframe['%-pct_res2_close'] = (dataframe['close'] - dataframe['%-res2_1d']) / dataframe['%-res2_1d']
        dataframe['%-pct_res3_close'] = (dataframe['close'] - dataframe['%-res3_1d']) / dataframe['%-res3_1d']
        
        # dataframe['up_or_lo'] = np.where((dataframe["close"] < dataframe["kc_2_lower_1h"]),-1,np.where((dataframe["close"] > dataframe["kc_2_upper_1h"]),1,0))
        dataframe['traget_num'] = dataframe['close'].rolling(window=40).apply(self.target_num, raw=True).shift(-40)
        dataframe['&s-cross_or_stay'] = np.where(dataframe['traget_num']==1, 'up', np.where(dataframe['traget_num']==-1, 'down', 'stay'))
        
        # print("ai:",dataframe.iloc[1:20, -10:-1])
        # dataframe.to_csv("user_data/data2.csv")
        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # kc_0 = qtpylib.keltner_channel(dataframe, window=36, atrs=0.5)
        # dataframe['kc_0_mid'] = kc_0['mid']
        # dataframe['kc_0_upper'] = kc_0['upper']
        # dataframe['kc_0_lower'] = kc_0['lower']

        # kc_2 = qtpylib.keltner_channel(dataframe, window=36, atrs=2.36)
        # dataframe['kc_2_upper'] = kc_2['upper']
        # dataframe['kc_2_lower'] = kc_2['lower']

        # kc_3 = qtpylib.keltner_channel(dataframe, window=36, atrs=3.28)
        # dataframe['kc_3_upper'] = kc_3['upper']
        # dataframe['kc_3_lower'] = kc_3['lower']

        # kc_5 = qtpylib.keltner_channel(dataframe, window=36, atrs=5)
        # dataframe['kc_5_upper'] = kc_5['upper']
        # dataframe['kc_5_lower'] = kc_5['lower']

        # kc_6 = qtpylib.keltner_channel(dataframe, window=36, atrs=6.18)
        # dataframe['kc_6_upper'] = kc_6['upper']
        # dataframe['kc_6_lower'] = kc_6['lower']

        # kc_10 = qtpylib.keltner_channel(dataframe, window=36, atrs=10)
        # dataframe['kc_10_upper'] = kc_10['upper']
        # dataframe['kc_10_lower'] = kc_10['lower']

        dataframe["rsi_12"] = ta.RSI(dataframe, timeperiod=12)
        return dataframe

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Pivots
        dataframe['%-pivot'], dataframe['%-res1'], dataframe['%-res2'], dataframe['%-res3'], dataframe['%-sup1'], dataframe['%-sup2'], dataframe['%-sup3'] = pivot_points(dataframe, mode='fibonacci')

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # print("final 1 :",dataframe.columns.values)
        dataframe = self.freqai.start(dataframe, metadata, self)
        # print("final:",dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # LONG
        dataframe.loc[((dataframe['&s-cross_or_stay'] == 'up') &
                (dataframe['rsi_12_1h'] > self.buy_params['rsi_low']) # &
                # (qtpylib.crossed_above(dataframe['close'], dataframe['kc_0_upper_1h']))
            ), 
            ['enter_long','enter_tag']] = (1,'closs cross')
        
        
        # SHORT
        dataframe.loc[((dataframe['&s-cross_or_stay'] == 'down') &
                (dataframe['rsi_12_1h'] < self.sell_params['rsi_high']) # &
                # (qtpylib.crossed_below(dataframe['close'], dataframe['kc_0_lower_1h']))
            ), 
            ['enter_short','enter_tag']] = (1,'closs cross')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # pair = metadata["pair"].replace(":", "")

        # dataframe.loc[(
        #         (qtpylib.crossed_above(dataframe['high'], dataframe['kc_10_upper_1h']))
        #     ), 
        #     ['exit_long','exit_tag']] = (1,'cross kc_10 upper')
        
        # dataframe.loc[(
        #         (qtpylib.crossed_below(dataframe['low'], dataframe[f'kc_10_lower_1h']))
        #     ), 
        #     ['exit_short','exit_tag']] = (1,'cross kc_10 lower')

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
        return 1.0
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        # if trade.is_short:
        #     if current_rate>last_candle['kc_0_upper_1h']:
        #         return 'hit kc_0 upper'
        # else:
        #     if current_rate<last_candle['kc_0_lower_1h']:
        #         return 'hit kc_0 lower'
    
        if 0.01 < current_profit:
            return '0.01'
        
        elif -0.01 > current_profit:
            return '-0.01'


    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:

    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

    #     last_candle = dataframe.iloc[-1].squeeze()
        
    #     # evaluate highest to lowest, so that highest possible stop is used

    #     if trade.is_short:
    #         if current_rate <= last_candle['kc_6_lower_1h']:
    #             return stoploss_from_absolute(last_candle['kc_5_lower_1h'], current_rate, is_short=trade.is_short)
    #         elif current_rate <= last_candle['kc_3_lower_1h']:
    #             return stoploss_from_absolute(last_candle['kc_2_lower_1h'], current_rate, is_short=trade.is_short)
    #         elif current_rate <= last_candle['kc_2_lower_1h']:
    #             return stoploss_from_absolute(last_candle['kc_0_lower_1h'], current_rate, is_short=trade.is_short)
            
    #     else:
    #         if current_rate >= last_candle['kc_6_upper_1h']:
    #             return stoploss_from_absolute(last_candle['kc_5_upper_1h'], current_rate, is_short=trade.is_short)
    #         elif current_rate >= last_candle['kc_3_upper_1h']:
    #             return stoploss_from_absolute(last_candle['kc_2_upper_1h'], current_rate, is_short=trade.is_short)
    #         elif current_rate >= last_candle['kc_2_upper_1h']:
    #             return stoploss_from_absolute(last_candle['kc_0_upper_1h'], current_rate, is_short=trade.is_short)

    #     # return maximum stoploss value, keeping current stoploss price unchanged
    #     return 0.99

    # @property
    # def protections(self):
    #     return [
    #         {
    #             "method": "CooldownPeriod",
    #             "stop_duration_candles": 1
    #         }
    #     ]

# Pivot Points - 3 variants - daily recommended
def pivot_points(dataframe: DataFrame, mode = 'fibonacci') -> Series:
    if mode == 'simple':
        hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
        res1 = hlc3_pivot * 2 - dataframe['low'].shift(1)
        sup1 = hlc3_pivot * 2 - dataframe['high'].shift(1)
        res2 = hlc3_pivot + (dataframe['high'] - dataframe['low']).shift()
        sup2 = hlc3_pivot - (dataframe['high'] - dataframe['low']).shift()
        res3 = hlc3_pivot * 2 + (dataframe['high'] - 2 * dataframe['low']).shift()
        sup3 = hlc3_pivot * 2 - (2 * dataframe['high'] - dataframe['low']).shift()
        return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
    elif mode == 'fibonacci':
        hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
        hl_range = (dataframe['high'] - dataframe['low']).shift(1)
        res1 = hlc3_pivot + 0.382 * hl_range
        sup1 = hlc3_pivot - 0.382 * hl_range
        res2 = hlc3_pivot + 0.618 * hl_range
        sup2 = hlc3_pivot - 0.618 * hl_range
        res3 = hlc3_pivot + 1 * hl_range
        sup3 = hlc3_pivot - 1 * hl_range
        return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
    elif mode == 'DeMark':
        demark_pivot_lt = (dataframe['low'] * 2 + dataframe['high'] + dataframe['close'])
        demark_pivot_eq = (dataframe['close'] * 2 + dataframe['low'] + dataframe['high'])
        demark_pivot_gt = (dataframe['high'] * 2 + dataframe['low'] + dataframe['close'])
        demark_pivot = np.where((dataframe['close'] < dataframe['open']), demark_pivot_lt, np.where((dataframe['close'] > dataframe['open']), demark_pivot_gt, demark_pivot_eq))
        dm_pivot = demark_pivot / 4
        dm_res = demark_pivot / 2 - dataframe['low']
        dm_sup = demark_pivot / 2 - dataframe['high']
        return dm_pivot, dm_res, dm_sup

# Range midpoint acts as Support
def is_support(row_data) -> bool:
    conditions = []
    for row in range(len(row_data)-1):
        if row < len(row_data)//2:
            conditions.append(row_data[row] > row_data[row+1])
        else:
            conditions.append(row_data[row] < row_data[row+1])
    result = reduce(lambda x, y: x & y, conditions)
    return result

# Range midpoint acts as Resistance
def is_resistance(row_data) -> bool:
    conditions = []
    for row in range(len(row_data)-1):
        if row < len(row_data)//2:
            conditions.append(row_data[row] < row_data[row+1])
        else:
            conditions.append(row_data[row] > row_data[row+1])
    result = reduce(lambda x, y: x & y, conditions)
    return result

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

# Peak Percentage Change
def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
    """
    Rolling Percentage Change Maximum across interval.

    :param dataframe: DataFrame The original OHLC dataframe
    :param method: High to Low / Open to Close
    :param length: int The length to look back
    """
    if method == 'HL':
        return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
    elif method == 'OC':
        return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
    else:
        raise ValueError(f"Method {method} not defined!")
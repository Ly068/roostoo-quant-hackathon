"""
波动率突破策略（Volatility Breakout Strategy）
==============================================
核心逻辑：Bollinger Bands挤压 → 突破跟进 → ATR止损

研究依据：
- 波动率具有周期性：低波动（挤压）后必然跟随高波动（突破）
- 挤压期间Bollinger Bands宽度进入历史最低20%分位
- 突破时收盘价穿越上/下轨，成交量确认
- 止损用1.5-2.0倍ATR，止盈用2-3倍ATR

适用行情：爆发市、趋势启动点
与动量策略互补：动量抓"已经在趋势中"，突破抓"趋势刚启动"
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from rooster_trader.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    波动率突破策略。

    参数:
        symbols: 交易对列表
        bb_window: Bollinger Bands窗口（默认20）
        bb_std: Bollinger Bands标准差倍数（默认2.0）
        squeeze_percentile: 挤压判定分位数（默认0.2，即带宽低于历史20%算挤压）
        squeeze_lookback: 挤压判定的历史窗口（默认100）
        atr_window: ATR窗口（默认14）
        atr_stop_mult: ATR止损倍数（默认2.0）
        atr_target_mult: ATR止盈倍数（默认3.0）
        volume_confirm: 是否需要成交量确认（默认True）
        max_positions: 最大同时持仓数（默认2）
    """

    def __init__(
        self,
        symbols: list,
        bb_window: int = 20,
        bb_std: float = 2.0,
        squeeze_percentile: float = 0.2,
        squeeze_lookback: int = 100,
        atr_window: int = 14,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        volume_confirm: bool = True,
        max_positions: int = 2,
    ):
        super().__init__(name="VolatilityBreakout", symbols=symbols)
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.squeeze_percentile = squeeze_percentile
        self.squeeze_lookback = squeeze_lookback
        self.atr_window = atr_window
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.volume_confirm = volume_confirm
        self.max_positions = max_positions

        # 持仓状态跟踪
        self.positions: Dict[str, dict] = {}  # {symbol: {side, entry_price, stop_loss, take_profit}}

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # Bollinger Bands
        df["sma"] = df["close"].rolling(self.bb_window).mean()
        df["std"] = df["close"].rolling(self.bb_window).std()
        df["upper_band"] = df["sma"] + self.bb_std * df["std"]
        df["lower_band"] = df["sma"] - self.bb_std * df["std"]
        df["band_width"] = (df["upper_band"] - df["lower_band"]) / df["sma"]

        # 挤压判定：当前带宽是否低于历史squeeze_lookback窗口的squeeze_percentile分位
        df["bandwidth_min"] = df["band_width"].rolling(self.squeeze_lookback).min()
        df["bandwidth_max"] = df["band_width"].rolling(self.squeeze_lookback).max()
        df["bandwidth_pct"] = (df["band_width"] - df["bandwidth_min"]) / (
            df["bandwidth_max"] - df["bandwidth_min"] + 1e-10
        )
        df["is_squeeze"] = df["bandwidth_pct"] < self.squeeze_percentile

        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_window).mean()

        # 成交量确认：当前成交量 > 20周期平均
        if self.volume_confirm:
            df["vol_sma"] = df["volume"].rolling(20).mean()
            df["vol_confirm"] = df["volume"] > df["vol_sma"]
        else:
            df["vol_confirm"] = True

        return df

    def on_bar(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        每根K线调用，返回目标仓位权重。

        逻辑：
        1. 对每个币种计算指标
        2. 检查现有持仓是否触发止损/止盈
        3. 寻找新的突破信号（挤压后突破）
        4. 按突破强度排序，选top_n
        """
        target_weights = {s: 0.0 for s in self.symbols}
        signals = []

        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < self.squeeze_lookback + 10:
                continue

            df = self._calculate_indicators(data[symbol])
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(latest["atr"]) or pd.isna(latest["upper_band"]):
                continue

            price = latest["close"]
            atr = latest["atr"]

            # ---- 检查现有持仓 ----
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos["side"] == "LONG":
                    if price <= pos["stop_loss"]:
                        logger.info("[突破策略] %s 触发止损 %.2f", symbol, price)
                        del self.positions[symbol]
                        continue
                    if price >= pos["take_profit"]:
                        logger.info("[突破策略] %s 触发止盈 %.2f", symbol, price)
                        del self.positions[symbol]
                        continue
                elif pos["side"] == "SHORT":
                    if price >= pos["stop_loss"]:
                        logger.info("[突破策略] %s 空头触发止损 %.2f", symbol, price)
                        del self.positions[symbol]
                        continue
                    if price <= pos["take_profit"]:
                        logger.info("[突破策略] %s 空头触发止盈 %.2f", symbol, price)
                        del self.positions[symbol]
                        continue

                # 持仓中，保持权重
                target_weights[symbol] = 0.15 if pos["side"] == "LONG" else -0.15
                continue

            # ---- 寻找新突破信号 ----
            # 条件1：前一根K线处于挤压状态
            # 条件2：当前收盘价突破上轨（做多）或下轨（做空）
            # 条件3：成交量确认
            was_squeeze = prev["is_squeeze"] if not pd.isna(prev["is_squeeze"]) else False
            vol_ok = latest["vol_confirm"] if self.volume_confirm else True

            if was_squeeze and vol_ok:
                # 向上突破
                if price > latest["upper_band"] and prev["close"] <= prev["upper_band"]:
                    strength = (price - latest["upper_band"]) / atr
                    signals.append({
                        "symbol": symbol,
                        "side": "LONG",
                        "strength": strength,
                        "entry_price": price,
                        "stop_loss": price - self.atr_stop_mult * atr,
                        "take_profit": price + self.atr_target_mult * atr,
                    })
                # 向下突破
                elif price < latest["lower_band"] and prev["close"] >= prev["lower_band"]:
                    strength = (latest["lower_band"] - price) / atr
                    signals.append({
                        "symbol": symbol,
                        "side": "SHORT",
                        "strength": strength,
                        "entry_price": price,
                        "stop_loss": price + self.atr_stop_mult * atr,
                        "take_profit": price - self.atr_target_mult * atr,
                    })

        # ---- 按突破强度排序，选top_n ----
        signals.sort(key=lambda x: x["strength"], reverse=True)
        selected = signals[: self.max_positions]

        for sig in selected:
            symbol = sig["symbol"]
            self.positions[symbol] = {
                "side": sig["side"],
                "entry_price": sig["entry_price"],
                "stop_loss": sig["stop_loss"],
                "take_profit": sig["take_profit"],
            }
            weight = 0.15 if sig["side"] == "LONG" else -0.15
            target_weights[symbol] = weight
            logger.info(
                "[突破策略] 开仓 %s %s @ %.2f (强度%.2f, 止损%.2f, 止盈%.2f)",
                sig["side"], symbol, sig["entry_price"],
                sig["strength"], sig["stop_loss"], sig["take_profit"]
            )

        return target_weights

"""
市场状态检测模块（Market Regime Detection）
============================================
识别当前市场处于哪种状态，动态调整策略权重。

三种核心状态：
1. TREND（趋势市）：ADX≥25，价格有明确方向性 → 动量策略权重提升
2. RANGE（震荡市）：ADX<20，价格在区间内波动 → 配对交易权重提升
3. VOLATILE（高波动市）：ATR>历史80分位 → 整体降仓，突破策略权重提升

检测方法：
- ADX（Average Directional Index）：衡量趋势强度
- ATR百分位：衡量波动率相对水平
- Bollinger Bands宽度：辅助确认
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    市场状态检测器。

    参数:
        adx_window: ADX计算窗口（默认14）
        adx_trend_threshold: ADX趋势阈值（默认25）
        adx_range_threshold: ADX震荡阈值（默认20）
        atr_window: ATR窗口（默认14）
        atr_lookback: ATR百分位历史窗口（默认100）
        high_vol_percentile: 高波动分位（默认0.8）
        low_vol_percentile: 低波动分位（默认0.2）
    """

    def __init__(
        self,
        adx_window: int = 14,
        adx_trend_threshold: float = 25,
        adx_range_threshold: float = 20,
        atr_window: int = 14,
        atr_lookback: int = 100,
        high_vol_percentile: float = 0.8,
        low_vol_percentile: float = 0.2,
    ):
        self.adx_window = adx_window
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.atr_window = atr_window
        self.atr_lookback = atr_lookback
        self.high_vol_percentile = high_vol_percentile
        self.low_vol_percentile = low_vol_percentile

    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """计算ADX（Average Directional Index）"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM, -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # ATR
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        tr = np.maximum(np.maximum(high_low, high_close), low_close)

        # 平滑
        atr = pd.Series(tr).rolling(self.adx_window).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(self.adx_window).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(self.adx_window).mean().values / (atr + 1e-10)

        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # ADX = DX的平滑
        adx = pd.Series(dx).rolling(self.adx_window).mean().values

        return float(adx[-1]) if not np.isnan(adx[-1]) else 0

    def _calculate_atr_percentile(self, df: pd.DataFrame) -> float:
        """计算当前ATR在历史中的百分位"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()

        # 当前ATR在最近atr_lookback窗口中的百分位
        recent_atr = atr.tail(self.atr_lookback).dropna()
        if len(recent_atr) < 10:
            return 0.5

        current_atr = atr.iloc[-1]
        percentile = (recent_atr < current_atr).sum() / len(recent_atr)
        return float(percentile)

    def detect(self, data: Dict[str, pd.DataFrame], reference_symbol: str = "BTC/USD") -> Dict:
        """
        检测当前市场状态。

        返回:
            {
                "regime": "TREND" | "RANGE" | "VOLATILE",
                "adx": float,
                "atr_percentile": float,
                "trend_strength": float,  # 0-1
                "volatility_level": float,  # 0-1
                "strategy_weights_adj": {  # 策略权重调整系数
                    "momentum": float,
                    "pairs": float,
                    "breakout": float,
                    "xgboost": float,
                },
                "global_position_adj": float,  # 全局仓位调整系数
            }
        """
        if reference_symbol not in data or len(data[reference_symbol]) < self.atr_lookback + 20:
            # 数据不足，默认中性
            return {
                "regime": "RANGE",
                "adx": 20,
                "atr_percentile": 0.5,
                "trend_strength": 0.3,
                "volatility_level": 0.5,
                "strategy_weights_adj": {
                    "momentum": 1.0,
                    "pairs": 1.0,
                    "breakout": 1.0,
                    "xgboost": 1.0,
                },
                "global_position_adj": 1.0,
            }

        df = data[reference_symbol]
        adx = self._calculate_adx(df)
        atr_pct = self._calculate_atr_percentile(df)

        # 趋势强度（0-1）：ADX从0到50映射
        trend_strength = min(max((adx - 15) / 35, 0), 1)

        # 波动率水平（0-1）
        volatility_level = atr_pct

        # 判断主状态
        if atr_pct >= self.high_vol_percentile:
            regime = "VOLATILE"
        elif adx >= self.adx_trend_threshold:
            regime = "TREND"
        elif adx < self.adx_range_threshold:
            regime = "RANGE"
        else:
            # 过渡区，按趋势强度判断
            regime = "TREND" if trend_strength > 0.5 else "RANGE"

        # 策略权重调整系数
        if regime == "TREND":
            weights_adj = {
                "momentum": 1.5,    # 趋势市动量加50%
                "pairs": 0.5,       # 配对交易减50%
                "breakout": 1.2,    # 突破加20%
                "xgboost": 1.0,     # ML不变
            }
            global_adj = 1.0
        elif regime == "RANGE":
            weights_adj = {
                "momentum": 0.5,    # 动量减50%
                "pairs": 1.5,       # 配对加50%
                "breakout": 0.8,    # 突破减20%
                "xgboost": 1.0,
            }
            global_adj = 0.9
        else:  # VOLATILE
            weights_adj = {
                "momentum": 0.7,
                "pairs": 0.8,
                "breakout": 1.5,    # 高波动突破加50%
                "xgboost": 1.2,     # ML加20%（适应能力强）
            }
            global_adj = 0.7        # 高波动整体降仓30%

        result = {
            "regime": regime,
            "adx": adx,
            "atr_percentile": atr_pct,
            "trend_strength": trend_strength,
            "volatility_level": volatility_level,
            "strategy_weights_adj": weights_adj,
            "global_position_adj": global_adj,
        }

        logger.info(
            "[市场状态] %s (ADX=%.1f, ATR分位=%.0f%%, 趋势强度=%.0f%%, 全局仓位=%.0f%%)",
            regime, adx, atr_pct * 100, trend_strength * 100, global_adj * 100
        )

        return result

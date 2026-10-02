"""
策略三：美股趋势轮动
====================
逻辑：在代币化美股池中，只做多。价格在20日均线上方 + 放量 → 买入。
月度调仓，选趋势最强的几只持有。
"""

import numpy as np
import pandas as pd
from typing import Dict

from rooster_trader.strategy.base import BaseStrategy


class RotationStrategy(BaseStrategy):
    """
    美股趋势轮动策略。

    参数:
        ma_window:       均线窗口（默认20天）
        vol_factor:      成交量放量倍数阈值（默认1.5倍均量）
        top_n:           选几只股票做多（默认3只）
        rebalance_days:  调仓频率（默认30天）
    """

    def __init__(
        self,
        symbols: list,
        ma_window: int = 20,
        vol_factor: float = 1.5,
        top_n: int = 3,
        rebalance_days: int = 30,
    ):
        super().__init__(name="USStockRotation", symbols=symbols)
        self.ma_window = ma_window
        self.vol_factor = vol_factor
        self.top_n = top_n
        self.rebalance_days = rebalance_days

        self.last_rebalance_day = -999
        self.current_weights: Dict[str, float] = {}

    def warmup_bars(self) -> int:
        return self.ma_window + 5

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        n_bars = len(list(bars.values())[0])

        # 调仓判断：每30根日线调一次
        if n_bars - self.last_rebalance_day < self.rebalance_days and self.current_weights:
            return self.current_weights

        scores = {}
        for symbol in self.symbols:
            df = bars[symbol]
            close = df["close"]
            volume = df["volume"]

            if len(close) < self.ma_window:
                continue

            ma = close.rolling(self.ma_window).mean()
            current_price = close.iloc[-1]
            current_ma = ma.iloc[-1]

            # 趋势过滤：价格必须在均线上方
            if current_price < current_ma:
                continue

            # 评分1：价格相对均线的强度（偏离度）
            above_ma_pct = (current_price - current_ma) / current_ma

            # 评分2：成交量确认
            avg_volume = volume.iloc[-self.ma_window:].mean()
            latest_volume = volume.iloc[-1]
            if avg_volume > 0:
                vol_ratio = latest_volume / avg_volume
            else:
                vol_ratio = 1.0

            # 只选放量突破的
            if vol_ratio < self.vol_factor:
                continue

            # 综合得分
            scores[symbol] = above_ma_pct * vol_ratio

        if len(scores) < self.top_n:
            return {s: 0.0 for s in self.symbols}

        # 选得分最高的top_n只
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in ranked[:self.top_n]]

        # 等权配置
        weights = {}
        for s in self.symbols:
            weights[s] = 0.25 if s in selected else 0.0  # 每只25%，3只合计75%

        self.last_rebalance_day = n_bars
        self.current_weights = weights
        return weights

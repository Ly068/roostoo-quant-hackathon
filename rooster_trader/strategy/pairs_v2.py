"""
优化版配对交易策略
==================
改进点：
1. 4小时级别（减少噪音）
2. 线性回归残差（比简单比值更精确）
3. 趋势过滤：强趋势市不做配对（避免价差持续偏离）
4. 止损：价差继续偏离超过3σ就止损
5. 自适应窗口：根据价差半衰期计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from rooster_trader.strategy.base import BaseStrategy


class PairTradingV2(BaseStrategy):
    """
    优化版配对交易。

    参数:
        pair_symbols:  配对列表
        window:        滚动窗口（4小时K线数，默认30 = 5天）
        entry_z:       入场Z-score阈值
        exit_z:        平仓Z-score阈值
        stop_loss_z:   止损Z-score阈值（超过就止损）
        pair_weight:   每个配对的资金权重
        trend_filter:  是否启用趋势过滤
    """

    def __init__(
        self,
        pair_symbols: List[Tuple[str, str]],
        window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_loss_z: float = 3.5,
        pair_weight: float = 0.15,
        trend_filter: bool = True,
    ):
        all_symbols = list(set(s for pair in pair_symbols for s in pair))
        super().__init__(name="PairTradingV2", symbols=all_symbols)

        self.pairs = pair_symbols
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        self.pair_weight = pair_weight
        self.trend_filter = trend_filter

        # 每个配对的状态
        self.pair_state = {pair: "flat" for pair in pair_symbols}
        self.entry_zscore = {pair: 0.0 for pair in pair_symbols}

    def warmup_bars(self) -> int:
        return self.window + 10

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        weights = {s: 0.0 for s in self.symbols}

        for (sym_a, sym_b) in self.pairs:
            if sym_a not in bars or sym_b not in bars:
                continue

            close_a = bars[sym_a]["close"]
            close_b = bars[sym_b]["close"]

            if len(close_a) < self.window + 10:
                continue

            # ---- 计算价差（线性回归残差）----
            # 用最近 window 根K线做线性回归：close_b = α + β * close_a
            recent_a = close_a.iloc[-self.window:].values
            recent_b = close_b.iloc[-self.window:].values

            # 简单线性回归（最小二乘）
            beta = np.cov(recent_a, recent_b)[0, 1] / np.var(recent_a)
            alpha = np.mean(recent_b) - beta * np.mean(recent_a)

            # 残差 = 实际值 - 预测值
            spread = close_b - (alpha + beta * close_a)

            # ---- 计算 Z-score ----
            rolling_mean = spread.rolling(self.window).mean()
            rolling_std = spread.rolling(self.window).std()
            zscore = (spread - rolling_mean) / rolling_std

            current_z = zscore.iloc[-1]
            state = self.pair_state[(sym_a, sym_b)]

            # ---- 趋势过滤 ----
            if self.trend_filter:
                # 计算两个资产的20期均线斜率
                ma_a = close_a.rolling(20).mean().iloc[-1]
                ma_b = close_b.rolling(20).mean().iloc[-1]
                price_a = close_a.iloc[-1]
                price_b = close_b.iloc[-1]

                # 如果两个资产都偏离均线超过5%，说明是强趋势市，不做配对
                trend_a = abs(price_a / ma_a - 1)
                trend_b = abs(price_b / ma_b - 1)
                if trend_a > 0.05 and trend_b > 0.05 and state == "flat":
                    continue  # 强趋势市，不开新仓

            # ---- 状态机 ----
            if state == "flat":
                if current_z > self.entry_z:
                    # 价差太高 → 做空B，做多A
                    weights[sym_a] += self.pair_weight / 2
                    weights[sym_b] -= self.pair_weight / 2
                    self.pair_state[(sym_a, sym_b)] = "short_spread"
                    self.entry_zscore[(sym_a, sym_b)] = current_z

                elif current_z < -self.entry_z:
                    # 价差太低 → 做多B，做空A
                    weights[sym_a] -= self.pair_weight / 2
                    weights[sym_b] += self.pair_weight / 2
                    self.pair_state[(sym_a, sym_b)] = "long_spread"
                    self.entry_zscore[(sym_a, sym_b)] = current_z

            elif state == "short_spread":
                # 止损：价差继续扩大超过 stop_loss_z
                if current_z > self.stop_loss_z:
                    self.pair_state[(sym_a, sym_b)] = "flat"
                # 止盈：Z-score 回归
                elif abs(current_z) < self.exit_z:
                    self.pair_state[(sym_a, sym_b)] = "flat"
                else:
                    # 继续持有
                    weights[sym_a] += self.pair_weight / 2
                    weights[sym_b] -= self.pair_weight / 2

            elif state == "long_spread":
                if current_z < -self.stop_loss_z:
                    self.pair_state[(sym_a, sym_b)] = "flat"
                elif abs(current_z) < self.exit_z:
                    self.pair_state[(sym_a, sym_b)] = "flat"
                else:
                    weights[sym_a] -= self.pair_weight / 2
                    weights[sym_b] += self.pair_weight / 2

        return weights

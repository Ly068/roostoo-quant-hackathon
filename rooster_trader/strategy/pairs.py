"""
策略二：配对交易（Pairs Trading）
=================================
逻辑：找两个长期高度相关的资产，当它们的价差偏离均值太远时入场，回归时离场。

适用场景：震荡市。跟动量策略低相关，平滑组合净值。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from rooster_trader.strategy.base import BaseStrategy


class PairTradingStrategy(BaseStrategy):
    """
    配对交易策略。

    参数:
        pair_symbols:  要交易的配对，如 [("BTC-USDT", "ETH-USDT")]
        window:        滚动均值/标准差窗口（小时数）
        entry_z:        入场 Z-score 阈值（默认2.0）
        exit_z:         平仓 Z-score 阈值（默认0.5）
        pair_weight:    每个配对分配的资金比例（默认0.15，即15%）
    """

    def __init__(
        self,
        pair_symbols: List[Tuple[str, str]],
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        pair_weight: float = 0.15,
    ):
        # 展开成所有涉及的 symbol
        all_symbols = list(set(s for pair in pair_symbols for s in pair))
        super().__init__(name="PairTrading", symbols=all_symbols)

        self.pairs = pair_symbols
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.pair_weight = pair_weight

        # 每个配对的当前状态: "flat" / "long_spread" / "short_spread"
        self.pair_state = {pair: "flat" for pair in pair_symbols}

    def warmup_bars(self) -> int:
        return self.window + 5

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        weights = {s: 0.0 for s in self.symbols}

        for (sym_a, sym_b) in self.pairs:
            if sym_a not in bars or sym_b not in bars:
                continue

            close_a = bars[sym_a]["close"]
            close_b = bars[sym_b]["close"]

            if len(close_a) < self.window + 5:
                continue

            # ---- 计算价差比值 ----
            ratio = close_a / close_b

            # ---- 滚动计算 Z-score ----
            rolling_mean = ratio.rolling(self.window).mean()
            rolling_std = ratio.rolling(self.window).std()
            zscore = (ratio - rolling_mean) / rolling_std

            current_z = zscore.iloc[-1]
            state = self.pair_state[(sym_a, sym_b)]

            # ---- 状态机：根据 Z-score 切换仓位 ----
            if state == "flat":
                if current_z > self.entry_z:
                    # 价差太高 → 做空A，做多B
                    weights[sym_a] -= self.pair_weight / 2
                    weights[sym_b] += self.pair_weight / 2
                    self.pair_state[(sym_a, sym_b)] = "short_spread"

                elif current_z < -self.entry_z:
                    # 价差太低 → 做多A，做空B
                    weights[sym_a] += self.pair_weight / 2
                    weights[sym_b] -= self.pair_weight / 2
                    self.pair_state[(sym_a, sym_b)] = "long_spread"

            elif state == "short_spread":
                if abs(current_z) < self.exit_z:
                    # Z-score 回归，平仓
                    self.pair_state[(sym_a, sym_b)] = "flat"

            elif state == "long_spread":
                if abs(current_z) < self.exit_z:
                    self.pair_state[(sym_a, sym_b)] = "flat"

        return weights

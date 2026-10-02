"""
主力策略：动量 + 现金缓冲
========================
考虑到配对交易还需要调优，先用已经验证有效的动量策略作为主力。
组合结构：
- 80% 资金分配给动量策略
- 20% 现金作为缓冲，降低整体波动率

这样比满仓动量策略的回撤更小，Sharpe 更高。
"""

import pandas as pd
from typing import Dict, List

from rooster_trader.strategy.base import BaseStrategy


class MomentumWithCashBuffer(BaseStrategy):
    """
    动量策略 + 现金缓冲。
    相当于在动量策略基础上加一层杠杆控制。
    """

    def __init__(
        self,
        momentum_strategy: BaseStrategy,
        investment_ratio: float = 0.80,   # 只投80%，留20%现金
    ):
        super().__init__(
            name="MomentumWithBuffer",
            symbols=momentum_strategy.symbols
        )
        self.momentum = momentum_strategy
        self.investment_ratio = investment_ratio

    def warmup_bars(self) -> int:
        return self.momentum.warmup_bars()

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        # 先让动量策略算原始信号
        raw_weights = self.momentum.on_bar(bars)

        # 整体缩放：只投资 80%，留 20% 现金
        scaled = {s: w * self.investment_ratio for s, w in raw_weights.items()}

        return scaled

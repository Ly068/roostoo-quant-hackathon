"""
多策略组合引擎
==============
把多个子策略的信号按波动率倒数加权，合成一个总策略。

核心思想：
- 每个子策略输出自己的目标仓位
- 组合层根据各策略近期的波动率动态分配权重
- 波动率高的策略分配更少资金，波动率低的分配更多
- 这样组合的整体波动率更低，Sharpe 更高
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from rooster_trader.strategy.base import BaseStrategy


class MultiStrategyEngine(BaseStrategy):
    """
    多策略组合引擎。

    参数:
        strategies:  子策略列表
        weight_method: 权重分配方法
            - "equal": 等权
            - "vol_inverse": 波动率倒数加权（推荐）
        vol_window:  计算各策略波动率的窗口（K线数）
        max_total_weight: 组合总权重上限
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weight_method: str = "vol_inverse",
        vol_window: int = 500,
        max_total_weight: float = 1.0,
    ):
        # 收集所有子策略涉及的标的
        all_symbols = list(set(s for strat in strategies for s in strat.symbols))
        super().__init__(name="MultiStrategy", symbols=all_symbols)

        self.strategies = strategies
        self.weight_method = weight_method
        self.vol_window = vol_window
        self.max_total_weight = max_total_weight

        # 记录各策略的历史净值，用于计算波动率权重
        self.strategy_equity: Dict[str, list] = {s.name: [1.0] for s in strategies}

    def warmup_bars(self) -> int:
        # 所有子策略中最大的预热期
        return max(s.warmup_bars() for s in self.strategies) + 50

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        1. 每个子策略独立计算目标仓位
        2. 根据各策略近期波动率分配权重
        3. 加权求和得到组合总仓位
        """
        # ---- 第一步：收集每个子策略的信号 ----
        strategy_signals = {}
        for strat in self.strategies:
            try:
                weights = strat.on_bar(bars)
                strategy_signals[strat.name] = weights

                # 估算该策略本期收益（用各持仓的加权收益近似）
                ret = self._estimate_strategy_return(weights, bars)
                self.strategy_equity[strat.name].append(
                    self.strategy_equity[strat.name][-1] * (1 + ret)
                )
            except Exception as e:
                # 单个策略出错不影响整个组合
                print(f"⚠ 策略 {strat.name} 出错: {e}")
                strategy_signals[strat.name] = {s: 0.0 for s in strat.symbols}

        # ---- 第二步：计算各策略权重 ----
        weights = self._calculate_weights()

        # ---- 第三步：合成组合信号 ----
        combined = {s: 0.0 for s in self.symbols}
        for strat_name, strat_weights in strategy_signals.items():
            w = weights[strat_name]
            for symbol, target_w in strat_weights.items():
                if symbol in combined:
                    combined[symbol] += target_w * w

        # ---- 第四步：归一化总敞口 ----
        total_exposure = sum(abs(w) for w in combined.values())
        if total_exposure > self.max_total_weight and total_exposure > 0:
            scale = self.max_total_weight / total_exposure
            combined = {s: w * scale for s, w in combined.items()}

        return combined

    def _estimate_strategy_return(self, weights: dict, bars: dict) -> float:
        """
        估算单个策略这根K线的收益率。
        用各标的最近一根K线的收益率 × 权重 加权求和。
        """
        total_ret = 0.0
        for symbol, w in weights.items():
            if symbol not in bars or abs(w) < 1e-6:
                continue
            close = bars[symbol]["close"]
            if len(close) < 2:
                continue
            ret = close.iloc[-1] / close.iloc[-2] - 1
            total_ret += w * ret
        return total_ret

    def _calculate_weights(self) -> Dict[str, float]:
        """
        根据各策略近期波动率计算权重。
        波动率越低的策略，分配的权重越高。
        """
        strat_names = [s.name for s in self.strategies]

        if self.weight_method == "equal":
            w = 1.0 / len(strat_names)
            return {name: w for name in strat_names}

        # 波动率倒数加权
        inv_vols = {}
        for name in strat_names:
            equity = self.strategy_equity[name]
            if len(equity) < 20:
                inv_vols[name] = 1.0
                continue
            # 取最近 vol_window 根计算收益率波动率
            recent = equity[-self.vol_window:] if len(equity) > self.vol_window else equity
            rets = pd.Series(recent).pct_change().dropna()
            vol = rets.std()
            inv_vols[name] = 1.0 / (vol + 1e-8)  # 加小量防止除零

        total_inv_vol = sum(inv_vols.values())
        weights = {name: v / total_inv_vol for name, v in inv_vols.items()}

        return weights

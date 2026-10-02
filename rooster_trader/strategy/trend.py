"""
时序动量 / 趋势跟踪策略（Time-Series Momentum + Trend Filter）
============================================================
与横截面"多空对冲"不同，本策略只做多、且带市场状态保护：

  做多条件（两者同时满足）：
    1. 时序动量为正：close[t] > close[t-lookback]（过去N天上涨）
    2. 趋势过滤（可选）：close[t] > MA(ma_window)（价格在均线上方）

  - 满足条件的币种等权做多，总多头敞口 = max_total_exposure
  - 无任何币种满足时 → 全部转为现金（自动避险，不硬扛熊市）
  - 每 rebalance_hours 根据最新趋势更新

设计依据：
  - 加密市场有长期上涨偏差，做空山寨币易被挤压 → 只做多
  - Moskowitz, Ooi & Pedersen (2012) 时序动量在多市场稳健
  - "无趋势空仓"规避了动量在震荡/下跌市的崩溃
"""
import numpy as np
import pandas as pd
from typing import Dict

from rooster_trader.strategy.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    def __init__(
        self,
        symbols: list,
        lookback_hours: int = 168,       # 时序动量窗口（7天）
        ma_hours: int = 168,             # 均线过滤窗口
        use_ma_filter: bool = True,      # 是否要求价格在均线上方
        max_total_exposure: float = 1.0, # 有趋势时的总多头敞口上限
        rebalance_hours: int = 72,
    ):
        super().__init__(name="TrendFollowing", symbols=symbols)
        self.lookback = lookback_hours
        self.ma_hours = ma_hours
        self.use_ma_filter = use_ma_filter
        self.max_total_exposure = max_total_exposure
        self.rebalance_interval = rebalance_hours

        self.last_rebalance_idx = -999
        self.current_weights: Dict[str, float] = {}

    def warmup_bars(self) -> int:
        return max(self.lookback, self.ma_hours) + 5

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        n_bars = len(list(bars.values())[0])
        bars_since = n_bars - self.last_rebalance_idx
        if bars_since < self.rebalance_interval and self.current_weights:
            return self.current_weights

        eligible = []
        for symbol in self.symbols:
            close = bars[symbol]["close"]
            if len(close) < max(self.lookback, self.ma_hours) + 1:
                continue

            ts_positive = close.iloc[-1] > close.iloc[-self.lookback]
            if self.use_ma_filter:
                ma = close.rolling(self.ma_hours).mean().iloc[-1]
                trend_ok = close.iloc[-1] > ma
            else:
                trend_ok = True

            if ts_positive and trend_ok:
                eligible.append(symbol)

        weights = {s: 0.0 for s in self.symbols}
        if eligible:
            # 等权，总敞口=max_total_exposure；信号越多单币越分散
            per = self.max_total_exposure / len(eligible)
            for s in eligible:
                weights[s] = per

        self.last_rebalance_idx = n_bars
        self.current_weights = weights
        return weights

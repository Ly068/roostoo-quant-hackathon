"""
策略一：多因子加密动量策略
==========================
逻辑：在加密货币池中，每周计算综合动量得分，做多前2、做空后2。

三个因子：
1. 收益率动量：过去7天涨跌幅
2. 波动率调整动量：收益率 / 近期波动率（单位风险收益）
3. 成交量确认：上涨时是否放量
"""

import numpy as np
import pandas as pd
from typing import Dict

from rooster_trader.strategy.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    加密货币多因子动量策略。

    参数:
        lookback_days:  动量回看天数（默认7天 = 一周）
        top_n:          做多/做空各选几个（默认2个）
        rebalance_hours: 再平衡间隔小时数（默认168 = 7天）
        vol_window:     波动率计算窗口（默认20小时）
        vol_target:     单标的目标波动率（用于仓位缩放）
    """

    def __init__(
        self,
        symbols: list,
        lookback_hours: int = 168,      # 7天 = 168小时（加密市场反应快）
        top_n: int = 1,                  # top1做多+top1做空（更集中）
        rebalance_hours: int = 72,       # 3天再平衡
        vol_window: int = 20,
        vol_target: float = 0.02,
        position_size: float = 0.25,     # 单个仓位25%（top1多+top1空=50%总敞口）
        sizing_method: str = "equal",    # "equal"=等权；"inverse_vol"=波动率倒数加权
        long_only: bool = False,         # True=只做多top_n（上涨市不做空，避免空头挤压）
    ):
        super().__init__(name="CryptoMomentum", symbols=symbols)
        self.lookback = lookback_hours
        self.top_n = top_n
        self.rebalance_interval = rebalance_hours
        self.vol_window = vol_window
        self.vol_target = vol_target
        self.position_size = position_size
        self.sizing_method = sizing_method
        self.long_only = long_only

        # 内部状态：记录上次再平衡时间、当前持仓
        self.last_rebalance_idx = -999
        self.current_weights: Dict[str, float] = {}

    def warmup_bars(self) -> int:
        """需要预热的K线数：动量窗口 + 波动率窗口"""
        return max(self.lookback, self.vol_window) + 5

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        每根K线收盘后调用。
        返回 {symbol: target_weight}，正=做多，负=做空。
        """
        n_bars = len(list(bars.values())[0])  # 当前总K线数

        # ---- 判断是否到再平衡时间 ----
        bars_since_rebalance = n_bars - self.last_rebalance_idx
        if bars_since_rebalance < self.rebalance_interval and self.current_weights:
            # 还没到再平衡日，保持当前仓位不变
            return self.current_weights

        # ---- 计算每个标的的三个因子 ----
        scores = {}
        vols = {}   # 记录近期波动率，供逆波动加权使用
        for symbol in self.symbols:
            df = bars[symbol]
            close = df["close"]
            volume = df["volume"]

            if len(close) < self.lookback + self.vol_window:
                continue

            # 因子1：过去 lookback 小时的累计收益率
            ret = close.iloc[-1] / close.iloc[-self.lookback] - 1

            # 因子2：波动率调整动量 = 收益率 / 近期波动率
            hourly_returns = close.pct_change().dropna()
            recent_vol = hourly_returns.iloc[-self.vol_window:].std()
            vols[symbol] = recent_vol if recent_vol and recent_vol > 0 else 1e-6
            if recent_vol > 0:
                vol_adj_ret = ret / recent_vol
            else:
                vol_adj_ret = 0

            # 因子3：成交量确认 —— 最近20小时平均量 / 之前20小时平均量
            recent_vol_avg = volume.iloc[-self.vol_window:].mean()
            prev_vol_avg = volume.iloc[-2*self.vol_window:-self.vol_window].mean()
            if prev_vol_avg > 0:
                volume_ratio = recent_vol_avg / prev_vol_avg
            else:
                volume_ratio = 1.0
            # 截断极端比值，避免 log(0)=-inf 或异常放大
            volume_ratio = float(np.clip(volume_ratio, 0.1, 10.0))

            # 合成综合得分：三因子等权
            # 注意：volume_ratio 只给"上涨"加分，下跌时不额外惩罚
            momentum_component = ret * (1 + np.log(volume_ratio))
            combined_score = 0.5 * vol_adj_ret + 0.5 * momentum_component

            scores[symbol] = combined_score

        if len(scores) < self.top_n * 2:
            # 数据不够，保持空仓
            return {s: 0.0 for s in self.symbols}

        # ---- 排名：选前2做多，后2做空 ----
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        long_symbols = [s for s, _ in ranked[:self.top_n]]
        short_symbols = [] if self.long_only else [s for s, _ in ranked[-self.top_n:]]

        # ---- 计算目标权重 ----
        def allocate(selected, sign):
            """对选出的标的分配权重，总敞口固定为 top_n*position_size"""
            if self.sizing_method != "inverse_vol":
                return {s: sign * self.position_size for s in selected}
            # 波动率倒数加权：高波动给小仓、低波动给大仓
            inv = {s: 1.0 / max(vols.get(s, 1e-6), 1e-6) for s in selected}
            total_inv = sum(inv.values())
            total_exposure = self.top_n * self.position_size
            return {s: sign * (inv[s] / total_inv) * total_exposure for s in selected}

        long_alloc = allocate(long_symbols, +1)
        short_alloc = allocate(short_symbols, -1)
        weights = {
            s: long_alloc.get(s, short_alloc.get(s, 0.0))
            for s in self.symbols
        }

        # 记录本次再平衡
        self.last_rebalance_idx = n_bars
        self.current_weights = weights

        return weights

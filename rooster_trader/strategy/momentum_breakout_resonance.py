"""
方案E：动量+突破共振加仓策略
逻辑：
  - 基础：30天动量top1做多 + top1做空
  - 正常仓位：30%单仓位（60%总敞口）
  - 共振加仓：当动量信号与波动率突破信号共振时，加仓到50%单仓位（100%总敞口）
  - 突破信号：Bollinger Band挤压后价格突破轨道
"""
import numpy as np
import pandas as pd
from typing import Dict

from .base import BaseStrategy


class MomentumBreakoutResonance(BaseStrategy):
    """
    动量+突破共振加仓策略

    信号生成：
    1. 计算30天动量综合得分（收益率+波动率调整+成交量确认）
    2. 计算波动率突破信号（BB挤压后突破）
    3. 动量top1做多，bottom1做空
    4. 如果该币种同时有突破信号 → 共振加仓（50%仓位）
    5. 否则 → 正常仓位（30%仓位）
    """

    def __init__(
        self,
        symbols: list,
        lookback_hours: int = 720,      # 30天动量窗口
        rebalance_hours: int = 72,       # 3天再平衡
        normal_position: float = 0.30,   # 正常单仓位30%（60%总敞口）
        resonance_position: float = 0.50, # 共振单仓位50%（100%总敞口）
        bb_window: int = 20,              # Bollinger Band窗口
        bb_std: float = 2.0,              # BB标准差倍数
        squeeze_percentile: float = 0.2,  # 挤压阈值（20%分位）
        vol_window: int = 20,
    ):
        super().__init__(name="MomentumBreakoutResonance", symbols=symbols)
        self.lookback = lookback_hours
        self.rebalance_interval = rebalance_hours
        self.normal_position = normal_position
        self.resonance_position = resonance_position
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.squeeze_percentile = squeeze_percentile
        self.vol_window = vol_window

        # 内部状态
        self.last_rebalance_idx = -999
        self.current_weights: Dict[str, float] = {}

    def warmup_bars(self) -> int:
        return max(self.lookback, self.bb_window * 5) + 5

    def _compute_momentum_scores(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """计算30天动量综合得分"""
        scores = {}
        for symbol in self.symbols:
            df = bars[symbol]
            if len(df) < self.lookback + self.vol_window:
                scores[symbol] = 0.0
                continue

            close = df["close"]
            volume = df["volume"]

            # 因子1：过去lookback小时累计收益率
            ret = close.iloc[-1] / close.iloc[-self.lookback] - 1

            # 因子2：波动率调整动量
            hourly_returns = close.pct_change().dropna()
            recent_vol = hourly_returns.iloc[-self.vol_window:].std()
            vol_adj_ret = ret / recent_vol if recent_vol > 0 else 0

            # 因子3：成交量确认
            recent_vol_avg = volume.iloc[-20:].mean()
            prev_vol_avg = volume.iloc[-40:-20].mean()
            volume_ratio = recent_vol_avg / prev_vol_avg if prev_vol_avg > 0 else 1.0
            momentum_component = ret * (1 + np.log(max(volume_ratio, 0.01)))

            # 合成综合得分
            combined_score = 0.5 * vol_adj_ret + 0.5 * momentum_component
            scores[symbol] = combined_score

        return scores

    def _compute_breakout_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        计算波动率突破信号
        返回：{symbol: +1(向上突破), -1(向下突破), 0(无信号)}
        """
        signals = {}
        for symbol in self.symbols:
            df = bars[symbol]
            if len(df) < self.bb_window * 5:
                signals[symbol] = 0
                continue

            close = df["close"]
            high = df["high"]
            low = df["low"]

            # Bollinger Band
            bb_sma = close.rolling(self.bb_window).mean()
            bb_std_val = close.rolling(self.bb_window).std()
            bb_upper = bb_sma + self.bb_std * bb_std_val
            bb_lower = bb_sma - self.bb_std * bb_std_val
            bb_width = (bb_upper - bb_lower) / bb_sma

            # 挤压判断：当前BB宽度 < 历史分位
            bb_width_history = bb_width.iloc[-100:]
            squeeze_threshold = bb_width_history.quantile(self.squeeze_percentile)
            is_squeeze = bb_width.iloc[-1] < squeeze_threshold

            # 突破判断
            current_close = close.iloc[-1]
            current_upper = bb_upper.iloc[-1]
            current_lower = bb_lower.iloc[-1]

            breakout_up = current_close > current_upper
            breakout_down = current_close < current_lower

            # 只有在挤压状态下的突破才算有效信号
            if is_squeeze and breakout_up:
                signals[symbol] = 1
            elif is_squeeze and breakout_down:
                signals[symbol] = -1
            else:
                signals[symbol] = 0

        return signals

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """每根K线收盘后调用，返回目标权重"""
        n_bars = len(list(bars.values())[0])

        # 再平衡间隔控制
        bars_since_rebalance = n_bars - self.last_rebalance_idx
        if bars_since_rebalance < self.rebalance_interval and self.current_weights:
            return self.current_weights

        # 计算动量得分
        momentum_scores = self._compute_momentum_scores(bars)

        # 计算突破信号
        breakout_signals = self._compute_breakout_signals(bars)

        # 排名
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        long_symbol = ranked[0][0] if ranked else None
        short_symbol = ranked[-1][0] if ranked else None

        # 计算目标权重
        weights = {s: 0.0 for s in self.symbols}

        if long_symbol:
            # 检查是否共振（动量做多 + 突破向上）
            if breakout_signals.get(long_symbol, 0) == 1:
                weights[long_symbol] = self.resonance_position  # 共振加仓
            else:
                weights[long_symbol] = self.normal_position  # 正常仓位

        if short_symbol and short_symbol != long_symbol:
            # 检查是否共振（动量做空 + 突破向下）
            if breakout_signals.get(short_symbol, 0) == -1:
                weights[short_symbol] = -self.resonance_position  # 共振加仓
            else:
                weights[short_symbol] = -self.normal_position  # 正常仓位

        self.current_weights = weights
        self.last_rebalance_idx = n_bars
        return weights

"""
动态仓位策略
============
根据比赛阶段自动调整仓位比例：
- 第1-3天（观察期）：50%仓位，小仓位试跑
- 第4-10天（确认期）：80%仓位，策略验证有效后加仓
- 第11-14天（收尾期）：60%仓位，降仓位保收益、拉Sharpe

同时根据策略信号强度微调：
- 信号强（多空分化明显）：仓位 +10%
- 信号弱（多空不明显）：仓位 -10%
"""

import pandas as pd
from typing import Dict
from datetime import datetime, timedelta

from rooster_trader.strategy.base import BaseStrategy


class DynamicPositionStrategy(BaseStrategy):
    """
    动态仓位包装器：在子策略外面套一层仓位控制。
    """

    def __init__(
        self,
        underlying_strategy: BaseStrategy,
        competition_start_date: str,   # 比赛开始日期，格式 "2026-10-04"
        competition_days: int = 14,
    ):
        super().__init__(
            name=f"Dynamic_{underlying_strategy.name}",
            symbols=underlying_strategy.symbols,
        )
        self.strategy = underlying_strategy
        self.start_date = datetime.strptime(competition_start_date, "%Y-%m-%d")
        self.competition_days = competition_days

    def warmup_bars(self) -> int:
        return self.strategy.warmup_bars()

    def _get_position_ratio(self, current_date: datetime) -> float:
        """
        根据比赛天数返回目标仓位比例。
        """
        # 统一时区：如果current_date有时区，转为naive（避免offset-naive/aware比较错误）
        if hasattr(current_date, 'tzinfo') and current_date.tzinfo is not None:
            current_date = current_date.replace(tzinfo=None)

        days_elapsed = (current_date - self.start_date).days + 1

        if days_elapsed <= 3:
            # 观察期：60%仓位（稍微提高，避免太保守）
            return 0.60
        elif days_elapsed <= 10:
            # 确认期：100%仓位（满仓进攻）
            return 1.00
        else:
            # 收尾期：70%仓位（锁定利润）
            return 0.70

    def _get_signal_strength(self, weights: Dict[str, float]) -> float:
        """
        估算信号强度：多空分化越明显，信号越强。
        返回 0.8 - 1.2 的乘数。
        """
        long_weights = [w for w in weights.values() if w > 0]
        short_weights = [w for w in weights.values() if w < 0]

        # 只做多策略（无空头）：单边是设计使然，不做"信号弱"惩罚
        if not short_weights:
            return 1.0
        if not long_weights:
            return 1.0

        long_total = sum(long_weights)
        short_total = sum(abs(w) for w in short_weights)

        # 多空均衡度：越接近1:1，信号越可靠
        balance = min(long_total, short_total) / max(long_total, short_total)

        if balance > 0.8:
            return 1.1  # 多空均衡，信号强，加10%
        elif balance > 0.5:
            return 1.0  # 正常
        else:
            return 0.85  # 多空不均衡，信号弱，降15%

    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        # 1. 子策略计算原始信号
        raw_weights = self.strategy.on_bar(bars)

        # 2. 获取当前日期（从数据索引取，或用系统时间）
        try:
            current_date = list(bars.values())[0].index[-1]
            if hasattr(current_date, 'to_pydatetime'):
                current_date = current_date.to_pydatetime()
        except Exception:
            current_date = datetime.now()

        # 3. 根据比赛阶段确定基础仓位
        base_ratio = self._get_position_ratio(current_date)

        # 4. 根据信号强度微调
        signal_multiplier = self._get_signal_strength(raw_weights)

        # 5. 最终仓位比例（限制在 0.3 - 1.0 之间）
        final_ratio = max(0.3, min(1.0, base_ratio * signal_multiplier))

        # 6. 缩放所有仓位
        scaled = {s: w * final_ratio for s, w in raw_weights.items()}

        return scaled

    def get_current_position_info(self) -> dict:
        """获取当前仓位信息，用于日志记录"""
        now = datetime.now()
        days_elapsed = (now - self.start_date).days + 1
        base_ratio = self._get_position_ratio(now)

        phase = "观察期" if days_elapsed <= 3 else ("确认期" if days_elapsed <= 10 else "收尾期")

        return {
            "days_elapsed": days_elapsed,
            "phase": phase,
            "base_position_ratio": base_ratio,
        }

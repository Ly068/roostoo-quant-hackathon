"""
增强版多策略组合层（Enhanced Multi-Strategy Portfolio）
======================================================
核心创新：
1. 波动率倒数加权（Risk Parity思想）
2. 市场状态自适应权重调整（趋势市加动量，震荡市加配对）
3. 策略级熔断（单个策略连续亏损自动降权）
4. XGBoost信号过滤（ML预测方向与策略信号一致才开仓）
5. 全局仓位系数（高波动降仓）

四个子策略：
- momentum: 多因子动量（30天窗口，3天再平衡）
- pairs: 统计套利（BTC/ETH配对，4小时级别）
- breakout: 波动率突破（Bollinger挤压+突破，ATR止损）
- xgboost: 机器学习择时（13特征→XGBoost→方向预测）
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from rooster_trader.strategy.base import BaseStrategy
from rooster_trader.strategy.regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)


class EnhancedMultiStrategy(BaseStrategy):
    """
    增强版多策略组合。

    参数:
        symbols: 交易对列表
        strategies: 策略实例字典 {name: strategy_instance}
        base_weights: 基础权重 {name: weight}，默认等权
        vol_lookback: 波动率计算窗口（默认48小时）
        use_regime_adjust: 是否启用市场状态自适应（默认True）
        use_xgb_filter: 是否启用XGBoost信号过滤（默认True）
        use_strategy_circuit_breaker: 是否启用策略级熔断（默认True）
        cb_consecutive_losses: 熔断触发的连续亏损次数（默认3）
        cb_reduction: 熔断后权重降低比例（默认0.5）
        xgb_filter_threshold: XGBoost过滤阈值（默认0.3，方向一致性>0.3才通过）
    """

    def __init__(
        self,
        symbols: list,
        strategies: Dict[str, BaseStrategy],
        base_weights: Optional[Dict[str, float]] = None,
        vol_lookback: int = 48,
        use_regime_adjust: bool = True,
        use_xgb_filter: bool = True,
        use_strategy_circuit_breaker: bool = True,
        cb_consecutive_losses: int = 3,
        cb_reduction: float = 0.5,
        xgb_filter_threshold: float = 0.3,
    ):
        super().__init__(name="EnhancedMultiStrategy", symbols=symbols)
        self.strategies = strategies
        self.vol_lookback = vol_lookback
        self.use_regime_adjust = use_regime_adjust
        self.use_xgb_filter = use_xgb_filter
        self.use_strategy_circuit_breaker = use_strategy_circuit_breaker
        self.cb_consecutive_losses = cb_consecutive_losses
        self.cb_reduction = cb_reduction
        self.xgb_filter_threshold = xgb_filter_threshold

        # 基础权重（默认等权）
        if base_weights is None:
            n = len(strategies)
            self.base_weights = {name: 1.0 / n for name in strategies}
        else:
            total = sum(base_weights.values())
            self.base_weights = {name: w / total for name, w in base_weights.items()}

        # 市场状态检测器
        self.regime_detector = MarketRegimeDetector()

        # 策略级熔断状态
        self.consecutive_losses: Dict[str, int] = {name: 0 for name in strategies}
        self.circuit_breaker_active: Dict[str, bool] = {name: False for name in strategies}
        self.last_portfolio_value: Optional[float] = None

        # 策略收益历史（用于波动率倒数加权）
        self.strategy_returns: Dict[str, List[float]] = {name: [] for name in strategies}
        self.last_weights: Dict[str, Dict[str, float]] = {}

    def _calculate_volatility_weights(self) -> Dict[str, float]:
        """
        波动率倒数加权：低波动策略多分资金。
        Weight_i = (1/σ_i) / Σ(1/σ_j)
        """
        volatilities = {}
        for name, returns in self.strategy_returns.items():
            if len(returns) >= 10:
                volatilities[name] = np.std(returns[-self.vol_lookback:]) + 1e-10
            else:
                volatilities[name] = 0.05  # 默认波动率

        # 波动率倒数
        inv_vol = {name: 1.0 / vol for name, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        return {name: w / total_inv_vol for name, w in inv_vol.items()}

    def _apply_xgb_filter(
        self,
        weights: Dict[str, float],
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        XGBoost信号过滤：只有当ML预测方向与策略信号一致时才保留仓位。
        如果没有xgboost策略，直接返回原权重。
        """
        if not self.use_xgb_filter or "xgboost" not in self.strategies:
            return weights

        xgb_strategy = self.strategies["xgboost"]
        if not hasattr(xgb_strategy, "get_signal_filter"):
            return weights

        filters = xgb_strategy.get_signal_filter(data)
        filtered = {}

        for symbol, weight in weights.items():
            if weight == 0:
                filtered[symbol] = 0
                continue

            ml_direction = filters.get(symbol, 0)
            strategy_direction = 1 if weight > 0 else -1

            # 方向一致性：ML方向 × 策略方向，正=一致，负=矛盾
            consistency = ml_direction * strategy_direction

            if consistency >= self.xgb_filter_threshold:
                # 方向一致，保留仓位，且按置信度微调
                filtered[symbol] = weight * min(1.0, 0.5 + abs(ml_direction))
            elif consistency <= -self.xgb_filter_threshold:
                # 方向矛盾，过滤掉这个信号
                logger.info("[XGB过滤] %s 策略方向与ML矛盾(ML=%.2f)，过滤",
                            symbol, ml_direction)
                filtered[symbol] = 0
            else:
                # 中性，保留半仓
                filtered[symbol] = weight * 0.5

        return filtered

    def _update_circuit_breaker(self, current_value: float):
        """更新策略级熔断状态"""
        if self.last_portfolio_value is None:
            self.last_portfolio_value = current_value
            return

        portfolio_return = (current_value - self.last_portfolio_value) / (self.last_portfolio_value + 1e-10)

        # 简化版：组合亏损时所有策略计数+1，盈利时清零
        # 更精确的做法是跟踪每个策略的单独收益，但这里简化处理
        if portfolio_return < -0.001:  # 亏损超过0.1%
            for name in self.strategies:
                self.consecutive_losses[name] += 1
                if (self.consecutive_losses[name] >= self.cb_consecutive_losses
                        and not self.circuit_breaker_active[name]):
                    self.circuit_breaker_active[name] = True
                    logger.warning("[策略熔断] %s 连续亏损%d次，权重降低%.0f%%",
                                name, self.consecutive_losses[name], self.cb_reduction * 100)
        elif portfolio_return > 0.001:
            for name in self.strategies:
                self.consecutive_losses[name] = 0
                if self.circuit_breaker_active[name]:
                    self.circuit_breaker_active[name] = False
                    logger.info("[策略熔断恢复] %s", name)

        self.last_portfolio_value = current_value

    def on_bar(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        组合策略主逻辑：
        1. 每个子策略计算目标权重
        2. 波动率倒数加权 + 市场状态调整 + 熔断调整
        3. 汇总所有策略的权重
        4. XGBoost信号过滤
        5. 全局仓位系数调整
        """
        # ---- 1. 市场状态检测 ----
        if self.use_regime_adjust:
            regime = self.regime_detector.detect(data)
            regime_adj = regime["strategy_weights_adj"]
            global_adj = regime["global_position_adj"]
        else:
            regime_adj = {name: 1.0 for name in self.strategies}
            global_adj = 1.0

        # ---- 2. 波动率倒数加权 ----
        vol_weights = self._calculate_volatility_weights()

        # ---- 3. 计算最终策略权重 ----
        final_strategy_weights = {}
        for name in self.strategies:
            # 基础权重 × 波动率倒数 × 市场状态调整
            w = self.base_weights[name] * vol_weights[name] * regime_adj[name]

            # 策略级熔断
            if self.use_strategy_circuit_breaker and self.circuit_breaker_active[name]:
                w *= self.cb_reduction

            final_strategy_weights[name] = w

        # 归一化
        total_w = sum(final_strategy_weights.values())
        if total_w > 0:
            final_strategy_weights = {name: w / total_w for name, w in final_strategy_weights.items()}

        # ---- 4. 每个子策略计算目标权重并加权汇总 ----
        combined_weights = {s: 0.0 for s in self.symbols}

        for name, strategy in self.strategies.items():
            strategy_weight = final_strategy_weights[name]
            if strategy_weight < 0.01:
                continue

            try:
                sub_weights = strategy.on_bar(data)
                self.last_weights[name] = sub_weights

                for symbol, w in sub_weights.items():
                    combined_weights[symbol] += w * strategy_weight

            except Exception as e:
                logger.error("[组合] 策略 %s 执行失败: %s", name, e)

        # ---- 5. XGBoost信号过滤 ----
        combined_weights = self._apply_xgb_filter(combined_weights, data)

        # ---- 6. 全局仓位系数调整 ----
        combined_weights = {s: w * global_adj for s, w in combined_weights.items()}

        # ---- 7. 记录策略权重（用于调试） ----
        active = {k: f"{v:.2%}" for k, v in combined_weights.items() if abs(v) > 0.001}
        logger.info("[组合] 策略权重: %s", {k: f"{v:.1%}" for k, v in final_strategy_weights.items()})
        logger.info("[组合] 目标仓位: %s", active)

        return combined_weights

    def get_strategy_weights(self) -> Dict[str, float]:
        """获取当前各策略的权重（用于调试和日志）"""
        return self._calculate_volatility_weights()

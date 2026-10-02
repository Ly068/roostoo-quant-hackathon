"""风控管理器：在每笔交易前检查，在每日收盘后检查组合风险"""
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控参数配置，与 config.yaml 对应"""
    max_position_pct: float = 0.10       # 单标的最大仓位
    max_total_exposure: float = 1.0      # 总净敞口上限
    daily_drawdown_limit: float = 0.03   # 日内回撤触发减仓
    hard_drawdown_limit: float = 0.05    # 日内回撤触发全平
    consecutive_loss_pause: int = 3      # 连亏N笔暂停
    pause_minutes: int = 30              # 暂停时长


class RiskManager:
    """
    风控层。三层防护：
    1. 下单前：检查单标的仓位、总敞口
    2. 每日收盘后：检查日内回撤，触发减仓或平仓
    3. 异常熔断：连续亏损自动暂停
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.peak_value: float = 100_000.0   # 初始净值
        self.daily_start_value: float = 100_000.0
        self.consecutive_losses: int = 0
        self.paused_until: float = 0.0       # 暂停到期时间戳
        self.reduced: bool = False           # 是否已减仓

    def check_order(
        self,
        symbol: str,
        target_weight: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
    ) -> tuple[bool, float, str]:
        """
        下单前检查。
        返回: (是否允许下单, 修正后的目标权重, 原因说明)
        """
        import time

        # 熔断中
        if time.time() < self.paused_until:
            return False, 0.0, "风控暂停中"

        # 1. 单标的仓位上限
        if abs(target_weight) > self.config.max_position_pct:
            capped = self.config.max_position_pct * (1 if target_weight > 0 else -1)
            logger.warning(
                "%s 仓位 %.2f%% 超过上限 %.2f%%，已截断",
                symbol, abs(target_weight) * 100, self.config.max_position_pct * 100
            )
            target_weight = capped

        # 2. 总敞口检查（做多+做空绝对值之和不超过 max_total_exposure）
        total_exposure = sum(abs(w) for w in current_positions.values()) + abs(target_weight)
        if total_exposure > self.config.max_total_exposure:
            scale = self.config.max_total_exposure / total_exposure
            target_weight *= scale
            logger.warning("总敞口超限，整体缩放至 %.0f%%", scale * 100)

        return True, target_weight, "OK"

    def check_daily_risk(self, current_value: float) -> str:
        """
        每日收盘后检查组合风险。
        返回动作: "HOLD" / "REDUCE" / "FLATTEN"
        """
        drawdown = (self.daily_start_value - current_value) / self.daily_start_value

        if drawdown >= self.config.hard_drawdown_limit:
            logger.critical("日内回撤 %.2f%% 触发硬止损，全部平仓！", drawdown * 100)
            self._pause_trading()
            return "FLATTEN"

        if drawdown >= self.config.daily_drawdown_limit and not self.reduced:
            logger.warning("日内回撤 %.2f%% 触发减仓，仓位减半", drawdown * 100)
            self.reduced = True
            return "REDUCE"

        return "HOLD"

    def record_trade_result(self, pnl: float):
        """记录每笔交易盈亏，用于连亏熔断"""
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.consecutive_loss_pause:
                logger.warning("连续亏损 %d 笔，暂停交易 %d 分钟",
                               self.consecutive_losses, self.config.pause_minutes)
                self._pause_trading()
        else:
            self.consecutive_losses = 0

    def _pause_trading(self):
        import time
        self.paused_until = time.time() + self.config.pause_minutes * 60

    def reset_daily(self, portfolio_value: float):
        """新交易日开始时调用，重置日级风控"""
        self.daily_start_value = portfolio_value
        self.reduced = False
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

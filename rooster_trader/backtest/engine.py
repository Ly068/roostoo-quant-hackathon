"""回测引擎：用历史K线数据测试策略，计算收益、Sharpe、Sortino、Calmar等指标"""
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

from rooster_trader.strategy.base import BaseStrategy
from rooster_trader.risk.manager import RiskManager, RiskConfig


@dataclass
class BacktestResult:
    """回测结果汇总"""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series


class Backtester:
    """
    事件驱动回测引擎。
    逐根K线喂给策略，模拟下单、扣手续费、计算净值。
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        fee_rate: float = 0.001,       # 单边手续费 0.1%
        bars_per_year: int = 24 * 365, # 加密24h×365天，每根bar=1小时
        min_rebalance_bars: int = 1,    # 最小调仓间隔（根K线），实盘72小时=72
        rebalance_threshold: float = 0.0,  # 权重变化超过阈值才调仓（0=每次都调）
        stop_loss_pct: float = 0.0,     # 周期内单标的止损阈值（0=不启用，如0.08=反向8%平仓）
        dynamic_stop: bool = False,     # 启用基于策略Parkinson波动率的3σ动态止损
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.bars_per_year = bars_per_year
        self.min_rebalance_bars = min_rebalance_bars
        self.rebalance_threshold = rebalance_threshold
        self.stop_loss_pct = stop_loss_pct
        self.dynamic_stop = dynamic_stop
        self.stop_loss_count = 0

    def _stop_pct_for(self, symbol: str) -> float:
        """返回该标的止损比例：动态3σ Parkinson（保底4%/封顶15%），否则固定值"""
        if self.dynamic_stop and hasattr(self.strategy, "latest_volatilities"):
            sigma = self.strategy.latest_volatilities.get(symbol)
            if sigma is not None and np.isfinite(sigma):
                return float(np.clip(3.0 * sigma, 0.04, 0.15))
        return self.stop_loss_pct

    def run(self, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        运行回测。

        参数:
            data: {symbol: DataFrame[open, high, low, close, volume]}
                  所有symbol必须有相同的时间索引

        返回:
            BacktestResult 包含所有指标和净值曲线
        """
        symbols = list(data.keys())
        # 对齐所有数据到同一时间索引
        idx = data[symbols[0]].index
        for s in symbols:
            assert len(data[s]) == len(idx), f"{s} 数据长度不一致"

        capital = self.initial_capital
        positions = {s: 0.0 for s in symbols}  # 各标的持仓数量
        current_weights = {s: 0.0 for s in symbols}  # 当前仓位权重
        equity_curve = []
        trades = []
        last_rebalance_bar = -9999  # 上次调仓的bar索引
        entry_prices = {s: 0.0 for s in symbols}  # 各持仓的开仓价（用于周期内止损）
        self.stop_loss_count = 0

        warmup = self.strategy.warmup_bars()

        for i in range(warmup, len(idx)):
            timestamp = idx[i]
            # 取截止到当前的历史K线
            bars = {s: data[s].iloc[:i+1] for s in symbols}

            # 策略输出目标仓位比例
            target_weights = self.strategy.on_bar(bars)

            # 计算当前组合市值
            current_prices = {s: data[s]["close"].iloc[i] for s in symbols}
            position_value = sum(positions[s] * current_prices[s] for s in symbols)
            total_equity = capital + position_value

            # ---- 调仓判断：间隔 + 阈值 ----
            bars_since_rebalance = i - last_rebalance_bar
            need_rebalance = bars_since_rebalance >= self.min_rebalance_bars

            if need_rebalance and self.rebalance_threshold > 0:
                # 检查权重变化是否超过阈值
                max_change = max(abs(target_weights.get(s, 0) - current_weights.get(s, 0)) for s in symbols)
                if max_change < self.rebalance_threshold:
                    need_rebalance = False

            if need_rebalance:
                # 执行调仓：计算需要交易的数量
                for s in symbols:
                    target_value = total_equity * target_weights.get(s, 0.0)
                    target_qty = target_value / current_prices[s] if current_prices[s] > 0 else 0
                    delta_qty = target_qty - positions[s]
                    old_qty = positions[s]

                    if abs(delta_qty * current_prices[s]) > total_equity * 0.001:
                        # 交易金额 > 0.1% 总资产才执行，避免频繁小额交易
                        trade_value = abs(delta_qty) * current_prices[s]
                        fee = trade_value * self.fee_rate
                        # 现金流：买入(delta>0)现金减少，卖出(delta<0)现金增加
                        cash_flow = -delta_qty * current_prices[s]
                        capital += cash_flow - fee  # 现金 = 现金 + 买卖现金流 - 手续费
                        positions[s] = target_qty
                        # 记录开仓价：新建仓或多空反手时以当前价为新成本
                        if abs(target_qty) > 1e-12 and (
                            abs(old_qty) < 1e-12 or np.sign(target_qty) != np.sign(old_qty)
                        ):
                            entry_prices[s] = current_prices[s]
                        elif abs(target_qty) < 1e-12:
                            entry_prices[s] = 0.0
                        trades.append({
                            "time": timestamp, "symbol": s,
                            "qty": delta_qty, "price": current_prices[s], "pnl": 0
                        })

                current_weights = target_weights.copy()
                last_rebalance_bar = i

            elif self.stop_loss_pct > 0 or self.dynamic_stop:
                # ---- 周期内止损：不死等72h，单标的反向偏离阈值即单独平仓 ----
                for s in symbols:
                    pos = positions[s]
                    ep = entry_prices[s]
                    if abs(pos) < 1e-12 or ep <= 0:
                        continue
                    ret_from_entry = current_prices[s] / ep - 1
                    sp = self._stop_pct_for(s)  # 动态3σ或固定
                    if sp <= 0:
                        continue
                    # 多头：相对开仓价下跌超阈值；空头：相对开仓价上涨超阈值
                    hit_stop = (
                        (pos > 0 and ret_from_entry <= -sp)
                        or (pos < 0 and ret_from_entry >= sp)
                    )
                    if hit_stop:
                        trade_value = abs(pos) * current_prices[s]
                        fee = trade_value * self.fee_rate
                        # 平多现金增加、平空现金减少（pos带符号，统一处理）
                        capital += pos * current_prices[s] - fee
                        positions[s] = 0.0
                        entry_prices[s] = 0.0
                        current_weights[s] = 0.0
                        self.stop_loss_count += 1
                        trades.append({
                            "time": timestamp, "symbol": s,
                            "qty": -pos, "price": current_prices[s],
                            "pnl": 0, "stop": True,
                        })

            # 记录每日净值
            position_value = sum(positions[s] * current_prices[s] for s in symbols)
            equity = capital + position_value
            equity_curve.append({"time": timestamp, "equity": equity})

        # 计算指标
        eq_df = pd.DataFrame(equity_curve).set_index("time")
        equity_series = eq_df["equity"]
        returns = equity_series.pct_change().dropna()

        total_return = equity_series.iloc[-1] / self.initial_capital - 1
        n_bars = len(returns)
        annual_factor = self.bars_per_year / (n_bars / n_bars) if n_bars > 0 else 1

        annual_return = (1 + total_return) ** (self.bars_per_year / max(n_bars, 1)) - 1
        annual_vol = returns.std() * np.sqrt(self.bars_per_year)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Sortino: 只用下行波动率
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(self.bars_per_year) if len(downside) > 1 else 1
        sortino = annual_return / downside_vol if downside_vol > 0 else 0

        # 最大回撤
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = drawdown.min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=0.0,  # 简化版，可补充
            total_trades=len(trades),
            equity_curve=equity_series,
        )

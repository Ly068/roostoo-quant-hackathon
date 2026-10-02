"""
金奖策略组合回测验证脚本
========================
验证四策略组合（动量+配对+突破+XGBoost）的历史表现。

回测设置：
- 数据：Yahoo Finance 60天1小时K线（5币种）
- 初始资金：$50,000
- 手续费：市价0.1%
- 对比：单动量策略 vs 四策略组合
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.pairs_v2 import PairTradingV2
from rooster_trader.strategy.volatility_breakout import VolatilityBreakoutStrategy
from rooster_trader.strategy.xgboost_timing import XGBoostTimingStrategy
from rooster_trader.strategy.enhanced_multi import EnhancedMultiStrategy

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history")
FEE_RATE = 0.001  # 0.1% 市价单手续费


def load_data() -> Dict[str, pd.DataFrame]:
    """加载历史数据"""
    data = {}
    coins = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    for coin in coins:
        filepath = os.path.join(DATA_DIR, f"{coin}_1h.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            symbol = f"{coin}/USD"
            data[symbol] = df
            print(f"  加载 {symbol}: {len(df)} 根")
    return data


def run_backtest(
    strategy,
    data: Dict[str, pd.DataFrame],
    initial_capital: float = 50000,
    fee_rate: float = 0.001,
    warmup_bars: int = 100,
    rebalance_threshold: float = 0.03,  # 权重变化超过3%才调仓
    min_rebalance_bars: int = 24,  # 最小调仓间隔24根K线（1天）
) -> dict:
    """
    简化版回测引擎：按目标权重调仓，计算净值曲线。

    逻辑：
    - 每根K线调用strategy.on_bar获取目标权重
    - 如果目标权重变化超过阈值，执行调仓（扣手续费）
    - 按当前持仓和价格计算市值
    """
    symbols = list(data.keys())
    # 对齐时间索引
    all_index = None
    for symbol, df in data.items():
        if all_index is None:
            all_index = df.index
        else:
            all_index = all_index.intersection(df.index)

    # 截取共同时间段
    aligned_data = {}
    for symbol, df in data.items():
        aligned_data[symbol] = df.loc[all_index].copy()

    n_bars = len(all_index)
    print(f"  回测区间: {all_index[0].strftime('%Y-%m-%d')} ~ {all_index[-1].strftime('%Y-%m-%d')} ({n_bars}根)")

    # 状态
    cash = initial_capital
    positions = {s: 0.0 for s in symbols}  # 持仓数量
    current_weights = {s: 0.0 for s in symbols}
    portfolio_values = []
    trade_count = 0
    last_rebalance_bar = -999  # 确保第一次就能调仓

    for i in range(warmup_bars, n_bars):
        # 截至当前的历史数据
        hist_data = {}
        for symbol, df in aligned_data.items():
            hist_data[symbol] = df.iloc[:i+1].copy()

        # 当前价格
        prices = {s: aligned_data[s]["close"].iloc[i] for s in symbols}

        # 策略计算目标权重
        try:
            target_weights = strategy.on_bar(hist_data)
        except Exception as e:
            target_weights = current_weights.copy()

        # 计算当前市值
        portfolio_value = cash + sum(positions[s] * prices[s] for s in symbols)

        # 检查是否需要调仓（权重变化超过阈值且距上次调仓超过最小间隔）
        need_rebalance = False
        if i - last_rebalance_bar >= min_rebalance_bars:
            for s in symbols:
                if abs(target_weights.get(s, 0) - current_weights.get(s, 0)) > rebalance_threshold:
                    need_rebalance = True
                    break

        if need_rebalance and portfolio_value > 0:
            # 执行调仓
            for s in symbols:
                target_w = target_weights.get(s, 0)
                target_value = portfolio_value * target_w
                current_value = positions[s] * prices[s]
                delta_value = target_value - current_value

                if abs(delta_value) > portfolio_value * 0.005:  # 最小交易额0.5%
                    delta_qty = delta_value / prices[s] if prices[s] > 0 else 0
                    positions[s] += delta_qty
                    cash -= delta_value
                    # 扣手续费
                    fee = abs(delta_value) * fee_rate
                    cash -= fee
                    trade_count += 1

            current_weights = target_weights.copy()
            last_rebalance_bar = i

        # 记录市值
        portfolio_value = cash + sum(positions[s] * prices[s] for s in symbols)
        portfolio_values.append(portfolio_value)

    # 计算绩效指标
    returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = (portfolio_values[-1] / initial_capital - 1) if portfolio_values else 0

    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)  # 年化（小时级）
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(24 * 365) if (returns < 0).std() > 0 else 0
        max_drawdown = (pd.Series(portfolio_values) / pd.Series(portfolio_values).cummax() - 1).min()
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    else:
        sharpe = sortino = calmar = 0
        max_drawdown = 0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "final_value": portfolio_values[-1] if portfolio_values else initial_capital,
        "trade_count": trade_count,
        "portfolio_values": portfolio_values,
    }


def main():
    print("=" * 70)
    print("金奖策略组合回测验证")
    print("=" * 70)

    # 加载数据
    print("\n[1/4] 加载历史数据...")
    data = load_data()
    symbols = list(data.keys())

    # 策略1：单动量（基准）
    print("\n[2/4] 回测单动量策略（基准）...")
    momentum = MomentumStrategy(
        symbols=symbols,
        lookback_hours=720,
        top_n=2,
        rebalance_hours=72,
        vol_window=20,
    )
    result_momentum = run_backtest(momentum, data)

    # 策略2：四策略组合（金奖）
    print("\n[3/4] 回测四策略组合（金奖策略）...")
    mom = MomentumStrategy(
        symbols=symbols,
        lookback_hours=720,
        top_n=2,
        rebalance_hours=72,
        vol_window=20,
    )
    pairs = PairTradingV2(
        pair_symbols=[("BTC/USD", "ETH/USD")],
        window=80,
        entry_z=1.5,
        exit_z=0.5,
        stop_loss_z=4.0,
    )
    breakout = VolatilityBreakoutStrategy(
        symbols=symbols,
        bb_window=20,
        bb_std=2.0,
        squeeze_percentile=0.2,
        atr_window=14,
        atr_stop_mult=2.0,
        atr_target_mult=3.0,
        max_positions=2,
    )
    xgboost = XGBoostTimingStrategy(
        symbols=symbols,
        forecast_hours=12,
        long_threshold=0.60,   # 提高阈值，只在高置信度时开仓（样本外准确率有限）
        short_threshold=0.60,
        max_positions=1,        # 最多1个ML仓位
        position_size=0.10,     # 单个ML仓位更小
        load_pretrained=False,  # 回测时不加载预训练模型（防未来数据泄漏）
    )

    strategies = {
        "momentum": mom,
        "pairs": pairs,
        "breakout": breakout,
        "xgboost": xgboost,
    }

    # 权重调整：ML独立策略降到5%（主要靠信号过滤器），动量45%+配对15%+突破35%
    combo = EnhancedMultiStrategy(
        symbols=symbols,
        strategies=strategies,
        base_weights={"momentum": 0.45, "pairs": 0.15, "breakout": 0.35, "xgboost": 0.05},
        use_regime_adjust=True,
        use_xgb_filter=True,    # ML信号过滤器仍然启用（这是ML的主要价值）
        use_strategy_circuit_breaker=True,
    )

    result_combo = run_backtest(combo, data)

    # 结果对比
    print("\n" + "=" * 70)
    print("回测结果对比")
    print("=" * 70)
    print(f"{'指标':<20} {'单动量':>15} {'四策略组合':>15} {'提升':>10}")
    print("-" * 70)
    print(f"{'总收益率':<20} {result_momentum['total_return']*100:>14.2f}% {result_combo['total_return']*100:>14.2f}% {(result_combo['total_return']-result_momentum['total_return'])*100:>+9.2f}%")
    print(f"{'Sharpe比率':<20} {result_momentum['sharpe']:>15.3f} {result_combo['sharpe']:>15.3f} {result_combo['sharpe']-result_momentum['sharpe']:>+10.3f}")
    print(f"{'Sortino比率':<20} {result_momentum['sortino']:>15.3f} {result_combo['sortino']:>15.3f} {result_combo['sortino']-result_momentum['sortino']:>+10.3f}")
    print(f"{'最大回撤':<20} {result_momentum['max_drawdown']*100:>14.2f}% {result_combo['max_drawdown']*100:>14.2f}% {(abs(result_combo['max_drawdown'])-abs(result_momentum['max_drawdown']))*100:>+9.2f}%")
    print(f"{'Calmar比率':<20} {result_momentum['calmar']:>15.3f} {result_combo['calmar']:>15.3f} {result_combo['calmar']-result_momentum['calmar']:>+10.3f}")
    print(f"{'最终市值':<20} ${result_momentum['final_value']:>14,.2f} ${result_combo['final_value']:>14,.2f}")
    print(f"{'交易次数':<20} {result_momentum['trade_count']:>15d} {result_combo['trade_count']:>15d}")
    print("=" * 70)

    # 保存净值曲线
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    pd.DataFrame({
        "momentum": result_momentum["portfolio_values"],
        "combo": result_combo["portfolio_values"],
    }).to_csv(os.path.join(results_dir, "gold_strategy_backtest.csv"))

    print(f"\n净值曲线已保存: {results_dir}/gold_strategy_backtest.csv")


if __name__ == "__main__":
    main()

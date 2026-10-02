"""
完整回测脚本
============
加载历史数据 → 初始化策略 → 运行回测 → 输出所有绩效指标
"""

import os
import sys
import pandas as pd

# 把项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.backtest.engine import Backtester
from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.pairs import PairTradingStrategy
from rooster_trader.strategy.rotation import RotationStrategy


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data(symbol: str, interval: str = "1h") -> pd.DataFrame:
    """从CSV加载历史数据"""
    filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    if not os.path.exists(filepath):
        print(f"⚠ 找不到 {filepath}，请先运行 scripts/download_data.py")
        sys.exit(1)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def print_result(name: str, result):
    """格式化打印回测结果"""
    print(f"\n{'='*60}")
    print(f"策略: {name}")
    print(f"{'='*60}")
    print(f"  总收益率:       {result.total_return:>8.2%}")
    print(f"  年化收益率:     {result.annual_return:>8.2%}")
    print(f"  年化波动率:     {result.annual_volatility:>8.2%}")
    print(f"  Sharpe 比率:    {result.sharpe_ratio:>8.3f}")
    print(f"  Sortino 比率:   {result.sortino_ratio:>8.3f}")
    print(f"  Calmar 比率:    {result.calmar_ratio:>8.3f}")
    print(f"  最大回撤:       {result.max_drawdown:>8.2%}")
    print(f"  总交易次数:     {result.total_trades:>8d}")


def main():
    print("=" * 60)
    print("APAC Quant Hackathon - 策略回测")
    print("=" * 60)

    # ---- 加载数据 ----
    crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    data = {}
    for sym in crypto_symbols:
        data[sym] = load_data(sym)

    # 对齐所有数据到同一时间索引
    min_len = min(len(df) for df in data.values())
    for sym in data:
        data[sym] = data[sym].iloc[-min_len:]

    print(f"\n加载完成: {len(crypto_symbols)} 个标的，{min_len} 根1小时K线")

    # ---- 测试策略1：多因子动量 ----
    print("\n" + "→ 正在回测：多因子动量策略...")
    momentum = MomentumStrategy(
        symbols=crypto_symbols,
        lookback_hours=168,   # 7天
        top_n=2,
        rebalance_hours=168, # 每周
    )
    bt = Backtester(momentum, initial_capital=100_000, fee_rate=0.001)
    result_mom = bt.run(data)
    print_result("多因子动量策略", result_mom)

    # ---- 测试策略2：配对交易 ----
    print("\n" + "→ 正在回测：配对交易策略...")
    pairs = PairTradingStrategy(
        pair_symbols=[("BTCUSDT", "ETHUSDT")],
        window=20,
        entry_z=2.0,
        exit_z=0.5,
    )
    bt2 = Backtester(pairs, initial_capital=100_000, fee_rate=0.001)
    result_pair = bt2.run(data)
    print_result("配对交易策略", result_pair)

    # ---- 对比总结 ----
    print(f"\n{'='*60}")
    print("对比总结")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'动量策略':>12} {'配对交易':>12}")
    print("-" * 50)
    print(f"{'总收益率':<20} {result_mom.total_return:>12.2%} {result_pair.total_return:>12.2%}")
    print(f"{'Sharpe 比率':<20} {result_mom.sharpe_ratio:>12.3f} {result_pair.sharpe_ratio:>12.3f}")
    print(f"{'最大回撤':<20} {result_mom.max_drawdown:>12.2%} {result_pair.max_drawdown:>12.2%}")
    print(f"{'Calmar 比率':<20} {result_mom.calmar_ratio:>12.3f} {result_pair.calmar_ratio:>12.3f}")


if __name__ == "__main__":
    main()

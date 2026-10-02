"""
大universe横截面动量回测
对比：
  - 5币种动量（之前）
  - 大universe动量 top3/bottom3
  - 大universe动量 top5/bottom5
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.dynamic_position import DynamicPositionStrategy
from rooster_trader.backtest.engine import Backtester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LARGE_DIR = os.path.join(PROJECT_ROOT, "data", "history_large")
SMALL_DIR = os.path.join(PROJECT_ROOT, "data", "history")

def load_universe(data_dir, coins=None):
    data = {}
    if coins is None:
        # 读取universe.txt
        uf = os.path.join(data_dir, "universe.txt")
        if os.path.exists(uf):
            coins = [l.strip() for l in open(uf) if l.strip()]
        else:
            coins = [f.replace("_1h.csv","") for f in os.listdir(data_dir) if f.endswith("_1h.csv")]
    for coin in coins:
        fp = os.path.join(data_dir, f"{coin}_1h.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            data[f"{coin}/USD"] = df
    return data

def align_data(data):
    """对齐所有币种到共同时间索引"""
    common_idx = None
    for s, df in data.items():
        idx = df.index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    return {s: df.loc[common_idx] for s, df in data.items()}

def run_bt(strategy, data, name):
    bt = Backtester(strategy=strategy, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01)
    r = bt.run(data)
    eq = r.equity_curve
    n = len(eq)
    daily = r.total_return / (n/24)
    print(f"  {name:42s} | 收益{r.total_return*100:+6.2f}% | Sharpe{r.sharpe_ratio:+6.2f} | "
          f"回撤{r.max_drawdown*100:+6.2f}% | Calmar{r.calmar_ratio:+6.2f} | 交易{r.total_trades:3d} | "
          f"14天预期{daily*14*0.679*100:+.2f}%")
    return r

def main():
    print("=" * 100)
    print("大universe横截面动量回测（72h调仓，0.1%手续费）")
    print("=" * 100)

    # 加载大universe
    large_raw = load_universe(LARGE_DIR)
    large_data = align_data(large_raw)
    large_symbols = list(large_data.keys())
    print(f"\n大universe: {len(large_symbols)}个币种，{len(large_data[large_symbols[0]])}根共同K线")

    # 加载5币种（对比基准）
    small_raw = load_universe(SMALL_DIR, ["BTC","ETH","SOL","BNB","XRP"])
    small_data = align_data(small_raw)
    small_symbols = list(small_data.keys())
    print(f"小universe: {len(small_symbols)}个币种")

    # ================================================================
    print("\n【基准】5币种动量")
    print("-" * 100)
    for top_n, ps in [(1, 0.50), (2, 0.25)]:
        m = MomentumStrategy(symbols=small_symbols, lookback_hours=720, top_n=top_n,
                              rebalance_hours=72, position_size=ps)
        s = DynamicPositionStrategy(underlying_strategy=m, competition_start_date="2026-10-04", competition_days=14)
        run_bt(s, small_data, f"5币种 top{top_n} {ps*100:.0f}%仓位")

    # ================================================================
    print("\n【大universe】30天动量，不同top_n和仓位")
    print("-" * 100)
    for lookback in [720]:
        for top_n, ps in [(1, 0.50), (2, 0.25), (3, 0.17), (5, 0.10)]:
            m = MomentumStrategy(symbols=large_symbols, lookback_hours=lookback, top_n=top_n,
                                  rebalance_hours=72, position_size=ps)
            s = DynamicPositionStrategy(underlying_strategy=m, competition_start_date="2026-10-04", competition_days=14)
            run_bt(s, large_data, f"{len(large_symbols)}币种 30天 top{top_n} {ps*100:.0f}%仓位")

    # ================================================================
    print("\n【大universe】不同动量窗口（top3，17%仓位）")
    print("-" * 100)
    for lb in [168, 336, 504, 720]:
        m = MomentumStrategy(symbols=large_symbols, lookback_hours=lb, top_n=3,
                              rebalance_hours=72, position_size=0.17)
        s = DynamicPositionStrategy(underlying_strategy=m, competition_start_date="2026-10-04", competition_days=14)
        run_bt(s, large_data, f"{len(large_symbols)}币种 {lb//24}天 top3 17%仓位")

    # ================================================================
    print("\n【大universe激进】top3 30%仓位（总敞口180%→需要确认风控）")
    print("-" * 100)
    m = MomentumStrategy(symbols=large_symbols, lookback_hours=720, top_n=3,
                          rebalance_hours=72, position_size=0.30)
    s = DynamicPositionStrategy(underlying_strategy=m, competition_start_date="2026-10-04", competition_days=14)
    run_bt(s, large_data, f"{len(large_symbols)}币种 30天 top3 30%仓位(激进)")

    print("\n" + "=" * 100)
    print("回测完成")
    print("=" * 100)

if __name__ == "__main__":
    main()

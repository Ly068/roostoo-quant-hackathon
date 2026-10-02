"""
方案E回测：动量+突破共振加仓策略
对比方案D（纯动量满仓）和方案B（纯动量30%仓位）
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.momentum_breakout_resonance import MomentumBreakoutResonance
from rooster_trader.strategy.dynamic_position import DynamicPositionStrategy
from rooster_trader.backtest.engine import Backtester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "history")

def load_data():
    data = {}
    for coin in ["BTC", "ETH", "SOL", "BNB", "XRP"]:
        fp = os.path.join(DATA_DIR, f"{coin}_1h.csv")
        if os.path.exists(fp):
            data[f"{coin}/USD"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    return data

def run_backtest(strategy, data, name):
    bt = Backtester(strategy=strategy, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01)
    r = bt.run(data)
    eq = r.equity_curve
    n = len(eq)
    daily_ret = r.total_return / (n / 24)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  60天收益:    {r.total_return*100:+.2f}%")
    print(f"  年化收益:    {r.annual_return*100:+.2f}%")
    print(f"  年化波动:    {r.annual_volatility*100:.2f}%")
    print(f"  Sharpe:      {r.sharpe_ratio:+.2f}")
    print(f"  Sortino:     {r.sortino_ratio:+.2f}")
    print(f"  最大回撤:    {r.max_drawdown*100:+.2f}%")
    print(f"  Calmar:      {r.calmar_ratio:+.2f}")
    print(f"  交易次数:    {r.total_trades}")
    print(f"  最终净值:    ${eq.iloc[-1]:,.2f}")

    # 分阶段
    print(f"\n  分阶段收益（每10天）:")
    for i in range(0, n, 240):
        end = min(i+240, n-1)
        if end > i:
            ret = (eq.iloc[end] / eq.iloc[i] - 1) * 100
            ds = eq.index[i].strftime('%m-%d') if hasattr(eq.index[i], 'strftime') else str(eq.index[i])[:5]
            de = eq.index[end].strftime('%m-%d') if hasattr(eq.index[end], 'strftime') else str(eq.index[end])[:5]
            print(f"    {ds} ~ {de}: {ret:+.2f}%")

    # 14天预期
    print(f"\n  14天比赛预期（动态仓位67.9%）: {daily_ret*14*0.679*100:+.2f}%")
    print(f"  14天比赛预期（满仓100%）:      {daily_ret*14*100:+.2f}%")

    return r

def main():
    data = load_data()
    symbols = list(data.keys())
    print("="*70)
    print("方案E回测：动量+突破共振加仓 vs 纯动量")
    print("回测区间：2026-08-03 ~ 2026-10-01（59天）")
    print("调仓间隔：72小时，手续费：0.1%/边")
    print("="*70)

    # 方案B：30天动量top1 + 30%仓位 + 动态仓位
    mom_b = MomentumStrategy(symbols=symbols, lookback_hours=720, top_n=1,
                              rebalance_hours=72, position_size=0.30)
    strat_b = DynamicPositionStrategy(underlying_strategy=mom_b,
                                       competition_start_date="2026-10-04", competition_days=14)
    run_backtest(strat_b, data, "方案B：30天动量top1 + 30%仓位 + 动态仓位")

    # 方案D：30天动量top1 + 50%仓位（满仓）+ 动态仓位
    mom_d = MomentumStrategy(symbols=symbols, lookback_hours=720, top_n=1,
                              rebalance_hours=72, position_size=0.50)
    strat_d = DynamicPositionStrategy(underlying_strategy=mom_d,
                                       competition_start_date="2026-10-04", competition_days=14)
    run_backtest(strat_d, data, "方案D：30天动量top1 + 50%仓位(满仓) + 动态仓位")

    # 方案E：动量+突破共振加仓（正常30%，共振50%）+ 动态仓位
    res_e = MomentumBreakoutResonance(
        symbols=symbols,
        lookback_hours=720,
        rebalance_hours=72,
        normal_position=0.30,
        resonance_position=0.50,
        bb_window=20,
        bb_std=2.0,
        squeeze_percentile=0.2,
    )
    strat_e = DynamicPositionStrategy(underlying_strategy=res_e,
                                       competition_start_date="2026-10-04", competition_days=14)
    run_backtest(strat_e, data, "方案E：动量+突破共振加仓（正常30%/共振50%）+ 动态仓位")

    # 方案E激进版：正常40%，共振60%
    res_e2 = MomentumBreakoutResonance(
        symbols=symbols,
        lookback_hours=720,
        rebalance_hours=72,
        normal_position=0.40,
        resonance_position=0.60,
        bb_window=20,
        bb_std=2.0,
        squeeze_percentile=0.2,
    )
    strat_e2 = DynamicPositionStrategy(underlying_strategy=res_e2,
                                        competition_start_date="2026-10-04", competition_days=14)
    run_backtest(strat_e2, data, "方案E激进：动量+突破共振加仓（正常40%/共振60%）+ 动态仓位")

    print("\n" + "="*70)
    print("回测完成")
    print("="*70)

if __name__ == "__main__":
    main()

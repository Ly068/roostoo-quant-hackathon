"""
审计大universe top2策略的真实性
- 实际交易天数（预热期问题）
- 交易明细
- 净值曲线特征
- 多空敞口
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

def load_universe():
    data = {}
    uf = os.path.join(LARGE_DIR, "universe.txt")
    coins = [l.strip() for l in open(uf) if l.strip()]
    for coin in coins:
        fp = os.path.join(LARGE_DIR, f"{coin}_1h.csv")
        if os.path.exists(fp):
            data[f"{coin}/USD"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    # 对齐
    common = None
    for s, df in data.items():
        common = df.index if common is None else common.intersection(df.index)
    return {s: df.loc[common] for s, df in data.items()}

def main():
    data = load_universe()
    symbols = list(data.keys())
    n_bars = len(data[symbols[0]])
    print("="*80)
    print("大universe top2策略真实性审计")
    print("="*80)
    print(f"币种数: {len(symbols)}, 总K线: {n_bars}")
    print(f"数据区间: {data[symbols[0]].index[0]} ~ {data[symbols[0]].index[-1]}")
    print(f"30天动量预热需要720根K线")
    print(f"预计实际可交易K线: {n_bars-720}根 = {(n_bars-720)/24:.1f}天")

    mom = MomentumStrategy(symbols=symbols, lookback_hours=720, top_n=2,
                            rebalance_hours=72, position_size=0.25)
    strat = DynamicPositionStrategy(underlying_strategy=mom,
                                     competition_start_date="2026-10-04", competition_days=14)

    bt = Backtester(strategy=strat, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01)
    r = bt.run(data)

    eq = r.equity_curve
    print(f"\n回测结果:")
    print(f"  净值曲线点数: {len(eq)}")
    print(f"  净值首日: {eq.index[0]}  净值末日: {eq.index[-1]}")
    actual_days = (eq.index[-1] - eq.index[0]).total_seconds()/86400
    print(f"  净值曲线实际跨度: {actual_days:.1f}天")
    print(f"  总收益: {r.total_return*100:+.2f}%")
    print(f"  交易次数: {r.total_trades}")

    # 检查净值曲线是否包含预热期（flat段）
    first_nonflat = None
    for i in range(len(eq)):
        if abs(eq.iloc[i] - eq.iloc[0]) > 0.01:
            first_nonflat = i
            break
    if first_nonflat:
        print(f"  首次净值变动位置: 第{first_nonflat}根 = {eq.index[first_nonflat]}")
        real_eq = eq.iloc[first_nonflat:]
        real_days = (real_eq.index[-1]-real_eq.index[first_nonflat]).total_seconds()/86400
        real_ret = real_eq.iloc[-1]/real_eq.iloc[0]-1
        real_vol = real_eq.pct_change().std()*np.sqrt(24*365)
        real_sharpe = (real_eq.pct_change().mean()*24*365)/real_vol if real_vol>0 else 0
        real_dd = (real_eq/real_eq.cummax()-1).min()
        print(f"\n  === 剔除预热flat段后的真实指标（{real_days:.1f}天）===")
        print(f"  真实收益: {real_ret*100:+.2f}%")
        print(f"  真实年化波动率: {real_vol*100:.2f}%")
        print(f"  真实Sharpe: {real_sharpe:.2f}")
        print(f"  真实最大回撤: {real_dd*100:+.2f}%")
        print(f"  真实Calmar(年化收益/回撤): {(real_ret*(365/real_days))/abs(real_dd):.2f}")

    # 输出交易明细（前20笔+后10笔）
    print(f"\n交易明细审计:")
    if hasattr(bt, 'trade_log'):
        trades = bt.trade_log
        print(f"  总交易记录: {len(trades)}")
        for t in trades[:15]:
            print(f"    {t}")
        print("    ...")
        for t in trades[-8:]:
            print(f"    {t}")

    # 检查最终持仓的多空敞口
    print(f"\n最终持仓敞口:")
    final_weights = mom.current_weights
    longs = {s:w for s,w in final_weights.items() if w>0}
    shorts = {s:w for s,w in final_weights.items() if w<0}
    print(f"  做多: {longs}  合计{sum(longs.values())*100:.0f}%")
    print(f"  做空: {shorts}  合计{sum(shorts.values())*100:.0f}%")
    print(f"  净敞口: {(sum(longs.values())+sum(shorts.values()))*100:.0f}%")
    print(f"  总敞口: {(sum(longs.values())-sum(shorts.values()))*100:.0f}%")

if __name__ == "__main__":
    main()

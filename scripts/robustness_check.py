"""
大universe动量分段稳健性验证
把更长的历史数据分成多个时段（walk-forward窗口），分别验证策略表现
同时测试参数敏感性
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.backtest.engine import Backtester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LARGE_DIR = os.path.join(PROJECT_ROOT, "data", "history_binance")

def load_universe():
    data = {}
    uf = os.path.join(LARGE_DIR, "universe.txt")
    coins = [l.strip() for l in open(uf) if l.strip()]
    for coin in coins:
        fp = os.path.join(LARGE_DIR, f"{coin}_1h.csv")
        if os.path.exists(fp):
            data[f"{coin}/USD"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    common = None
    for s, df in data.items():
        common = df.index if common is None else common.intersection(df.index)
    return {s: df.loc[common] for s, df in data.items()}

def slice_data(data, start_bar, end_bar):
    """截取K线区间（每段需要自带30天预热，所以段起点要往前推720根）"""
    return {s: df.iloc[start_bar:end_bar] for s, df in data.items()}

def run_window(data, symbols, lookback, top_n, position_size, label):
    mom = MomentumStrategy(symbols=symbols, lookback_hours=lookback, top_n=top_n,
                            rebalance_hours=72, position_size=position_size)
    bt = Backtester(strategy=mom, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01)
    r = bt.run(data)
    eq = r.equity_curve
    # 剔除预热flat段
    fnf = 0
    for i in range(len(eq)):
        if abs(eq.iloc[i]-eq.iloc[0]) > 0.01:
            fnf = i; break
    real = eq.iloc[fnf:]
    if len(real) < 10:
        print(f"  {label:28s} | 无有效交易")
        return None
    days = (real.index[-1]-real.index[fnf]).total_seconds()/86400
    ret = real.iloc[-1]/real.iloc[0]-1
    vol = real.pct_change().std()*np.sqrt(24*365)
    sharpe = (real.pct_change().mean()*24*365)/vol if vol>0 else 0
    dd = (real/real.cummax()-1).min()
    print(f"  {label:28s} | {days:4.0f}天 | 收益{ret*100:+6.2f}% | Sharpe{sharpe:+5.2f} | 回撤{dd*100:+6.2f}% | 交易{r.total_trades:3d}")
    return {"days":days,"ret":ret,"sharpe":sharpe,"dd":dd}

def main():
    data = load_universe()
    symbols = list(data.keys())
    total = len(data[symbols[0]])
    print("="*90)
    print("大universe动量分段稳健性验证")
    print("="*90)
    print(f"币种: {len(symbols)}, 总K线: {total} ({total/24:.0f}天)")
    print(f"区间: {data[symbols[0]].index[0]} ~ {data[symbols[0]].index[-1]}")

    # ================================================================
    # 分段：每段约30天交易 + 30天预热
    # ================================================================
    print(f"\n【分段验证】30天动量 top2 25%仓位，滚动窗口")
    print("-"*90)
    window_trade = 720   # 30天交易
    warmup = 720         # 30天预热
    step = window_trade
    results = []
    start = 0
    seg = 0
    while start + warmup + 100 < total:
        end = min(start + warmup + window_trade, total)
        sub = slice_data(data, start, end)
        seg += 1
        ref_idx = max(warmup-1, 0)
        ref_date = sub[symbols[0]].index[ref_idx]
        ref_str = ref_date.strftime('%m-%d') if hasattr(ref_date, 'strftime') else str(ref_date)[:5]
        date_label = f"段{seg} ({ref_str}起)"
        r = run_window(sub, symbols, 720, 2, 0.25, date_label)
        if r: results.append(r)
        start += step

    # 汇总
    if results:
        print(f"\n  分段汇总（{len(results)}段）:")
        rets = [r["ret"] for r in results]
        sharpes = [r["sharpe"] for r in results]
        dds = [r["dd"] for r in results]
        print(f"    各段收益: {[f'{x*100:+.1f}%' for x in rets]}")
        print(f"    正收益段占比: {sum(1 for x in rets if x>0)/len(rets)*100:.0f}%")
        print(f"    平均段Sharpe: {np.mean(sharpes):.2f}")
        print(f"    最差段回撤: {min(dds)*100:+.2f}%")
        print(f"    平均段收益: {np.mean(rets)*100:+.2f}%")

    # ================================================================
    # 参数敏感性（全区间）
    # ================================================================
    print(f"\n【参数敏感性】全区间")
    print("-"*90)
    print("  动量窗口:")
    for lb in [480, 600, 720, 840, 960]:
        run_window(data, symbols, lb, 2, 0.25, f"窗口{lb//24}天 top2")
    print("  top_n/仓位:")
    for tn, ps in [(1,0.40),(1,0.50),(2,0.25),(2,0.30),(3,0.17),(3,0.22)]:
        run_window(data, symbols, 720, tn, ps, f"top{tn} {ps*100:.0f}%仓位")

    print("\n"+"="*90)
    print("稳健性验证完成")
    print("="*90)

if __name__ == "__main__":
    main()

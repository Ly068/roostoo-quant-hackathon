"""
下载大universe历史数据（30个高流动性加密货币，60天1h）
用于横截面动量策略（大universe效应更强）
"""
import os, sys, time
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "history_large")
os.makedirs(OUT_DIR, exist_ok=True)
PERIOD = "120d"  # 下载120天，扣除30天预热后可回测约90天

# 30个高流动性币种（Roostoo上市 + Yahoo有数据）
COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "DOGE", "ADA", "AVAX", "LINK", "AAVE",
    "SUI", "DOT", "LTC", "TRX", "NEAR",
    "UNI", "APT", "ARB", "FIL", "HBAR",
    "ICP", "ENA", "ONDO", "TAO", "TON",
    "SEI", "FET", "WLD", "XLM", "CRV",
]

def main():
    print("=" * 70)
    print(f"下载大universe数据：{len(COINS)}个币种，{PERIOD} 1h K线")
    print("=" * 70)

    tickers = [f"{c}-USD" for c in COINS]
    valid_coins = []

    # 分批下载（每批10个，避免请求过大）
    batch_size = 10
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"\n下载批次 {i//batch_size+1}: {[t.replace('-USD','') for t in batch]}")
        try:
            raw = yf.download(batch, period=PERIOD, interval="1h",
                              progress=False, group_by="ticker", threads=True)
            for ticker in batch:
                coin = ticker.replace("-USD", "")
                try:
                    if len(batch) == 1:
                        df = raw
                    else:
                        df = raw[ticker]
                    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    if len(df) >= 1500:  # 至少约62天数据
                        df.to_csv(os.path.join(OUT_DIR, f"{coin}_1h.csv"))
                        all_data[coin] = df
                        valid_coins.append(coin)
                        print(f"  ✓ {coin:6s} {len(df):5d}根K线  "
                              f"{df.index[0].strftime('%m-%d')}~{df.index[-1].strftime('%m-%d')}")
                    else:
                        print(f"  ✗ {coin:6s} 数据不足({len(df)}根)，剔除")
                except Exception as e:
                    print(f"  ✗ {coin:6s} 解析失败: {str(e)[:60]}")
        except Exception as e:
            print(f"  批次下载失败: {e}")
        time.sleep(2)

    # 对齐时间索引（取所有币种的共同时间窗口）
    print("\n" + "=" * 70)
    print(f"有效币种: {len(valid_coins)}个")
    print("=" * 70)

    # 保存universe列表
    with open(os.path.join(OUT_DIR, "universe.txt"), "w") as f:
        for c in valid_coins:
            f.write(c + "\n")

    print(f"\n数据保存至: {OUT_DIR}")
    print(f"universe列表: {valid_coins}")

if __name__ == "__main__":
    main()

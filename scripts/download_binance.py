"""
Binance 公开K线历史数据下载（替代有volume缺陷的 yfinance）
=========================================================
数据源: https://data-api.binance.vision （Binance公共数据，无需API Key、无geo限制）
优点:
  - 1h OHLCV 完整真实（yfinance 的 1h volume 约50%为0）
  - 价格与 Roostoo 等 CEX 一致（ARB 纠正为 $0.20，yfinance 错为 $0.0006）
  - 25 币种全部可映射（coin -> coinUSDT）

输出: data/history_binance/<COIN>_1h.csv （120天，约2880根）
"""
import os
import sys
import time

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "history_binance")
os.makedirs(OUT_DIR, exist_ok=True)

BASE = "https://data-api.binance.vision"
DAYS = 120
INTERVAL = "1h"

# 25 币种（Roostoo universe，全部已验证可映射到 Binance <COIN>USDT）
COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK",
    "AAVE", "DOT", "LTC", "TRX", "NEAR", "ARB", "FIL", "HBAR", "ICP",
    "ENA", "ONDO", "SEI", "FET", "WLD", "XLM", "CRV",
]


def fetch_klines(symbol: str, start_ms: int, end_ms: int):
    """分页拉取1h K线，每次最多1000根，按startTime向前翻页"""
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol, "interval": INTERVAL,
            "startTime": cur, "limit": 1000,
        }
        r = requests.get(BASE + "/api/v3/klines", params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        cur = last_open + 3600_000  # 下一根K线开盘时间
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return rows


def to_frame(rows):
    df = pd.DataFrame(rows, columns=[
        "openTime", "open", "high", "low", "close", "volume",
        "closeTime", "quoteVolume", "trades", "takerBase", "takerQuote", "ignore",
    ])
    # 去重（翻页边界可能重复）
    df = df.drop_duplicates(subset="openTime")
    df["time"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def main():
    print("=" * 72)
    print(f"Binance 历史K线下载：{len(COINS)}币种，{DAYS}天 1h")
    print("=" * 72)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - DAYS * 24 * 3600 * 1000
    valid = []

    for coin in COINS:
        symbol = coin + "USDT"
        try:
            rows = fetch_klines(symbol, start_ms, end_ms)
            df = to_frame(rows)
            zero_vol = (df["volume"] == 0).mean()
            if len(df) >= 2000 and zero_vol < 0.05:
                fp = os.path.join(OUT_DIR, f"{coin}_1h.csv")
                df.to_csv(fp)
                valid.append(coin)
                print(f"  ✓ {coin:5s} {len(df):5d}根  volume=0占比{zero_vol*100:4.1f}%  "
                      f"{df.index[0]:%m-%d}~{df.index[-1]:%m-%d}")
            else:
                print(f"  ✗ {coin:5s} 数据不足或volume异常（{len(df)}根, 0占比{zero_vol*100:.0f}%）")
        except Exception as e:
            print(f"  ✗ {coin:5s} 失败: {str(e)[:70]}")
        time.sleep(0.2)

    with open(os.path.join(OUT_DIR, "universe.txt"), "w") as f:
        f.write("\n".join(valid) + "\n")

    print("\n" + "=" * 72)
    print(f"完成：{len(valid)}币种 -> {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()

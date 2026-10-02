"""
历史数据下载脚本
================
从 Binance 公开 API 免费下载加密货币历史K线数据。
用于本地回测，不需要 Rooster API。
"""

import os
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_binance_klines(symbol: str, interval: str = "1h", lookback_days: int = 180):
    """
    从 Binance 下载历史K线数据。

    参数:
        symbol: 交易对，如 "BTCUSDT"
        interval: K线周期，如 "1h", "4h", "1d"
        lookback_days: 下载多少天历史数据
    """
    url = "https://api.binance.com/api/v3/klines"

    # 计算需要多少根K线
    interval_hours = {"1h": 1, "4h": 4, "1d": 24}[interval]
    total_bars = int(lookback_days * 24 / interval_hours)

    all_data = []
    end_time = None

    print(f"下载 {symbol} {interval} 历史数据（约{lookback_days}天）...")

    while len(all_data) < total_bars:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,  # 每次最多1000根
        }
        if end_time:
            params["endTime"] = end_time

        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if not data:
            break

        all_data = data + all_data
        end_time = data[0][0] - 1  # 往前翻一页

        print(f"  已下载 {len(all_data)} 根K线...")

        if len(data) < 1000:
            break  # 没有更多数据了

    # 转成 DataFrame
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # 类型转换
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]]

    # 保存
    filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    df.to_csv(filepath)
    print(f"✓ 保存到 {filepath}（{len(df)} 根K线）")

    return df


if __name__ == "__main__":
    # 下载比赛用的加密货币数据
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    for sym in symbols:
        download_binance_klines(sym, interval="1h", lookback_days=180)

    print("\n全部下载完成！")

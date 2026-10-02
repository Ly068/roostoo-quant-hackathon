"""
从 Yahoo Finance 下载加密货币历史数据（1小时级别）
====================================================
Ticker格式: BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD
保存为CSV，供回测和实盘预填充使用
"""

import os
import sys
import time
import yaml
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history")


def download_crypto_data(ticker: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """
    从Yahoo Finance下载单个加密货币的历史数据。

    参数:
        ticker: 如 "BTC-USD"
        period: 数据周期 "60d", "3mo", "6mo", "1y", "2y", "max"
        interval: K线间隔 "1h", "1d"

    返回: DataFrame with columns [open, high, low, close, volume]
    """
    print(f"  下载 {ticker} ({period}, {interval})...")

    # Yahoo Finance 1h数据最多支持730天，但60天内质量最好
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        print(f"  ⚠️ {ticker} 下载失败，数据为空")
        return pd.DataFrame()

    # 统一列名（yfinance返回MultiIndex columns）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # 只保留需要的列
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # 去除NaN
    df = df.dropna()

    # 索引重命名为datetime
    df.index.name = "datetime"

    print(f"  ✅ {ticker}: {len(df)} 根K线, "
          f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 交易对格式转换: BTC/USD -> BTC-USD (Yahoo Finance格式)
    symbols = config["SYMBOL_UNIVERSE"]
    yf_tickers = [s.replace("/", "-") for s in symbols]

    print("=" * 60)
    print("从 Yahoo Finance 下载历史数据")
    print(f"交易对: {yf_tickers}")
    print(f"保存目录: {DATA_DIR}")
    print("=" * 60)

    all_data = {}

    for ticker in yf_tickers:
        try:
            df = download_crypto_data(ticker, period="60d", interval="1h")
            if not df.empty:
                # 保存为CSV（文件名用BTC_USD_1h.csv格式）
                coin = ticker.replace("-USD", "")
                filepath = os.path.join(DATA_DIR, f"{coin}_1h.csv")
                df.to_csv(filepath)
                all_data[coin] = df
                print(f"  已保存: {filepath}")
            time.sleep(1)  # 避免请求过快
        except Exception as e:
            print(f"  ❌ {ticker} 下载失败: {e}")

    print("\n" + "=" * 60)
    print(f"下载完成: {len(all_data)}/{len(yf_tickers)} 个交易对")
    print("=" * 60)

    # 打印数据概览
    print("\n数据概览:")
    for coin, df in all_data.items():
        print(f"  {coin}: {len(df)}根, "
              f"最新价 ${df['close'].iloc[-1]:.2f}, "
              f"区间涨跌 {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.1f}%")


if __name__ == "__main__":
    main()

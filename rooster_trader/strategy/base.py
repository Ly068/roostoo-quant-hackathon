"""策略基类：所有自定义策略都继承这个类，实现 on_bar() 方法"""
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class BaseStrategy(ABC):
    """
    策略基类。

    子类必须实现：
        on_bar(bars: pd.DataFrame) -> Dict[str, float]
            输入最近N根K线，输出 {symbol: target_weight} 的目标仓位字典。
            正权重=做多，负权重=做空，0=空仓。
    """

    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols

    @abstractmethod
    def on_bar(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        每根K线收盘后调用，返回目标仓位。

        参数:
            bars: {symbol: DataFrame[open, high, low, close, volume]}
                  每个 symbol 对应最近 window 根 K 线

        返回:
            {"BTC-USDT": 0.1, "ETH-USDT": -0.1, ...}
            每个值是目标仓位占总资金的比例
        """
        pass

    def warmup_bars(self) -> int:
        """策略需要多少根预热K线才能计算信号。子类可覆盖。"""
        return 30

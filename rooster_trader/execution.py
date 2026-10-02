"""
执行引擎（Execution Engine）v5.2
================================
把策略输出的"目标仓位权重"转换成实际下单。

v5.2 防护：
  1. rebalance 对"当前持有但不在 target_weights"的币种自动补目标0 → 不漏平仓
  2. 卖出数量以可用 Free 为上限（Free+Lock 中的 Lock 无法直接卖）
  3. floor 向下截断，永不超卖
  4. 提供公开 adjust_quantity()，避免外部调用私有方法
  5. 先卖后买，归集现金
注：本系统只用市价单、不挂限价单，故 Lock 通常为0。
"""

import logging
import math
import time
from typing import Dict

from rooster_trader.client import RoosterClient

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(
        self,
        client: RoosterClient,
        min_trade_pct: float = 0.005,
        order_type: str = "MARKET",
        sleep_between_orders: float = 0.5,
    ):
        self.client = client
        self.min_trade_pct = min_trade_pct
        self.order_type = order_type
        self.sleep_between_orders = sleep_between_orders
        self._amount_precisions: Dict[str, int] = {}
        self._precisions_loaded = False

    # ---------- 精度 ----------
    def _load_precisions(self):
        if self._precisions_loaded:
            return
        try:
            info = self.client.get_exchange_info()
            if info and "TradePairs" in info:
                for pair, data in info["TradePairs"].items():
                    self._amount_precisions[pair] = int(data.get("AmountPrecision", 4))
            logger.info("已加载 %d 个交易对数量精度", len(self._amount_precisions))
        except Exception as e:
            logger.warning("加载精度失败，用默认4位: %s", e)
        finally:
            self._precisions_loaded = True

    def _adjust_quantity(self, pair: str, quantity: float) -> float:
        """按精度 floor 向下截断，永不超卖/超限"""
        self._load_precisions()
        precision = self._amount_precisions.get(pair, 4)
        factor = 10 ** precision
        return math.floor(quantity * factor) / factor

    def adjust_quantity(self, pair: str, quantity: float) -> float:
        """公开接口：按交易对精度截断数量"""
        return self._adjust_quantity(pair, quantity)

    # ---------- 状态 ----------
    def _get_current_positions(self) -> Dict[str, Dict[str, float]]:
        """返回 {coin: {"free":可用, "lock":冻结, "total":总量}}"""
        positions = {}
        try:
            spot = self.client.get_balance().get("SpotWallet", {})
            for coin, data in spot.items():
                free = float(data.get("Free", 0))
                lock = float(data.get("Lock", 0))
                if free + lock > 0:
                    positions[coin] = {"free": free, "lock": lock,
                                       "total": free + lock}
        except Exception as e:
            logger.error("获取持仓失败: %s", e)
        return positions

    def _get_prices(self) -> Dict[str, float]:
        prices = {}
        try:
            ticker = self.client.get_ticker()
            for pair, data in ticker.get("Data", {}).items():
                prices[pair.split("/")[0]] = float(data.get("LastPrice", 0))
        except Exception as e:
            logger.error("获取行情失败: %s", e)
        return prices

    # ---------- 调仓 ----------
    def rebalance(self, target_weights: Dict[str, float]) -> dict:
        logger.info("=" * 50)
        logger.info("开始调仓")
        active = {k: f"{v:.2%}" for k, v in target_weights.items() if v != 0}
        logger.info("目标仓位: %s", active)

        portfolio_value = self.client.get_portfolio_value()
        positions = self._get_current_positions()
        prices = self._get_prices()
        logger.info("组合市值: %.2f USD", portfolio_value)

        # 关键修复：当前持仓与 target 取并集，缺失目标补0 → 防止漏平仓
        full_targets = dict(target_weights)
        for coin in positions:
            if coin == "USD":
                continue
            pair = f"{coin}/USD"
            full_targets.setdefault(pair, 0.0)

        orders, short_orders = [], []

        for pair, tw in full_targets.items():
            coin = pair.split("/")[0]
            p = positions.get(coin, {"free": 0.0, "lock": 0.0, "total": 0.0})
            cur_total, cur_free = p["total"], p["free"]

            if abs(tw) < 0.001:
                # 目标0 → 清仓，卖出量以 free 为上限
                sell_qty = min(cur_total, cur_free)
                if sell_qty > 0 and coin in prices:
                    value = sell_qty * prices[coin]
                    if value > portfolio_value * self.min_trade_pct:
                        orders.append({"pair": pair, "side": "SELL",
                                       "quantity": sell_qty, "price": prices[coin],
                                       "value": value})
                continue

            if coin not in prices:
                logger.warning("跳过 %s: 无行情", pair)
                continue

            price = prices[coin]
            target_qty = portfolio_value * abs(tw) / price

            if tw > 0:
                delta = target_qty - cur_total
                if abs(delta * price) > portfolio_value * self.min_trade_pct:
                    if delta > 0:
                        orders.append({"pair": pair, "side": "BUY",
                                       "quantity": abs(delta), "price": price,
                                       "value": abs(delta) * price})
                    else:
                        sell_qty = min(abs(delta), cur_free)  # 不超free
                        if sell_qty > 0:
                            orders.append({"pair": pair, "side": "SELL",
                                           "quantity": sell_qty, "price": price,
                                           "value": sell_qty * price})
            else:
                # 做空：先平多头（卖free），再开空
                sell_qty = min(cur_total, cur_free)
                if sell_qty > 0:
                    orders.append({"pair": pair, "side": "SELL",
                                   "quantity": sell_qty, "price": price,
                                   "value": sell_qty * price})
                short_orders.append({"pair": pair,
                                      "collateral": portfolio_value * abs(tw),
                                      "price": price})

        # 先卖后买
        orders = [o for o in orders if o["side"] == "SELL"] + \
                 [o for o in orders if o["side"] == "BUY"]

        executed, failed = [], []
        for order in orders:
            try:
                q = self._adjust_quantity(order["pair"], order["quantity"])
                if q <= 0:
                    continue
                result = self.client.place_order(
                    pair=order["pair"], side=order["side"],
                    quantity=q, order_type=self.order_type)
                if result.get("Success"):
                    executed.append({**order, "order_id":
                                     result.get("OrderDetail", {}).get("OrderID")})
                else:
                    failed.append({**order, "error": result.get("ErrMsg")})
                time.sleep(self.sleep_between_orders)
            except Exception as e:
                logger.error("下单失败 %s: %s", order["pair"], e)
                failed.append({**order, "error": str(e)})

        short_executed = []
        for short in short_orders:
            try:
                result = self.client.short_open(pair=short["pair"],
                                                collateral=short["collateral"])
                if result.get("Success"):
                    short_executed.append(short)
                else:
                    failed.append({**short, "error": result.get("ErrMsg")})
                time.sleep(self.sleep_between_orders)
            except Exception as e:
                failed.append({**short, "error": str(e)})

        summary = {
            "portfolio_value_before": portfolio_value,
            "orders_planned": len(orders) + len(short_orders),
            "orders_executed": len(executed),
            "short_executed": len(short_executed),
            "orders_failed": len(failed),
            "executed": executed, "failed": failed,
        }
        logger.info("调仓完成: 成功%d(空%d), 失败%d",
                    summary["orders_executed"], summary["short_executed"],
                    summary["orders_failed"])
        return summary

    # ---------- 全平 ----------
    def flatten_all(self):
        """紧急全平：现货（以free为上限）+ 空头"""
        logger.warning("执行全部平仓")
        positions = self._get_current_positions()
        prices = self._get_prices()

        for coin, p in positions.items():
            if coin == "USD":
                continue
            pair = f"{coin}/USD"
            sell_qty = min(p["total"], p["free"])
            if sell_qty <= 0 or coin not in prices:
                continue
            try:
                q = self._adjust_quantity(pair, sell_qty)
                if q > 0:
                    self.client.place_order(pair=pair, side="SELL",
                                            quantity=q, order_type="MARKET")
                    logger.info("  现货全平 %s %.6f", pair, q)
                    time.sleep(self.sleep_between_orders)
            except Exception as e:
                logger.error("  全平现货失败 %s: %s", pair, e)

        try:
            for pos in self.client.get_short_positions().get("Positions", []):
                pair = pos.get("Pair")
                self.client.short_close(pair=pair, close_pct=100)
                logger.info("  空头全平 %s", pair)
                time.sleep(self.sleep_between_orders)
        except Exception as e:
            logger.error("  全平空头失败: %s", e)

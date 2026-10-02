"""
Roostoo REST API 客户端（最终版，匹配官方文档）
==================================================
Base URL: https://mock-api.roostoo.com
认证: RST-API-KEY + MSG-SIGNATURE (HMAC SHA256)
POST Content-Type: application/x-www-form-urlencoded
交易对格式: BTC/USD
"""

import time
import hmac
import hashlib
import logging
from typing import Optional, Dict, List
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


class RoostooAPIError(Exception):
    def __init__(self, status_code: int, message: str, raw_response: dict = None):
        self.status_code = status_code
        self.raw_response = raw_response or {}
        super().__init__(f"[HTTP {status_code}] {message}")


class RoosterClient:
    """
    Roostoo 交易平台 REST API 客户端。

    用法:
        client = RoosterClient(api_key="...", api_secret="...")
        server_time = client.get_server_time()
        ticker = client.get_ticker("BTC/USD")
        order = client.place_order(pair="BTC/USD", side="BUY", quantity=0.01)
    """

    BASE_URL = "https://mock-api.roostoo.com"
    MAX_RETRIES = 3
    BASE_BACKOFF = 2
    TIMEOUT = 10

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.session = requests.Session()

    # ================================================================
    # 签名工具
    # ================================================================

    def _generate_signature(self, params: dict) -> str:
        """
        生成 HMAC SHA256 签名。
        1. 参数按 key 字母排序
        2. 用 & 连接成 key=value 格式
        3. 用 api_secret 做 HMAC SHA256
        """
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
        return hmac.new(
            self.api_secret,
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _get_signed_headers(self, params: dict) -> dict:
        """生成带签名的请求头"""
        signature = self._generate_signature(params)
        return {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }

    def _timestamp(self) -> int:
        """13位毫秒时间戳"""
        return int(time.time() * 1000)

    # ================================================================
    # 统一请求方法
    # ================================================================

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        signed: bool = True,
    ) -> dict:
        url = f"{self.BASE_URL}{path}"
        params = params or {}

        if signed:
            params["timestamp"] = self._timestamp()
            headers = self._get_signed_headers(params)
        else:
            headers = {}

        for attempt in range(self.MAX_RETRIES):
            try:
                if method.upper() == "GET":
                    resp = self.session.get(
                        url, params=params, headers=headers, timeout=self.TIMEOUT
                    )
                else:
                    # POST 用 form-urlencoded
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    resp = self.session.post(
                        url, data=params, headers=headers, timeout=self.TIMEOUT
                    )

                if 200 <= resp.status_code < 300:
                    return resp.json()

                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    wait = self.BASE_BACKOFF ** attempt
                    logger.warning("API 临时错误 HTTP %d，%d秒后重试", resp.status_code, wait)
                    time.sleep(wait)
                    continue

                try:
                    err_body = resp.json()
                except Exception:
                    err_body = {"raw": resp.text}
                raise RoostooAPIError(
                    status_code=resp.status_code,
                    message=err_body.get("ErrMsg", resp.text),
                    raw_response=err_body,
                )

            except requests.RequestException as e:
                wait = self.BASE_BACKOFF ** attempt
                logger.warning("网络异常: %s，%d秒后重试", e, wait)
                time.sleep(wait)

        raise RoostooAPIError(500, "重试多次后仍然失败")

    # ================================================================
    # 公开接口（无需签名）
    # ================================================================

    def get_server_time(self) -> dict:
        """获取服务器时间"""
        return self._request("GET", "/v3/serverTime", signed=False)

    def get_exchange_info(self) -> dict:
        """获取交易所信息：所有交易对、精度、最小下单量"""
        return self._request("GET", "/v3/exchangeInfo", signed=False)

    # ================================================================
    # 行情接口（需要时间戳）
    # ================================================================

    def get_ticker(self, pair: Optional[str] = None) -> dict:
        """
        获取实时行情。
        pair: 如 "BTC/USD"，不传则返回所有交易对
        """
        params = {}
        if pair:
            params["pair"] = pair
        return self._request("GET", "/v3/ticker", params=params, signed=True)

    # ================================================================
    # 账户接口（需要签名）
    # ================================================================

    def get_balance(self) -> dict:
        """查询账户余额"""
        return self._request("GET", "/v3/balance", signed=True)

    def get_pending_count(self) -> dict:
        """查询挂单数量"""
        return self._request("GET", "/v3/pending_count", signed=True)

    # ================================================================
    # 交易接口
    # ================================================================

    def place_order(
        self,
        pair: str,           # 如 "BTC/USD"
        side: str,           # "BUY" 或 "SELL"
        quantity: float,
        order_type: str = "MARKET",  # "MARKET" 或 "LIMIT"
        price: Optional[float] = None,
    ) -> dict:
        """
        下单。
        MARKET: 市价单，立即成交
        LIMIT:  限价单，需要传 price
        """
        params = {
            "pair": pair,
            "side": side.upper(),
            "quantity": str(quantity),
            "type": order_type.upper(),
        }
        if order_type.upper() == "LIMIT" and price is not None:
            params["price"] = str(price)

        logger.info("下单: %s %s %s @ %s", side, quantity, pair, price or "MARKET")
        return self._request("POST", "/v3/place_order", params=params, signed=True)

    def query_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
    ) -> dict:
        """查询订单"""
        params = {}
        if order_id is not None:
            params["order_id"] = order_id
        if pair:
            params["pair"] = pair
        if pending_only is not None:
            params["pending_only"] = str(pending_only)
        return self._request("POST", "/v3/query_order", params=params, signed=True)

    def cancel_order(self, pair: str, order_id: Optional[int] = None) -> dict:
        """撤销订单"""
        params = {"pair": pair}
        if order_id is not None:
            params["order_id"] = order_id
        return self._request("POST", "/v3/cancel_order", params=params, signed=True)

    def cancel_all_orders(self, pair: Optional[str] = None) -> dict:
        """撤销所有挂单（紧急风控用）"""
        # API 没有批量撤单，逐个撤
        pending = self.get_pending_count()
        logger.warning("撤销全部订单: %s", pair or "ALL")
        return pending

    # ================================================================
    # 做空接口（v6）
    # ================================================================

    def short_open(
        self,
        pair: str,
        collateral: float,       # 抵押品金额（USD）
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> dict:
        """
        开空仓。
        注意：做空用 collateral（抵押金额），不是 quantity！
        """
        params = {
            "pair": pair,
            "collateral": str(collateral),
        }
        if order_type.upper() == "LIMIT" and price is not None:
            params["order_type"] = "LIMIT"
            params["price"] = str(price)

        logger.info("开空: %s 抵押$%.2f", pair, collateral)
        return self._request("POST", "/v6/short_open", params=params, signed=True)

    def short_close(
        self,
        pair: str,
        close_qty: Optional[float] = None,
        close_pct: Optional[float] = None,
    ) -> dict:
        """
        平空仓。
        不传参数则平掉全部。
        close_qty 和 close_pct 二选一。
        """
        params = {"pair": pair}
        if close_qty is not None:
            params["close_qty"] = str(close_qty)
        elif close_pct is not None:
            params["close_pct"] = str(close_pct)

        logger.info("平空: %s", pair)
        return self._request("POST", "/v6/short_close", params=params, signed=True)

    def get_short_positions(self) -> dict:
        """查询当前空头持仓"""
        return self._request("GET", "/v6/short_positions", signed=True)

    # ================================================================
    # 便捷方法
    # ================================================================

    def get_portfolio_value(self) -> float:
        """计算当前投资组合总市值（USD），包含所有币种持仓"""
        balance = self.get_balance()
        try:
            spot = balance.get("SpotWallet", {})
            total = 0.0

            # 获取所有交易对的行情
            ticker = self.get_ticker()
            prices = {}
            if ticker and "Data" in ticker:
                for pair, data in ticker["Data"].items():
                    coin = pair.split("/")[0]
                    prices[coin] = data.get("LastPrice", 0)

            # 计算每个币种的市值
            for coin, data in spot.items():
                free = float(data.get("Free", 0))
                lock = float(data.get("Lock", 0))
                qty = free + lock
                if coin == "USD":
                    total += qty
                elif coin in prices:
                    total += qty * prices[coin]

            return total
        except Exception as e:
            logger.warning("计算组合市值失败: %s", e)
            return 0.0

    def get_available_pairs(self) -> list:
        """获取所有可交易的交易对"""
        info = self.get_exchange_info()
        try:
            return list(info.get("TradePairs", {}).keys())
        except Exception:
            return []

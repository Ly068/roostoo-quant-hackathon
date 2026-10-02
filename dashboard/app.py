"""
量化交易 Dashboard（Streamlit）v5：时序动量只做多
决赛Pitch与比赛期间实时监控两用。

运行：
  streamlit run dashboard/app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLD_DIR = os.path.join(BASE_DIR, "data", "history_binance")
LIVE_DIR = os.path.join(BASE_DIR, "data", "live")

st.set_page_config(page_title="Team48-Y 时序动量系统", layout="wide", page_icon="📊")


@st.cache_data(ttl=300)
def load_universe_coins():
    uf = os.path.join(COLD_DIR, "universe.txt")
    return [l.strip() for l in open(uf, encoding="utf-8") if l.strip()]


@st.cache_data(ttl=120)
def load_prices(coin):
    for d in [LIVE_DIR, COLD_DIR]:
        fp = os.path.join(d, f"{coin}_1h.csv")
        if os.path.exists(fp):
            return pd.read_csv(fp, index_col=0, parse_dates=True)
    return None


def trend_table(coins, lookback=720):
    rows = []
    for coin in coins:
        df = load_prices(coin)
        if df is None or len(df) < lookback + 5:
            continue
        close = df["close"]
        ret = close.iloc[-1] / close.iloc[-lookback] - 1
        up = close.iloc[-1] > close.iloc[-lookback]
        rows.append({"币种": coin, "30天收益": ret,
                     "趋势状态": "上升(做多)" if up else "无趋势(空仓)"})
    return pd.DataFrame(rows).sort_values("30天收益", ascending=False).reset_index(drop=True)


st.sidebar.title("Team48-Y · HKU")
st.sidebar.caption("APAC Quant Trading Hackathon")
page = st.sidebar.radio("导航", ["策略总览", "趋势状态", "行情", "回测验证", "风控体系"])

coins = load_universe_coins()

# ================================================================
if page == "策略总览":
    st.title("时序动量 · 趋势跟踪（只做多）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("交易币种", f"{len(coins)}")
    c2.metric("动量窗口", "30天")
    c3.metric("再平衡", "每3天")
    c4.metric("方向", "只做多")

    st.subheader("策略逻辑")
    st.markdown("""
    **核心信号：** 对每个币种，若 **现价 > 30天前价格**（过去30天处于上升趋势），
    则纳入做多池，池内等权、总多头敞口100%；**无任何币种满足 → 100%现金，自动避险。**

    **为什么只做多：** 加密市场有长期上涨偏差；做空弱势山寨在轮动市极易被空头挤压
    （实测多空横截面动量在6–10月上涨轮动市亏损 -18%）。

    **为什么无趋势空仓：** 时序动量在震荡/下跌市会因假趋势反复止损，空仓机制不硬扛，天然控制下行。

    **风险特征（非对称）：** 震荡/下跌月下行约 -5%，上涨月充分参与（历史 +23%~40%）。
    """)
    st.info("动态仓位：观察期(1-3天)60% → 确认期(4-10天)100% → 收尾期(11-14天)70%")

# ================================================================
elif page == "趋势状态":
    st.title("币种趋势状态（实时）")
    df = trend_table(coins)
    if len(df):
        longs = df[df["趋势状态"] == "上升(做多)"]["币种"].tolist()
        fig = px.bar(df, x="币种", y="30天收益", color="趋势状态",
                     color_discrete_map={"上升(做多)": "#2ca02c", "无趋势(空仓)": "#bbbbbb"},
                     title="30天收益与趋势状态（绿色=做多池，灰色=空仓）")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"**做多池（{len(longs)}币，等权）：** {', '.join(longs)}")
        show = df.copy()
        show["30天收益"] = show["30天收益"].map(lambda x: f"{x*100:+.2f}%")
        st.dataframe(show, use_container_width=True)

# ================================================================
elif page == "行情":
    st.title("币种行情")
    coin = st.selectbox("选择币种", coins,
                        index=coins.index("BTC") if "BTC" in coins else 0)
    df = load_prices(coin)
    if df is not None:
        sub = df.tail(720)
        fig = go.Figure(data=[go.Candlestick(
            x=sub.index, open=sub["open"], high=sub["high"],
            low=sub["low"], close=sub["close"], name="K线")])
        fig.update_layout(title=f"{coin}/USD 近30天", height=480,
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("最新价", f"{df['close'].iloc[-1]:,.4f}")
        c2.metric("30天涨跌", f"{(df['close'].iloc[-1]/df['close'].iloc[-720]-1)*100:+.2f}%")
        c3.metric("近20h年化波动",
                  f"{df['close'].pct_change().iloc[-20:].std()*np.sqrt(24*365)*100:.1f}%")

# ================================================================
elif page == "回测验证":
    st.title("回测与稳健性验证")
    st.caption("数据：Binance 1h K线（volume完整），手续费0.1%/边，72h调仓")

    st.subheader("三段独立窗口（每段约29天，互不重叠）")
    seg = pd.DataFrame([
        {"窗口": "段1 (7月震荡)", "收益": -4.8, "Sharpe": -1.3},
        {"窗口": "段2 (8月涨)", "收益": 23.5, "Sharpe": 4.7},
        {"窗口": "段3 (9月山寨季)", "收益": 40.5, "Sharpe": 6.4},
    ])
    fig = px.bar(seg, x="窗口", y="收益", color="收益",
                 color_continuous_scale=["#d62728", "#eeeeee", "#2ca02c"],
                 title="三段收益（非对称：震荡月小亏，上涨月充分参与）")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg, use_container_width=True)

    st.subheader("全周期（120天）")
    m1, m2, m3 = st.columns(3)
    m1.metric("总收益", "+65.4%")
    m2.metric("Sharpe", "12.4")
    m3.metric("最大回撤", "-13.4%")

    st.warning("诚实说明：历史高Sharpe是特定趋势行情产物，不能外推；"
               "若14天持续无趋势，可能不赚或小亏。")

# ================================================================
elif page == "风控体系":
    st.title("风控体系（独立 RiskManager）")
    st.markdown("""
    | 层级 | 规则 | 动作 |
    |------|------|------|
    | 单标的止损 | 相对开仓价反向偏离10% | 立即单独平仓 |
    | 日回撤熔断 | 日内回撤>4% / >8% | 减仓预警 / 全平暂停1h |
    | 连亏熔断 | 连续亏损3笔 | 暂停30分钟 |
    | 敞口约束 | 总敞口≤100%（无杠杆） | 硬约束 |
    """)
    st.markdown("""
    **空仓机制的风控意义：** 无上升趋势时组合自动转为现金，是第一道、也是最有效的下行保护；
    10%止损应对单币黑天鹅，8%日回撤全平应对系统性急跌。
    """)

st.sidebar.caption("初始资金 $50,000 · 周期14天 · 10月4-17日")

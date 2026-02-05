import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# --- 1. 頁面設定 ---
st.set_page_config(page_title="台股 AI 隨身分析師", layout="wide")

# --- 2. 左側邊欄：輸入與警語 ---
st.sidebar.title("🔍 股票設定")
stock_code = st.sidebar.text_input("輸入台股代碼", "2330")
full_code = f"{stock_code}.TW"

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **【免責聲明】**\n\n"
    "本工具僅供技術研究與程式教學使用。\n"
    "AI 預測結果與技術分析內容**不保證準確性**，"
    "股市有風險，請勿僅依賴本程式作為買賣依據。"
)

# --- 3. 核心函數：抓取資料與計算 ---
@st.cache_data
def load_data_and_predict(code):
    # 建立 Ticker 物件
    ticker = yf.Ticker(code)
    try:
        # 嘗試取得簡稱
        stock_name = ticker.info.get('shortName') or ticker.info.get('longName') or code
    except:
        stock_name = code
    
    # 抓取歷史數據
    df = ticker.history(start="2020-01-01")
    
    # 順便抓取新聞 (將資料取出，不要回傳 Ticker 物件本身)
    try:
        stock_news = ticker.news
    except:
        stock_news = []
    
    if df.empty:
        return None, None, None, None

    # --- 特徵工程 (計算指標) ---
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- XGBoost 預測模型 ---
    df_clean = df.dropna().copy()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI']
    X = df_clean[features]
    y = df_clean['Close'].shift(-1).dropna()
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X[:-1], y)
    
    last_row = X.tail(1)
    prediction = model.predict(last_row)[0]

    # 修改回傳值：只回傳數據 (df, name, prediction, news)，不回傳物件
    return df, stock_name, prediction, stock_news

# --- 4. 主程式執行邏輯 ---

with st.status(f"正在分析 {stock_code} 的大數據...", expanded=True) as status:
    st.write("📥 下載最新股價與基本資料...")
    # 這裡接收的第四個參數變成了 stock_news
    df, name, pred_price, stock_news = load_data_and_predict(full_code)
    
    if df is not None:
        st.write("🧮 計算技術指標 (RSI, MACD, 均線)...")
        st.write("🤖 AI 模型正在進行趨勢預測...")
        status.update(label=f"✅ {name} 分析完成！", state="complete", expanded=False)
    else:
        status.update(label="❌ 找不到股票", state="error")
        st.error(f"找不到代碼 {stock_code}，請確認是否為上市櫃股票。")
        st.stop()

# --- 5. 顯示結果區 ---

st.title(f"🚀 {name} ({stock_code}) AI 分析報告")

current_price = df['Close'].iloc[-1]
yesterday_price = df['Close'].iloc[-2]
price_change = current_price - yesterday_price
pct_change = (price_change / yesterday_price) * 100
vol_change = df['Volume'].iloc[-1] / df['Volume'].iloc[-2]

c1, c2, c3, c4 = st.columns(4)
c1.metric("目前股價", f"{current_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
c2.metric("AI 預測明日", f"{pred_price:.2f}", f"{pred_price - current_price:.2f}")
c3.metric("RSI 熱度", f"{df['RSI'].iloc[-1]:.1f}")
c4.metric("成交量變化", "放量" if vol_change > 1.2 else "縮量" if vol_change < 0.8 else "持平")

# --- 6. 口語化分析引擎 ---
st.subheader("💡 簡單白話分析 (近況與趨勢)")

rsi = df['RSI'].iloc[-1]
ma20 = df['MA20'].iloc[-1]
ma60 = df['MA60'].iloc[-1]
trend = ""
rsi_status = ""

# A. 判斷趨勢
if current_price > ma20 and current_price > ma60:
    trend = "目前股價站穩在生命線（季線）之上，整體格局偏向**多頭（上漲趨勢）**，主力做多意願強。"
elif current_price < ma20 and current_price < ma60:
    trend = "目前股價跌破生命線，整體格局偏向**空頭（下跌趨勢）**，上方賣壓可能比較重。"
elif current_price > ma20 and current_price < ma60:
    trend = "股價正在嘗試反彈，雖然站上月線，但還沒突破長期的壓力，目前處於**震盪整理**階段。"
else:
    trend = "股價短期回檔，跌破了月線支撐，需要觀察能否守住長期的季線，目前走勢比較**糾結**。"

# B. 判斷熱度
if rsi > 80:
    rsi_status = "🔥 **市場過熱警告**：現在大家都在搶買，RSI 指標衝太高了，短線隨時可能會有獲利了結的賣壓，追高要小心！"
elif rsi > 60:
    rsi_status = "💪 **人氣很旺**：買盤力道強勁，市場氣氛樂觀，是個強勢的表現。"
elif rsi < 20:
    rsi_status = "🧊 **市場結凍**：跌太深了，現在大家都在恐慌拋售，RSI 進入超賣區，反而可能會有「反彈」的機會喔。"
elif rsi < 40:
    rsi_status = "😰 **人氣渙散**：買氣不足，市場氣氛偏弱，大家還在觀望。"
else:
    rsi_status = "⚖️ **冷熱適中**：目前多空雙方力道差不多，沒有明顯過熱或恐慌，股價走勢比較平穩。"

# C. AI 預測解讀
ai_gap = ((pred_price - current_price) / current_price) * 100
if ai_gap > 1:
    ai_msg = f"🤖 **AI 模型看法**：模型偵測到上漲訊號，預估明日有 **{ai_gap:.2f}%** 左右的潛在漲幅。"
elif ai_gap < -1:
    ai_msg = f"🤖 **AI 模型看法**：模型偵測到下跌風險，預估明日可能會有 **{abs(ai_gap):.2f}%** 的修正。"
else:
    ai_msg = "🤖 **AI 模型看法**：模型認為明日走勢**持平震盪**，可能變化不大。"

with st.container():
    st.info(f"""
    **【趨勢解讀】** {trend}
    
    **【市場情緒】** {rsi_status}
    
    {ai_msg}
    """)

# --- 7. 圖表展示 ---
tab1, tab2 = st.tabs(["📊 互動 K 線圖", "📰 相關新聞"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="月線 (20MA)", line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name="季線 (60MA)", line=dict(color='blue', width=1.5)), row=1, col=1)

    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"📰 {name} 最新動態")
    # 直接使用回傳的 stock_news 列表，而不是呼叫物件
    if stock_news:
        for n in stock_news[:5]:
            # 處理時間格式，防止報錯
            try:
                pub_time = pd.to_datetime(n.get('providerPublishTime'), unit='s').strftime('%Y-%m-%d %H:%M')
            except:
                pub_time = "未知時間"
                
            st.markdown(f"**[{n.get('title', '無標題')}]({n.get('link', '#')})**")
            st.caption(f"發布時間: {pub_time} | 來源: {n.get('publisher', '未知')}")
            st.divider()
    else:
        st.write("目前沒有抓取到相關新聞。")
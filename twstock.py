import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# 頁面設定
st.set_page_config(page_title="台股 AI 預測與技術分析", layout="wide")
st.title("🚀 台股 AI 綜合分析儀表板")

# 側邊欄輸入
stock_code = st.sidebar.text_input("輸入台股代碼", "2330")
full_code = f"{stock_code}.TW"
period = st.sidebar.selectbox("查看區間", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

# 1. 抓取數據
@st.cache_data
def load_data(code):
    df = yf.download(code, start="2020-01-01")
    return df

df = load_data(full_code)

if not df.empty:
    # --- 技術指標計算 (特徵工程) ---
    # 移動平均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # RSI 指標
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- AI 預測模型 (XGBoost) ---
    df_clean = df.dropna().copy()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']
    X = df_clean[features]
    y = df_clean['Close'].shift(-1).dropna()
    X_train = X[:-1]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y)
    
    last_row = X.tail(1)
    prediction = model.predict(last_row)[0]

    # --- 頂部數據卡片 ---
    curr_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    delta_price = curr_price - prev_price
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("目前股價", f"{curr_price:.2f}", f"{delta_price:.2f}")
    col2.metric("AI 預測明日", f"{float(prediction):.2f}", f"{float(prediction)-curr_price:.2f}")
    col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
    col4.metric("成交量", f"{int(df['Volume'].iloc[-1]):,}")

   # --- 繪製多圖表 (K線 + 指標) ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2])

    # 主圖：K線與均線 (注意這裡是用 col=1 而不是 col1)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name="MA5", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(width=1)), row=1, col=1)

    # 副圖：RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    # 加入 RSI 上下限橫線
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color="red", dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color="green", dash="dash"), row=2, col=1)

    # 副圖：MACD
    fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal'], name="MACD柱狀圖"), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- 分析建議 ---
    st.subheader("💡 技術面綜合分析")
    advice = []
    if df['RSI'].iloc[-1] > 70: advice.append("⚠️ RSI 超過 70，進入超買區，注意過熱風險。")
    elif df['RSI'].iloc[-1] < 30: advice.append("✅ RSI 低於 30，進入超跌區，可能存在反彈機會。")
    
    if curr_price > df['MA5'].iloc[-1] and df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
        advice.append("📈 目前均線呈現多頭排列，短期走勢強勁。")
    
    for item in advice:
        st.write(item)

else:
    st.error("無法取得數據，請確認代碼是否正確。")
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

st.title("🚀 台股 AI 預測檢視器 V2.0")
stock_code = st.text_input("請輸入台股代碼", "2330")
full_code = f"{stock_code}.TW"

# 1. 抓取較長的歷史數據供訓練
df = yf.download(full_code, start="2020-01-01")

if not df.empty:
    # --- 數據前處理 (特徵工程) ---
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    # 預測目標：明天的收盤價 (將收盤價向上平移一天)
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    # 選取特徵與目標
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'Daily_Return']
    X = df[features]
    y = df['Target']

    # 2. 簡單模型訓練 (使用 XGBoost)
    # 在實際應用中，模型應事先訓練好並儲存，這裡為了演示在網頁即時訓練
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X[:-1], y[:-1]) # 保留最後一筆用來預測未來

    # 3. 預測未來
    last_data = X.tail(1)
    prediction = model.predict(last_data)[0]
    
    # --- 視覺化 ---
    fig = go.Figure()
    # 歷史 K 線
    fig.add_trace(go.Candlestick(x=df.index[-30:], 
                  open=df['Open'], high=df['High'],
                  low=df['Low'], close=df['Close'], name="近期走勢"))
    
    # 預測點
    next_day = df.index[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(x=[next_day], y=[prediction], 
                             mode='markers+text', 
                             text=[f"預測明日: {prediction:.2f}"],
                             textposition="top center",
                             marker=dict(color='red', size=12), name="AI 預測"))

    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"📊 **分析結果：**")
    st.write(f"- 今日收盤：{df['Close'].iloc[-1]:.2f}")
    st.write(f"- AI 預測明日收盤：{prediction:.2f}")
    
    direction = "🔴 上漲" if prediction > df['Close'].iloc[-1] else "🟢 下跌"
    st.info(f"模型預測方向：{direction}")

else:
    st.error("代碼錯誤或無數據")
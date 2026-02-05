import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import pytz

# --- 1. 頁面設定 ---
st.set_page_config(page_title="台股 AI 操盤手 Pro+", layout="wide")

# --- 2. 左側邊欄：設定與刷新 ---
st.sidebar.title("🔍 操盤控制台")
stock_code = st.sidebar.text_input("輸入台股代碼", "2603") 
full_code = f"{stock_code}.TW"

# --- 新增：日期區間篩選功能 ---
st.sidebar.subheader("📅 檢視區間設定")
date_option = st.sidebar.selectbox(
    "快速選擇區間", 
    ["近 1 個月", "近 3 個月", "近 6 個月", "近 1 年", "近 3 年", "全部", "自訂範圍"], 
    index=2 # 預設選 6 個月，看起來最舒服
)

start_date_filter = None
end_date_filter = None

if date_option == "自訂範圍":
    # 讓使用者自己選日期
    today = datetime.today()
    start_input = st.sidebar.date_input("開始日期", today - timedelta(days=180))
    end_input = st.sidebar.date_input("結束日期", today)
    start_date_filter = pd.Timestamp(start_input).tz_localize("Asia/Taipei") if start_input else None
    end_date_filter = pd.Timestamp(end_input).tz_localize("Asia/Taipei") + timedelta(days=1) if end_input else None
else:
    # 自動計算日期
    days_map = {
        "近 1 個月": 30,
        "近 3 個月": 90,
        "近 6 個月": 180,
        "近 1 年": 365,
        "近 3 年": 1095
    }
    if date_option != "全部":
        # 計算開始日期 (注意時區處理)
        start_date_filter = datetime.now(pytz.timezone('Asia/Taipei')) - timedelta(days=days_map[date_option])
    else:
        start_date_filter = None # 全部就不設限

if st.sidebar.button("🔄 立即更新股價"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 **小提醒**：\nAI 預測僅供參考，請搭配下方技術指標（如成交量、OBV）一起判斷，準確率更高。")

# --- 3. 核心函數：計算指標與 AI 預測 ---
@st.cache_data
def load_data_and_predict(code):
    ticker = yf.Ticker(code)
    
    # A. 嘗試抓取名稱
    try:
        info = ticker.info
        stock_name = info.get('longName') or info.get('shortName') or code
    except:
        stock_name = code
    
    # 抓取歷史數據 (抓多一點，確保指標計算準確，之後再篩選顯示)
    df = ticker.history(start="2018-01-01")
    
    # 抓取新聞
    try:
        stock_news = ticker.news
    except:
        stock_news = []
    
    if df.empty:
        return None, None, None, None, None

    # B. 基礎指標
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()   
    df['MA60'] = df['Close'].rolling(window=60).mean()   

    # C. 量能指標
    df['VolMA5'] = df['Volume'].rolling(window=5).mean()
    df['VolMA20'] = df['Volume'].rolling(window=20).mean()
    # OBV 能量潮
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # D. 震盪指標
    # RSI
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

    # KD
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

    # E. XGBoost 預測模型
    df_clean = df.dropna().copy()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'K', 'D', 'OBV']
    X = df_clean[features]
    y = df_clean['Close'].shift(-1).dropna()
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X[:-1], y)
    
    prediction = model.predict(X.tail(1))[0]

    # 時間處理
    last_time = df.index[-1]
    if last_time.tzinfo is None:
        tz = pytz.timezone('Asia/Taipei')
        last_time = last_time.replace(tzinfo=pytz.utc).astimezone(tz)
    else:
        last_time = last_time.astimezone(pytz.timezone('Asia/Taipei'))
    update_time_str = last_time.strftime('%Y-%m-%d %H:%M')

    return df, stock_name, prediction, stock_news, update_time_str

# --- 4. 主程式邏輯 ---

with st.status(f"🚀 AI 正在分析 {stock_code} 的走勢與籌碼...", expanded=True) as status:
    df, name, pred_price, stock_news, update_time = load_data_and_predict(full_code)
    
    if df is not None:
        st.write("🤖 預測明日股價中...")
        st.write("📊 計算 OBV 與主力籌碼...")
        status.update(label=f"✅ {name} 分析完成！", state="complete", expanded=False)
    else:
        status.update(label="❌ 讀取失敗", state="error")
        st.error(f"找不到代碼 {stock_code}，請確認輸入正確。")
        st.stop()

# --- 5. 資料篩選邏輯 (關鍵步驟) ---
# 先備份完整的 df 用於計算最新數據
df_full = df.copy()

# 進行區間篩選 (只影響圖表顯示，不影響最新價與 AI 預測)
if start_date_filter:
    # 確保索引有時區資訊以便比較
    if df.index.tzinfo is None:
         df.index = df.index.tz_localize("Asia/Taipei")
    
    df_view = df[df.index >= start_date_filter]
    
    if end_date_filter and date_option == "自訂範圍":
        df_view = df_view[df_view.index <= end_date_filter]
else:
    df_view = df

# 如果篩選後沒資料，就顯示全部
if df_view.empty:
    st.warning("⚠️ 選定的日期區間沒有數據，已自動切換顯示全部資料。")
    df_view = df

# --- 6. 儀表板顯示 ---
st.title(f"📊 {name} ({stock_code}) AI 診斷報告")
st.caption(f"🕒 資料時間：{update_time} | ⚠️ 僅供參考，投資盈虧自負")

# 使用 df_full (完整數據) 來抓取最新狀態，確保數據是最新的
curr_close = df_full['Close'].iloc[-1]
diff = curr_close - df_full['Close'].iloc[-2]
pct = (diff / df_full['Close'].iloc[-2]) * 100
vol_today = df_full['Volume'].iloc[-1]
vol_avg = df_full['VolMA20'].iloc[-1]
vol_ratio = vol_today / vol_avg

# 預測漲跌幅計算
pred_diff = pred_price - curr_close
pred_pct = (pred_diff / curr_close) * 100

# 頂部指標卡
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("目前股價", f"{curr_close:.2f}", f"{diff:.2f} ({pct:.2f}%)")
c2.metric("AI 預測明日", f"{pred_price:.2f}", f"{pred_diff:.2f} ({pred_pct:.2f}%)")
c3.metric("成交量 (張)", f"{int(vol_today/1000):,}K", f"{(vol_today-vol_avg)/1000:.1f}K")
c4.metric("OBV 能量", "⬆️ 累積中" if df_full['OBV'].iloc[-1] > df_full['OBV'].iloc[-2] else "⬇️ 流失中")
c5.metric("KD 指標", f"K:{df_full['K'].iloc[-1]:.0f}")

# --- 7. 白話文診斷室 (含 AI 解讀) ---
st.subheader("💡 買賣訊號與 AI 觀點")

# 訊號判斷邏輯 (使用最新數據)
k = df_full['K'].iloc[-1]
d = df_full['D'].iloc[-1]
rsi = df_full['RSI'].iloc[-1]
obv_trend = "上升" if df_full['OBV'].iloc[-1] > df_full['OBV'].iloc[-5] else "下降"
signals = []

if k < 20 and k > d: signals.append("🟢 **KD 黃金交叉**：在低檔出現買進訊號，反彈機會高。")
if k > 80 and k < d: signals.append("🔴 **KD 死亡交叉**：在高檔轉弱，建議獲利了結。")
if rsi > 75: signals.append("🔴 **RSI 過熱**：買氣太瘋狂了，小心隨時回檔。")
if vol_ratio > 2.0 and pct > 0: signals.append("🔥 **爆量上漲**：主力帶量攻擊，行情可能還沒結束。")
if obv_trend == "上升" and diff < 0: signals.append("✨ **量價背離 (多)**：股價跌但 OBV 在漲，代表有人在偷偷吃貨！")

# 顯示區塊
col_text1, col_text2 = st.columns([1.5, 1])

with col_text1:
    st.info("📊 **【技術面訊號】**")
    if not signals:
        st.write("⚖️ 目前技術指標呈現**中性**，無明顯強烈買賣訊號，建議區間操作。")
    else:
        for s in signals:
            st.write(s)
            
    if vol_today < vol_avg * 0.6:
        st.warning("🧊 **量能狀態**：今日量縮，市場觀望氣氛濃厚。")
    elif vol_today > vol_avg * 1.5:
        st.success("🚀 **量能狀態**：今日出量，市場交投熱絡。")

with col_text2:
    st.info("🤖 **【AI 預測解讀】**")
    st.write(f"模型預測目標價：**{pred_price:.2f}**")
    
    if pred_pct > 1.5:
        st.write(f"🚀 看法：**強勢看漲** (預估漲幅 +{pred_pct:.2f}%)")
        st.write("建議：AI 認為明日動能強勁，可偏多思考。")
    elif pred_pct > 0:
        st.write(f"📈 看法：**小幅上漲** (預估漲幅 +{pred_pct:.2f}%)")
        st.write("建議：趨勢溫和向上，持股續抱。")
    elif pred_pct > -1.5:
        st.write(f"📉 看法：**小幅震盪** (預估跌幅 {pred_pct:.2f}%)")
        st.write("建議：可能面臨整理，多看少做。")
    else:
        st.write(f"🩸 看法：**顯著下跌** (預估跌幅 {pred_pct:.2f}%)")
        st.write("建議：AI 偵測到賣壓風險，建議避險。")

# --- 8. 圖表區 (使用 df_view 篩選後的資料繪圖) ---
tab1, tab2 = st.tabs(["📈 趨勢技術圖", "📰 相關新聞"])

with tab1:
    # 建立 4 個子圖表 (K線, 成交量, KD, OBV)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2])

    # 1. K線 + 均線 (使用 df_view)
    fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['Open'], high=df_view['High'], 
                                low=df_view['Low'], close=df_view['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MA20'], name="月線", line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MA60'], name="季線", line=dict(color='blue', width=1)), row=1, col=1)

    # 2. 成交量
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df_view.iterrows()]
    fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], marker_color=colors, name="成交量"), row=2, col=1)

    # 3. KD
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['K'], name="K值", line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['D'], name="D值", line=dict(color='blue')), row=3, col=1)

    # 4. OBV
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['OBV'], name="OBV 能量", line=dict(color='purple', width=1.5)), row=4, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"📰 {name} 最新消息")
    if stock_news:
        for n in stock_news[:10]:
            try:
                pub_time = pd.to_datetime(n.get('providerPublishTime'), unit='s').strftime('%Y-%m-%d %H:%M')
            except:
                pub_time = "未知時間"
            
            st.markdown(f"> **[{n.get('title', '無標題')}]({n.get('link', '#')})** \n> *{pub_time}* | {n.get('publisher', 'Yahoo Finance')}")
            st.markdown("---")
    else:
        st.write("⚠️ 暫無新聞")
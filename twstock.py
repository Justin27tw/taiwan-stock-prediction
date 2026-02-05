import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta, time
import pytz
import twstock

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球股市 AI 戰情室", layout="wide")

# --- 2. 左側邊欄：設定 ---
st.sidebar.title("🔍 戰情控制室")

# 市場選擇
market_type = st.sidebar.selectbox(
    "選擇市場", 
    ["🇹🇼 台股", "🇭🇰 港股", "🇺🇸 美股"],
    index=0,
    key="market_selector"
)

# 根據市場預設代碼
default_code = "2603"
if market_type == "🇭🇰 港股":
    default_code = "9988" # 阿里巴巴
elif market_type == "🇺🇸 美股":
    default_code = "NVDA" # 輝達

stock_code = st.sidebar.text_input("輸入股票代碼", default_code, key="sidebar_stock_code") 

# 自動處理代碼後綴
is_tw_stock = False
if "台股" in market_type:
    full_code = f"{stock_code}.TW"
    is_tw_stock = True
elif "港股" in market_type:
    # 港股自動補0
    if len(stock_code) < 4:
        clean_code = stock_code.zfill(4)
    else:
        clean_code = stock_code
    full_code = f"{clean_code}.HK"
    is_tw_stock = False
else:
    full_code = stock_code # 美股
    is_tw_stock = False

# 日期區間篩選
st.sidebar.subheader("📅 趨勢圖區間")
date_option = st.sidebar.selectbox(
    "選擇顯示範圍", 
    ["近 3 個月", "近 6 個月", "近 1 年", "近 3 年", "全部"], 
    index=1,
    key="sidebar_date_option"
)

if st.sidebar.button("🔄 立即全盤掃描", key="sidebar_refresh_btn"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 **顯示設定**：\n🔴 紅色 = 上漲 (Bullish)\n🟢 綠色 = 下跌 (Bearish)")

# --- 3. 輔助函數：判斷開休市狀態 ---
def check_market_status(market):
    """
    回傳: (is_open: bool, status_text: str, status_color: str)
    """
    utc_now = datetime.now(pytz.utc)
    
    if "台股" in market:
        tz = pytz.timezone('Asia/Taipei')
        local_now = utc_now.astimezone(tz)
        # 台股交易時間: 週一至週五 09:00 - 13:30
        if 0 <= local_now.weekday() <= 4:
            current_time = local_now.time()
            start = time(9, 0)
            end = time(13, 30)
            if start <= current_time <= end:
                return True, "🟢 開盤中 (交易進行中)", "#22c55e"
    
    elif "港股" in market:
        tz = pytz.timezone('Asia/Hong_Kong')
        local_now = utc_now.astimezone(tz)
        # 港股交易時間: 週一至週五 09:30 - 16:00
        if 0 <= local_now.weekday() <= 4:
            current_time = local_now.time()
            start = time(9, 30)
            end = time(16, 0)
            if start <= current_time <= end:
                return True, "🟢 開盤中 (交易進行中)", "#22c55e"

    elif "美股" in market:
        tz = pytz.timezone('America/New_York')
        local_now = utc_now.astimezone(tz)
        # 美股交易時間: 週一至週五 09:30 - 16:00 (當地時間)
        if 0 <= local_now.weekday() <= 4:
            current_time = local_now.time()
            start = time(9, 30)
            end = time(16, 0)
            if start <= current_time <= end:
                return True, "🟢 開盤中 (美股盤中)", "#22c55e"

    return False, "🔴 已收盤 (Market Closed)", "#ef4444"

# --- 4. 核心函數：全方位資料抓取 ---
@st.cache_data
def load_comprehensive_data(raw_code, yf_code, is_taiwan):
    # 1. 名稱抓取邏輯
    stock_name = raw_code
    industry = "未知產業"
    
    if is_taiwan:
        try:
            if raw_code in twstock.codes:
                info_tw = twstock.codes[raw_code]
                stock_name = info_tw.name
                industry = info_tw.type
        except:
            pass

    # 2. 透過 yfinance 抓取數據
    ticker = yf.Ticker(yf_code)
    
    try:
        info_yf = ticker.info
        if not is_taiwan or stock_name == raw_code:
            stock_name = info_yf.get('longName') or info_yf.get('shortName') or raw_code
        if industry == "未知產業":
            industry = info_yf.get('industry', 'N/A')
    except:
        info_yf = {}
    
    # B. 歷史股價
    df = ticker.history(start="2019-01-01")
    
    # C. 財報資料
    try:
        financials = ticker.financials.T
        balance_sheet = ticker.balance_sheet.T
    except:
        financials = pd.DataFrame()
        balance_sheet = pd.DataFrame()
        
    # D. 國際指數
    indices = {
        'S&P 500 (美)': '^GSPC',
        '費城半導體 (美)': '^SOX',
        '恒生指數 (港)': '^HSI',
        '上證指數 (中)': '000001.SS'
    }
    
    global_data = {}
    if not df.empty:
        try:
            start_date = df.index[-250].strftime('%Y-%m-%d')
            for name, idx_code in indices.items():
                try:
                    idx_df = yf.download(idx_code, start=start_date, progress=False)
                    if not idx_df.empty:
                        idx_series = idx_df['Close']
                        if isinstance(idx_series, pd.DataFrame): 
                            idx_series = idx_series.iloc[:, 0]
                        global_data[name] = idx_series
                except:
                    pass
        except:
            pass

    # E. 新聞
    try:
        news = ticker.news
    except:
        news = []

    if df.empty:
        return None, None, None, None, None, None, None, None, None

    # --- 資料計算 (技術指標) ---
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    df['VolMA20'] = df['Volume'].rolling(window=20).mean()
    
    # 乖離率
    df['Bias20'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # KD
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # --- F. AI 預測 ---
    prediction = df['Close'].iloc[-1]
    try:
        df_clean = df.dropna().copy()
        if len(df_clean) > 30:
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'K', 'D', 'OBV']
            X = df_clean[features]
            y = df_clean['Close'].shift(-1).dropna()
            
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
            model.fit(X[:-1], y)
            prediction = model.predict(X.tail(1))[0]
    except:
        pass

    # 時間格式 (最新資料時間)
    last_time = df.index[-1]
    if last_time.tzinfo is None:
        tz = pytz.timezone('Asia/Taipei')
        last_time = last_time.replace(tzinfo=pytz.utc).astimezone(tz)
    else:
        last_time = last_time.astimezone(pytz.timezone('Asia/Taipei'))
    data_time_str = last_time.strftime('%Y-%m-%d %H:%M')

    return df, stock_name, prediction, news, data_time_str, financials, balance_sheet, global_data, industry

# --- 5. 主程式執行 ---

with st.status(f"🚀 正在啟動 {stock_code} 深度分析引擎...", expanded=True) as status:
    # 傳入 is_taiwan 參數
    data = load_comprehensive_data(stock_code, full_code, is_tw_stock)
    
    if data[0] is None:
        status.update(label="❌ 查無資料", state="error")
        st.error(f"找不到代碼 {full_code}，請確認代碼與市場選擇是否正確。")
        st.stop()
        
    df, name, pred_price, news, data_time, fin_df, bal_df, glob_data, industry = data
    status.update(label=f"✅ {name} 分析報告生成完畢！", state="complete", expanded=False)

# --- 6. 數據與狀態計算 ---
last_row = df.iloc[-1]
prev_row = df.iloc[-2]

# 價格數據
curr_price = last_row['Close']
open_price = last_row['Open']
high_price = last_row['High']
low_price = last_row['Low']

# 漲跌計算
diff = curr_price - prev_row['Close']
pct = (diff / prev_row['Close']) * 100

# 配色邏輯
if diff > 0:
    main_color = "#e11d48" # 紅
    bg_color = "rgba(225, 29, 72, 0.1)"
    arrow = "▲"
elif diff < 0:
    main_color = "#10b981" # 綠
    bg_color = "rgba(16, 185, 129, 0.1)"
    arrow = "▼"
else:
    main_color = "#9ca3af" # 灰
    bg_color = "rgba(156, 163, 175, 0.1)"
    arrow = "-"

# 取得現在的台北時間
taipei_tz = pytz.timezone('Asia/Taipei')
now_taipei = datetime.now(taipei_tz)
current_time_str = now_taipei.strftime('%Y-%m-%d %H:%M:%S')

# 判斷市場狀態
is_open, status_text, status_color = check_market_status(market_type)
price_label = "⚡ 目前成交價" if is_open else "🔒 今日收盤股價"

# AI 與量能數據
vol = last_row['Volume']
vol_ma = last_row['VolMA20']
pred_diff = pred_price - curr_price
pred_pct = (pred_diff / curr_price) * 100

# --- 7. 🏆 置頂大看板 (修正縮排問題) ---
st.title(f"📊 {name} ({stock_code})")

# 這裡移除了 f-string 中的所有縮排，確保 HTML 能正確渲染
st.markdown(f"""
<div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; margin-bottom: 25px; border: 2px solid {main_color}; text-align: center; position: relative;">
<div style="position: absolute; top: 10px; right: 15px; text-align: right;">
<div style="font-size: 0.9rem; color: #6b7280; font-weight: bold;">🇹🇼 台北時間</div>
<div style="font-size: 1.1rem; color: #333; font-family: monospace;">{current_time_str}</div>
<div style="margin-top: 5px; background-color: {status_color}; color: white; padding: 2px 8px; border-radius: 5px; font-size: 0.8rem; display: inline-block;">
{status_text}
</div>
</div>
<span style="color: {main_color}; font-size: 1.2rem; font-weight: bold;">{price_label}</span>
<h1 style="color: {main_color}; margin: 5px 0; font-size: 4.5rem; font-weight: 800; line-height: 1;">{curr_price:.2f}</h1>
<h2 style="color: {main_color}; margin: 0; font-size: 2rem;">{arrow} {abs(diff):.2f} ({abs(pct):.2f}%)</h2>
<p style="color: #6b7280; font-size: 0.9rem; margin-top: 15px;">
📅 數據更新時間: {data_time} | 昨收: {prev_row['Close']:.2f}
</p>
</div>
""", unsafe_allow_html=True)

# --- 8. 詳細行情數據 ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("開盤價", f"{open_price:.2f}")
m2.metric("最高價", f"{high_price:.2f}")
m3.metric("最低價", f"{low_price:.2f}")
m4.metric("成交量", f"{int(vol/1000):,}K", f"{(vol-vol_ma)/1000:.1f}K", delta_color="inverse")

st.markdown("---")

# --- 9. AI 預測與關鍵指標 ---
st.subheader("🤖 AI 預測與關鍵指標")
c1, c2, c3, c4 = st.columns(4)
c1.metric("AI 預測明日", f"{pred_price:.2f}", f"{pred_diff:.2f} ({pred_pct:.2f}%)", delta_color="inverse")
c2.metric("乖離率 (月線)", f"{last_row['Bias20']:.2f}%")
c3.metric("RSI (14)", f"{last_row['RSI']:.1f}")
c4.metric("KD 指標", f"K:{last_row['K']:.0f} / D:{last_row['D']:.0f}")

# --- 10. 🕵️‍♂️ 深度分析報告區 ---
st.markdown("---")
st.subheader("🕵️‍♂️ 深度戰略分析報告")

ma20 = df['MA20'].iloc[-1]
ma60 = df['MA60'].iloc[-1]
bias20 = df['Bias20'].iloc[-1]
k_val = df['K'].iloc[-1]
d_val = df['D'].iloc[-1]
rsi_val = df['RSI'].iloc[-1]

trend_text = ""
if curr_price > ma20 and curr_price > ma60:
    trend_text = "✅ **多頭排列**：股價位於月線與季線之上，趨勢偏多。"
elif curr_price < ma20 and curr_price < ma60:
    trend_text = "❌ **空頭排列**：股價遭月線與季線反壓，趨勢偏空。"
elif curr_price > ma60 and curr_price < ma20:
    trend_text = "⚠️ **回檔整理**：跌破月線但守住季線，長多短空。"
else:
    trend_text = "⚠️ **反彈格局**：站上月線但受制於季線，短多長空。"

bias_text = ""
if bias20 > 10:
    bias_text = "🔥 **乖離過大**：股價離月線太遠，不宜追高。"
elif bias20 < -10:
    bias_text = "💎 **負乖離過大**：股價超跌，醞釀反彈。"
else:
    bias_text = "⚖️ **乖離正常**：股價沿著均線穩步運行。"

ai_text = ""
if pred_pct > 1.5:
    ai_text = f"🚀 **AI 強力看漲**：預測明日漲幅 > 1.5%。"
elif pred_pct < -1.5:
    ai_text = f"🩸 **AI 示警風險**：預測明日跌幅 > 1.5%。"
else:
    ai_text = "⚖️ **AI 預測盤整**：預期波動不大。"

with st.container():
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.info(f"**【趨勢結構】** {trend_text}\n\n**【乖離檢測】** {bias_text}")
    with col_a2:
        st.success(f"**【AI 觀點】** {ai_text}")
        kd_cross = "黃金交叉 (買進)" if (k_val > d_val and df['K'].iloc[-2] < df['D'].iloc[-2]) else "死亡交叉 (賣出)" if (k_val < d_val and df['K'].iloc[-2] > df['D'].iloc[-2]) else "無交叉"
        st.write(f"**【關鍵訊號】** KD目前呈現 **{kd_cross}**，RSI 為 **{rsi_val:.1f}**。")

# --- 11. 多分頁圖表區 ---
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 深度技術分析", 
    "💰 財報與基本面", 
    "🌏 國際連動", 
    "📰 相關新聞"
])

# === Tab 1: 技術分析 (現價標註) ===
with tab1:
    days_map = {"近 3 個月": 90, "近 6 個月": 180, "近 1 年": 365, "近 3 年": 1095, "全部": 9999}
    start_dt = datetime.now(pytz.timezone('Asia/Taipei')) - timedelta(days=days_map[date_option])
    if df.index.tzinfo is None: df.index = df.index.tz_localize("Asia/Taipei")
    df_view = df[df.index >= start_dt] if date_option != "全部" else df

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=("股價 K 線與均線趨勢", "市場成交量能", "KD 隨機指標", "OBV 籌碼能量潮")
    )
    
    # 1. 主圖
    fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MA20'], name="月線 (20MA)", line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MA60'], name="季線 (60MA)", line=dict(color='blue', width=1.5)), row=1, col=1)

    # --- 現價標註 ---
    last_idx = df_view.index[-1]
    last_val = df_view['Close'].iloc[-1]
    
    fig.add_shape(type="line", x0=df_view.index[0], x1=df_view.index[-1], y0=last_val, y1=last_val, line=dict(color="red", width=1, dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[last_idx], y=[last_val], mode="markers+text", marker=dict(color="red", size=8), text=[f"現價 {last_val:.2f}"], textposition="top center", name="目前股價", showlegend=False), row=1, col=1)
    
    # 2. 成交量
    colors = ['red' if r['Open'] - r['Close'] >= 0 else 'green' for i, r in df_view.iterrows()]
    fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
    
    # 3. KD
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['K'], name="K值 (快線)", line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['D'], name="D值 (慢線)", line=dict(color='blue')), row=3, col=1)
    fig.add_shape(type="line", x0=df_view.index[0], x1=df_view.index[-1], y0=80, y1=80, line=dict(color="red", dash="dot"), row=3, col=1)
    fig.add_shape(type="line", x0=df_view.index[0], x1=df_view.index[-1], y0=20, y1=20, line=dict(color="green", dash="dot"), row=3, col=1)
    
    # 4. OBV
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['OBV'], name="OBV 累積能量", line=dict(color='purple', width=2)), row=4, col=1)
    
    fig.update_layout(height=1100, xaxis_rangeslider_visible=False, title_text=f"<b>{name} ({stock_code}) 綜合技術分析圖表</b>")
    st.plotly_chart(fig, use_container_width=True)

# === Tab 2: 基本面 ===
with tab2:
    st.subheader("📊 財務體質分析")
    if not fin_df.empty:
        rev_col = [c for c in fin_df.columns if 'Total Revenue' in c or 'Revenue' in c]
        inc_col = [c for c in fin_df.columns if 'Net Income' in c]
        if rev_col and inc_col:
            fin_plot = fin_df.iloc[:4]
            fig_fin = go.Figure()
            fig_fin.add_trace(go.Bar(x=fin_plot.index.astype(str), y=fin_plot[rev_col[0]], name="總營收"))
            fig_fin.add_trace(go.Scatter(x=fin_plot.index.astype(str), y=fin_plot[inc_col[0]], name="稅後淨利", yaxis='y2', line=dict(color='red', width=3)))
            fig_fin.update_layout(yaxis=dict(title="營收"), yaxis2=dict(title="淨利", overlaying='y', side='right'), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_fin, use_container_width=True)
        else:
            st.warning("無法解析財報欄位")
    else:
        st.info("暫無詳細財報資料")

# === Tab 3: 國際連動 ===
with tab3:
    st.subheader("🌏 國際股市連動性")
    if glob_data:
        stock_close = df['Close'].tz_localize(None)
        target_len = min(len(stock_close), 250)
        base_series = stock_close.iloc[-target_len:]
        corrs = {}
        for name, series in glob_data.items():
            aligned = series.reindex(base_series.index, method='ffill')
            corrs[name] = base_series.corr(aligned)
        cols = st.columns(len(corrs))
        for i, (name, val) in enumerate(corrs.items()):
            cols[i].metric(name, f"{val:.2f}", delta="高度正相關" if val > 0.7 else "負相關" if val < -0.3 else None)
        fig_glob = go.Figure()
        norm_base = (base_series / base_series.iloc[0]) * 100
        fig_glob.add_trace(go.Scatter(x=base_series.index, y=norm_base, name=f"{stock_code}", line=dict(color='red', width=3)))
        for name, series in glob_data.items():
            aligned = series.reindex(base_series.index, method='ffill')
            norm = (aligned / aligned.iloc[0]) * 100
            fig_glob.add_trace(go.Scatter(x=base_series.index, y=norm, name=name, line=dict(dash='dot')))
        st.plotly_chart(fig_glob, use_container_width=True)
    else:
        st.warning("暫無國際指數資料")

# === Tab 4: 新聞 ===
with tab4:
    st.subheader(f"📰 {name} 最新動態")
    if news:
        for n in news[:8]:
            try:
                raw_time = n.get('providerPublishTime')
                pub_time = pd.to_datetime(raw_time, unit='s').strftime('%Y-%m-%d %H:%M') if raw_time else "未知時間"
            except: pub_time = "未知時間"
            st.markdown(f"➤ **[{n.get('title', '無標題')}]({n.get('link', '#')})**")
            st.caption(f"來源：{n.get('publisher', '未知')} | 時間：{pub_time}")
            st.markdown("---")
    else:
        st.write("暫無相關新聞")
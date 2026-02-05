import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import pytz
import twstock # 引入台灣股市套件 (專門解決名稱問題)

# --- 1. 頁面設定 ---
st.set_page_config(page_title="台股 AI 旗艦分析系統 (中文增強版)", layout="wide")

# --- 2. 左側邊欄：設定 ---
st.sidebar.title("🔍 戰情控制室")
stock_code = st.sidebar.text_input("輸入台股代碼", "2603") 
full_code = f"{stock_code}.TW"

# 日期區間篩選
st.sidebar.subheader("📅 趨勢圖區間")
date_option = st.sidebar.selectbox(
    "選擇顯示範圍", 
    ["近 3 個月", "近 6 個月", "近 1 年", "近 3 年", "全部"], 
    index=1
)

if st.sidebar.button("🔄 立即全盤掃描"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 **教學模式啟動中**：\n點擊各個圖表下方的 **「🎓 小學堂」** 展開按鈕，即可查看該指標的計算公式與判讀祕訣！")

# --- 3. 核心函數：全方位資料抓取 ---
@st.cache_data
def load_comprehensive_data(raw_code, yf_code):
    # 1. 優先解決名稱問題 (使用 twstock 套件)
    stock_name = raw_code # 預設為代碼
    industry = "未知產業"
    sector = "未知板塊"
    
    # 嘗試從 twstock 抓取精準中文名稱
    try:
        if raw_code in twstock.codes:
            info_tw = twstock.codes[raw_code]
            stock_name = info_tw.name # 抓到中文名 (如：長榮)
            industry = info_tw.type   # 抓到產業 (如：航運業)
    except:
        pass

    # 2. 透過 yfinance 抓取數據
    ticker = yf.Ticker(yf_code)
    
    # 補充 yfinance 的資訊 (若 twstock 沒抓到，嘗試用 yfinance 補救)
    try:
        info_yf = ticker.info
        if stock_name == raw_code: # 如果上面沒抓到中文名
            stock_name = info_yf.get('longName') or info_yf.get('shortName') or raw_code
        if industry == "未知產業":
            industry = info_yf.get('industry', 'N/A')
            sector = info_yf.get('sector', 'N/A')
    except:
        info_yf = {}
    
    # B. 歷史股價 (技術面)
    df = ticker.history(start="2019-01-01")
    
    # C. 財報資料 (基本面)
    try:
        financials = ticker.financials.T # 損益表
        balance_sheet = ticker.balance_sheet.T # 資產負債表
    except:
        financials = pd.DataFrame()
        balance_sheet = pd.DataFrame()
        
    # D. 國際指數 (市場面)
    indices = {
        'S&P 500 (美)': '^GSPC',
        'Nasdaq (美)': '^IXIC',
        '日經 225 (日)': '^N225',
        'KOSPI (韓)': '^KS11',
        '恆生指數 (港)': '^HSI'
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

    # 防呆機制：若無數據，回傳 None
    if df.empty:
        return None, None, None, None, None, None, None, None, None

    # --- 資料計算 (技術指標) ---
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['VolMA20'] = df['Volume'].rolling(window=20).mean()
    
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

    # 時間格式
    last_time = df.index[-1]
    if last_time.tzinfo is None:
        tz = pytz.timezone('Asia/Taipei')
        last_time = last_time.replace(tzinfo=pytz.utc).astimezone(tz)
    else:
        last_time = last_time.astimezone(pytz.timezone('Asia/Taipei'))
    update_time = last_time.strftime('%Y-%m-%d %H:%M')

    return df, stock_name, prediction, news, update_time, financials, balance_sheet, global_data, industry

# --- 4. 主程式執行 ---

with st.status(f"🚀 正在啟動 {stock_code} 深度分析引擎...", expanded=True) as status:
    # 傳入 原始代碼(給twstock用) 和 完整代碼(給yfinance用)
    data = load_comprehensive_data(stock_code, full_code)
    
    if data[0] is None:
        status.update(label="❌ 查無資料", state="error")
        st.error(f"找不到代碼 {stock_code}，請確認是否為上市櫃股票。")
        st.stop()
        
    df, name, pred_price, news, up_time, fin_df, bal_df, glob_data, industry = data
    status.update(label=f"✅ {name} 分析報告生成完畢！", state="complete", expanded=False)

# --- 5. 儀表板頭部 ---
st.title(f"📊 {name} ({stock_code}) 投資戰情室")
st.caption(f"🕒 最後更新：{up_time} | 🏢 產業：{industry}")

# 最新數據
curr = df['Close'].iloc[-1]
diff = curr - df['Close'].iloc[-2]
pct = (diff / df['Close'].iloc[-2]) * 100
vol = df['Volume'].iloc[-1]
vol_ma = df['VolMA20'].iloc[-1]

pred_diff = pred_price - curr
pred_pct = (pred_diff / curr) * 100

# 頂部指標
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("目前股價", f"{curr:.2f}", f"{diff:.2f} ({pct:.2f}%)")
c2.metric("AI 預測明日", f"{pred_price:.2f}", f"{pred_diff:.2f} ({pred_pct:.2f}%)")
c3.metric("成交量", f"{int(vol/1000):,}K", f"{(vol-vol_ma)/1000:.1f}K")
c4.metric("RSI 強弱", f"{df['RSI'].iloc[-1]:.1f}")
c5.metric("KD 指標", f"K: {df['K'].iloc[-1]:.0f}")

# --- 6. 多分頁分析區 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 技術與 AI", 
    "💰 財報基本面", 
    "🌏 國際連動分析", 
    "🏢 公司與產業", 
    "📰 新聞快訊"
])

# === Tab 1: 技術分析 ===
with tab1:
    days_map = {"近 3 個月": 90, "近 6 個月": 180, "近 1 年": 365, "近 3 年": 1095, "全部": 9999}
    start_dt = datetime.now(pytz.timezone('Asia/Taipei')) - timedelta(days=days_map[date_option])
    if df.index.tzinfo is None: df.index = df.index.tz_localize("Asia/Taipei")
    df_view = df[df.index >= start_dt] if date_option != "全部" else df

    st.subheader("🤖 AI 趨勢解讀")
    col_ai1, col_ai2 = st.columns([2, 1])
    with col_ai1:
        if pred_pct > 1: st.success(f"🚀 **強勢看漲**：AI 預測明日漲幅達 {pred_pct:.2f}%，動能強勁。")
        elif pred_pct < -1: st.error(f"🩸 **修正風險**：AI 預測明日跌幅達 {abs(pred_pct):.2f}%，建議避險。")
        else: st.info(f"⚖️ **盤整格局**：預測漲跌幅在 {pred_pct:.2f}% 之間，區間震盪。")
    
    with col_ai2:
        k = df['K'].iloc[-1]
        if k < 20: st.write("💎 KD 指標：**低檔超賣**")
        elif k > 80: st.write("🔥 KD 指標：**高檔過熱**")
        else: st.write("⚖️ KD 指標：**中性整理**")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2])
    fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MA20'], name="月線 (20MA)", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MA60'], name="季線 (60MA)", line=dict(color='blue')), row=1, col=1)
    
    colors = ['red' if r['Open'] - r['Close'] >= 0 else 'green' for i, r in df_view.iterrows()]
    fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['K'], name="K值", line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['D'], name="D值", line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['OBV'], name="OBV 能量", line=dict(color='purple')), row=4, col=1)
    fig.update_layout(height=1000, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🎓 小學堂：圖表看不懂？點這裡學看盤！", expanded=False):
        st.markdown("""
        ### 1. 移動平均線 (MA)
        * **月線 (橘線)**：短期生命線，股價在上面代表短期強勢。
        * **季線 (藍線)**：中期趨勢線，跌破季線通常代表趨勢轉空。
        ### 2. KD 指標
        * **黃金交叉**：K值由下往上穿過D值 ➔ 買進訊號 🟢。
        * **死亡交叉**：K值由上往下穿過D值 ➔ 賣出訊號 🔴。
        ### 3. OBV 能量潮
        * **量價背離**：股價沒漲，但紫色 OBV 線一直在漲 ➔ 主力偷吃貨，後市看漲！
        """)

# === Tab 2: 基本面財報 ===
with tab2:
    st.subheader("📊 財務健康度")
    if not fin_df.empty:
        rev_col = [c for c in fin_df.columns if 'Total Revenue' in c or 'Revenue' in c]
        inc_col = [c for c in fin_df.columns if 'Net Income' in c]
        
        if rev_col and inc_col:
            fin_plot = fin_df.iloc[:4]
            fig_fin = go.Figure()
            fig_fin.add_trace(go.Bar(x=fin_plot.index.astype(str), y=fin_plot[rev_col[0]], name="總營收"))
            fig_fin.add_trace(go.Scatter(x=fin_plot.index.astype(str), y=fin_plot[inc_col[0]], name="淨利", yaxis='y2', line=dict(color='red', width=3)))
            fig_fin.update_layout(yaxis=dict(title="營收"), yaxis2=dict(title="淨利", overlaying='y', side='right'), title="近年營收與獲利")
            st.plotly_chart(fig_fin, use_container_width=True)
            
            with st.expander("🎓 小學堂：財報名詞解釋", expanded=False):
                st.markdown("""
                * **總營收**：做生意的總收入。
                * **淨利**：扣掉成本後真正賺的錢。
                """)
        else:
            st.warning("無法解析詳細財報欄位。")
    else:
        st.info("暫無詳細財報資料。")

# === Tab 3: 國際連動 ===
with tab3:
    st.subheader("🌏 全球股市連動性")
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
            cols[i].metric(name, f"{val:.2f}", delta="高度連動" if val > 0.7 else None)

        fig_glob = go.Figure()
        norm_base = (base_series / base_series.iloc[0]) * 100
        fig_glob.add_trace(go.Scatter(x=base_series.index, y=norm_base, name=f"{stock_code}", line=dict(color='red', width=3)))
        for name, series in glob_data.items():
            aligned = series.reindex(base_series.index, method='ffill')
            norm = (aligned / aligned.iloc[0]) * 100
            fig_glob.add_trace(go.Scatter(x=base_series.index, y=norm, name=name, line=dict(dash='dot')))
        st.plotly_chart(fig_glob, use_container_width=True)
        
        with st.expander("🎓 小學堂：連動係數是什麼？", expanded=False):
            st.markdown("""
            * **接近 1.0**：同進退（如：美股漲，這支也漲）。
            * **接近 0**：沒關係。
            * **接近 -1.0**：唱反調（通常是避險股）。
            """)

# === Tab 4 & 5 ===
with tab4:
    st.subheader("🏢 公司檔案")
    st.info(f"產業：{industry}")
    
with tab5:
    st.subheader(f"📰 最新消息")
    if news:
        for n in news[:8]:
            st.markdown(f"[{n.get('title')}]({n.get('link')})")
            st.markdown("---")
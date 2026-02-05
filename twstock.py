import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import pytz
import twstock

# --- 1. 頁面設定 ---
st.set_page_config(page_title="台股 AI 旗艦分析系統 (深度版)", layout="wide")

# --- 2. 左側邊欄：設定 ---
st.sidebar.title("🔍 戰情控制室")

# 輸入框 (已有 Key 防止 ID 衝突)
stock_code = st.sidebar.text_input("輸入台股代碼", "2603", key="sidebar_stock_code") 
full_code = f"{stock_code}.TW"

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
st.sidebar.info("💡 **系統提示**：\n已在 K 線圖加入「目前股價」紅色虛線與標記，方便即時判讀位階。")

# --- 3. 核心函數：全方位資料抓取 ---
@st.cache_data
def load_comprehensive_data(raw_code, yf_code):
    # 1. 優先解決名稱問題 (使用 twstock)
    stock_name = raw_code
    industry = "未知產業"
    
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
        if stock_name == raw_code:
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
        '日經 225 (日)': '^N225',
        'KOSPI (韓)': '^KS11'
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
    df['Bias60'] = ((df['Close'] - df['MA60']) / df['MA60']) * 100

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
    data = load_comprehensive_data(stock_code, full_code)
    
    if data[0] is None:
        status.update(label="❌ 查無資料", state="error")
        st.error(f"找不到代碼 {stock_code}，請確認是否為上市櫃股票。")
        st.stop()
        
    df, name, pred_price, news, up_time, fin_df, bal_df, glob_data, industry = data
    status.update(label=f"✅ {name} 分析報告生成完畢！", state="complete", expanded=False)

# --- 5. 儀表板頭部 ---
st.title(f"📊 {name} ({stock_code}) 投資戰情室")
st.caption(f"🕒 資料最後更新：{up_time} | 🏢 產業類別：{industry}")

# 最新數據
curr = df['Close'].iloc[-1]
diff = curr - df['Close'].iloc[-2]
pct = (diff / df['Close'].iloc[-2]) * 100
vol = df['Volume'].iloc[-1]
vol_ma = df['VolMA20'].iloc[-1]
pred_diff = pred_price - curr
pred_pct = (pred_diff / curr) * 100

# 頂部指標卡
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("目前股價", f"{curr:.2f}", f"{diff:.2f} ({pct:.2f}%)")
c2.metric("AI 預測明日", f"{pred_price:.2f}", f"{pred_diff:.2f} ({pred_pct:.2f}%)")
c3.metric("成交量", f"{int(vol/1000):,}K", f"{(vol-vol_ma)/1000:.1f}K")
c4.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
c5.metric("KD 指標", f"K:{df['K'].iloc[-1]:.0f} / D:{df['D'].iloc[-1]:.0f}")

# --- 6. 🕵️‍♂️ 深度分析報告區 ---
st.markdown("---")
st.subheader("🕵️‍♂️ 深度戰略分析報告")

ma20 = df['MA20'].iloc[-1]
ma60 = df['MA60'].iloc[-1]
bias20 = df['Bias20'].iloc[-1]
k_val = df['K'].iloc[-1]
d_val = df['D'].iloc[-1]
rsi_val = df['RSI'].iloc[-1]

trend_text = ""
if curr > ma20 and curr > ma60:
    trend_text = "✅ **多頭排列**：股價位於月線與季線之上，中長期趨勢看漲，主力控盤穩健。"
elif curr < ma20 and curr < ma60:
    trend_text = "❌ **空頭排列**：股價遭月線與季線反壓，趨勢偏弱，建議保守看待。"
elif curr > ma60 and curr < ma20:
    trend_text = "⚠️ **回檔整理**：股價跌破月線但守住季線，屬於漲多回檔，觀察季線支撐。"
else:
    trend_text = "⚠️ **反彈格局**：股價站上月線但仍受制於季線，尚未完全翻多。"

bias_text = ""
if bias20 > 10:
    bias_text = "🔥 **乖離過大**：股價離月線太遠（乖離率 > 10%），短線容易拉回修正，不宜追高。"
elif bias20 < -10:
    bias_text = "💎 **負乖離過大**：股價超跌（乖離率 < -10%），隨時有機會出現技術性反彈。"
else:
    bias_text = "⚖️ **乖離正常**：股價沿著均線穩步運行，無過熱或超跌跡象。"

ai_text = ""
if pred_pct > 1.5:
    ai_text = f"🚀 **AI 強力看漲**：模型預測明日有 {pred_pct:.2f}% 的潛在漲幅，動能強勁。"
elif pred_pct < -1.5:
    ai_text = f"🩸 **AI 示警風險**：模型預測明日可能修正 {abs(pred_pct):.2f}%，留意賣壓。"
else:
    ai_text = "⚖️ **AI 預測盤整**：預期波動不大，建議區間操作。"

with st.container():
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.info(f"**【趨勢結構】**\n\n{trend_text}\n\n**【乖離檢測】**\n\n{bias_text}")
    with col_a2:
        st.success(f"**【AI 觀點】**\n\n{ai_text}")
        kd_cross = "黃金交叉 (買進訊號)" if (k_val > d_val and df['K'].iloc[-2] < df['D'].iloc[-2]) else "死亡交叉 (賣出訊號)" if (k_val < d_val and df['K'].iloc[-2] > df['D'].iloc[-2]) else "無交叉"
        st.write(f"**【關鍵訊號】** KD目前呈現 **{kd_cross}**，RSI 數值為 **{rsi_val:.1f}**。")

# --- 7. 多分頁圖表區 ---
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 深度技術分析", 
    "💰 財報與基本面", 
    "🌏 國際連動", 
    "📰 相關新聞"
])

# === Tab 1: 技術分析 (新增現價標註) ===
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

    # --- 新增：現價標註 (水平線 + 氣泡) ---
    last_idx = df_view.index[-1]
    last_val = df_view['Close'].iloc[-1]
    
    # 水平虛線
    fig.add_shape(
        type="line", 
        x0=df_view.index[0], x1=df_view.index[-1], 
        y0=last_val, y1=last_val,
        line=dict(color="red", width=1, dash="dash"),
        row=1, col=1
    )
    # 標記點與文字
    fig.add_trace(go.Scatter(
        x=[last_idx], y=[last_val],
        mode="markers+text",
        marker=dict(color="red", size=8),
        text=[f"現價 {last_val:.2f}"],
        textposition="top center",
        name="目前股價",
        showlegend=False
    ), row=1, col=1)
    # ------------------------------------
    
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
    
    with st.expander("🎓 圖表教學：如何看懂這些線？", expanded=False):
        st.markdown("""
        * **現價紅線**：畫面上的紅色虛線與標記點，代表這檔股票現在的價格位置。
        * **K線與均線**：K 線代表股價，月線(橘)代表短期成本，季線(藍)代表長期成本。站上季線通常代表多頭。
        * **成交量**：紅柱代表跌、綠柱代表漲（台股慣例紅色為漲，若設定不同請見諒）。有量才有價。
        * **KD 指標**：80以上過熱（可能跌），20以下超賣（可能漲）。黃金交叉（橘穿藍往上）為買點。
        * **OBV**：如果股價盤整但 OBV 往上衝，代表有人在偷偷吃貨。
        """)

# === Tab 2: 基本面 ===
with tab2:
    st.subheader("📊 財務體質分析")
    if not fin_df.empty:
        rev_col = [c for c in fin_df.columns if 'Total Revenue' in c or 'Revenue' in c]
        inc_col = [c for c in fin_df.columns if 'Net Income' in c]
        
        if rev_col and inc_col:
            fin_plot = fin_df.iloc[:4]
            fig_fin = go.Figure()
            fig_fin.add_trace(go.Bar(x=fin_plot.index.astype(str), y=fin_plot[rev_col[0]], name="總營收 (Revenue)"))
            fig_fin.add_trace(go.Scatter(x=fin_plot.index.astype(str), y=fin_plot[inc_col[0]], name="稅後淨利 (Net Income)", yaxis='y2', line=dict(color='red', width=3)))
            
            fig_fin.update_layout(
                title_text="<b>近年營收與獲利趨勢圖</b>",
                yaxis=dict(title="營收金額"), 
                yaxis2=dict(title="淨利金額", overlaying='y', side='right'),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_fin, use_container_width=True)
            st.caption("註：柱狀圖為營收，紅線為公司真正賺進口袋的淨利。")
        else:
            st.warning("無法解析財報欄位")
    else:
        st.info("暫無詳細財報資料")

# === Tab 3: 國際連動 ===
with tab3:
    st.subheader("🌏 國際股市連動性矩陣")
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
        fig_glob.add_trace(go.Scatter(x=base_series.index, y=norm_base, name=f"{stock_code} (本股)", line=dict(color='red', width=3)))
        for name, series in glob_data.items():
            aligned = series.reindex(base_series.index, method='ffill')
            norm = (aligned / aligned.iloc[0]) * 100
            fig_glob.add_trace(go.Scatter(x=base_series.index, y=norm, name=name, line=dict(dash='dot')))
            
        fig_glob.update_layout(title_text="<b>近一年走勢疊加比較圖 (基期=100)</b>")
        st.plotly_chart(fig_glob, use_container_width=True)
    else:
        st.warning("暫無國際指數資料")

# === Tab 4: 新聞 (防呆機制) ===
with tab4:
    st.subheader(f"📰 {name} 最新動態")
    if news:
        for n in news[:8]:
            try:
                raw_time = n.get('providerPublishTime')
                if raw_time:
                    pub_time = pd.to_datetime(raw_time, unit='s').strftime('%Y-%m-%d %H:%M')
                else:
                    pub_time = "未知時間"
            except:
                pub_time = "未知時間"
            
            title = n.get('title', '無標題')
            link = n.get('link', '#')
            publisher = n.get('publisher', '未知來源')
            
            st.markdown(f"➤ **[{title}]({link})**")
            st.caption(f"來源：{publisher} | 時間：{pub_time}")
            st.markdown("---")
    else:
        st.write("暫無相關新聞")
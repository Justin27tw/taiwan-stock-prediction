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
from streamlit_autorefresh import st_autorefresh
from deep_translator import GoogleTranslator # 新增：翻譯套件
import feedparser # 新增：RSS 解析套件
import urllib.parse

# --- 1. 頁面設定與 CSS 美化 ---
st.set_page_config(page_title="全球股市 AI 戰情室", layout="wide", page_icon="📈")

def local_css():
    st.markdown("""
    <style>
        /* 引入現代字體 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans TC', sans-serif;
            background-color: #0e1117;
        }

        /* 隱藏預設元件 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 數據卡片樣式 (Glassmorphism) */
        .metric-card {
            background-color: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #3b82f6;
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
        }
        
        .card-title {
            color: #94a3b8; 
            font-size: 0.9rem; 
            margin-bottom: 8px;
            font-weight: 500;
        }
        .card-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .card-delta {
            font-size: 0.9rem;
            margin-top: 5px;
            font-weight: 600;
        }

        /* 置頂看板樣式 */
        .hero-container {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
            text-align: center;
            position: relative;
        }

        /* Tab 優化 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: #1e293b;
            border-radius: 8px;
            color: #cbd5e1;
            border: 1px solid rgba(255,255,255,0.05);
            padding: 0 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563eb !important;
            color: white !important;
            border-color: #2563eb !important;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 2. 輔助函數 ---
def check_market_status(market):
    utc_now = datetime.now(pytz.utc)
    tz_map = {
        "台股": 'Asia/Taipei',
        "港股": 'Asia/Hong_Kong',
        "美股": 'America/New_York'
    }
    
    target_tz = pytz.timezone(next((v for k, v in tz_map.items() if k in market), 'Asia/Taipei'))
    local_now = utc_now.astimezone(target_tz)
    
    # 簡易判斷 (週一至週五)
    if 0 <= local_now.weekday() <= 4:
        current_time = local_now.time()
        # 簡單定義開盤區間 (可再細修)
        start = time(9, 0) if "美股" not in market else time(9, 30)
        end = time(13, 30) if "台股" in market else (time(16, 0) if "美股" in market else time(16, 0))
        
        if start <= current_time <= end:
            return True, "🟢 交易進行中", "#22c55e"
            
    return False, "🔴 已收盤", "#ef4444"

# --- 新增：中文翻譯與新聞抓取函數 ---
def get_chinese_name_and_news(raw_name, raw_code):
    # 1. 嘗試翻譯名稱
    zh_name = raw_name
    try:
        # 如果本身不含中文，才進行翻譯
        if not any("\u4e00" <= char <= "\u9fff" for char in raw_name):
            zh_name = GoogleTranslator(source='auto', target='zh-TW').translate(raw_name)
    except:
        pass

    # 2. 抓取 Google News (繁體中文)
    news_list = []
    try:
        # 搜尋關鍵字：中文名稱 + 股票代碼 (增加精準度)
        query = f"{zh_name} {raw_code}"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:6]: # 取前 6 則
            pub_date = entry.published_parsed
            if pub_date:
                dt = datetime(*pub_date[:6])
                fmt_date = dt.strftime('%Y-%m-%d %H:%M')
            else:
                fmt_date = ""
            
            news_list.append({
                'title': entry.title,
                'link': entry.link,
                'publisher': entry.source.title if hasattr(entry, 'source') else 'Google News',
                'time': fmt_date
            })
    except Exception as e:
        print(f"News Error: {e}")
        
    return zh_name, news_list

# --- 3. 核心資料載入 (新增基本面與翻譯) ---
@st.cache_data(ttl=60)
def load_data(stock_code, market_type, is_tw):
    # 處理代碼
    yf_code = stock_code
    if is_tw: yf_code = f"{stock_code}.TW"
    elif "港股" in market_type: 
        yf_code = f"{stock_code.zfill(4)}.HK" if len(stock_code) < 5 else f"{stock_code}.HK"
    
    # 1. 抓取基礎資料
    ticker = yf.Ticker(yf_code)
    try:
        history = ticker.history(period="2y") # 抓 2 年以利計算長期均線
        info = ticker.info
    except:
        return None

    if history.empty:
        return None

    # 2. 名稱處理
    stock_name = yf_code
    industry = "未知產業"
    
    # 優先使用 yfinance 的資訊
    long_name = info.get('longName', info.get('shortName', yf_code))
    industry = info.get('industry', info.get('sector', 'N/A'))
    
    # 台股特別處理：嘗試用 twstock 修正名稱 (有時候 yf 的中文名怪怪的)
    if is_tw and stock_code in twstock.codes:
        stock_name = twstock.codes[stock_code].name
        industry = twstock.codes[stock_code].type
    else:
        stock_name = long_name

    # 3. 執行翻譯與抓新聞 (新增功能)
    zh_name, news_data = get_chinese_name_and_news(stock_name, stock_code)

    # 4. 整理基本面數據 (新增需求)
    fundamentals = {
        'PE': info.get('trailingPE', 'N/A'),
        'ForwardPE': info.get('forwardPE', 'N/A'),
        'PB': info.get('priceToBook', 'N/A'),
        'Yield': info.get('dividendYield', 0), # 通常是 0.05 代表 5%
        'MarketCap': info.get('marketCap', 'N/A'),
        'ROE': info.get('returnOnEquity', 'N/A'),
        'TargetPrice': info.get('targetMeanPrice', 'N/A')
    }

    # 5. 技術指標計算
    df = history.copy()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # KD
    rsv = (df['Close'] - df['Low'].rolling(9).min()) / (df['High'].rolling(9).max() - df['Low'].rolling(9).min()) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # OBV (量能)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # 6. AI 預測 (XGBoost)
    pred_price = 0
    try:
        if len(df) > 60:
            df_ml = df.dropna()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'K', 'D']
            X = df_ml[features]
            y = df_ml['Close'].shift(-1).dropna()
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
            model.fit(X[:-1], y)
            pred_price = model.predict(X.tail(1))[0]
    except:
        pred_price = df['Close'].iloc[-1]

    # 時間格式
    last_time = df.index[-1]
    if last_time.tzinfo is None:
        last_time = pytz.utc.localize(last_time).astimezone(pytz.timezone('Asia/Taipei'))
    else:
        last_time = last_time.astimezone(pytz.timezone('Asia/Taipei'))
        
    return {
        'df': df,
        'info': info,
        'name_zh': zh_name,
        'name_en': stock_name,
        'industry': industry,
        'news': news_data,
        'fund': fundamentals,
        'pred': pred_price,
        'time': last_time.strftime('%Y-%m-%d %H:%M'),
        'yf_code': yf_code
    }

# --- 4. 側邊欄 ---
st.sidebar.title("🎛️ 戰情控制中心")
market_type = st.sidebar.selectbox("選擇市場", ["🇹🇼 台股", "🇺🇸 美股", "🇭🇰 港股"])

default_code = "2330"
if "美股" in market_type: default_code = "NVDA"
elif "港股" in market_type: default_code = "9988"

stock_input = st.sidebar.text_input("輸入代碼", default_code)
is_tw = "台股" in market_type

# 自動刷新
is_open, msg, color_status = check_market_status(market_type)
if is_open:
    st_autorefresh(interval=60000, key="auto_refresh")
    st.sidebar.success(f"⚡ 市場開啟中 | {msg}")
else:
    st.sidebar.warning(f"💤 市場已收盤 | {msg}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **新增功能**：\n1. 自動翻譯英文股名\n2. 抓取 Google 中文新聞\n3. 基本面數據 (本益比/殖利率)")

# --- 5. 主程式 ---
if stock_input:
    data = load_data(stock_input, market_type, is_tw)
    
    if not data:
        st.error(f"❌ 找不到代碼 {stock_input}，請檢查輸入是否正確。")
        st.stop()
        
    df = data['df']
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    
    # 顏色邏輯
    color = "#ef4444" if change > 0 else "#22c55e" if change < 0 else "#94a3b8"
    arrow = "▲" if change > 0 else "▼" if change < 0 else "-"
    
    # --- UI: 置頂英雄區 (Hero Section) ---
    st.markdown(f"""
    <div class="hero-container" style="border-top: 5px solid {color};">
        <div style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 5px;">{market_type} | {data['industry']}</div>
        <h1 style="font-size: 3.5rem; margin: 0; font-weight: 800; color: #f8fafc;">
            {data['name_zh']} <span style="font-size: 1.5rem; color: #64748b;">({stock_input})</span>
        </h1>
        <div style="display: flex; justify-content: center; align-items: baseline; gap: 20px; margin-top: 15px;">
            <span style="font-size: 4rem; font-weight: bold; color: {color};">{last['Close']:.2f}</span>
            <span style="font-size: 2rem; font-weight: 600; color: {color};">
                {arrow} {abs(change):.2f} ({abs(pct):.2f}%)
            </span>
        </div>
        <div style="margin-top: 15px; color: #64748b;">
            🕒 更新時間: {data['time']} | 昨收: {prev['Close']:.2f} | 成交量: {int(last['Volume']/1000):,} K
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- UI: 數據卡片區 (Metric Cards) ---
    c1, c2, c3, c4 = st.columns(4)
    
    # 自訂卡片顯示函數
    def card(col, title, value, delta=None, prefix="", suffix=""):
        delta_html = ""
        if delta:
            d_color = "#ef4444" if "▲" in delta else "#22c55e" if "▼" in delta else "#94a3b8"
            delta_html = f'<div class="card-delta" style="color: {d_color};">{delta}</div>'
            
        col.markdown(f"""
        <div class="metric-card">
            <div class="card-title">{title}</div>
            <div class="card-value">{prefix}{value}{suffix}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

    # 計算 AI 漲跌
    pred_diff = data['pred'] - last['Close']
    pred_pct = (pred_diff / last['Close']) * 100
    ai_arrow = "▲" if pred_diff > 0 else "▼"
    
    card(c1, "AI 預測明日價格", f"{data['pred']:.2f}", f"{ai_arrow} {abs(pred_pct):.2f}%")
    
    # 基本面/籌碼面數據
    pe_val = data['fund']['PE']
    pe_str = f"{pe_val:.1f}" if isinstance(pe_val, (int, float)) else "N/A"
    
    yield_val = data['fund']['Yield']
    yield_str = f"{yield_val*100:.2f}%" if isinstance(yield_val, (int, float)) else "N/A"
    
    card(c2, "本益比 (P/E)", pe_str)
    card(c3, "殖利率 (Yield)", yield_str)
    
    kd_status = "黃金交叉" if last['K'] > last['D'] else "死亡交叉"
    kd_color = "▲" if last['K'] > last['D'] else "▼"
    card(c4, "技術指標 (KD)", f"K{last['K']:.0f}", f"{kd_color} {kd_status}")

    st.markdown("---")

    # --- UI: 分頁內容 ---
    tab1, tab2, tab3 = st.tabs(["📊 深度技術分析", "📰 智能新聞解析", "💰 籌碼與基本面"])

    with tab1:
        # 繪圖
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # K線
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="股價"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1.5), name="月線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1.5), name="季線"), row=1, col=1)
        
        # 成交量
        colors = ['red' if r['Open'] < r['Close'] else 'green' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
        
        fig.update_layout(
            height=600, 
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"📰 {data['name_zh']} 最新相關新聞 (AI 聚合)")
        if data['news']:
            for n in data['news']:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #3b82f6;">
                    <a href="{n['link']}" target="_blank" style="text-decoration: none; color: #f8fafc; font-size: 1.1rem; font-weight: 600;">{n['title']}</a>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">
                        📅 {n['time']} | 📢 {n['publisher']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暫無抓取到近期相關中文新聞")

    with tab3:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.subheader("📋 關鍵財務數據")
            fund = data['fund']
            
            # 使用 DataFrame 展示表格
            f_data = {
                "指標": ["本益比 (P/E)", "預估本益比 (Fwd P/E)", "股價淨值比 (P/B)", "股東權益報酬率 (ROE)", "分析師目標價"],
                "數值": [
                    fund['PE'], 
                    fund['ForwardPE'], 
                    fund['PB'], 
                    f"{fund['ROE']*100:.2f}%" if isinstance(fund['ROE'], float) else 'N/A',
                    fund['TargetPrice']
                ]
            }
            st.dataframe(pd.DataFrame(f_data), hide_index=True, use_container_width=True)

        with col_f2:
            st.subheader("🐳 籌碼面/法人邏輯 (Lite)")
            st.markdown("""
            * **外資持股與動向**：對於大型股 (如台積電) 影響極大，建議觀察 `yfinance` 內的 Institutional Holders (多為美股數據) 或透過券商軟體查看三大法人。
            * **OBV 能量潮**：目前圖表中已計算 OBV，若 OBV 創新高但股價未過高，代表主力在吸籌；反之則為出貨。
            """)
            st.metric("目前 OBV 數值", f"{int(last['OBV']/1000):,} K")
            
            # 簡單的大戶訊號
            if last['Volume'] > df['Volume'].mean() * 2:
                st.warning("⚠️ **爆量訊號**：今日成交量大於平均 2 倍，請留意主力換手或變盤。")
            else:
                st.success("⚖️ **量能溫和**：成交量在正常範圍內。")
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
from deep_translator import GoogleTranslator
import feedparser
import urllib.parse
import requests 

# --- 1. 頁面設定與 CSS 美化 ---
st.set_page_config(page_title="全球股市 AI 戰情室", layout="wide", page_icon="📈")

def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans TC', sans-serif;
            background-color: #0e1117;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
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
        }
        .card-title { color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px; font-weight: 500; }
        .card-value { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
        .card-delta { font-size: 0.9rem; margin-top: 5px; font-weight: 600; }

        /* AI 分析報告區塊樣式 */
        .ai-report-box {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
        }
        .ai-report-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #34d399;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .ai-report-content {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #e2e8f0;
        }
        .highlight {
            color: #fbbf24;
            font-weight: bold;
            padding: 0 5px;
        }
        
        /* 搜尋結果按鈕樣式 */
        .stButton button {
            width: 100%;
            text-align: left;
            border: 1px solid #334155;
            background-color: #1e293b;
            color: #e2e8f0;
        }
        .stButton button:hover {
            border-color: #3b82f6;
            color: #3b82f6;
        }

        .hero-container {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        
        /* 買賣盤計量條 */
        .vol-bar-bg {
            background-color: #334155;
            height: 10px;
            border-radius: 5px;
            width: 100%;
            margin-top: 5px;
            overflow: hidden;
        }
        .vol-bar-fill {
            height: 100%;
        }
        
        /* 警語樣式 */
        .disclaimer-box {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            padding: 15px;
            border-radius: 10px;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 2. 輔助函數 ---

def get_market_timing_info(market_type):
    tz_map = { "台股": 'Asia/Taipei', "港股": 'Asia/Hong_Kong', "美股": 'America/New_York' }
    tz_name = next((v for k, v in tz_map.items() if k in market_type), 'Asia/Taipei')
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    
    if "美股" in market_type:
        open_time = time(9, 30)
        close_time = time(16, 0)
    elif "台股" in market_type:
        open_time = time(9, 0)
        close_time = time(13, 30)
    else: # 港股
        open_time = time(9, 30)
        close_time = time(16, 0)

    current_time = now.time()
    weekday = now.weekday() 
    
    is_trading_day = weekday <= 4
    is_open = False
    countdown_msg = ""
    target_dt = None
    
    if is_trading_day:
        if current_time < open_time:
            target_dt = datetime.combine(now.date(), open_time).replace(tzinfo=tz)
            is_open = False
            state_label = "距離開盤"
        elif open_time <= current_time <= close_time:
            target_dt = datetime.combine(now.date(), close_time).replace(tzinfo=tz)
            is_open = True
            state_label = "距離收盤"
        else:
            is_open = False
            state_label = "距離開盤"
            days_add = 1
            if weekday == 4: days_add = 3
            target_dt = datetime.combine(now.date() + timedelta(days=days_add), open_time).replace(tzinfo=tz)
    else:
        is_open = False
        state_label = "距離開盤"
        days_add = (7 - weekday)
        target_dt = datetime.combine(now.date() + timedelta(days=days_add), open_time).replace(tzinfo=tz)

    diff = target_dt - now
    total_seconds = int(diff.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days := diff.days:
        time_str = f"{days}天 {hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    countdown_msg = f"{state_label}: {time_str}"
    
    if weekday == 4 or weekday == 5: ai_date_str = "下週一"
    elif weekday == 6: ai_date_str = "明日 (週一)"
    else: ai_date_str = "明日"
        
    return is_open, countdown_msg, ai_date_str

def search_symbols(query):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 5, "newsCount": 0}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, params=params, headers=headers)
        data = r.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes']
    except Exception as e:
        print(f"Search Error: {e}")
    return []

def get_market_indices(market_type):
    index_map = {
        "台股": {"加權指數 (TAIEX)": "^TWII"},
        "港股": {"恒生指數 (HSI)": "^HSI"},
        "美股": {"道瓊工業": "^DJI", "納斯達克": "^IXIC", "標普 500": "^GSPC"}
    }
    target_indices = {}
    for key in index_map:
        if key in market_type:
            target_indices = index_map[key]
            break
    results = []
    if target_indices:
        for name, ticker_code in target_indices.items():
            try:
                ticker = yf.Ticker(ticker_code)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    last = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = last - prev
                    pct = (change / prev) * 100
                    results.append({"name": name, "price": last, "change": change, "pct": pct})
            except: pass
    return results

def get_buy_sell_volume_estimate(ticker):
    try:
        df_intra = ticker.history(period="1d", interval="5m")
        if df_intra.empty: return 0, 0
        buy_vol = df_intra[df_intra['Close'] >= df_intra['Open']]['Volume'].sum()
        sell_vol = df_intra[df_intra['Close'] < df_intra['Open']]['Volume'].sum()
        return buy_vol, sell_vol
    except: return 0, 0

def get_chinese_name_and_news(raw_name, raw_code):
    zh_name = raw_name
    translated = False
    try:
        if not any("\u4e00" <= char <= "\u9fff" for char in raw_name):
            zh_name = GoogleTranslator(source='auto', target='zh-TW').translate(raw_name)
            translated = True
    except: pass

    def fetch_news(query_name):
        n_list = []
        try:
            query = f"{query_name} {raw_code}"
            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            feed = feedparser.parse(rss_url)
            sorted_entries = sorted(feed.entries, key=lambda x: x.published_parsed, reverse=True)
            for entry in sorted_entries[:8]:
                pub_date = entry.published_parsed
                fmt_date = datetime(*pub_date[:6]).strftime('%Y-%m-%d %H:%M') if pub_date else ""
                n_list.append({'title': entry.title, 'link': entry.link, 'publisher': entry.source.title if hasattr(entry, 'source') else 'Google News', 'time': fmt_date})
        except: pass
        return n_list

    news_list = fetch_news(zh_name)
    if not news_list and translated:
        news_list = fetch_news(raw_name)
        zh_name = raw_name
    return zh_name, news_list

def generate_layman_analysis(df, fund, pred_price, date_str):
    last_close = df['Close'].iloc[-1]
    ma5 = df['MA5'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    k = df['K'].iloc[-1]
    d = df['D'].iloc[-1]
    
    analysis = []
    if last_close > ma20 and ma20 > ma60: trend = "📈 **強勢多頭**：股價站穩月線與季線之上，長期趨勢看漲。"
    elif last_close < ma20 and ma20 < ma60: trend = "📉 **弱勢空頭**：股價位於均線下方，賣壓較重。"
    elif last_close > ma20: trend = "🌤️ **短期反彈**：股價重新站回月線，短期有轉強跡象。"
    else: trend = "☁️ **震盪整理**：股價在均線附近徘徊，方向尚未明確。"
    analysis.append(trend)
    
    if rsi > 75: heat = "🔥 **市場過熱**：RSI 指標顯示買盤過於擁擠，請勿盲目追高。"
    elif rsi < 25: heat = "❄️ **市場超賣**：RSI 指標顯示股價已跌深，有機會反彈。"
    else: heat = "⚖️ **交易健康**：目前買賣力道平衡，走勢屬於健康範圍。"
    analysis.append(heat)
    
    pred_diff = pred_price - last_close
    pred_pct = (pred_diff / last_close) * 100
    direction = "上漲" if pred_diff > 0 else "下跌"
    
    reasons = []
    if pred_diff > 0:
        if last_close > ma20: reasons.append("股價位於月線之上")
        if rsi < 40: reasons.append("RSI 相對低檔")
        if k > d: reasons.append("KD 黃金交叉")
        if not reasons: reasons.append("技術指標醞釀反彈")
    else:
        if last_close < ma20: reasons.append("股價跌破月線")
        if rsi > 70: reasons.append("RSI 過熱")
        if k < d: reasons.append("KD 死亡交叉")
        if not reasons: reasons.append("上方壓力較大")
        
    reason_str = "、".join(reasons)
    
    ai_msg = f"""
    🤖 **AI 模型預測**：根據大數據演算，預測<span class='highlight'>{date_str}</span>股價可能來到 <span class='highlight'>{pred_price:.2f}</span>，潛在{direction}幅度約 <span class='highlight'>{abs(pred_pct):.2f}%</span>。<br>
    <div style='margin-top: 10px; font-size: 0.95rem; color: #cbd5e1;'>
        💡 <b>AI 判斷主要依據：</b>{reason_str}。
    </div>
    """
    return analysis, ai_msg

# --- [升級版] 自動化同業推薦邏輯 ---
def get_peers_list(stock_code, info, market_type):
    # 1. 取得該股票的產業或板塊資訊
    industry = info.get('industry', '').lower()
    sector = info.get('sector', '').lower()
    
    # 用於判斷台股產業 (如果是台股)
    tw_category = ""
    if "台股" in market_type and stock_code in twstock.codes:
        tw_category = twstock.codes[stock_code].type
    
    # 2. 定義「產業 -> 競爭對手」的自動對照表
    # 這裡列出了熱門產業的代表性股票 (台美股混合)
    industry_map = {
        # --- 半導體與晶片 (Semiconductors) ---
        "semiconductors": ["2330.TW", "2454.TW", "NVDA", "AMD", "INTC", "TSM"],
        "半導體業": ["2330.TW", "2454.TW", "2303.TW", "3711.TWO"],
        
        # --- 電腦與消費電子 (Consumer Electronics) ---
        "consumer electronics": ["AAPL", "2317.TW", "2382.TW", "3231.TW", "MSFT"],
        "電腦及週邊設備業": ["2382.TW", "3231.TW", "2357.TW", "2324.TW"],
        "電子零組件業": ["2317.TW", "2308.TW", "3008.TW"],

        # --- 航運與物流 (Marine Shipping) ---
        "marine shipping": ["2603.TW", "2609.TW", "2615.TW", "ZIM"],
        "航運業": ["2603.TW", "2609.TW", "2615.TW", "2618.TW", "2610.TW"],

        # --- 金融銀行 (Banks & Financial) ---
        "banks": ["2881.TW", "2882.TW", "2891.TW", "JPM", "BAC"],
        "金融保險業": ["2881.TW", "2882.TW", "2891.TW", "2886.TW", "2892.TW"],

        # --- 電動車與汽車 (Auto Manufacturers) ---
        "auto manufacturers": ["TSLA", "2201.TW", "F", "TM"],
        "汽車工業": ["2201.TW", "2207.TW", "1319.TW"],

        # --- AI 與 軟體 (Software) ---
        "software": ["MSFT", "GOOGL", "PLTR", "AI"],
        "information technology": ["MSFT", "AAPL", "NVDA"],
    }
    
    # 3. 開始自動匹配
    peers = set() # 使用 set 避免重複

    # 策略 A: 根據 twstock 的分類 (準確度高，針對台股)
    if tw_category and tw_category in industry_map:
        peers.update(industry_map[tw_category])

    # 策略 B: 根據 Yahoo 的 industry 關鍵字模糊比對
    for key, val in industry_map.items():
        if key in industry or key in sector:
            peers.update(val)
            
    # 策略 C: 如果都沒對應到，加入該市場的大盤指數作為基準
    if not peers:
        if "台股" in market_type:
            peers.update(["^TWII", "2330.TW"]) # 沒對手就跟台積電比
        elif "美股" in market_type:
            peers.update(["^GSPC", "NVDA", "AAPL"])
        elif "港股" in market_type:
            peers.update(["^HSI", "0700.HK"])

    # 4. 移除自己 (不跟自己比) 並轉回 list
    clean_input = stock_code.upper()
    final_peers = [p for p in peers if p.replace(".TW", "").replace(".TWO", "") not in clean_input.replace(".TW", "").replace(".TWO", "")]
    
    return list(final_peers)[:5] # 最多只取前 5 檔，避免跑太久
# --- 3. 核心資料載入 ---
# [優化] 將快取時間 (ttl) 改為 45 秒。
# 這樣做是因為我們主程式每 60 秒會刷新一次，設定 45 秒可以確保
# 當網頁在第 60 秒刷新時，快取肯定已經過期，強迫系統去抓取最新資料。
@st.cache_data(ttl=45)
def load_data(stock_code, market_type, is_tw, ai_date_str):
    fetch_time = datetime.now()

    tickers_to_try = []
    clean_input = stock_code.strip().upper()
    if is_tw:
        base_code = clean_input.replace(".TW", "").replace(".TWO", "")
        tickers_to_try = [f"{base_code}.TW", f"{base_code}.TWO"]
    elif "港股" in market_type:
        base_code = clean_input.replace(".HK", "")
        tickers_to_try = [f"{base_code.zfill(4)}.HK"]
    else:
        tickers_to_try = [clean_input]

    ticker = None
    history = pd.DataFrame()
    yf_code_used = ""

    for yf_code in tickers_to_try:
        temp_ticker = yf.Ticker(yf_code)
        try:
            check = temp_ticker.history(period="5d")
            if not check.empty:
                history = temp_ticker.history(period="2y")
                ticker = temp_ticker
                yf_code_used = yf_code
                break 
        except: continue

    if history.empty: return None

    buy_vol, sell_vol = get_buy_sell_volume_estimate(ticker)

    info = {}
    try: info = ticker.info
    except: pass
    
    # 抓取公司/指數簡介並進行自動翻譯
    raw_summary = info.get('longBusinessSummary', info.get('description', '暫無相關簡介資訊。'))
    summary = raw_summary
    try:
        if raw_summary and raw_summary != '暫無相關簡介資訊。':
            summary = GoogleTranslator(source='auto', target='zh-TW').translate(raw_summary)
    except Exception as e:
        print(f"Summary Translation Error: {e}")
        pass

    fundamentals = {
        '本益比 (P/E)': info.get('trailingPE', 'N/A'),
        '預估本益比 (Fwd P/E)': info.get('forwardPE', 'N/A'),
        '股價淨值比 (P/B)': info.get('priceToBook', 'N/A'),
        '股東權益報酬率 (ROE)': info.get('returnOnEquity', 'N/A'),
        '分析師目標價': info.get('targetMeanPrice', 'N/A')
    }

    stock_name = info.get('longName', info.get('shortName', yf_code_used))
    if is_tw and stock_code in twstock.codes:
        stock_name = twstock.codes[stock_code].name

    zh_name, news_data = get_chinese_name_and_news(stock_name, stock_code)

    df = history.copy()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    rsv = (df['Close'] - df['Low'].rolling(9).min()) / (df['High'].rolling(9).max() - df['Low'].rolling(9).min()) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    pred_price = df['Close'].iloc[-1]
    try:
        if len(df) > 60:
            df_ml = df.dropna()
            X = df_ml[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'K', 'D']]
            y = df_ml['Close'].shift(-1).dropna()
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
            model.fit(X[:-1], y)
            pred_price = model.predict(X.tail(1))[0]
    except: pass

    txt_analysis, ai_msg = generate_layman_analysis(df, info, pred_price, ai_date_str)

    last_time = df.index[-1]
    if last_time.tzinfo is None:
        last_time = pytz.utc.localize(last_time).astimezone(pytz.timezone('Asia/Taipei'))
    else:
        last_time = last_time.astimezone(pytz.timezone('Asia/Taipei'))
        
    return {
        'df': df,
        'info': info,
        'name_zh': zh_name,
        'news': news_data,
        'fund': fundamentals,
        'pred': pred_price,
        'time': last_time.strftime('%Y-%m-%d %H:%M'),
        'industry': info.get('industry', 'N/A'),
        'analysis': txt_analysis,
        'ai_msg': ai_msg,
        'buy_vol': buy_vol,
        'sell_vol': sell_vol,
        'fetch_time': fetch_time,
        'summary': summary
    }

# --- 4. 側邊欄 ---
st.sidebar.title("🎛️ 戰情控制中心")
market_type = st.sidebar.selectbox("選擇市場", ["🇹🇼 台股", "🇺🇸 美股", "🇭🇰 港股"])

@st.fragment(run_every=1)
def show_sidebar_timers(market_type, data_fetch_time):
    # To this:
    is_open, time_msg, ai_date_str = get_market_timing_info(market_type)
    status_color = "#22c55e" if is_open else "#ef4444"
    status_text = "🟢 交易進行中" if is_open else "🔴 已收盤"

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid {status_color}; margin-bottom: 20px;">
        <div style="font-weight: bold; font-size: 1.1rem; color: #f8fafc; margin-bottom: 5px;">{status_text}</div>
        <div style="font-size: 0.9rem; color: #cbd5e1;">⏳ {time_msg}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if data_fetch_time:
        seconds_elapsed = (datetime.now() - data_fetch_time).total_seconds()
        seconds_remaining = int(60 - seconds_elapsed)
        if seconds_remaining < 0: seconds_remaining = 0
        
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3); margin-bottom: 20px; text-align: center;">
            <div style="font-size: 0.8rem; color: #93c5fd;">數據下一次更新於</div>
            <div style="font-size: 1.2rem; font-weight: bold; color: #3b82f6;">{seconds_remaining} 秒</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("等待數據載入...")

default_code = "2330"
if "美股" in market_type: default_code = "NVDA"
elif "港股" in market_type: default_code = "9988"

with st.sidebar.expander("🔍 不知道代碼？點此搜尋"):
    search_query = st.text_input("輸入公司名稱", key="search_input")
    if search_query:
        results = search_symbols(search_query)
        for res in results:
            if st.button(f"{res.get('symbol')} - {res.get('shortname')}", key=res.get('symbol')):
                st.session_state.stock_code = res.get('symbol')
                st.rerun()

if 'stock_code' not in st.session_state:
    st.session_state.stock_code = default_code
stock_input = st.sidebar.text_input("輸入代碼", key="stock_code")
is_tw = "台股" in market_type

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **免責聲明**\n\n本工具僅供學術研究，AI 預測與買賣盤估算僅供參考，不代表未來走勢。")

# --- 5. 主程式 ---
# 獲取當前市場是否開盤
# [修正] 這裡要把 ai_date_str 接出來，下面 load_data 才讀得到
is_open, time_msg, ai_date_str = get_market_timing_info(market_type)

# 動態設定更新間隔 (毫秒)
# 開盤 60,000ms (1分) / 收盤 3,600,000ms (7.5min)
refresh_interval = 60000 if is_open else 450000

# 套用動態間隔
st_autorefresh(interval=refresh_interval, key="data_refresh")

if stock_input:
    # 這裡現在就能正常讀取到 ai_date_str 了
    data = load_data(stock_input, market_type, is_tw, ai_date_str)
    
    if not data:
        st.error(f"❌ 找不到代碼 {stock_input}，請檢查輸入是否正確。")
        show_sidebar_timers(market_type, None)
        st.stop()

    show_sidebar_timers(market_type, data['fetch_time'])

    df = data['df']
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    color = "#ef4444" if change > 0 else "#22c55e" if change < 0 else "#94a3b8"
    arrow = "▲" if change > 0 else "▼" if change < 0 else "-"
    
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
            🕒 更新時間: {data['time']} | 昨收: {prev['Close']:.2f} | 總量: {int(last['Volume']/1000):,} K
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🏢 查看公司/指數簡介 (Business Summary)"):
        st.markdown(f"<div style='line-height: 1.6; color: #e2e8f0;'>{data['summary']}</div>", unsafe_allow_html=True)

    total_est_vol = data['buy_vol'] + data['sell_vol']
    if total_est_vol > 0:
        buy_pct = (data['buy_vol'] / total_est_vol) * 100
        sell_pct = (data['sell_vol'] / total_est_vol) * 100
    else:
        buy_pct, sell_pct = 50, 50
        
    c_vol1, c_vol2 = st.columns(2)
    with c_vol1:
        st.markdown(f"""
        <div style="text-align: center; background: rgba(239, 68, 68, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <div style="color: #fca5a5; font-size: 0.9rem;">🔴 預估買盤 (主動買進)</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #ef4444;">{int(data['buy_vol']/1000):,} K</div>
            <div class="vol-bar-bg"><div class="vol-bar-fill" style="width: {buy_pct}%; background-color: #ef4444;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    with c_vol2:
        st.markdown(f"""
        <div style="text-align: center; background: rgba(34, 197, 94, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(34, 197, 94, 0.3);">
            <div style="color: #86efac; font-size: 0.9rem;">🟢 預估賣盤 (主動賣出)</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #22c55e;">{int(data['sell_vol']/1000):,} K</div>
            <div class="vol-bar-bg"><div class="vol-bar-fill" style="width: {sell_pct}%; background-color: #22c55e;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    st.caption("註：買賣盤數據為使用當日分時K線估算之近似值，僅供參考力道方向。")
    st.markdown("<br>", unsafe_allow_html=True)

    market_indices = get_market_indices(market_type)
    if market_indices:
        st.markdown(f"###### 📊 {market_type} 重點指數")
        idx_cols = st.columns(len(market_indices))
        for i, idx in enumerate(market_indices):
            color = "#ef4444" if idx['change'] > 0 else "#22c55e" if idx['change'] < 0 else "#94a3b8"
            arrow = "▲" if idx['change'] > 0 else "▼" if idx['change'] < 0 else "-"
            with idx_cols[i]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); 
                            padding: 10px; border-radius: 10px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.8rem;">{idx['name']}</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #f8fafc;">{idx['price']:,.2f}</div>
                    <div style="color: {color}; font-size: 0.85rem;">
                        {arrow} {abs(idx['change']):,.2f} ({abs(idx['pct']):.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-report-box">
        <div class="ai-report-title">🤖 AI 投資顧問報告 (Beta)</div>
        <div class="ai-report-content">
            {data['ai_msg']}<br><br>
            <ul style="margin-top: 10px;">
                <li>{data['analysis'][0]}</li>
                <li>{data['analysis'][1]}</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. 定義 Helper 函數 (為了下方使用)
    def card(col, title, value, delta=None, prefix="", color=None):
        d_html = ""
        if delta:
            d_color = "#ef4444" if "▲" in delta else "#22c55e"
            d_html = f'<div class="card-delta" style="color: {d_color};">{delta}</div>'
        
        # 自訂值顏色 (若無則預設白色)
        val_color = color if color else "#f8fafc"
        
        col.markdown(f"""
        <div class="metric-card">
            <div class="card-title">{title}</div>
            <div class="card-value" style="color: {val_color}">{prefix}{value}</div>
            {d_html}
        </div>""", unsafe_allow_html=True)

    # [新增] 2. 顯示 當日最高 / 當日最低 / 開盤價
    # 這段代碼放在 AI 報告下方，一般數據卡片上方
    st.subheader("📊 本日行情摘要")
    c_high, c_low, c_open = st.columns(3)
    
    # 取得最新一筆資料
    card(c_high, "最高價 (High)", f"{last['High']:.2f}", color="#ef4444")  # 紅色代表高點
    card(c_low, "最低價 (Low)", f"{last['Low']:.2f}", color="#22c55e")    # 綠色代表低點
    card(c_open, "開盤價 (Open)", f"{last['Open']:.2f}")

    st.markdown("---") # 分隔線

    c1, c2, c3, c4 = st.columns(4)
    pred_diff = data['pred'] - last['Close']
    card(c1, f"AI 預測{ai_date_str}價格", f"{data['pred']:.2f}", f"{'▲' if pred_diff>0 else '▼'} {abs((pred_diff/last['Close'])*100):.2f}%")
    
    pe = data['info'].get('trailingPE', 'N/A')
    pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) else "N/A"
    card(c2, "本益比 (P/E)", pe_str)
    
    dy = data['info'].get('dividendYield', 0)
    dy_str = f"{dy*100:.2f}%" if isinstance(dy, (int, float)) else "N/A"
    card(c3, "殖利率 (Yield)", dy_str)
    
    k_val, d_val = last['K'], last['D']
    card(c4, "技術指標 (KD)", f"K{k_val:.0f}", f"{'▲' if k_val>d_val else '▼'} {'黃金交叉' if k_val>d_val else '死亡交叉'}")

    st.markdown("---")
    # 修改為：
    tab1, tab2, tab3, tab4 = st.tabs(["📊 技術分析", "📰 智能新聞", "💰 籌碼基本面", "🤝 同業與相關性"])

    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="股價"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1.5), name="月線"), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=['red' if r['Open'] < r['Close'] else 'green' for i, r in df.iterrows()], name="成交量"), row=2, col=1)
        fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if data['news']:
            for n in data['news']:
                st.markdown(f"""<div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #3b82f6;"><a href="{n['link']}" target="_blank" style="text-decoration: none; color: #f8fafc; font-size: 1.1rem; font-weight: 600;">{n['title']}</a><div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">📅 {n['time']} | 📢 {n['publisher']}</div></div>""", unsafe_allow_html=True)
        else: st.info("暫無相關新聞")

    with tab3:
        st.subheader("📋 關鍵財務數據")
        fund_df = pd.DataFrame(list(data['fund'].items()), columns=['指標', '數值'])
        fund_df['數值'] = fund_df['數值'].astype(str)
        st.dataframe(fund_df, hide_index=True, use_container_width=True)
    with tab4:
        st.subheader("🔗 同業股價與相關性分析")
        st.caption("相關係數 (Correlation) 代表股價走勢的連動程度，1 為完全正相關（同漲同跌），-1 為完全負相關，0 為無關。")
        
        # 1. 取得同業清單
        # 1. 取得同業清單 (新版：傳入 info 讓它自動判斷產業)
        peers = get_peers_list(stock_input, data['info'], market_type)
        
        # 2. 抓取資料 (顯示載入中動畫)
        with st.spinner('正在分析同業數據...'):
            peer_df = load_peer_data(df, peers)
        
        if peer_df is not None and not peer_df.empty:
            # 顯示相關性長條圖
            peer_df['color'] = peer_df['corr'].apply(lambda x: '#ef4444' if x > 0 else '#22c55e')
            
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Bar(
                x=peer_df['corr'],
                y=peer_df['name'],
                orientation='h',
                marker_color=peer_df['color'],
                text=peer_df['corr'].apply(lambda x: f"{x:.2f}"),
                textposition='auto'
            ))
            fig_corr.update_layout(
                title="與本股之價格走勢相關性 (近60日)",
                xaxis_title="相關係數 (-1 ~ 1)",
                yaxis={'categoryorder':'total ascending'},
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 顯示同業股價卡片
            st.markdown("###### 🏦 同業即時報價")
            p_cols = st.columns(len(peer_df))
            for idx, (code, row) in enumerate(peer_df.iterrows()):
                # 若同業太多，換行顯示
                col_idx = idx % 4 
                if col_idx == 0 and idx > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    p_cols = st.columns(4)
                
                with p_cols[col_idx]:
                    p_color = "#ef4444" if row['pct'] > 0 else "#22c55e" if row['pct'] < 0 else "#94a3b8"
                    p_arrow = "▲" if row['pct'] > 0 else "▼" if row['pct'] < 0 else ""
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border: 1px solid {p_color}40; 
                                padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 5px;">{row['name']}</div>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #f8fafc;">{row['price']:.2f}</div>
                        <div style="color: {p_color}; font-size: 0.9rem; font-weight: 500;">
                            {p_arrow} {abs(row['pct']):.2f}%
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">
                            相關性: {row['corr']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ 無法取得足夠的同業數據或無相關同業清單。")
    st.markdown("---")
    st.markdown("""<div class="disclaimer-box">⚠️ 免責聲明：所有數據僅供參考，投資盈虧自負。</div>""", unsafe_allow_html=True)
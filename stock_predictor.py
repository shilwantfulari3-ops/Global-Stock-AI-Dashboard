import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# --- Page Setup ---
st.set_page_config(page_title="Stock AI Dashboard", layout="wide", page_icon="📈")

# --- Custom CSS for cleaner look ---
st.markdown("""
<style>
    .metric-container {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 10px 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# 🧠 CORE PREDICTION ENGINE
# =========================================
def predict_next_day(ticker):
    """
    Fetches data, trains model, and predicts the next day's close.
    Returns a dictionary with all necessary data for plotting.
    """
    try:
        # Get 2 years of data for a good trend
        df = yf.Ticker(ticker).history(period="2y")
        if df.empty or len(df) < 60:
            return None

        # Prepare data
        data = df[['Close']].copy()
        data['Target'] = data['Close'].shift(-1)
        
        # Train (drop today's row for training)
        train_data = data.dropna()
        X_train = train_data[['Close']]
        y_train = train_data['Target']
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict using ACTUAL latest close
        latest_close = df['Close'].iloc[-1]
        latest_date = df.index[-1]
        predicted_price = model.predict([[latest_close]])[0]
        
        # Calculate next trading day
        next_date = latest_date + pd.tseries.offsets.BDay(1)
        
        return {
            "ticker": ticker,
            "history_df": df,
            "latest_date": latest_date,
            "latest_close": latest_close,
            "next_date": next_date,
            "predicted_close": predicted_price,
            "pct_change": ((predicted_price - latest_close) / latest_close) * 100
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# =========================================
# 📊 CHARTING FUNCTION (Interactive)
# =========================================
def plot_interactive_chart(prediction_data):
    """Creates a Plotly candlestick chart with prediction overlay."""
    df = prediction_data['history_df']
    
    # Create subplots: Row 1 for Price, Row 2 for Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=('Price Movement', 'Volume'),
                        row_heights=[0.7, 0.3])

    # 1. Candlestick Chart (Last 3 months for clarity initially)
    plot_df = df.tail(90) 
    fig.add_trace(go.Candlestick(x=plot_df.index,
                                 open=plot_df['Open'], high=plot_df['High'],
                                 low=plot_df['Low'], close=plot_df['Close'],
                                 name='OHLC'), row=1, col=1)

    # 2. Add Prediction Line (Connecting today to tomorrow)
    # We create a small 2-point line from Last Close -> Predicted Close
    pred_x = [prediction_data['latest_date'], prediction_data['next_date']]
    pred_y = [prediction_data['latest_close'], prediction_data['predicted_close']]
    
    color = 'green' if prediction_data['pct_change'] > 0 else 'red'
    fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines+markers',
                             line=dict(color=color, width=3, dash='dot'),
                             marker=dict(size=8),
                             name='AI Prediction'), row=1, col=1)

    # 3. Volume Bar Chart
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color='teal', showlegend=False), 
                  row=2, col=1)

    # Layout updates
    fig.update_layout(
        title=f"{prediction_data['ticker']} - AI Trend Forecast",
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark"
    )
    return fig

# =========================================
# 🌍 SCREENER HELPER FUNCTIONS (REVISED)
# =========================================
@st.cache_data
def get_tickers(market_type):
    """
    Fetches tickers from Wikipedia based on the selected market.
    This version is more robust and checks all tables.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        if market_type == 'USA':
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(requests.get(url, headers=headers).text)
            # Find the table with the 'Symbol' column
            for t in tables:
                if 'Symbol' in t.columns:
                    return t['Symbol'].str.replace('.', '-', regex=False).tolist()
            # If no table found
            st.error("Could not find S&P 500 table on Wikipedia.")
            return []

        elif market_type == 'India':
            url = 'https://en.wikipedia.org/wiki/NIFTY_50'
            tables = pd.read_html(requests.get(url, headers=headers).text)
            # Find the table with the 'Symbol' column
            for t in tables:
                if 'Symbol' in t.columns:
                    return (t['Symbol'] + '.NS').tolist()
            st.error("Could not find Nifty 50 table on Wikipedia.")
            return []

        elif market_type == 'UK':
            url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
            tables = pd.read_html(requests.get(url, headers=headers).text)
            # Find the table with the 'Ticker' column
            for t in tables:
                if 'Ticker' in t.columns:
                    return (t['Ticker'] + '.L').tolist()
            st.error("Could not find FTSE 100 table on Wikipedia.")
            return []
            
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return []
        
    return [] # Default return an empty list

# =========================================
# 📱 MAIN APP LAYOUT
# =========================================
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose Mode:", ["🔍 Single Stock Search", "📋 Market Screener"])

# ---------------------------------------------------
# MODE 1: SINGLE STOCK SEARCH
# ---------------------------------------------------
if app_mode == "🔍 Single Stock Search":
    st.title("AI Stock Inspector 🔍")
    st.write("Analyze any stock globally. Type the ticker symbol below.")
    
    # Ticker Input with Examples
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Enter Ticker Symbol:", value="RELIANCE.NS", help="USA: AAPL, TSLA | India: TCS.NS, INFY.NS | UK: HSBA.L, RR.L").upper()
    with col2:
        st.write("##") # Spacer
        search_button = st.button("🔮 Analyze Trend", use_container_width=True)

    if search_button or ticker_input:
        with st.spinner(f"Analyzing {ticker_input}..."):
            result = predict_next_day(ticker_input)
            
            if result:
                # --- METRICS SECTION ---
                st.write("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Last Date", f"{result['latest_date']:%Y-%m-%d}")
                m2.metric("Last Close Price", f"{result['latest_close']:.2f}")
                m3.metric("Predicted Next Close", f"{result['predicted_close']:.2f}", 
                          delta=f"{result['pct_change']:.2f}%")
                m4.metric("Prediction Date", f"{result['next_date']:%Y-%m-%d}")

                # --- INTERACTIVE CHART SECTION ---
                st.plotly_chart(plot_interactive_chart(result), use_container_width=True)
                
            else:
                st.error(f"Could not fetch data for '{ticker_input}'. Check the ticker symbol and try again.")

# ---------------------------------------------------
# MODE 2: MARKET SCREENER (Previous Functionality)
# ---------------------------------------------------
elif app_mode == "📋 Market Screener":
    st.title("Global Market Screener 🌎")
    market = st.sidebar.selectbox("Select Market", ["USA", "India", "UK"])
    num = st.sidebar.slider("Stocks to Scan", 5, 50, 10)
    
    if st.sidebar.button("Run Screener"):
        tickers = get_tickers(market)[:num]
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, t in enumerate(tickers):
            status.text(f"Scanning {i+1}/{num}: {t}")
            progress.progress((i+1)/num)
            res = predict_next_day(t)
            if res: results.append(res)
        
        progress.empty()
        status.success("Done!")
        
        if results:
            df = pd.DataFrame(results)
            # Clean up dataframe for display
            display_df = pd.DataFrame({
                "Ticker": df['ticker'],
                "Last Date": df['latest_date'],
                "Last Price": df['latest_close'],
                "Pred Date": df['next_date'],
                "Pred Price": df['predicted_close'],
                "% Move": df['pct_change']
            }).sort_values("% Move", ascending=False)
            
            st.dataframe(display_df.style.format({
                "Last Date": "{:%Y-%m-%d}", "Pred Date": "{:%Y-%m-%d}",
                "Last Price": "{:.2f}", "Pred Price": "{:.2f}", "% Move": "{:+.2f}%"
            }).background_gradient(subset=['% Move'], cmap='RdYlGn'))
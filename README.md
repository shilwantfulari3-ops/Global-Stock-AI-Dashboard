# 📈 Global Stock AI Dashboard

An interactive web application that fetches global stock market data and uses **Linear Regression Machine Learning** to predict the next day's closing price.

## 🚀 Features
* **Single Stock Search:** Analyze any ticker symbol globally (e.g., `RELIANCE.NS`, `AAPL`, `TSLA`).
* **Global Market Screener:** Scan entire indices like **Nifty 50** (India), **S&P 500** (USA), and **FTSE 100** (UK) for top predicted gainers.
* **Interactive Charts:** Dynamic candlestick charts built with **Plotly** featuring an AI-trend forecast line.
* **Live Market Data:** Real-time data fetching using the `yfinance` API.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Data Handling:** Pandas, NumPy
* **Visualization:** Plotly
* **Web Scraping:** Requests, LXML

---

## 🧠 Machine Learning Logic


### 1. Data Transformation
To predict the future, the time-series stock data is transformed into a **supervised learning** format. We create a "Target" column by shifting the closing prices back by one day ($Price_{Tomorrow}$).
* **Feature ($X$):** Today's Closing Price.
* **Label ($y$):** Tomorrow's Closing Price.

### 2. The Algorithm
The model uses **Ordinary Least Squares (OLS) Linear Regression**. It finds the "Line of Best Fit" by minimizing the sum of the squares of the vertical deviations between each data point and the line.
* **Equation:** $$y = mx + c$$ 

### 3. Training & Prediction
* The model is trained on **2 years** of historical data.
* The current day's closing price is fed into the trained model to generate the forecast for the next trading session.

---

## 🔧 Installation & Usage
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/shilwantfulari3-ops/Global-Stock-AI-Dashboard.git](https://github.com/shilwantfulari3-ops/Global-Stock-AI-Dashboard.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install streamlit yfinance pandas scikit-learn plotly lxml requests
    ```
3.  **Run the Application:**
    ```bash
    python -m streamlit run stock_predictor.py
    ```

## 📜 Disclaimer
This tool is for **educational purposes only**. Stock market predictions are inherently risky, and this simple linear model does not account for market news, sentiment, or external economic factors.

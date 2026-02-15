# Financial Market Analyses

A comprehensive Streamlit dashboard for stock market analysis, forecasting, and portfolio optimization.

## Features

### Home
- Overview of the platform and supported asset classes

### Technical Analysis
- **Time Series**: Open/Close price evolution with interactive range slider
- **Candlestick Patterns**: 19 pattern detections (Doji, Hammer, Engulfing, Morning Star, etc.)
- **RSI**: Relative Strength Index with overbought/oversold zones
- **Moving Averages**: SMA, EMA, WMA, DEMA, TEMA
- **MACD**: Moving Average Convergence Divergence with histogram
- **Bollinger Bands**: Upper, middle, lower bands with candlestick overlay
- **Forecasting**: Prophet-based price prediction (1-4 years) with MSE, RMSE, MAE, MAPE metrics

### Fundamental Analysis
- Company performance metrics (market cap, revenue, enterprise value)
- Growth indicators (earnings, revenue, margins)
- Risk scores (audit, board, compensation, shareholder rights)
- Dividend information
- Article summarizer with keyword extraction and sentiment analysis

### Portfolio Optimization
- Compare 2-4 stocks simultaneously
- Descriptive statistics and coefficient of variation
- Box plots and log-return distributions
- Correlation matrix and pair plots
- **Monte Carlo Simulation** (10,000 portfolios):
  - Maximum Sharpe Ratio portfolio
  - Minimum Volatility portfolio
  - Optimal weight allocation
  - Investment return calculator

## Supported Asset Classes

| Type | Examples |
|------|----------|
| **Stocks** | AAPL, MSFT, GOOGL, TSLA, META, NVDA, JPM, BA, etc. |
| **Forex** | USD/EUR, USD/JPY, USD/GBP, USD/CAD, etc. |
| **Commodities** | Gold, Silver, Crude Oil, Natural Gas, Copper, etc. |
| **Crypto** | BTC, ETH, XRP, DOGE, ADA, DOT, etc. |

## Installation

### Prerequisites
- Python 3.9+

### Setup

```bash
# Clone or navigate to the project directory
cd Project

# Install dependencies
pip install streamlit yfinance prophet plotly pandas numpy scikit-learn pandas-ta matplotlib scipy nltk newspaper3k beautifulsoup4 requests

# Run the app
streamlit run "stocks pred.py"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `yfinance` | Yahoo Finance market data |
| `prophet` | Time series forecasting |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | Model evaluation metrics |
| `pandas-ta` | Technical indicators (RSI, SMA, EMA, MACD, Bollinger Bands) |
| `scipy` | Statistical analysis |
| `nltk` | Sentiment analysis (VADER) |
| `newspaper3k` | Article scraping and summarization |
| `beautifulsoup4` | HTML parsing |

## Usage

1. Run the app with `streamlit run "stocks pred.py"`
2. Open the browser at `http://localhost:8501`
3. Navigate between tabs:
   - **Home**: Overview
   - **Technical Analysis**: Select equity type, stock, period, and interval. Explore charts and forecasting.
   - **Fundamental Analysis**: View company fundamentals and summarize articles.
   - **Portfolio Optimization**: Compare stocks and find optimal portfolio allocation.

## Notes

- Internet connection required for live market data
- Forecasting uses Facebook Prophet for time series prediction
- Candlestick pattern detection uses custom implementations (no TA-Lib dependency)
- Portfolio optimization runs 10,000 Monte Carlo simulations

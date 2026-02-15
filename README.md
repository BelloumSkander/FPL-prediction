# FPL ML Predictor

A machine learning-powered Streamlit dashboard for predicting Fantasy Premier League (FPL) player performance and points. This tool helps FPL managers make data-driven decisions by analyzing player statistics, form trends, and upcoming fixture difficulty.

## 🎯 Features

### Player Predictions
- **ML-based Point Predictions**: XGBoost, Neural Networks, and ensemble models predict player points for upcoming gameweeks
- **Form Analysis**: Rolling window statistics (3 and 5 gameweek trends)
- **Performance Metrics**: Goals, assists, clean sheets, bonus points, and efficiency per 90 minutes
- **Injury & Availability Tracking**: Real-time status updates from official FPL API
- **Fixture Difficulty Rating (FDR)**: Automated analysis of upcoming opponent difficulty

### Dashboard Features
- **Player Search & Filtering**: Filter by position, team, price range, and availability
- **Predictions Ranking**: Top predicted scorers for the next gameweek
- **Player Comparison**: Side-by-side performance analysis
- **Trend Visualization**: Interactive charts showing player form over time
- **Model Performance**: View accuracy metrics (MAE, RMSE, etc.) for selected models
- **Custom Predictions**: Generate predictions for specific players or gameweeks

### Data Integration
- **Live FPL API Integration**: Automatically fetches current player data, fixtures, and standings
- **Comprehensive Player History**: 5+ years of historical gameweek data
- **Team & Fixture Information**: All Premier League teams and scheduled fixtures
- **Efficient Caching**: Data cached locally to reduce API calls

## 📋 Prerequisites

- Python 3.9+
- pip (Python package manager)
- Internet connection for FPL API access

## ⚙️ Installation

### Step 1: Clone or navigate to the project directory

```bash
cd FPL-prediction
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.40.0 | Web dashboard framework |
| `pandas` | ≥2.2.0 | Data manipulation and analysis |
| `numpy` | ≥1.26.0 | Numerical computing |
| `scikit-learn` | ≥1.4.0 | Machine learning utilities |
| `xgboost` | ≥2.0.0 | Gradient boosting models |
| `torch` | ≥2.0.0 | Neural network models |
| `plotly` | ≥5.18.0 | Interactive visualizations |
| `requests` | ≥2.31.0 | HTTP API requests |

## 🚀 Usage

1. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate the dashboard**:
   - View predictions for upcoming gameweeks
   - Search for specific players
   - Compare player performance metrics
   - Explore form trends and fixture difficulty
   - Review model performance and confidence scores

3. **Use predictions for FPL decisions**:
   - Identify undervalued high-performers
   - Plan transfers based on fixtures
   - Spot emerging talent and form trends
   - Monitor injury/availability status

## 📁 Project Structure

```
FPL-prediction/
├── app.py                      # Main Streamlit dashboard
├── config.py                   # Configuration constants & settings
├── requirements.txt            # Python dependencies
├── styles.html                 # Custom CSS styling
├── README.md                   # Project documentation
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         # FPL API data retrieval
│   ├── feature_engineering.py  # ML feature creation
│   ├── models.py               # Model architectures & training
│   └── predictor.py            # Main prediction pipeline
├── data/
│   ├── bootstrap_static.json   # FPL static data (players, teams, positions)
│   ├── fixtures.json           # Upcoming fixtures
│   └── players/                # Historical data for each player (JSON format)
└── models/                     # Trained model artifacts
```

## ⚙️ Configuration

Key settings can be adjusted in [config.py](config.py):

- **`BASE_URL`**: FPL API endpoint
- **`CACHE_EXPIRY_HOURS`**: How often to refresh cached data (default: 6 hours)
- **`ROLLING_WINDOWS`**: Window sizes for form calculation (default: 3, 5 gameweeks)
- **`MIN_MINUTES_THRESHOLD`**: Minimum minutes required to count a gameweek (default: 30)
- **`FEATURE_COLUMNS`**: ML features used for predictions
- **`MODEL_NAMES`**: Available prediction models

## 🔧 Development

### Adding New Models

1. Implement a new model class in [src/models.py](src/models.py)
2. Add to `MODEL_NAMES` in [config.py](config.py)
3. Update training pipeline in [src/predictor.py](src/predictor.py)

### Customizing Features

Modify feature engineering in [src/feature_engineering.py](src/feature_engineering.py) and update `FEATURE_COLUMNS` in [config.py](config.py)

## 📊 Data Sources

- **Official FPL API**: https://fantasy.premierleague.com/api/
  - Player statistics and rankings
  - Team information and fixtures
  - Real-time injury/availability status

## 📝 Notes

- Prediction accuracy improves as the season progresses and more historical data becomes available
- Early-season predictions (Gameweek 1-3) are less reliable due to limited recent form data
- Internet connection is required for live data updates
- Fixture Difficulty Ratings (FDR) are calculated based on opponent defensive strength
- The dashboard automatically refreshes data based on cache settings

## ⚖️ Disclaimer

This tool provides data-driven insights for FPL decision-making but is not a guarantee of future performance. Always do your own research and consider multiple factors when making FPL transfers and lineup decisions.

## 📄 License

This project is for educational and personal use in Fantasy Premier League management.

"""Configuration constants for the FPL ML Predictor."""

import os

# --- API Configuration ---
BASE_URL = "https://fantasy.premierleague.com/api/"
BOOTSTRAP_URL = f"{BASE_URL}bootstrap-static/"
FIXTURES_URL = f"{BASE_URL}fixtures/"
PLAYER_SUMMARY_URL = f"{BASE_URL}element-summary/{{player_id}}/"

# --- Cache Configuration ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PLAYERS_CACHE_DIR = os.path.join(DATA_DIR, "players")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
CACHE_EXPIRY_HOURS = 6

# --- Position Mapping ---
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_IDS = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

# --- Feature Engineering ---
FORM_WINDOW = 5
MIN_MINUTES_THRESHOLD = 30
ROLLING_WINDOWS = [3, 5]

# --- ML Feature Columns ---
FEATURE_COLUMNS = [
    "rolling_3gw_points",
    "rolling_5gw_points",
    "rolling_3gw_minutes",
    "rolling_5gw_minutes",
    "rolling_3gw_goals",
    "rolling_3gw_assists",
    "rolling_3gw_clean_sheets",
    "rolling_3gw_bonus",
    "rolling_5gw_goals",
    "rolling_5gw_assists",
    "rolling_5gw_clean_sheets",
    "rolling_5gw_bonus",
    "opponent_difficulty",
    "is_home",
    "position_encoded",
    "now_cost",
    "selected_by_percent",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "goals_per_90",
    "assists_per_90",
    "minutes_trend",
    "clean_sheet_probability",
    "xg_overperformance",
]

TARGET_COLUMN = "next_gw_points"

# --- Model Hyperparameters ---
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

NEURAL_NET_PARAMS = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10,
}

MODEL_NAMES = ["XGBoost", "Random Forest", "Neural Network", "Ensemble"]

# --- Player Photos ---
PLAYER_PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/250x250/p{code}.png"

# --- Fixture Difficulty Rating Colors ---
FDR_COLORS = {
    1: "#375523",  # Very easy - dark green
    2: "#01FC7A",  # Easy - green
    3: "#E7E7E7",  # Medium - grey
    4: "#FF1751",  # Hard - red
    5: "#861D46",  # Very hard - dark red
}
FDR_LABELS = {
    1: "Very Easy",
    2: "Easy",
    3: "Medium",
    4: "Hard",
    5: "Very Hard",
}

"""End-to-end prediction pipeline orchestrating data, features, and models."""

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_fetcher import (
    fetch_all_player_histories,
    fetch_bootstrap_static,
    fetch_fixtures,
    get_current_gameweek,
    get_next_gameweek,
)
from src.feature_engineering import (
    build_fixture_df,
    build_player_base_df,
    build_prediction_features,
    build_training_dataset,
)
from src.models import get_model


class FPLPredictor:
    """End-to-end FPL prediction pipeline.

    Usage:
        predictor = FPLPredictor()
        predictor.load_data()
        predictor.prepare_training_data()
        predictor.train_model("XGBoost")
        predictions = predictor.predict()
    """

    def __init__(self):
        self.bootstrap_data = None
        self.fixtures_data = None
        self.all_histories = None
        self.fixtures_df = None
        self.training_df = None
        self.prediction_df = None
        self.current_model = None
        self.current_gw = None
        self.next_gw = None
        self.players_df = None

    def load_data(self, force_refresh=False, progress_callback=None):
        """Fetch all data from FPL API (or cache)."""
        self.bootstrap_data = fetch_bootstrap_static(force_refresh)
        self.fixtures_data = fetch_fixtures(force_refresh)
        self.fixtures_df = build_fixture_df(self.fixtures_data)
        self.players_df = build_player_base_df(self.bootstrap_data)

        self.current_gw = get_current_gameweek(self.bootstrap_data)
        self.next_gw = get_next_gameweek(self.bootstrap_data)

        # Only fetch histories for active players with some minutes played
        active_players = self.players_df[
            (self.players_df["status"] != "u") & (self.players_df["minutes"] > 0)
        ]["id"].tolist()

        self.all_histories = fetch_all_player_histories(
            active_players, force_refresh, progress_callback
        )

    def prepare_training_data(self):
        """Build training dataset and prediction features."""
        self.training_df = build_training_dataset(
            self.bootstrap_data, self.all_histories, self.fixtures_df
        )
        self.prediction_df = build_prediction_features(
            self.bootstrap_data, self.all_histories, self.fixtures_df, self.next_gw
        )

    def train_model(self, model_name: str) -> dict:
        """Train the specified model with a temporal train/test split.

        Returns evaluation metrics dict (MAE, RMSE, R2).
        """
        self.current_model = get_model(model_name)

        # Temporal split: train on earlier gameweeks, test on later ones
        max_gw = self.training_df["gameweek"].max()
        split_gw = int(max_gw * 0.8)

        train_mask = self.training_df["gameweek"] <= split_gw
        test_mask = self.training_df["gameweek"] > split_gw

        X_train = self.training_df[train_mask].copy()
        y_train = self.training_df[train_mask][TARGET_COLUMN].copy()
        X_test = self.training_df[test_mask].copy()
        y_test = self.training_df[test_mask][TARGET_COLUMN].copy()

        # Clean NaN/Inf values
        for col in FEATURE_COLUMNS:
            for df in [X_train, X_test]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                df[col] = df[col].replace([np.inf, -np.inf], 0)

        self.current_model.train(X_train, y_train)
        metrics = self.current_model.evaluate(X_test, y_test)

        # Save trained model
        self.current_model.save()

        return metrics

    def predict(self) -> pd.DataFrame:
        """Generate predictions for the next gameweek.

        Returns DataFrame sorted by predicted_points descending.
        """
        if not self.current_model or not self.current_model.is_trained:
            raise RuntimeError("Model not trained. Call train_model() first.")

        pred_df = self.prediction_df.copy()

        # Clean feature columns
        for col in FEATURE_COLUMNS:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").fillna(0)
            pred_df[col] = pred_df[col].replace([np.inf, -np.inf], 0)

        predictions = self.current_model.predict(pred_df)

        # Include all display columns
        display_cols = [
            "player_id", "web_name", "first_name", "second_name",
            "team", "team_full", "position", "now_cost", "photo_url",
            "next_opponent", "next_opponent_full", "next_difficulty", "next_is_home",
            "total_points_season", "goals_season", "assists_season",
            "minutes_season", "form_value", "points_per_game_value",
            "status", "news", "chance_next_round",
            "prev_season_name", "prev_season_points", "prev_season_minutes",
            "prev_season_goals", "prev_season_assists", "prev_season_cs",
        ]
        available_cols = [c for c in display_cols if c in pred_df.columns]
        result = pred_df[available_cols].copy()
        result["predicted_points"] = np.round(predictions, 2)
        result = result.sort_values("predicted_points", ascending=False).reset_index(drop=True)

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the current trained model."""
        fi = self.current_model.get_feature_importance()
        df = pd.DataFrame(list(fi.items()), columns=["feature", "importance"])
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_model_metrics(self) -> dict:
        """Return the evaluation metrics of the current model."""
        return self.current_model.metrics

    def get_actual_vs_predicted(self) -> pd.DataFrame:
        """Generate actual vs predicted comparison on the test set."""
        max_gw = self.training_df["gameweek"].max()
        split_gw = int(max_gw * 0.8)
        test_data = self.training_df[self.training_df["gameweek"] > split_gw].copy()

        # Clean features
        for col in FEATURE_COLUMNS:
            test_data[col] = pd.to_numeric(test_data[col], errors="coerce").fillna(0)
            test_data[col] = test_data[col].replace([np.inf, -np.inf], 0)

        test_data["predicted"] = self.current_model.predict(test_data)
        test_data["actual"] = test_data[TARGET_COLUMN]

        return test_data[["web_name", "gameweek", "actual", "predicted"]]

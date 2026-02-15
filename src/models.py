"""ML models for FPL point prediction: XGBoost, Random Forest, Neural Network, Ensemble."""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    NEURAL_NET_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)


# ============================================================
#  BASE CLASS
# ============================================================
class BaseModel:
    """Abstract base for all FPL prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> dict:
        raise NotImplementedError

    def save(self, path: str = None):
        raise NotImplementedError

    def load(self, path: str = None):
        raise NotImplementedError

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        predictions = self.predict(X_test)
        self.metrics = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R2": r2_score(y_test, predictions),
        }
        return self.metrics


# ============================================================
#  XGBOOST MODEL
# ============================================================
class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.model = XGBRegressor(**XGBOOST_PARAMS)

    def train(self, X_train, y_train):
        split_idx = int(len(X_train) * 0.9)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

        self.model.fit(
            X_tr[FEATURE_COLUMNS],
            y_tr,
            eval_set=[(X_val[FEATURE_COLUMNS], y_val)],
            verbose=False,
        )
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X[FEATURE_COLUMNS])

    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return dict(zip(FEATURE_COLUMNS, importance))

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, "xgboost_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path=None):
        path = path or os.path.join(MODELS_DIR, "xgboost_model.pkl")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


# ============================================================
#  RANDOM FOREST MODEL
# ============================================================
class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)

    def train(self, X_train, y_train):
        self.model.fit(X_train[FEATURE_COLUMNS], y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X[FEATURE_COLUMNS])

    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return dict(zip(FEATURE_COLUMNS, importance))

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, "random_forest_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path=None):
        path = path or os.path.join(MODELS_DIR, "random_forest_model.pkl")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


# ============================================================
#  NEURAL NETWORK MODEL
# ============================================================
class FPLNeuralNet(nn.Module):
    """PyTorch neural network for FPL point prediction."""

    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class NeuralNetModel(BaseModel):
    def __init__(self):
        super().__init__("Neural Network")
        self.scaler = StandardScaler()
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, X_train, y_train):
        params = NEURAL_NET_PARAMS
        X_np = X_train[FEATURE_COLUMNS].values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_np)
        y_np = y_train.values.astype(np.float32)

        split_idx = int(len(X_scaled) * 0.9)
        X_tr = torch.FloatTensor(X_scaled[:split_idx]).to(self.device)
        y_tr = torch.FloatTensor(y_np[:split_idx]).to(self.device)
        X_val = torch.FloatTensor(X_scaled[split_idx:]).to(self.device)
        y_val = torch.FloatTensor(y_np[split_idx:]).to(self.device)

        self.net = FPLNeuralNet(
            input_dim=len(FEATURE_COLUMNS),
            hidden_layers=params["hidden_layers"],
            dropout_rate=params["dropout_rate"],
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=params["learning_rate"])
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(params["epochs"]):
            self.net.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.net(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= params["patience"]:
                    break

        if best_state:
            self.net.load_state_dict(best_state)
        self.is_trained = True

    def predict(self, X):
        X_np = X[FEATURE_COLUMNS].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_np)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.net.eval()
        with torch.no_grad():
            predictions = self.net(X_tensor).cpu().numpy()
        return predictions

    def get_feature_importance(self):
        first_layer_weights = list(self.net.network.parameters())[0]
        importance = torch.abs(first_layer_weights).mean(dim=0).detach().cpu().numpy()
        importance = importance / (importance.sum() + 1e-10)
        return dict(zip(FEATURE_COLUMNS, importance))

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, "neural_net_model.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state": self.net.state_dict(),
                "scaler": self.scaler,
                "input_dim": len(FEATURE_COLUMNS),
                "hidden_layers": NEURAL_NET_PARAMS["hidden_layers"],
                "dropout_rate": NEURAL_NET_PARAMS["dropout_rate"],
            },
            path,
        )

    def load(self, path=None):
        path = path or os.path.join(MODELS_DIR, "neural_net_model.pt")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.scaler = checkpoint["scaler"]
        self.net = FPLNeuralNet(
            checkpoint["input_dim"],
            checkpoint["hidden_layers"],
            checkpoint["dropout_rate"],
        ).to(self.device)
        self.net.load_state_dict(checkpoint["model_state"])
        self.is_trained = True


# ============================================================
#  ENSEMBLE MODEL
# ============================================================
class EnsembleModel(BaseModel):
    """Stacking ensemble combining XGBoost, Random Forest, and Neural Network.

    Uses Ridge regression as a meta-learner to learn optimal weights.
    """

    def __init__(self):
        super().__init__("Ensemble")
        self.base_models = {
            "xgboost": XGBoostModel(),
            "random_forest": RandomForestModel(),
            "neural_net": NeuralNetModel(),
        }
        self.weights = None
        self.meta_model = None

    def train(self, X_train, y_train):
        split_idx = int(len(X_train) * 0.8)
        X_base_train = X_train.iloc[:split_idx]
        y_base_train = y_train.iloc[:split_idx]
        X_meta_train = X_train.iloc[split_idx:]
        y_meta_train = y_train.iloc[split_idx:]

        # Train base models
        for model in self.base_models.values():
            model.train(X_base_train, y_base_train)

        # Generate meta-features
        meta_features = np.column_stack(
            [model.predict(X_meta_train) for model in self.base_models.values()]
        )

        # Train meta-learner
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y_meta_train)

        # Extract normalized weights
        raw_weights = self.meta_model.coef_
        self.weights = np.abs(raw_weights) / (np.abs(raw_weights).sum() + 1e-10)

        self.is_trained = True

    def predict(self, X):
        base_predictions = np.column_stack(
            [model.predict(X) for model in self.base_models.values()]
        )
        return self.meta_model.predict(base_predictions)

    def get_feature_importance(self):
        combined = {}
        for (name, model), weight in zip(self.base_models.items(), self.weights):
            fi = model.get_feature_importance()
            for feature, importance in fi.items():
                combined[feature] = combined.get(feature, 0) + importance * weight
        return combined

    def save(self, path=None):
        for model in self.base_models.values():
            model.save()
        path = path or os.path.join(MODELS_DIR, "ensemble_meta.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"meta_model": self.meta_model, "weights": self.weights}, f)

    def load(self, path=None):
        for model in self.base_models.values():
            model.load()
        path = path or os.path.join(MODELS_DIR, "ensemble_meta.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.meta_model = data["meta_model"]
        self.weights = data["weights"]
        self.is_trained = True


# ============================================================
#  FACTORY
# ============================================================
def get_model(model_name: str) -> BaseModel:
    """Factory function to get a model instance by name."""
    models = {
        "XGBoost": XGBoostModel,
        "Random Forest": RandomForestModel,
        "Neural Network": NeuralNetModel,
        "Ensemble": EnsembleModel,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]()

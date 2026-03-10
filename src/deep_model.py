"""
deep_model.py - Deep Learning Model Zoo for XAI
================================================
Provides ready-to-use deep learning models with sklearn-compatible wrappers
so they can be seamlessly plugged into the XAI Engine.

Models:
- MLPClassifier    : Multi-layer Perceptron (tabular data)
- CNNClassifier    : Convolutional Neural Network (image data)
- LSTMClassifier   : LSTM for time-series / sequential data
- TransformerBlock : Lightweight transformer encoder
- AutoEncoder      : Anomaly detection via reconstruction error

Author: Pranay M Mahendrakar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASE PYTORCH SKLEARN WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class BaseTorchClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for PyTorch classifiers.
    Provides fit / predict / predict_proba interface.
    """

    def __init__(self, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
                 verbose: bool = True):
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.verbose    = verbose
        self.model_     = None
        self.classes_   = None
        self.n_features_ = None

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_classes = len(self.classes_)
        self.n_features_ = X.shape[1]

        self.model_ = self._build_model(self.n_features_, n_classes).to(DEVICE)
        optimizer   = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion   = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y_enc, dtype=torch.long))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}  loss={total_loss/len(loader):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.array(X, dtype=np.float32)).to(DEVICE)
            logits = self.model_(X_t)
            proba  = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(indices)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MLP CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(BaseTorchClassifier):
    """
    Multi-layer Perceptron for tabular classification.
    Default architecture: [128, 64, 32] hidden units.
    """

    def __init__(self, hidden_dims: List[int] = None, dropout: float = 0.3,
                 epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
                 verbose: bool = True):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout     = dropout

    def _build_model(self, input_dim, output_dim):
        return _MLP(input_dim, self.hidden_dims, output_dim, self.dropout)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CNN CLASSIFIER (1D - for tabular / sequence data)
# ─────────────────────────────────────────────────────────────────────────────

class _CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(16)
        self.fc    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, features)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(x)


class CNNClassifier(BaseTorchClassifier):
    """1-D CNN for feature-based or sequence classification."""

    def _build_model(self, input_dim, output_dim):
        return _CNN1D(input_dim, output_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: (B, features) -> treat as single time-step
        x, _ = self.lstm(x.unsqueeze(1))   # (B, 1, hidden)
        return self.fc(x[:, -1, :])


class LSTMClassifier(BaseTorchClassifier):
    """LSTM for sequential / time-series tabular data."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

    def _build_model(self, input_dim, output_dim):
        return _LSTMNet(input_dim, self.hidden_size, output_dim, self.num_layers)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSFORMER CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class _TransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj    = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)  # (B, 1, d_model)
        x = self.encoder(x)
        return self.fc(x.squeeze(1))


class TransformerClassifier(BaseTorchClassifier):
    """Lightweight Transformer for tabular classification."""

    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.d_model    = d_model
        self.nhead      = nhead
        self.num_layers = num_layers

    def _build_model(self, input_dim, output_dim):
        return _TransformerNet(input_dim, output_dim,
                               self.d_model, self.nhead, self.num_layers)


# ─────────────────────────────────────────────────────────────────────────────
# 6. AUTOENCODER (Anomaly Detection)
# ─────────────────────────────────────────────────────────────────────────────

class _AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoder:
    """
    Unsupervised Autoencoder for anomaly detection.
    Uses reconstruction error as an anomaly score.
    """

    def __init__(self, latent_dim: int = 16, epochs: int = 50,
                 lr: float = 1e-3, batch_size: int = 64):
        self.latent_dim  = latent_dim
        self.epochs      = epochs
        self.lr          = lr
        self.batch_size  = batch_size
        self.model_      = None
        self.threshold_  = None

    def fit(self, X: np.ndarray) -> "AutoEncoder":
        X = torch.tensor(X, dtype=torch.float32)
        self.model_ = _AE(X.shape[1], self.latent_dim).to(DEVICE)
        optimizer   = optim.Adam(self.model_.parameters(), lr=self.lr)
        loader      = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            for (xb,) in loader:
                xb = xb.to(DEVICE)
                optimizer.zero_grad()
                loss = nn.MSELoss()(self.model_(xb), xb)
                loss.backward()
                optimizer.step()

        # Set threshold as 95th-percentile reconstruction error on training data
        errors = self.reconstruction_error(X.numpy())
        self.threshold_ = float(np.percentile(errors, 95))
        logger.info(f"Anomaly threshold set at: {self.threshold_:.4f}")
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_t  = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            recon = self.model_(X_t).cpu().numpy()
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomalies, 0 for normal."""
        errors = self.reconstruction_error(X)
        return (errors > self.threshold_).astype(int)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self.reconstruction_error(X)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mlp":         MLPClassifier,
    "cnn":         CNNClassifier,
    "lstm":        LSTMClassifier,
    "transformer": TransformerClassifier,
}


def get_model(name: str, **kwargs) -> BaseTorchClassifier:
    """Factory function to instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY)}")
    logger.info(f"Creating model: {name}")
    return MODEL_REGISTRY[name](**kwargs)

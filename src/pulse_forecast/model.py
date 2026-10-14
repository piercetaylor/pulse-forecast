"""A direct multi-step GRU forecaster and a linear baseline."""

import numpy as np
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(self, hidden_size: int, horizon: int):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1]) + x[:, -1, :]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    design = np.column_stack([x[..., 0], np.ones(len(x))]).astype(np.float64)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0  # Leave the intercept unpenalized.
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([x[..., 0], np.ones(len(x))]) @ coefficients

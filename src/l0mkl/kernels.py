"""Kernel construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler


def combine_kernels(theta: np.ndarray, kernels: Sequence[np.ndarray]) -> np.ndarray:
    """Return the weighted sum of a kernel list."""
    theta = np.asarray(theta, dtype=float).ravel()
    if len(theta) != len(kernels):
        raise ValueError("theta and kernels must have the same length.")
    return sum(theta[m] * np.asarray(kernels[m]) for m in range(len(theta)))


class FeatureLinearKernelizer:
    """Create one trace-normalized linear kernel per input feature.

    Standardization is fitted only on the training data. Evaluation kernels have shape ``(n_eval, n_train)`` and use the corresponding training trace.
    """

    def fit(self, X: np.ndarray) -> "FeatureLinearKernelizer":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        self.scaler_ = StandardScaler().fit(X)
        self.X_train_scaled_ = self.scaler_.transform(X)
        self.traces_ = np.sum(self.X_train_scaled_ ** 2, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def training_kernels(self) -> list[np.ndarray]:
        """Return square kernels for the fitted training features."""
        if not hasattr(self, "X_train_scaled_"):
            raise RuntimeError("Call fit before requesting kernels.")
        kernels = []
        for m in range(self.n_features_in_):
            column = self.X_train_scaled_[:, m:m + 1]
            kernel = column @ column.T
            if self.traces_[m] > 0:
                kernel = kernel / self.traces_[m]
            kernels.append(kernel)
        return kernels

    def transform(self, X: np.ndarray) -> list[np.ndarray]:
        """Return evaluation-vs-training kernels for new observations."""
        if not hasattr(self, "X_train_scaled_"):
            raise RuntimeError("Call fit before transform.")
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        kernels = []
        for m in range(self.n_features_in_):
            train_column = self.X_train_scaled_[:, m:m + 1]
            eval_column = X_scaled[:, m:m + 1]
            kernel = eval_column @ train_column.T
            if self.traces_[m] > 0:
                kernel = kernel / self.traces_[m]
            kernels.append(kernel)
        return kernels

    def fit_transform(self, X: np.ndarray) -> list[np.ndarray]:
        """Fit preprocessing and return training kernels."""
        return self.fit(X).training_kernels()

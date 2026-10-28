"""Synthetic data generator used in the paper."""

from __future__ import annotations

import numpy as np


def make_copy_classification(
    n_samples: int,
    n_features: int,
    *,
    snr: float = 2.0,
    informative_ratio: float = 0.25,
    corr: float = 0.8,
    label_noise: float = 0.5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the COPY binary-classification design from the paper.

    Redundant features are noisy copies of randomly selected informative features. Labels depend only on the informative features.
    """
    # --------------------------------------------------
    # 1. Validate the data-generation parameters
    # --------------------------------------------------
    if n_samples <= 0 or n_features <= 1:
        raise ValueError("n_samples must be positive and n_features > 1.")
    if snr <= 0:
        raise ValueError("snr must be positive.")
    if not 0 < informative_ratio <= 1:
        raise ValueError("informative_ratio must be in (0, 1].")
    if not 0 <= corr < 1:
        raise ValueError("corr must be in [0, 1).")

    # --------------------------------------------------
    # 2. Select the informative and redundant features
    # --------------------------------------------------
    rng = np.random.default_rng(random_state)

    n_informative = max(1, int(round(n_features * informative_ratio)))
    mask = np.zeros(n_features, dtype=int)
    informative = rng.choice(n_features, size=n_informative, replace=False)
    
    mask[informative] = 1
    redundant = np.flatnonzero(mask == 0)

    # --------------------------------------------------
    # 3. Generate the informative features
    # --------------------------------------------------
    X_signal = np.zeros((n_samples, n_features))
    X_informative = rng.normal(size=(n_samples, n_informative))
    X_signal[:, informative] = X_informative

    # --------------------------------------------------
    # 4. Generate noisy copies as redundant features
    # --------------------------------------------------
    if len(redundant):
        sources = rng.integers(0, n_informative, size=len(redundant))
        for local_index, feature_index in enumerate(redundant):
            X_signal[:, feature_index] = (
                np.sqrt(corr) * X_informative[:, sources[local_index]]
                + np.sqrt(1.0 - corr) * rng.normal(size=n_samples)
            )

    # --------------------------------------------------
    # 5. Add feature noise according to the requested SNR
    # --------------------------------------------------
    signal_variance = max(float(np.var(X_signal)), 1e-12)
    X = X_signal + np.sqrt(signal_variance / snr) * rng.normal(
        size=(n_samples, n_features)
    )

    # --------------------------------------------------
    # 6. Standardize every feature
    # --------------------------------------------------
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # --------------------------------------------------
    # 7. Generate labels from the informative features only
    # --------------------------------------------------
    coefficients = np.zeros(n_features)
    coefficients[informative] = rng.normal(size=n_informative)

    scores = X @ coefficients + label_noise * rng.normal(size=n_samples)
    
    y = np.sign(scores)
    y[y == 0] = 1

    # --------------------------------------------------
    # 8. Enforce roughly balanced classes when necessary
    # --------------------------------------------------
    if abs(float(y.mean())) > 0.2:
        y = np.where(scores > np.median(scores), 1, -1)

    # Return the feature matrix, labels, and true informative support.
    return X, y.astype(int), mask

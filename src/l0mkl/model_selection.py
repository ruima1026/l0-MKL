"""Cross-validation and parameter-sweep helpers."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils.validation import check_X_y


class MKLGridSearchCV:
    """Small, explicit grid search for :class:`L0MKLClassifier`.

    Preprocessing and kernels are fitted independently inside every fold, so validation data do not leak into scaling or kernel construction.
    """

    def __init__(
        self,
        estimator,
        param_grid: dict | list[dict],
        *,
        cv: int = 3,
        random_state: int = 0,
        refit: bool = True,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.random_state = random_state
        self.refit = refit
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MKLGridSearchCV":
        X, y = check_X_y(X, y, dtype=float)
        splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        rows = []
        best_score = -np.inf
        best_params = None
        for params in ParameterGrid(self.param_grid):
            scores = []
            start = perf_counter()
            for train_index, validation_index in splitter.split(X, y):
                candidate = clone(self.estimator).set_params(**params)
                candidate.fit(X[train_index], y[train_index])
                scores.append(candidate.score(X[validation_index], y[validation_index]))
            row = {
                "params": params,
                "mean_validation_accuracy": float(np.mean(scores)),
                "std_validation_accuracy": float(np.std(scores, ddof=1)),
                "fit_seconds": perf_counter() - start,
                "fold_scores": scores,
            }
            rows.append(row)
            if self.verbose:
                print(params, row["mean_validation_accuracy"])
            # ParameterGrid is deterministic; exact ties keep the first entry.
            if row["mean_validation_accuracy"] > best_score:
                best_score = row["mean_validation_accuracy"]
                best_params = params

        self.cv_results_ = pd.DataFrame(rows)
        self.best_score_ = float(best_score)
        self.best_params_ = dict(best_params)
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(
                **self.best_params_
            ).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "best_estimator_"):
            raise RuntimeError("Set refit=True and call fit before predict.")
        return self.best_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))


def parameter_sweep(
    estimator,
    param_grid: dict | list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> pd.DataFrame:
    """Fit every parameter configuration and report validation performance."""
    rows = []
    for params in ParameterGrid(param_grid):
        candidate = clone(estimator).set_params(**params)
        start = perf_counter()
        candidate.fit(X_train, y_train)
        rows.append({
            **params,
            "validation_accuracy": candidate.score(X_validation, y_validation),
            "n_selected_kernels": int(len(candidate.support_)),
            "l1_mass": float(candidate.theta_.sum()),
            "relative_gap": float(candidate.diagnostics_.relative_gap),
            "converged": bool(candidate.diagnostics_.converged),
            "fit_seconds": perf_counter() - start,
        })
    return pd.DataFrame(rows)

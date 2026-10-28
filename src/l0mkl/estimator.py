"""Scikit-learn style estimators for sparse multiple-kernel learning."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .kernels import FeatureLinearKernelizer, combine_kernels
from .oa import solve_oa


class L0MKLClassifier(ClassifierMixin, BaseEstimator):
    """Sparse multiple-kernel SVM classifier.

    By default, each input feature defines one trace-normalized linear kernel. 
    """

    def __init__(
        self,
        formulation: str = "l0l1",
        scale_control: str = "constraint",
        gamma: float = 1.0,
        mu1: float = 1.0,
        mu2: float = 1.0,
        C: float = 100.0,
        kernel_budget: float = 1.0,
        theta_upper_bound: float = 1.0,
        oa_gap: float = 1e-2,
        max_iter: int = 1000,
        time_limit: float | None = None,
        master_mip_gap: float | None = None,
        threads: int | None = None,
        support_tol: float = 1e-6,
        verbose: int = 0,
    ) -> None:
        self.formulation = formulation
        self.scale_control = scale_control
        self.gamma = gamma
        self.mu1 = mu1
        self.mu2 = mu2
        self.C = C
        self.kernel_budget = kernel_budget
        self.theta_upper_bound = theta_upper_bound
        self.oa_gap = oa_gap
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.master_mip_gap = master_mip_gap
        self.threads = threads
        self.support_tol = support_tol
        self.verbose = verbose

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        self.label_encoder_ = LabelEncoder().fit(y)
        if len(self.label_encoder_.classes_) != 2:
            raise ValueError("l0MKL currently supports binary classification only.")
        self.classes_ = self.label_encoder_.classes_
        encoded = self.label_encoder_.transform(y)
        return np.where(encoded == 0, -1.0, 1.0)

    def _fit_from_kernels(
        self, kernels: Sequence[np.ndarray], y: np.ndarray
    ) -> "L0MKLClassifier":
        y = np.asarray(y).ravel()
        y_internal = self._encode_labels(y)
        solution = solve_oa(
            kernels,
            y_internal,
            formulation=self.formulation,
            scale_control=self.scale_control,
            gamma=self.gamma,
            mu1=self.mu1,
            mu2=self.mu2,
            C=self.C,
            kernel_budget=self.kernel_budget,
            theta_upper_bound=self.theta_upper_bound,
            oa_gap=self.oa_gap,
            max_iter=self.max_iter,
            time_limit=self.time_limit,
            master_mip_gap=self.master_mip_gap,
            threads=self.threads,
            verbose=self.verbose,
        )
        self.theta_ = solution.theta
        self.support_ = np.flatnonzero(self.theta_ > self.support_tol)
        self.diagnostics_ = solution.diagnostics
        self.classifier_ = SVC(C=self.C, kernel="precomputed")
        self.classifier_.fit(combine_kernels(self.theta_, kernels), y_internal)
        self.n_kernels_ = len(kernels)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L0MKLClassifier":
        """Fit from raw features using one linear kernel per feature."""
        X, y = check_X_y(X, y, dtype=float)
        self.kernelizer_ = FeatureLinearKernelizer()
        training_kernels = self.kernelizer_.fit_transform(X)
        self.input_mode_ = "features"
        return self._fit_from_kernels(training_kernels, y)

    def fit_kernels(
        self, kernels: Sequence[np.ndarray], y: np.ndarray
    ) -> "L0MKLClassifier":
        """Fit from a user-provided list of square training kernels."""
        self.input_mode_ = "precomputed"
        return self._fit_from_kernels(kernels, y)

    def _predict_internal(self, evaluation_kernels: Sequence[np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["theta_", "classifier_", "label_encoder_"])
        if len(evaluation_kernels) != self.n_kernels_:
            raise ValueError(
                f"Expected {self.n_kernels_} evaluation kernels; "
                f"received {len(evaluation_kernels)}."
            )
        prediction = self.classifier_.predict(
            combine_kernels(self.theta_, evaluation_kernels)
        )
        encoded = (prediction > 0).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels from raw features."""
        check_is_fitted(self, ["input_mode_", "kernelizer_"])
        if self.input_mode_ != "features":
            raise RuntimeError("Use predict_kernels after fit_kernels.")
        X = check_array(X, dtype=float)
        return self._predict_internal(self.kernelizer_.transform(X))

    def predict_kernels(self, kernels: Sequence[np.ndarray]) -> np.ndarray:
        """Predict from evaluation-vs-training kernel matrices."""
        check_is_fitted(self, ["input_mode_"])
        if self.input_mode_ != "precomputed":
            raise RuntimeError("Use predict after fitting raw features.")
        return self._predict_internal(kernels)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


class L1MKLClassifier(L0MKLClassifier):
    """Convenience interface for the continuous l1-MKL baseline.

    The default is the conventional unit kernel-weight budget. Set ``scale_control='penalty'`` and provide ``mu1`` for its penalized counterpart.
    """

    def __init__(
        self,
        scale_control: str = "constraint",
        mu1: float = 1.0,
        C: float = 100.0,
        kernel_budget: float = 1.0,
        theta_upper_bound: float = 1.0,
        oa_gap: float = 1e-2,
        max_iter: int = 1000,
        time_limit: float | None = None,
        master_mip_gap: float | None = None,
        threads: int | None = None,
        support_tol: float = 1e-6,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            formulation="l1",
            scale_control=scale_control,
            gamma=0.0,
            mu1=mu1,
            mu2=0.0,
            C=C,
            kernel_budget=kernel_budget,
            theta_upper_bound=theta_upper_bound,
            oa_gap=oa_gap,
            max_iter=max_iter,
            time_limit=time_limit,
            master_mip_gap=master_mip_gap,
            threads=threads,
            support_tol=support_tol,
            verbose=verbose,
        )

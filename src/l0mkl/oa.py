"""Outer-approximation solver for sparse multiple-kernel SVMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sklearn.svm import SVC


_FORMULATIONS = {"l0", "l0l1", "l0l2", "l1"}
_SCALE_CONTROLS = {"constraint", "penalty"}


@dataclass
class OADiagnostics:
    """Numerical information returned by the OA algorithm."""

    converged: bool
    stop_reason: str
    iterations: int
    lower_bound: float
    upper_bound: float
    relative_gap: float
    elapsed_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class OASolution:
    """Kernel weights and diagnostics from one OA solve."""

    theta: np.ndarray
    diagnostics: OADiagnostics


def _validate_kernel_list(
    kernels: Sequence[np.ndarray], y: np.ndarray
) -> list[np.ndarray]:
    """Validate and materialize a list of square training kernels."""
    if not kernels:
        raise ValueError("At least one kernel matrix is required.")
    n = len(y)
    checked: list[np.ndarray] = []
    for index, kernel in enumerate(kernels):
        matrix = np.asarray(kernel, dtype=float)
        if matrix.shape != (n, n):
            raise ValueError(
                f"Kernel {index} has shape {matrix.shape}; expected {(n, n)}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Kernel {index} contains non-finite values.")
        if not np.allclose(matrix, matrix.T, atol=1e-8, rtol=1e-7):
            raise ValueError(f"Training kernel {index} is not symmetric.")
        checked.append(matrix)
    return checked


def solve_svm_subproblem(
    theta: np.ndarray,
    kernels: Sequence[np.ndarray],
    y: np.ndarray,
    C: float,
) -> tuple[float, np.ndarray, float]:
    """Solve the SVM subproblem and return the OA cut components.

    This is the public-repository counterpart of
    ``SVM.solve_svm_subproblem``. Labels must already be encoded as ``-1`` and
    ``+1``.
    """
    n_samples = len(y)
    n_kernels = len(kernels)
    combined_kernel = sum(theta[m] * kernels[m] for m in range(n_kernels))

    classifier = SVC(C=C, kernel="precomputed")
    classifier.fit(combined_kernel, y)

    alpha = np.zeros(n_samples)
    alpha[classifier.support_] = np.abs(classifier.dual_coef_.ravel())
    sum_alpha = float(alpha.sum())

    # Equivalent to alpha.T @ Y @ K_m @ Y @ alpha, without constructing Y.
    signed_alpha = alpha * y
    gradient = np.empty(n_kernels)
    for m, kernel in enumerate(kernels):
        gradient[m] = -0.5 * signed_alpha @ kernel @ signed_alpha

    objective = sum_alpha + float(theta @ gradient)
    return sum_alpha, gradient, objective


def solve_oa(
    kernels: Sequence[np.ndarray],
    y: np.ndarray,
    *,
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
    verbose: int = 0,
) -> OASolution:
    """Solve one sparse-MKL formulation with outer approximation.

    Parameters match the formulations. ``l0l1`` and ``l1`` can use either a unit-budget constraint or a scale penalty. ``l0l2`` is the penalized formulation. ``l0`` uses only the coefficient-wise box bound.
    """
    formulation = formulation.lower()
    scale_control = scale_control.lower()
    if formulation not in _FORMULATIONS:
        raise ValueError(f"Unknown formulation {formulation!r}.")
    if scale_control not in _SCALE_CONTROLS:
        raise ValueError(f"Unknown scale_control {scale_control!r}.")
    if formulation == "l0l2" and scale_control != "penalty":
        raise ValueError("The paper's l0-l2 formulation uses scale_control='penalty'.")
    if C <= 0 or gamma < 0 or mu1 < 0 or mu2 < 0:
        raise ValueError("C must be positive and penalties must be nonnegative.")
    if oa_gap <= 0 or max_iter <= 0:
        raise ValueError("oa_gap and max_iter must be positive.")

    y = np.asarray(y, dtype=float).ravel()
    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("The OA solver requires labels encoded as {-1, +1}.")
    kernel_list = _validate_kernel_list(kernels, y)
    n_kernels = len(kernel_list)

    # ------------------------------------------------------------------
    # 1. Build the formulation-specific master problem.
    # ------------------------------------------------------------------
    model = gp.Model("l0mkl_master")
    model.Params.OutputFlag = int(verbose > 1)
    if master_mip_gap is not None:
        model.Params.MIPGap = float(master_mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    theta = model.addVars(
        n_kernels, lb=0.0, ub=theta_upper_bound, name="theta"
    )
    
    eta = model.addVar(lb=-1e10, name="eta")
    uses_l0 = formulation in {"l0", "l0l1", "l0l2"}
    z = model.addVars(n_kernels, vtype=GRB.BINARY, name="z") if uses_l0 else None
    if uses_l0:
        for m in range(n_kernels):
            model.addConstr(theta[m] <= theta_upper_bound * z[m])

    if scale_control == "constraint" and formulation in {"l0l1", "l1"}:
        model.addConstr(
            gp.quicksum(theta[m] for m in range(n_kernels)) <= kernel_budget,
            name="kernel_weight_budget",
        )

    regularizer = gp.LinExpr(0.0)
    if uses_l0:
        regularizer += gamma * gp.quicksum(z[m] for m in range(n_kernels))
    if scale_control == "penalty" and formulation in {"l0l1", "l1"}:
        regularizer += mu1 * gp.quicksum(theta[m] for m in range(n_kernels))

    if formulation == "l0l2":
        objective = eta + regularizer + mu2 * gp.quicksum(
            theta[m] * theta[m] for m in range(n_kernels)
        )
    else:
        objective = eta + regularizer
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()

    best_upper_bound = float("inf")
    best_theta = np.zeros(n_kernels)
    lower_bound = 0.0
    relative_gap = float("inf")
    history: list[dict[str, float]] = []
    start = perf_counter()
    stop_reason = "max_iter"

    # ------------------------------------------------------------------
    # 2. OA loop: master -> SVM -> bounds/gap -> cut.
    # ------------------------------------------------------------------
    for iteration in range(max_iter):
        if time_limit is not None:
            remaining = time_limit - (perf_counter() - start)
            if remaining <= 0:
                stop_reason = "time_limit"
                break
            model.Params.TimeLimit = max(remaining, 1e-3)

        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"Master problem has no solution (status={model.Status}).")
        theta_value = np.array([theta[m].X for m in range(n_kernels)])

        sum_alpha, gradient, svm_value = solve_svm_subproblem(
            theta_value, kernel_list, y, C
        )

        lower_bound = float(model.ObjVal)
        regularization_value = 0.0
        if uses_l0:
            regularization_value += gamma * sum(z[m].X for m in range(n_kernels))
        if scale_control == "penalty" and formulation in {"l0l1", "l1"}:
            regularization_value += mu1 * float(theta_value.sum())
        if formulation == "l0l2":
            regularization_value += mu2 * float(theta_value @ theta_value)
        current_upper_bound = svm_value + regularization_value

        if current_upper_bound < best_upper_bound:
            best_upper_bound = current_upper_bound
            best_theta = theta_value.copy()

        relative_gap = max(
            0.0,
            (best_upper_bound - lower_bound) / (abs(best_upper_bound) + 1e-8),
        )
        elapsed = perf_counter() - start
        history.append({
            "iteration": float(iteration),
            "lower_bound": lower_bound,
            "upper_bound": best_upper_bound,
            "relative_gap": relative_gap,
            "elapsed_seconds": elapsed,
        })
        if verbose:
            print(
                f"OA iteration {iteration}: LB={lower_bound:.6f}, "
                f"UB={best_upper_bound:.6f}, gap={relative_gap:.6f}, "
                f"elapsed={elapsed:.2f}s"
            )
        if relative_gap <= oa_gap:
            stop_reason = "oa_gap"
            break

        cut = sum_alpha + gp.quicksum(
            float(gradient[m]) * theta[m] for m in range(n_kernels)
        )
        model.addConstr(eta >= cut)
        model.update()

    elapsed = perf_counter() - start
    diagnostics = OADiagnostics(
        converged=relative_gap <= oa_gap,
        stop_reason=stop_reason,
        iterations=len(history),
        lower_bound=lower_bound,
        upper_bound=best_upper_bound,
        relative_gap=relative_gap,
        elapsed_seconds=elapsed,
        history=history,
    )
    return OASolution(theta=best_theta, diagnostics=diagnostics)

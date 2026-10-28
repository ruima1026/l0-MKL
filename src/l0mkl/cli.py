"""Command-line entry point for small reproducible COPY experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .synthetic_data import make_copy_classification
from .estimator import L0MKLClassifier
from .model_selection import MKLGridSearchCV, parameter_sweep


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _grid(args: argparse.Namespace) -> dict[str, list[float]]:
    if args.formulation == "l0":
        return {"gamma": args.gamma_grid}
    if args.formulation == "l0l1":
        grid = {"gamma": args.gamma_grid}
        if args.scale_control == "penalty":
            grid["mu1"] = args.mu1_grid
        return grid
    if args.formulation == "l0l2":
        return {"gamma": args.gamma_grid, "mu2": args.mu2_grid}
    if args.scale_control == "penalty":
        return {"mu1": args.mu1_grid}
    return {}


def _base_estimator(args: argparse.Namespace) -> L0MKLClassifier:
    return L0MKLClassifier(
        formulation=args.formulation,
        scale_control=args.scale_control,
        gamma=args.gamma,
        mu1=args.mu1,
        mu2=args.mu2,
        C=args.C,
        oa_gap=args.oa_gap,
        max_iter=args.max_iter,
        time_limit=args.time_limit,
        threads=args.threads,
        verbose=args.verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run l0MKL on the paper's COPY synthetic dataset."
    )
    parser.add_argument("--mode", choices=("fit", "cv", "sweep"), default="fit")
    parser.add_argument(
        "--formulation", choices=("l0", "l0l1", "l0l2", "l1"), default="l0l1"
    )
    parser.add_argument(
        "--scale-control", choices=("constraint", "penalty"), default="constraint"
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--informative-ratio", type=float, default=0.2)
    parser.add_argument("--corr", type=float, default=0.2)
    parser.add_argument("--snr", type=float, default=2.0)
    parser.add_argument("--label-noise", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mu1", type=float, default=1.0)
    parser.add_argument("--mu2", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=100.0)
    parser.add_argument("--gamma-grid", type=_float_list, default=[0.1, 1.0, 10.0])
    parser.add_argument("--mu1-grid", type=_float_list, default=[1.0, 10.0])
    parser.add_argument("--mu2-grid", type=_float_list, default=[1.0, 10.0])
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--oa-gap", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument("--output", type=Path, default=Path("results/copy_example"))
    args = parser.parse_args()

    X, y, mask = make_copy_classification(
        args.n_samples,
        args.n_features,
        snr=args.snr,
        informative_ratio=args.informative_ratio,
        corr=args.corr,
        label_noise=args.label_noise,
        random_state=args.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_state
    )
    estimator = _base_estimator(args)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "fit":
        fitted = estimator.fit(X_train, y_train)
    elif args.mode == "cv":
        grid = _grid(args)
        if not grid:
            raise ValueError("The constrained l1 model has no sparsity grid to tune.")
        search = MKLGridSearchCV(
            estimator, grid, cv=args.cv, random_state=args.random_state,
            refit=True, verbose=args.verbose,
        ).fit(X_train, y_train)
        search.cv_results_.to_csv(args.output / "cv_results.csv", index=False)
        (args.output / "best_params.json").write_text(
            json.dumps(search.best_params_, indent=2) + "\n"
        )
        fitted = search.best_estimator_
    else:
        X_fit, X_validation, y_fit, y_validation = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train,
            random_state=args.random_state,
        )
        results = parameter_sweep(
            estimator, _grid(args), X_fit, y_fit, X_validation, y_validation
        )
        results.to_csv(args.output / "sweep_results.csv", index=False)
        print(results.to_string(index=False))
        return

    np.savetxt(args.output / "theta.csv", fitted.theta_, delimiter=",")
    np.savetxt(args.output / "informative_mask.csv", mask, delimiter=",", fmt="%d")
    (args.output / "diagnostics.json").write_text(
        json.dumps(asdict(fitted.diagnostics_), indent=2) + "\n"
    )
    metrics = {
        "test_accuracy": fitted.score(X_test, y_test),
        "n_selected_kernels": int(len(fitted.support_)),
        "selected_kernels": fitted.support_.tolist(),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

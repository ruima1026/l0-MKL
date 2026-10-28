# l0MKL

Research implementation accompanying **“On l0-Regularized Sparse Multiple Kernel Learning for Feature Selection.”** 
The package learns nonnegative kernel weights for binary SVM classification and supports the four models used in the paper:

| `formulation` | Kernel-weight model | Scale control |
|---|---|---|
| `l0` | l0-l-infinity | coefficient-wise bound |
| `l0l1` | l0-l1 | constraint or penalty |
| `l0l2` | l0-l2 | squared-l2 penalty |
| `l1` | continuous l1-MKL baseline | constraint or penalty |

The l0 models are solved by the paper's outer-approximation (OA) algorithm: each iteration solves a Gurobi master problem, evaluates its kernel weights by solving an SVM subproblem, updates valid lower and upper bounds, and adds a new cut. 

The continuous l1-MKL baseline is our implementation of the SILP-based l1-MKL method. 
It follows the same cutting-plane framework as the sparse l0 formulations, but optimizes only continuous kernel-weight variables and therefore contains no binary kernel-selection variables.

## Requirements

- Python 3.10 or newer
- NumPy, pandas, scikit-learn
- `gurobipy` and a working Gurobi license

Gurobi is proprietary software and is not distributed with this repository.
Academic users can obtain a license from Gurobi. Verify the installation with:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate l0mkl
pip install -e .
```

Or in an existing environment:

```bash
pip install -e .
```

## Main parameters

- `formulation`: `l0`, `l0l1`, `l0l2`, or `l1`.
- `scale_control`: `constraint` or `penalty` where supported.
- `gamma`: coefficient of the l0 support penalty.
- `mu1`, `mu2`: l1 and squared-l2 scale penalties.
- `C`: SVM regularization parameter. 
- `oa_gap`: relative OA stopping tolerance. Paper experiments use `0.01`.
- `max_iter`: maximum number of OA iterations. Paper experiments use `1000`.
- `support_tol`: a kernel is selected when `theta > 1e-6` by default.

The fitted estimator exposes `theta_`, `support_`, `classifier_`, and
`diagnostics_`. The latter records bounds, gap, stopping reason, runtime, and
the complete OA history.


## Quick start: l0-l1 MKL

```python
from sklearn.model_selection import train_test_split
from l0mkl import L0MKLClassifier, make_copy_classification

X, y, true_support = make_copy_classification(
    n_samples=100,
    n_features=10,
    informative_ratio=0.2,
    corr=0.2,
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

model = L0MKLClassifier(
    formulation="l0l1",
    scale_control="constraint",
    gamma=1.0,
    C=100.0,
    oa_gap=0.01,
    max_iter=1000,
)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.theta_)
print(model.support_)
print(model.diagnostics_)
```

For raw features, the estimator follows the paper: preprocessing is fitted on
the training set, and every standardized feature produces one
trace-normalized linear kernel.

## l1-MKL interface

The conventional l1-MKL baseline has a dedicated estimator:

```python
from l0mkl import L1MKLClassifier

model = L1MKLClassifier(
    C=100.0,
    kernel_budget=1.0,       # ||theta||_1 <= 1
    oa_gap=0.01,
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

The penalized l1 model is obtained with:

```python
model = L1MKLClassifier(scale_control="penalty", mu1=10.0, C=100.0)
```

## User-provided kernels

Training kernels must be square arrays of shape `(n_train, n_train)`.
Evaluation kernels must have shape `(n_eval, n_train)`.

```python
model = L0MKLClassifier(formulation="l0l1", gamma=1.0)
model.fit_kernels(K_train_list, y_train)
prediction = model.predict_kernels(K_test_list)
```

All lists must use the same kernel order. `model.theta_[m]` is the weight of
the `m`-th supplied kernel.

## Cross-validation

`MKLGridSearchCV` performs stratified CV entirely inside the data passed to
`fit`. Feature scaling and kernel construction are refitted independently in
every fold.

```python
from l0mkl import MKLGridSearchCV

search = MKLGridSearchCV(
    L0MKLClassifier(formulation="l0l1", C=100.0),
    {"gamma": [0.01, 0.1, 1.0, 10.0]},
    cv=3,
    random_state=0,
).fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
print(search.score(X_test, y_test))
```

## Parameter sweeps

`parameter_sweep` evaluates every grid point on a user-provided validation set and returns a pandas DataFrame containing accuracy, sparsity, kernel mass, runtime, convergence status, and final OA gap. 

See [`examples/cv_and_sweep.py`](examples/cv_and_sweep.py).

## Command-line COPY experiment

One command exposes the data generator, fitting, CV, and parameter sweep:

```bash
# One constrained l0-l1 fit
l0mkl-copy --mode fit --formulation l0l1 --gamma 1 --C 100

# Three-fold CV over gamma
l0mkl-copy --mode cv --formulation l0l1 \
  --gamma-grid 0.01,0.1,1,10 --cv 3 --output results/cv

# Validation-set solution path
l0mkl-copy --mode sweep --formulation l0l1 \
  --gamma-grid 0.1,1,10,100 --output results/sweep
```



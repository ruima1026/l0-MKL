"""Minimal constrained l0-l1 MKL example."""

from sklearn.model_selection import train_test_split

from l0mkl import L0MKLClassifier, make_copy_classification


X, y, informative = make_copy_classification(
    n_samples=80,
    n_features=8,
    informative_ratio=0.25,
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

print("test accuracy:", model.score(X_test, y_test))
print("theta:", model.theta_)
print("selected kernels:", model.support_)
print("true informative mask:", informative)
print("OA diagnostics:", model.diagnostics_)

"""Three-fold CV and explicit validation-path examples."""

from sklearn.model_selection import train_test_split

from l0mkl import (
    L0MKLClassifier,
    MKLGridSearchCV,
    make_copy_classification,
    parameter_sweep,
)


X, y, _ = make_copy_classification(80, 8, corr=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
base = L0MKLClassifier(
    formulation="l0l1", scale_control="constraint", C=100.0,
    oa_gap=0.05, max_iter=200,
)

search = MKLGridSearchCV(
    base,
    {"gamma": [0.1, 1.0]},
    cv=3,
    random_state=0,
).fit(X_train, y_train)
print("best parameters:", search.best_params_)
print("test accuracy after refit:", search.score(X_test, y_test))

# A sweep reports every point; it does not select parameters using X_test.
X_fit, X_validation, y_fit, y_validation = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
)
path = parameter_sweep(
    base,
    {"gamma": [0.1, 1.0, 10.0]},
    X_fit,
    y_fit,
    X_validation,
    y_validation,
)
print(path)

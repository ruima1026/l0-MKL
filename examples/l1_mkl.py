"""User-facing l1-MKL example."""

from sklearn.model_selection import train_test_split

from l0mkl import L1MKLClassifier, make_copy_classification


X, y, _ = make_copy_classification(80, 8, corr=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# Conventional l1-MKL under the unit kernel-weight budget ||theta||_1 <= 1.
model = L1MKLClassifier(C=100.0, oa_gap=0.01)
model.fit(X_train, y_train)

print("test accuracy:", model.score(X_test, y_test))
print("kernel weights:", model.theta_)
print("selected kernels (theta > 1e-6):", model.support_)

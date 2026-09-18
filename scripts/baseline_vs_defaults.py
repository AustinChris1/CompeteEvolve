"""Accuracy of each shipped baseline versus scikit-learn defaults on the same split.

    python scripts/baseline_vs_defaults.py
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

import sklearn_algorithms as sa  # noqa: E402

warnings.filterwarnings("ignore")

PAIRS = {
    "logistic_regression": (
        LogisticRegression(max_iter=5, C=0.005, solver="lbfgs", random_state=42),
        LogisticRegression(max_iter=1000),
    ),
    "knn": (KNeighborsClassifier(n_neighbors=75, algorithm="brute"), KNeighborsClassifier()),
    "decision_tree": (DecisionTreeClassifier(max_depth=1, random_state=42), DecisionTreeClassifier(random_state=42)),
    "random_forest": (
        RandomForestClassifier(n_estimators=1, max_depth=1, n_jobs=1, random_state=42),
        RandomForestClassifier(random_state=42),
    ),
    "svm": (SVC(kernel="sigmoid", C=0.01, random_state=42), SVC()),
}


def main() -> None:
    df = pd.read_csv(io.StringIO(sa.dataset_csv("logistic_regression")))
    X, y = df.drop(columns=["target"]), df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"test rows: {len(yte)} (accuracy resolution {1 / len(yte):.3f})")
    print(f"{'algorithm':<20}{'repo baseline':>15}{'sklearn default':>17}")
    for name, (crippled, default) in PAIRS.items():
        a = accuracy_score(yte, crippled.fit(Xtr, ytr).predict(Xte))
        b = accuracy_score(yte, default.fit(Xtr, ytr).predict(Xte))
        print(f"{name:<20}{a:>15.4f}{b:>17.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sklearn.datasets as sk_datasets


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str                              # registry key / "algorithm" tag used everywhere else
    display_name: str                      # human-readable name for prompts/reports
    dataset_name: str                      # human-readable dataset name
    dataset_loader: Callable[[], object]   # e.g. sklearn.datasets.load_iris
    baseline_code: str                     # annotated `def run(data_path) -> dict`


def _dataset_to_csv(loader: Callable[[], object]) -> str:
    """Turns a scikit-learn `Bunch` into one CSV: feature columns
    followed by a `target` column."""
    bunch = loader()
    feature_names = [
        str(n).replace(" ", "_").replace("(", "").replace(")", "") for n in bunch.feature_names
    ]
    lines = [",".join(feature_names + ["target"])]
    for row, target in zip(bunch.data, bunch.target):
        lines.append(",".join([repr(float(v)) for v in row] + [str(int(target))]))
    return "\n".join(lines) + "\n"


def dataset_csv(algorithm: str) -> str:
    """Builds the single dataset CSV for `algorithm`, ready to be
    remembered under category='dataset'."""
    return _dataset_to_csv(get_spec(algorithm).dataset_loader)


# ---------------------------------------------------------------------
# Baselines. Every one returns the rich dict the sandbox prefers, so
# TT/TI/L are measured rather than inferred.
# ---------------------------------------------------------------------

_PREAMBLE = '''\
def run(data_path: str) -> dict:
    import time
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
{imports}

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

{split}

{scaling}
    model = {model}

    # Training time (TT) is measured around fit() only.
    _t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - _t0

    # Inference time (TI) is measured around predict() only.
    _t1 = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - _t1

    accuracy = float(accuracy_score(y_test, predictions))

    # TRAIT: loss (L). log_loss needs calibrated probabilities; models
    # without predict_proba fall back to the error rate.
    try:
        probabilities = model.predict_proba(X_test)
        loss = float(log_loss(y_test, probabilities, labels=sorted(set(y))))
    except Exception:
        loss = 1.0 - accuracy

    return {{
        "accuracy": accuracy,
        "loss": loss,
        "training_time": training_time,
        "inference_time": inference_time,
    }}
'''

_DEFAULT_SPLIT = '''\
    # TRAIT: model accuracy, training time. The split itself is
    # evolvable — test_size, random_state and stratification all change
    # how much data the model sees and how stable the score is.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )'''

_NO_SCALING = '''\
    # TRAIT: model accuracy. No feature scaling is applied. For
    # distance- and margin-based models this is a significant handicap;
    # introducing a scaler here is a legitimate evolution.
'''

_SCALING_HOOK = '''\
    # TRAIT: model accuracy, training time. Preprocessing goes here.
    # Nothing is applied yet — adding a scaler, feature selection or
    # dimensionality reduction is an evolvable improvement.
'''


_LOGISTIC_REGRESSION_CODE = _PREAMBLE.format(
    imports="    from sklearn.linear_model import LogisticRegression",
    split=_DEFAULT_SPLIT,
    scaling=_NO_SCALING,
    model=(
        "LogisticRegression(\n"
        "        # TRAIT: model accuracy, training time, computational cost.\n"
        "        # Deliberately under-tuned: stops long before convergence and\n"
        "        # over-regularises, so it underfits badly (~0.70 on Iris).\n"
        "        max_iter=5,\n"
        "        C=0.005,\n"
        "        solver='lbfgs',\n"
        "        random_state=42,\n"
        "    )"
    ),
)

_KNN_CODE = _PREAMBLE.format(
    imports="    from sklearn.neighbors import KNeighborsClassifier",
    split=_DEFAULT_SPLIT,
    scaling=_NO_SCALING,
    model=(
        "KNeighborsClassifier(\n"
        "        # TRAIT: model accuracy, inference time, memory usage.\n"
        "        # k=75 is half the dataset, so the neighbourhood is far too\n"
        "        # wide to resolve class boundaries; brute-force search\n"
        "        # also makes inference slower than it needs to be.\n"
        "        n_neighbors=75,\n"
        "        weights='uniform',\n"
        "        algorithm='brute',\n"
        "    )"
    ),
)

_DECISION_TREE_CODE = _PREAMBLE.format(
    imports="    from sklearn.tree import DecisionTreeClassifier",
    split=_DEFAULT_SPLIT,
    scaling=_SCALING_HOOK,
    model=(
        "DecisionTreeClassifier(\n"
        "        # TRAIT: model accuracy, training time, loss.\n"
        "        # A depth-1 stump can only ever split into two groups, so one\n"
        "        # of the three Iris classes is always misclassified.\n"
        "        max_depth=1,\n"
        "        criterion='gini',\n"
        "        min_samples_split=2,\n"
        "        random_state=42,\n"
        "    )"
    ),
)

_RANDOM_FOREST_CODE = _PREAMBLE.format(
    imports="    from sklearn.ensemble import RandomForestClassifier",
    split=_DEFAULT_SPLIT,
    scaling=_SCALING_HOOK,
    model=(
        "RandomForestClassifier(\n"
        "        # TRAIT: model accuracy, training time, memory usage,\n"
        "        # computational cost. A single depth-1 stump is not an\n"
        "        # ensemble at all and badly underfits.\n"
        "        n_estimators=1,\n"
        "        max_depth=1,\n"
        "        n_jobs=1,\n"
        "        random_state=42,\n"
        "    )"
    ),
)

_SVM_CODE = _PREAMBLE.format(
    imports="    from sklearn.svm import SVC",
    split=_DEFAULT_SPLIT,
    scaling=_NO_SCALING,
    model=(
        "SVC(\n"
        "        # TRAIT: model accuracy, training time, loss.\n"
        "        # A sigmoid kernel on unscaled features with a tiny C is a\n"
        "        # poor fit for Iris (~0.30). Kernel, C, gamma and\n"
        "        # probability are all evolvable, as is adding a scaler.\n"
        "        kernel='sigmoid',\n"
        "        C=0.01,\n"
        "        probability=False,\n"
        "        random_state=42,\n"
        "    )"
    ),
)


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "logistic_regression": AlgorithmSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        dataset_name="Iris",
        dataset_loader=sk_datasets.load_iris,
        baseline_code=_LOGISTIC_REGRESSION_CODE,
    ),
    "knn": AlgorithmSpec(
        name="knn",
        display_name="K-Nearest Neighbors",
        dataset_name="Iris",
        dataset_loader=sk_datasets.load_iris,
        baseline_code=_KNN_CODE,
    ),
    "decision_tree": AlgorithmSpec(
        name="decision_tree",
        display_name="Decision Tree",
        dataset_name="Iris",
        dataset_loader=sk_datasets.load_iris,
        baseline_code=_DECISION_TREE_CODE,
    ),
    "random_forest": AlgorithmSpec(
        name="random_forest",
        display_name="Random Forest",
        dataset_name="Iris",
        dataset_loader=sk_datasets.load_iris,
        baseline_code=_RANDOM_FOREST_CODE,
    ),
    "svm": AlgorithmSpec(
        name="svm",
        display_name="Support Vector Machine",
        dataset_name="Iris",
        dataset_loader=sk_datasets.load_iris,
        baseline_code=_SVM_CODE,
    ),
}

# Presentation order matching the paper's §2 and §3.
ALGORITHM_ORDER = ["logistic_regression", "knn", "decision_tree", "random_forest", "svm"]


def get_spec(algorithm: str) -> AlgorithmSpec:
    try:
        return ALGORITHMS[algorithm]
    except KeyError:
        raise RuntimeError(
            f"unknown algorithm '{algorithm}'; available: {', '.join(ALGORITHM_ORDER)}"
        ) from None


def choose_algorithm() -> str:
    """Interactive picker over the registry."""
    names = [n for n in ALGORITHM_ORDER if n in ALGORITHMS]

    print("Select an algorithm to optimize:")
    for i, name in enumerate(names):
        spec = ALGORITHMS[name]
        print(f"  {i + 1}) {spec.display_name}  (dataset: {spec.dataset_name})")

    choice_raw = input("> ").strip()
    try:
        choice = int(choice_raw)
    except ValueError:
        raise RuntimeError("invalid selection") from None

    index = choice - 1
    if index < 0 or index >= len(names):
        raise RuntimeError("selection out of range")

    return names[index]

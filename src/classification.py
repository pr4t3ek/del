"""
Classification models for Late_delivery_risk (spec sections 12-16).

Answers: given what is known at the moment the shipping-mode decision is made, what is the
probability this order is delivered late?

Design notes
------------
Split strategy. The default is GroupShuffleSplit on Order Id rather than a plain
stratified split. The delivery outcome is recorded once per order but rows are line items
(2.75 per order on average), so a random split scatters line items of the same order
across train and test. The model would then be scored partly on orders it had already
seen the outcome for. A chronological split is available via config.SPLIT_STRATEGY.

Feature ablation. Three variants are trained so the contribution of the decision variable
is measurable rather than asserted:
    full        - every screened decision-time feature
    mode_only   - Shipping Mode alone
    without_mode- every screened feature EXCEPT Shipping Mode
The third doubles as a leakage check: with the decision variable removed, remaining
operational and external factors should carry almost no signal in this dataset, so an AUC
materially above 0.5 would indicate an unintended leak rather than a discovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from src.data_dictionary import ColumnResolver


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def build_preprocessor(df: pd.DataFrame, features: list[str], scale: bool) -> ColumnTransformer:
    """
    One ColumnTransformer for the whole feature set (spec section 42).

    Numerics are median-imputed, and scaled only for the linear model. Categoricals are
    most-frequent-imputed and one-hot encoded with handle_unknown='ignore', so a category
    never seen in training - including the 'Other' bucket - does not break prediction.
    """
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float32,
                                min_frequency=20,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable names for the transformed design matrix."""
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:                                       # pragma: no cover - defensive
        return []


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------
def make_split(df: pd.DataFrame, strategy: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Return train and test positional indices plus a description of what was done.

    'group' keeps every line item of an order on the same side of the split.
    'time'  trains on the earlier period and tests on the later one.
    """
    strategy = strategy or config.SPLIT_STRATEGY
    r = ColumnResolver(df.columns)
    n = len(df)
    positions = np.arange(n)

    if strategy == "time":
        order_date = r.resolve("order date (DateOrders)")
        if order_date is None:
            strategy = "group"
        else:
            cutoff = pd.Timestamp(config.TIME_SPLIT_CUTOFF)
            is_train = df[order_date] < cutoff
            train_idx = positions[is_train.to_numpy()]
            test_idx = positions[(~is_train).to_numpy()]
            if len(test_idx) and len(train_idx):
                return train_idx, test_idx, {
                    "strategy": "time",
                    "description": (
                        f"Chronological split at {config.TIME_SPLIT_CUTOFF}: the model is "
                        "trained on the earlier period and tested on the later one, which "
                        "mirrors how it would be deployed."
                    ),
                    "cutoff": config.TIME_SPLIT_CUTOFF,
                }
            strategy = "group"

    order_id = r.resolve("Order Id")
    if order_id is None:
        rng = np.random.default_rng(config.RANDOM_STATE)
        shuffled = rng.permutation(positions)
        cut = int(n * (1 - config.TEST_SIZE))
        return shuffled[:cut], shuffled[cut:], {
            "strategy": "random",
            "description": "Random split (Order Id unavailable for grouping).",
        }

    groups = df[order_id].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(positions, groups=groups))
    return train_idx, test_idx, {
        "strategy": "group",
        "description": (
            "80/20 split grouped on Order Id, so all line items of an order stay on the "
            "same side. A plain stratified split would place lines of the same order in "
            "both train and test - and because the delivery outcome is identical across "
            "an order's lines, the model would be scored on outcomes it had already seen."
        ),
    }


# --------------------------------------------------------------------------------------
# Model zoo (spec section 12)
# --------------------------------------------------------------------------------------
def model_zoo() -> dict:
    """The four required classifiers, each with a short rationale for the report."""
    seed = config.RANDOM_STATE
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=seed, C=1.0),
            "scale": True,
            "note": "Linear baseline; also the inference model for odds ratios.",
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=50, random_state=seed
            ),
            "scale": False,
            "note": "Interpretable non-linear baseline; depth-capped to limit overfitting.",
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200, max_depth=16, min_samples_leaf=20,
                n_jobs=-1, random_state=seed,
            ),
            "scale": False,
            "note": "Bagged trees; captures interactions without heavy tuning.",
        },
        "Gradient Boosting": {
            "estimator": HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.1, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=seed, early_stopping=True, validation_fraction=0.1,
            ),
            "scale": False,
            "note": "Histogram gradient boosting; strongest general-purpose learner here.",
        },
    }


# --------------------------------------------------------------------------------------
# Evaluation (spec sections 14-16)
# --------------------------------------------------------------------------------------
def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Full metric suite. Accuracy alone is never reported on its own."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "specificity": round(float(specificity), 4),
        "sensitivity": round(float(sensitivity), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "youden_j": round(float(sensitivity + specificity - 1), 4),
        "threshold": round(float(threshold), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def curve_payloads(y_true, y_prob, max_points: int = 300) -> dict:
    """ROC and PR curves, thinned so the browser receives a few hundred points."""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)

    def thin(*arrays):
        n = len(arrays[0])
        if n <= max_points:
            idx = np.arange(n)
        else:
            idx = np.linspace(0, n - 1, max_points).astype(int)
        return [np.asarray(a)[idx].round(5).tolist() for a in arrays]

    fpr_t, tpr_t = thin(fpr, tpr)
    prec_t, rec_t = thin(precision, recall)

    # Youden-optimal operating point (spec section 14).
    j = tpr - fpr
    best = int(np.argmax(j))

    return {
        "roc": {"fpr": fpr_t, "tpr": tpr_t},
        "pr": {"precision": prec_t, "recall": rec_t},
        "youden": {
            "threshold": round(float(roc_thresh[best]), 4),
            "j": round(float(j[best]), 4),
            "tpr": round(float(tpr[best]), 4),
            "fpr": round(float(fpr[best]), 4),
        },
    }


def threshold_sweep(y_true, y_prob, lo: float = 0.10, hi: float = 0.90, step: float = 0.01) -> list[dict]:
    """
    Precompute metrics across the threshold range for the section-16 slider.

    Computed once at training time so the slider is instant and never re-scores the model.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    out = []
    for t in np.arange(lo, hi + 1e-9, step):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        out.append(
            {
                "threshold": round(float(t), 2),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(sens), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "specificity": round(float(spec), 4),
                "sensitivity": round(float(sens), 4),
                "youden_j": round(float(sens + spec - 1), 4),
                "n_flagged": int(y_pred.sum()),
                "flagged_pct": round(100 * float(y_pred.mean()), 2),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            }
        )
    return out


def calibration_payload(y_true, y_prob, n_bins: int = 10) -> dict:
    """Reliability curve: predicted probability vs observed frequency (spec section 21)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "predicted": round(float(y_prob[mask].mean()), 4),
                "observed": round(float(y_true[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return {
        "bins": rows,
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train_variant(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    run_cv: bool = False,
) -> dict:
    """Fit every model in the zoo on one feature variant and evaluate on the held-out set."""
    X = df[features]
    y = df[target].to_numpy().astype(int)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    r = ColumnResolver(df.columns)
    order_id = r.resolve("Order Id")
    groups_train = (
        df[order_id].to_numpy()[train_idx] if order_id is not None else None
    )

    results = {}
    for name, spec in model_zoo().items():
        pipe = Pipeline(
            [
                ("prep", build_preprocessor(df, features, scale=spec["scale"])),
                ("model", spec["estimator"]),
            ]
        )
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_prob)
        metrics["note"] = spec["note"]

        if run_cv and groups_train is not None:
            cv = StratifiedGroupKFold(
                n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
            )
            scores = cross_val_score(
                pipe, X_train, y_train, groups=groups_train,
                cv=cv, scoring="roc_auc", n_jobs=1,
            )
            metrics["cv_auc_mean"] = round(float(scores.mean()), 4)
            metrics["cv_auc_std"] = round(float(scores.std()), 4)

        results[name] = {"pipeline": pipe, "metrics": metrics, "y_prob": y_prob}

    return {"results": results, "y_test": y_test, "test_idx": test_idx}


def select_best(results: dict) -> str:
    """
    Choose the preferred model on ROC-AUC, breaking ties toward the simpler estimator.

    ROC-AUC rather than accuracy: the decision layer consumes ranked probabilities, and
    accuracy at a fixed 0.5 threshold rewards whichever model happens to suit the base
    rate rather than the one that best separates late from on-time orders.
    """
    order = list(model_zoo().keys())
    best = max(
        results.items(),
        key=lambda kv: (kv[1]["metrics"]["roc_auc"], -order.index(kv[0])),
    )
    return best[0]

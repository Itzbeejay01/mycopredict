import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "RESULTS", "logs")
PLOT_DIR = os.path.join(BASE_DIR, "RESULTS", "plots")

FAVORABLE_OUTCOME = "Cured/Completed"
UNFAVORABLE_OUTCOMES = {
    "Died",
    "Lost to follow-up",
    "Treatment failure",
}
EXCLUDED_FEATURES = {
    "patient_id",
    "treatment_outcome",
    "treatment_started",
    "treatment_category",
    "died",
}


def prepare_split(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only resolved outcomes and remove leakage columns."""
    resolved_outcomes = {FAVORABLE_OUTCOME, *UNFAVORABLE_OUTCOMES}
    filtered = df.loc[df["treatment_outcome"].isin(resolved_outcomes)].copy()
    filtered["target"] = (filtered["treatment_outcome"] == FAVORABLE_OUTCOME).astype(int)
    filtered.drop(columns=[c for c in EXCLUDED_FEATURES if c in filtered.columns], inplace=True)
    return filtered


def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "sensitivity": tp / (tp + fn + 1e-9),
        "specificity": tn / (tn + fp + 1e-9),
    }


def best_threshold(y_true, probs):
    thresholds = np.arange(0.01, 0.99, 0.002)
    best_t = 0.5
    best_score = -1

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        precision = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        specificity = tn / (tn + fp + 1e-9)
        score = (0.5 * f1) + (0.3 * precision) + (0.2 * specificity)

        if score > best_score:
            best_score = score
            best_t = threshold

    return float(best_t)


def plot_roc(y_true, probs, name):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = roc_auc_score(y_true, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_roc.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_cm.png"))
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    train_df = prepare_split(pd.read_csv(os.path.join(DATA_DIR, "train_dataset.csv")))
    val_df = prepare_split(pd.read_csv(os.path.join(DATA_DIR, "val_dataset.csv")))
    test_df = prepare_split(pd.read_csv(os.path.join(DATA_DIR, "test_dataset.csv")))

    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
    X_val, y_val = val_df.drop(columns=["target"]), val_df["target"]
    X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, "columns.pkl"))

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    negative_count = max(int((y_train == 0).sum()), 1)
    positive_count = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = negative_count / positive_count

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=600,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=31,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\n================ {name.upper()} ================")

        pipeline = Pipeline(
            [
                ("prep", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, val_prob)

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        metrics = evaluate(y_test, test_pred, test_prob)

        print(metrics)
        print(f"Best threshold: {threshold:.3f}")

        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.pkl"))
        plot_roc(y_test, test_prob, name)
        plot_confusion_matrix(y_test, test_pred, name)

        results.append(
            {
                "model": name,
                **metrics,
                "threshold": threshold,
            }
        )

    pd.DataFrame(results).to_csv(os.path.join(LOG_DIR, "metrics.csv"), index=False)
    print("\nDONE: training + plots + logs saved")


if __name__ == "__main__":
    main()

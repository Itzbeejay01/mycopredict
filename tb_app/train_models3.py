import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "RESULTS", "logs")
PLOT_DIR = os.path.join(BASE_DIR, "RESULTS", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_dataset.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val_dataset.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_dataset.csv"))


def create_target(df):
    return (df["treatment_outcome"] == "Cured/Completed").astype(int)

for df in [train_df, val_df, test_df]:
    df["target"] = create_target(df)


leakage_cols = ["treatment_outcome", "treatment_started", "treatment_category", "died"]
for df in [train_df, val_df, test_df]:
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)


# =========================
# FEATURES
# =========================
X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
X_val, y_val     = val_df.drop(columns=["target"]), val_df["target"]
X_test, y_test   = test_df.drop(columns=["target"]), test_df["target"]


num_features = X_train.select_dtypes(include=np.number).columns.tolist()
cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()


# =========================
# PREPROCESSOR
# =========================
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", KNNImputer(n_neighbors=3)),
        ("scaler", StandardScaler())
    ]), num_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_features)
])


# =========================
# MODELS (FAST ONLY)
# =========================
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )
}


# =========================
# METRICS
# =========================
def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp)
    }


def best_threshold(y_true, probs):
    thresholds = np.linspace(0.2, 0.8, 200)

    best_t = 0.5
    best_score = 0

    for t in thresholds:
        preds = (probs >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        specificity = tn / (tn + fp + 1e-9)
        f1 = f1_score(y_true, preds, zero_division=0)

        # balanced metric (KEY FIX)
        score = (
            0.5 * f1 +
            0.3 * specificity +
            0.2 * precision
        )

        if score > best_score:
            best_score = score
            best_t = t

    return best_t

# =========================
# PLOT FUNCTIONS
# =========================
def plot_roc(y_true, probs, name):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = roc_auc_score(y_true, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()

    plt.savefig(os.path.join(PLOT_DIR, f"{name}_roc.png"))
    plt.close()

# =========================
# ENSEMBLE ROC PLOT
# =========================

def plot_ensemble_roc(models, X_test, y_test, preprocessor, plot_dir):

    plt.figure()

    ensemble_probs = np.zeros(len(y_test))

    for name, model in models.items():

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        # IMPORTANT: retrain pipeline on full train+val if needed
        # but here we assume models are already trained in memory
        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)

        plt.plot(fpr, tpr, label=f"{name} AUC = {auc_score:.4f}")

        ensemble_probs += probs

    # Average ensemble
    ensemble_probs /= len(models)

    fpr_e, tpr_e, _ = roc_curve(y_test, ensemble_probs)
    auc_e = roc_auc_score(y_test, ensemble_probs)

    plt.plot(fpr_e, tpr_e, linewidth=3, label=f"Ensemble AUC = {auc_e:.4f}")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (Models vs Ensemble)")
    plt.legend()

    plt.savefig(os.path.join(plot_dir, "ensemble_roc.png"))
    plt.close()


# CALL THIS AFTER TRAIN LOOP
plot_ensemble_roc(models, X_test, y_test, preprocessor, PLOT_DIR)

def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()

    plt.xticks([0,1], ["0","1"])
    plt.yticks([0,1], ["0","1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(os.path.join(PLOT_DIR, f"{name}_cm.png"))
    plt.close()


# =========================
# TRAIN LOOP
# =========================
results = []

for name, model in models.items():

    print(f"\n================ {name.upper()} ================")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    # TRAIN
    pipeline.fit(X_train, y_train)

    # VALIDATION threshold
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    thresh = best_threshold(y_val, val_prob)

    # TEST
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= thresh).astype(int)

    metrics = evaluate(y_test, test_pred, test_prob)

    print(metrics)

    # SAVE MODEL
    joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.pkl"))

    # PLOTS
    plot_roc(y_test, test_prob, name)
    plot_confusion_matrix(y_test, test_pred, name)

    # LOG
    results.append({
        "model": name,
        **metrics,
        "threshold": thresh
    })


# =========================
# SAVE LOGS
# =========================
pd.DataFrame(results).to_csv(
    os.path.join(LOG_DIR, "metrics.csv"),
    index=False
)

print("\nDONE: training + plots + logs saved")
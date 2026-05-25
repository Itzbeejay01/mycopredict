"""Flask web app: TB treatment outcome prediction via hard-voting ensemble.

Models (from tb_app/models/): RandomForest.pkl, XGBoost.pkl, LightGBM.pkl.
Each saved model is a full sklearn Pipeline (preprocessing + classifier).
"""

import logging
import os
from statistics import mean
from typing import Any, Dict

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tb_app")


def _load(name: str):
    return joblib.load(os.path.join(MODELS_DIR, name))


# Train models on first run if any required artifact is missing.
# (Matches tb_app/train_models.py output.)
required = ["RandomForest.pkl", "XGBoost.pkl", "LightGBM.pkl", "columns.pkl"]
if any(not os.path.exists(os.path.join(MODELS_DIR, f)) for f in required):
    log.info("Models not found, training now...")
    from train_models import main as train_main
    train_main()

# Load model pipelines
RF_MODEL = _load("RandomForest.pkl")
XGB_MODEL = _load("XGBoost.pkl")
LGBM_MODEL = _load("LightGBM.pkl")

# Not strictly needed for prediction because models contain preprocessing.
# Kept only as a reference artifact.
FEATURE_COLUMNS = _load("columns.pkl")


MODELS: Dict[str, Any] = {
    "RandomForest": RF_MODEL,
    "XGBoost": XGB_MODEL,
    "LightGBM": LGBM_MODEL,
}

MODEL_DISPLAY_NAMES = {
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}

# Form fields expected by the training dataset.
# NOTE: train_models.py derives feature types from the CSV columns; we pass
# these raw values and let each saved pipeline handle preprocessing.
NUMERIC_FIELDS = {
    "age": (40, int),
    "bmi": (22.0, float),
    "symptom_duration_weeks": (4, int),
    "hemoglobin_g_dl": (12.5, float),
    "esr_mm_hr": (40.0, float),
    "cd4_count": (350, int),
    "tb_probability_score": (0.5, float),
}

CATEGORICAL_FIELDS = {
    "sex": ("Male", ["Male", "Female"]),
    "residence": ("Urban", ["Urban", "Rural"]),
    "hiv_status": ("Negative", ["Negative", "Positive"]),
    "age_category": ("25-34", ["0-14", "15-24", "25-34", "35-49", "50+"]),
    "tb_status": ("Positive", ["Positive", "Negative"]),
    "tb_type": ("Pulmonary", ["Pulmonary", "Extrapulmonary", "Both"]),
    "smear_status": ("Positive", ["Positive", "Negative"]),
    "xray_findings": (
        "Normal",
        [
            "Normal",
            "Cavitation",
            "Consolidation",
            "Lymphadenopathy",
            "Miliary pattern",
            "Pleural effusion",
            "Upper lobe infiltrates",
        ],
    ),
}

BOOLEAN_FIELDS = [
    "underweight", "previous_tb", "cavitary_disease", "mdr_tb",
    "cough", "fever", "night_sweats", "weight_loss",
    "anemia", "xray_abnormal", "overcrowding", "culture_confirmed",
    "xdr_tb", "hemoptysis", "chest_pain", "fatigue",
]

DEFAULT_FAVORABLE_PROBA_THRESHOLD = 0.5
THRESHOLDS_BY_MODEL: Dict[str, float] = {}

try:
    metrics_path = os.path.join(BASE_DIR, "RESULTS", "logs", "metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    # metrics.csv rows contain: model, ..., threshold
    for _, row in metrics_df.iterrows():
        THRESHOLDS_BY_MODEL[str(row["model"])] = float(row["threshold"])
    log.info("Loaded per-model thresholds: %s", THRESHOLDS_BY_MODEL)
except Exception as e:
    log.warning("Could not load per-model thresholds; falling back to default. Error=%s", e)


def get_threshold(model_name: str) -> float:
    return THRESHOLDS_BY_MODEL.get(model_name, DEFAULT_FAVORABLE_PROBA_THRESHOLD)


def build_defaults(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    defaults = {field: config[0] for field, config in NUMERIC_FIELDS.items()}
    defaults.update({field: config[0] for field, config in CATEGORICAL_FIELDS.items()})
    defaults.update({field: 0 for field in BOOLEAN_FIELDS})
    if overrides:
        defaults.update(overrides)
    return defaults


def build_selected(defaults: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    return {
        field: {option: defaults.get(field) == option for option in config[1]}
        for field, config in CATEGORICAL_FIELDS.items()
    }


def render_predict_page(
    *,
    result: Dict[str, Any] | None = None,
    payload: Dict[str, Any] | None = None,
    error: str | None = None,
    status_code: int = 200,
):
    defaults = build_defaults(payload)
    return (
        render_template(
            "predict.html",
            defaults=defaults,
            selected=build_selected(defaults),
            boolean_fields=BOOLEAN_FIELDS,
            categorical_fields=CATEGORICAL_FIELDS,
            result=result,
            error=error,
        ),
        status_code,
    )



def parse_form(form) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    for field, (default, caster) in NUMERIC_FIELDS.items():
        raw = form.get(field, "").strip()
        try:
            data[field] = caster(raw) if raw != "" else default
        except (ValueError, TypeError):
            data[field] = default

    for field, config in CATEGORICAL_FIELDS.items():
        default = config[0]
        data[field] = form.get(field, default) or default

    for field in BOOLEAN_FIELDS:
        data[field] = 1 if form.get(field) in (
            "on", "true", "True", "1", "yes") else 0

    return data


def to_model_input(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Align incoming form data with training schema.
    Missing columns are filled with training-aligned defaults.
    """
    aligned = {}

    for col in FEATURE_COLUMNS:
        if col in payload:
            aligned[col] = payload[col]
        elif col in NUMERIC_FIELDS:
            aligned[col] = NUMERIC_FIELDS[col][0]
        elif col in CATEGORICAL_FIELDS:
            aligned[col] = CATEGORICAL_FIELDS[col][0]
        elif col in BOOLEAN_FIELDS:
            aligned[col] = 0
        else:
            aligned[col] = 0

    return pd.DataFrame([aligned])


def predict_ensemble(df_in: pd.DataFrame) -> Dict[str, Any]:
    individual_proba = {}
    individual_vote = {}

    for name, model in MODELS.items():
        class_labels = list(model.classes_)
        positive_index = class_labels.index(1)
        proba = float(model.predict_proba(df_in)[:, positive_index][0])
        individual_proba[name] = proba
        individual_vote[name] = int(proba >= get_threshold(name))

    favorable_votes = sum(individual_vote.values())
    total = len(individual_vote)
    final = 1 if favorable_votes > (total / 2) else 0
    agreement_votes = favorable_votes if final == 1 else total - favorable_votes
    favorable_probability = mean(individual_proba.values())
    confidence = round((favorable_probability if final == 1 else 1 - favorable_probability) * 100, 1)
    agreement = round((agreement_votes / total) * 100, 1)

    return {
        "final": final,
        "label": "Favorable" if final == 1 else "Unfavorable",
        "risk_level": "Low Risk" if final == 1 else "High Risk",
        "confidence": confidence,
        "agreement": agreement,
        "favorable_votes": favorable_votes,
        "total_models": total,
        "individual": {
            MODEL_DISPLAY_NAMES[name]: vote
            for name, vote in individual_vote.items()
        },
        "individual_proba": {
            MODEL_DISPLAY_NAMES[name]: round(proba * 100, 1)
            for name, proba in individual_proba.items()
        },
        "threshold": {
            MODEL_DISPLAY_NAMES[name]: round(get_threshold(name), 3)
            for name in MODELS.keys()
        },
    }




@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/predictor")
@app.route("/predict", methods=["GET"])
def predictor():
    return render_predict_page()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = parse_form(request.form)
        X = to_model_input(payload)
        result = predict_ensemble(X)
        return render_predict_page(result=result, payload=payload)
    except Exception as e:  # pragma: no cover
        log.exception("Prediction failed")
        return render_predict_page(
            error=f"Prediction failed: {e}",
            payload=parse_form(request.form),
            status_code=500,
        )


@app.route("/healthz")
def healthz():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# TB Treatment Outcome Predictor

Flask web app that predicts tuberculosis treatment outcomes (Favorable vs.
Unfavorable) using a hard-voting ensemble of Logistic Regression and Random
Forest classifiers. SVM is included automatically if `models/svm_model.pkl`
is present.

## Install requirements

```bash
# (recommended) create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

> **Training is optional** if you already uploaded the trained model files.

### Option A: use already-trained models (no retraining)

```bash
python tb_app/app.py
```

### Option B (optional): (re)train models

```bash
python tb_app/train_models.py
python tb_app/app.py
```

The app checks for these required artifacts in `tb_app/models/`:

- `RandomForest.pkl`
- `XGBoost.pkl`
- `LightGBM.pkl`
- `columns.pkl`

If any are missing, the app will auto-run the training script on first launch.

## Project layout

```
tb_app/
├── app.py                 # Flask backend (form, /predict, hard voting)
├── train_models.py        # Synthetic data + model training
├── models/                # Saved .pkl models + feature column schema
├── templates/
│   ├── index.html         # Input form
│   └── result.html        # Prediction result
└── static/style.css       # UI styling
```

## How hard voting works

Each model predicts 0 (Unfavorable) or 1 (Favorable). The class with the
majority of votes becomes the final prediction. Confidence is the share of
models that agreed with the final prediction.

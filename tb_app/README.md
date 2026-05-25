# TB Treatment Outcome Predictor

Flask web app that predicts tuberculosis treatment outcomes (Favorable vs.
Unfavorable) using a hard-voting ensemble of Logistic Regression and Random
Forest classifiers. SVM is included automatically if `models/svm_model.pkl`
is present.

## Run

```bash
# 1. Train models (creates models/*.pkl)
python tb_app/train_models.py

# 2. Start the Flask app
python tb_app/app.py
```

The app auto-trains the models on first launch if `.pkl` files are missing.

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

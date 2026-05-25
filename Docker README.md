# tb_app — Docker install & run guide (Flask + model training)

This folder (`tb_app/`) contains a Flask web app that predicts TB treatment outcomes using 3 persisted sklearn pipelines:

- `models/RandomForest.pkl`
- `models/XGBoost.pkl`
- `models/LightGBM.pkl`

On first launch, if the required model files are missing, the app will automatically run the training script (`train_models.py` imported from `app.py`).

---

## 1) Prerequisites

- Docker Desktop (Mac/Windows) or Docker Engine (Linux)
- (Optional) A working `docker buildx` installation is fine, but not required.

Verify Docker is installed:

```bash
docker --version
docker compose version
```

---

## 2) Build the Docker image

From the repository root, build using the `tb_app/Dockerfile`:

```bash
docker build -t tb-app:latest -f tb_app/Dockerfile tb_app
```

What this does:

- Creates a Python 3.10 slim image
- Installs system build deps (`build-essential`) for packages that need compilation
- Installs Python dependencies from `tb_app/requirements.txt`
- Exposes port `5000`

---

## 3) Run the container (start the app)

### Option A (recommended): publish port 5000

```bash
docker run --rm -p 5000:5000 \
  -e SESSION_SECRET='dev-secret-key' \
  tb-app:latest
```

Then open:

- http://localhost:5000/
- http://localhost:5000/predict

### Option B: choose a different host port

If you want the service accessible on `8080`:

```bash
docker run --rm -p 8080:5000 \
  tb-app:latest
```

---

## 4) Python requirements & model files (what you need to provide)

### Install requirements

The Docker image installs all Python requirements automatically from `tb_app/requirements.txt` during the image build.

If you run the app **outside Docker**, you must install them first:

```bash
pip install -r requirements.txt
```

### Models (training is optional)

Inside `tb_app/app.py` the app checks for these required artifacts (under `/app/models`):

- `models/RandomForest.pkl`
- `models/XGBoost.pkl`
- `models/LightGBM.pkl`
- `models/columns.pkl`

### How first-run / retraining works

- **If all of the above files are present**, the app **loads the models** and **does NOT retrain**.
- **If any file is missing**, the app imports and runs the training script **automatically** (inside the container):

```python
from train_models import main as train_main
train_main()
```

### Important

- If you already trained and uploaded/copied the `models/` files (or mount them via a Docker volume), the container start will be fast and will skip retraining.
- Training is optional and only happens when required artifacts are missing.

---

## 5) Persist `models/` to avoid re-training (volume)

Create a local folder to persist models:

```bash
mkdir -p tb_app_persist/models
```

Run with volumes:

```bash
docker run --rm \
  -p 5000:5000 \
  -v "$(pwd)/tb_app_persist/models:/app/models" \
  tb-app:latest
```

After the first successful container start, the trained `*.pkl` files will be stored in `tb_app_persist/models`.

---

## 6) Required dependencies

Dependencies are installed in the image using:

- `tb_app/requirements.txt`

Example list (high level):

- `flask`, `gunicorn`
- `scikit-learn`, `pandas`, `numpy`, `joblib`
- `xgboost`, `lightgbm`
- `imbalanced-learn`, `shap`
- plotting libs: `matplotlib`, `seaborn`

---

## 7) Start mode in Docker (gunicorn)

The `tb_app/Dockerfile` starts the app with:

```bash
gunicorn -b 0.0.0.0:5000 app:app
```

So the Flask app must be importable as `app.py` exposing `app = Flask(__name__)`.

---

## 8) Health check endpoint

Use:

```bash
curl http://localhost:5000/healthz
```

Response shape:

```json
{ "status": "ok", "models_loaded": ["RandomForest", "XGBoost", "LightGBM"] }
```

---

## 9) Common issues

### `ModuleNotFoundError` / missing model artifacts

- Ensure `models/` contains `RandomForest.pkl`, `XGBoost.pkl`, `LightGBM.pkl`, and `columns.pkl`.
- If not, allow the container to train on first launch, or persist `models/` with a volume.

### Build fails for native libs

- The Dockerfile installs `build-essential` before pip install.
- If you still see compilation errors, inspect the build log and verify Docker has enough resources.

---

## 10) Quick command summary

Build:

```bash
docker build -t tb-app:latest -f tb_app/Dockerfile tb_app
```

Run (no model persistence):

```bash
docker run --rm -p 5000:5000 tb-app:latest
```

Run (persist models):

```bash
docker run --rm -p 5000:5000 -v "$(pwd)/tb_app_persist/models:/app/models" tb-app:latest
```

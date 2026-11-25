# NutriPlan Body Fat Predictor: an end-to-end MLOps pipeline

Our NutriPlan app estimates body fat with the **U.S. Navy tape formula**, which is off by about ±4 % points.
This project trains a model that predicts body fat from the **same tape measurements**, and wraps it in a
minimal but complete **MLOps lifecycle**: validated data → tracked experiments → quality gate →
model registry → REST API in Docker → CI/CD → drift monitoring.

**Result:** the production model's average error is **3.21 % points**, compared with **4.03** for the Navy formula
(test set, 49 people). That's about 20 % more accurate.

---

## 1. Run it (Docker, recommended)

You need **Docker Desktop** running. Nothing else.

```bash
cd nutriplan-bodyfat-mlops
docker compose up --build
```

The first build takes 3-5 minutes. Then:

| Open | What you see |
|---|---|
| http://localhost:8000 | Web page: enter measurements → prediction, production model card, drift monitor |
| http://localhost:8000/docs | Swagger: try every API endpoint |
| http://localhost:5000 | MLflow: every training run, metrics, charts, the Model Registry |

What happens when you run `docker compose up`:
1. `mlflow` starts the tracking server (port 5000).
2. `trainer` runs the whole training pipeline once, logs everything to MLflow, registers the best model, then exits.
3. `api` starts **only if the trainer succeeded** (if the quality gate fails, nothing gets deployed).

### Demo commands (in a second terminal)

```bash
# run the unit tests + code check inside the container
docker compose run --rm trainer pytest -v
docker compose run --rm trainer flake8 src api tests scripts

# retrain -> a new model version appears in MLflow (v2, v3, ...)
docker compose run --rm trainer
curl -X POST http://localhost:8000/admin/reload-model        # API picks up the new version

# monitoring demo
docker compose exec api python scripts/simulate_traffic.py --mode normal    # status OK
docker compose exec api python -m src.monitor
docker compose exec api python scripts/simulate_traffic.py --mode shifted   # status DRIFT
docker compose exec api python -m src.monitor
```

Stop: `Ctrl+C`, then `docker compose down`. Wipe everything (MLflow history too): `docker compose down -v`,
then delete the `models/` and `logs/` folders.

## 2. Run it without Docker (Python 3.11)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.train                # pipeline -> mlflow.db + models/
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # (other terminal) MLflow UI
uvicorn api.main:app --port 8000   # (other terminal) API + web page
pytest -v
```

---

## 3. The pipeline

```
data/raw/bodyfat.txt
   │  src/data.py       ingest → kg/cm → VALIDATE (7 bad rows removed) → md5 = data version
   ▼
   │  src/features.py   BMI, waist/height, log(waist−neck)   (inside the model pipeline)
   ▼
   │  src/train.py      Navy baseline + 4 models, 5-fold CV → every run logged in MLflow
   ▼
   │  QUALITY GATE      MAE ≤ 4.5 AND better than Navy?  no → stop, keep old model
   ▼
   │  MLflow Registry   bodyfat-model vN, alias "production"   → models/model.joblib
   ▼
   │  api/main.py       FastAPI /predict  (Docker)  → every prediction logged
   ▼
   │  src/monitor.py    live data vs training data → OK / DRIFT → retrain
```

| MLOps concept | Where in this project |
|---|---|
| Data validation | `src/data.py::validate`, rules in `params.yaml` |
| Data versioning | md5 of the raw file logged with every run |
| Config management | `params.yaml` (no numbers hard-coded in code) |
| Reproducible training | fixed seed, pinned library versions, one Docker image |
| Experiment tracking | MLflow: params, metrics, plot, model file per run |
| Baseline comparison | Navy formula logged as its own run |
| Quality gate | `src/train.py` step 4 |
| Model registry + versioning | MLflow Model Registry, alias `production` |
| Model serving | FastAPI (`api/main.py`), input validation with Pydantic |
| Containerization | `Dockerfile`, `docker-compose.yml` |
| CI/CD | `.github/workflows/ci-cd.yml`: lint → tests → train + gate → docker build + smoke test → deploy |
| Monitoring | prediction log + drift report (`src/monitor.py`, `/monitoring/drift`) |
| Testing | 13 pytest tests: data, model, API |

## 4. Project structure

```
nutriplan-bodyfat-mlops/
├── params.yaml                 # all settings
├── data/raw/bodyfat.txt        # original dataset (never edited)
├── src/
│   ├── config.py               # loads params.yaml, folder paths
│   ├── data.py                 # ingest + validate + clean          (Member 1)
│   ├── features.py             # feature engineering                (Member 1)
│   ├── navy.py                 # baseline formula from NutriPlan    (Member 2)
│   ├── train.py                # training pipeline + MLflow + gate  (Member 2)
│   └── monitor.py              # drift detection                    (Member 4)
├── api/main.py, api/static/    # REST API + web page                (Member 3)
├── scripts/simulate_traffic.py # fake users for the monitoring demo (Member 4)
├── tests/                      # pytest                             (everyone)
├── Dockerfile, docker-compose.yml                                   (Member 3)
├── .github/workflows/ci-cd.yml                                      (Member 4)
└── docs/                       # study guide, screenshots, presentation
```

## 5. Dataset

"Body Fat" dataset, Johnson (1996), *Journal of Statistics Education*, from StatLib (Carnegie Mellon University):
252 adult men, body fat measured by **underwater weighing** (the lab reference method), plus age, weight, height
and body circumferences. Free for non-commercial and classroom use.

**Limitation:** men only, so the model is only valid for adult men. Adding female data is listed as future work.

## 6. Team

| Member | Owns | Files |
|---|---|---|
| 1 | Data engineering: ingestion, validation, data versioning, features | `src/data.py`, `src/features.py`, `params.yaml` (data part), `tests/test_data.py` |
| 2 | Model training: baseline, experiments, MLflow tracking, quality gate, registry | `src/train.py`, `src/navy.py`, `tests/test_model.py` |
| 3 | Deployment: FastAPI service, web page, Docker, Compose | `api/`, `Dockerfile`, `docker-compose.yml`, `tests/test_api.py` |
| 4 | Automation and monitoring: CI/CD, drift monitoring, traffic simulation, docs and presentation | `.github/workflows/`, `src/monitor.py`, `scripts/`, `README.md` |

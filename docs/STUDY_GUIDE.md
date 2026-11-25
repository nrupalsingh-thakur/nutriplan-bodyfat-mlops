# Study guide: MLOps from zero, using this project

Read this once before the presentation. It explains every idea in plain words and shows where it lives in the code.

---

## Part 1: What is MLOps?

**Machine learning (ML):** instead of writing the rule yourself, you give the computer many examples
(measurements → real body fat) and it *learns* the rule. The learned rule is called a **model**.

**The problem:** training a model in a notebook is maybe 10 % of the work. In a real company you also need answers to:

- Was the data correct? Which version of the data was used?
- We tried 4 algorithms and many settings. Which one was best, and can we prove it?
- How does an app (like NutriPlan) actually *use* the model?
- Will it run on another laptop or server?
- When someone changes the code, how do we know nothing broke?
- After a few months, is the model still good, or are the users different now?

**MLOps** (Machine Learning Operations) = the practices and tools that answer those questions.
It is DevOps (automating software delivery) applied to ML, plus extra care for **data** and **models**,
because those change over time while code does not.

> One-line definition for the viva: *"MLOps is the set of practices that take an ML model from a notebook
> to a reliable, monitored production service, and keep it working over time."*

### The MLOps lifecycle (our 7 stages)

```
 Data → Features → Train & Track → Quality Gate → Registry → Serve (Docker) → Monitor
   ↑                                                                             │
   └─────────────────────────── retrain when drift is found ────────────────────┘
                     CI/CD runs the whole thing automatically on every git push
```

---

## Part 2: The project in one paragraph

NutriPlan asks the user for age, weight, height, neck, waist and hip, and uses the U.S. Navy formula
to estimate body fat. The Navy formula has an average error of about 4 % points. We took a public dataset of
252 men whose body fat was measured **exactly** in a lab (underwater weighing). We trained models that
predict body fat from the same six measurements. The best one has an average error of **3.21** vs **4.03** for the Navy formula.
Then we built the MLOps around it: every step is automated, tracked, tested, containerized and monitored.

---

## Part 3: Each concept, where it is, how to explain it

### 1. Data validation (`src/data.py`, `params.yaml` → `validation`)
**Idea:** garbage in → garbage out. Check the data *before* training, automatically.
**In our project:** rules such as "height must be 140–210 cm" and "body fat must match body density". They caught
**7 bad records** in the real dataset, for example a man listed as **74.9 cm tall** (typo) and a man with **0 % body fat** (impossible).
**Say:** "The pipeline removes impossible records automatically and logs why, so nobody cleans data by hand in Excel."

### 2. Data versioning (`file_md5` in `src/data.py`)
**Idea:** if the data changes, results change. You must know *which* data made *which* model.
**In our project:** we compute an md5 fingerprint of the data file and store it with every MLflow run and in the
model metadata. Any change to the file → a different fingerprint.
**Say:** "It's like a git commit id, but for data. Tools like DVC do this at scale."

### 3. Configuration management (`params.yaml`)
**Idea:** no magic numbers inside code. All settings (test size, model settings, gate thresholds) live in one file.
**Say:** "To try a new setting we only edit params.yaml and rerun. MLflow records the new values."

### 4. Feature engineering inside the pipeline (`src/features.py`)
**Idea:** new useful inputs computed from raw inputs: BMI, waist-to-height ratio, log(waist − neck).
**The MLOps part:** this calculation is *inside* the saved model (scikit-learn `Pipeline`), so the API runs
exactly the same code as training. That prevents **training/serving skew** (training and serving doing slightly different maths).

### 5. Baseline (`src/navy.py`)
**Idea:** a new model is only useful if it beats what we already have.
**In our project:** the Navy formula, which NutriPlan uses today, is logged as its own MLflow run.

### 6. Experiment tracking with MLflow (`src/train.py`)
**Idea:** keep a lab notebook automatically. For every training run, save:
- **parameters** (which algorithm, which settings, data md5)
- **metrics** (MAE, RMSE, R², cross-validation MAE)
- **artifacts** (predicted-vs-actual plot, validation report, the model file itself)

We try 4 algorithms: Linear Regression, Ridge, Random Forest, Gradient Boosting.
The winner is chosen by **5-fold cross-validation MAE** (not by the test set, so the test score stays honest). The winner is **Ridge**.
**Say:** "Open http://localhost:5000 → every run since day one is there and can be compared."

**Metrics explained:**
- **MAE** (mean absolute error) = average error in % points. 3.21 means "usually within about 3 % body fat". Lower is better.
- **RMSE** = like MAE but punishes big mistakes more.
- **R²** = how much of the variation the model explains (1.0 = perfect, 0 = useless).
- **Cross-validation** = split the training data into 5 parts, train on 4 and test on 1, 5 times, then average. A more reliable score.

### 7. Quality gate (`src/train.py`, step 4)
**Idea:** an automatic "exam" the model must pass before it can be deployed.
**Our rules:** MAE ≤ 4.5 **and** better than the Navy formula. If it fails → the program exits with an error →
the model is NOT registered → Docker does not start the API with it, and CI/CD turns red.

### 8. Model registry and versioning (MLflow Model Registry)
**Idea:** one official shelf of approved models with version numbers.
**In our project:** each passing model becomes `bodyfat-model` version 1, 2, 3… The alias **`production`** points to
the one in use. Going back to an old version = moving the alias.

### 9. Model serving / REST API (`api/main.py`)
**Idea:** the model becomes a web service any app can call over HTTP.
**Endpoints:** `/predict` (POST measurements → body fat), `/health`, `/model-info`, `/monitoring/drift`, `/admin/reload-model`.
**Input validation:** Pydantic rejects impossible inputs (height 40 cm, waist smaller than neck) with error 422.
**Say:** "NutriPlan's backend could call this API instead of the Navy formula."

### 10. Containerization (`Dockerfile`, `docker-compose.yml`)
**Idea:** "it works on my machine" → package code + Python + libraries into an **image**, so it runs the same everywhere.
**docker compose** starts 3 containers in the right order: `mlflow` → `trainer` → `api`. The API starts only if the trainer succeeded.

### 11. CI/CD with GitHub Actions (`.github/workflows/ci-cd.yml`)
**CI (Continuous Integration):** on every push, automatically check code quality (flake8) and run tests (pytest).
**CD (Continuous Delivery/Deployment):** if everything passes, build and ship.
**Our 4 jobs:** `test` → `train` (with quality gate) → `build` (Docker image + smoke test calling `/predict`) → `deploy` (main branch only, simulated).
**Say:** "A broken commit or a worse model can never reach production, because the pipeline stops it."

### 12. Monitoring and data drift (`src/monitor.py`, `scripts/simulate_traffic.py`)
**Idea:** after deployment the world changes. If users become very different from the training data, predictions get less reliable. This is called **data drift**.
**In our project:** the API saves every request in `logs/predictions.csv`. The monitor compares the average of each
input with the training average. If it moved by more than **0.5 standard deviations** → **DRIFT** → "retrain".
**Demo:** normal users → OK. Older, heavier users (`--mode shifted`) → weight, waist, hip… flagged DRIFT.

### 13. Testing (`tests/`, 13 tests)
- data tests: 252 rows read, bad rows removed, units converted correctly
- model tests: Navy formula value, engineered features, predictions in a sensible range
- API tests: valid request works, invalid requests are rejected, drift detection works

---

## Part 4: Live demo script (about 5 minutes)

1. Before the presentation: `docker compose up --build` and wait for the API to start. Open three browser tabs.
2. **Terminal:** scroll the trainer logs: "7 bad rows removed… Navy MAE 4.03… 4 models… gate PASS… version N registered".
3. **MLflow tab (localhost:5000):** experiment `bodyfat-prediction` → runs table → click the chart icon to compare MAE → **Models** tab → `bodyfat-model` versions.
4. **App tab (localhost:8000):** press *Predict*: model 17.5 % vs Navy 20.1 %. Show the "Model in production" card.
5. **Swagger tab (localhost:8000/docs):** POST `/predict` with height 40 → 422 error (input validation).
6. **Terminal:** `docker compose exec api python scripts/simulate_traffic.py --mode shifted`, then refresh the drift card → DRIFT.
7. **Terminal:** `docker compose run --rm trainer` (retrain) → MLflow shows a new version → `curl -X POST localhost:8000/admin/reload-model`.
8. **GitHub (if pushed):** Actions tab → the 4 green jobs.

---

## Part 5: Likely viva questions

| Question | Short answer |
|---|---|
| What is MLOps? | Practices for taking ML from notebook to reliable, monitored production, and keeping it there. |
| Why not just use a Jupyter notebook? | Not repeatable, not tracked, not testable, can't be called by an app, no monitoring. |
| What does MLflow do here? | Tracks every run (params, metrics, artifacts) and holds the Model Registry. |
| Why did Ridge win if Random Forest had a lower test MAE? | We select by cross-validation MAE (3.76 vs 4.06). Picking by test score would mean "studying the exam paper". Small dataset (245 rows), so the simpler model generalizes better. |
| What is the quality gate? | Automatic rules (MAE ≤ 4.5 and better than Navy). Fail → not registered, not deployed. |
| What is data drift? | Live input data no longer looks like training data. We detect it by mean shift in standard deviations. |
| What is concept drift? | The relationship itself changes (same inputs, different correct answer). Needs fresh labelled data to detect. Future work. |
| Why Docker? | Same environment everywhere: laptop, CI server, cloud. |
| Why one image for 3 services? | Same code and library versions everywhere. Fewer moving parts. |
| CI vs CD? | CI = automatically test every change. CD = automatically build/deploy what passed. |
| How is the data versioned? | md5 hash of the raw file logged with every run. DVC would be the next step. |
| What is training/serving skew and how do you avoid it? | Different preprocessing in training vs API. We put feature code inside the saved Pipeline. |
| How do you roll back a bad model? | Move the `production` alias to the previous version in MLflow, re-export, reload. |
| Limitations? | Men only, 252 people, simple drift check, deployment simulated (no real cloud). |
| Future work? | Female data, DVC, scheduled retraining (Airflow), cloud deploy (AWS ECR/ECS), Evidently/Prometheus dashboards, integrate into NutriPlan backend. |
| Is this medical advice? | No. It's a planning aid, like NutriPlan itself. |

---

## Part 6: Who presents what (4 members)

| Member | Slides | Demo part | Must be able to explain |
|---|---|---|---|
| 1: Data | Problem, dataset, data validation, data versioning, features | Trainer log lines about removed rows | `data.py`, `features.py`, validation rules, md5 |
| 2: Model | Baseline, experiments, MLflow, model selection, quality gate, registry | MLflow UI | `train.py`, `navy.py`, metrics (MAE/RMSE/R²/CV) |
| 3: Deployment | API, Docker, Compose | Web page + Swagger 422 | `api/main.py`, `Dockerfile`, `docker-compose.yml` |
| 4: Automation | CI/CD, monitoring, results, future work | Drift demo + retrain + GitHub Actions | `ci-cd.yml`, `monitor.py`, `simulate_traffic.py` |

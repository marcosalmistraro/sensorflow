PYTHON     := venv/Scripts/python
PYTHONPATH := src
export PYTHONPATH

.PHONY: help ingest features train evaluate monitor serve dashboard \
        test lint typecheck docker-up docker-down clean all

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── pipeline ──────────────────────────────────────────────────────────────────

ingest:        ## Download SMAP data and write train/val/test parquet
	$(PYTHON) -m ingestion

features:      ## Scale, window, and lag-featurise all splits
	$(PYTHON) -m features

train:         ## Train Isolation Forest + LSTM Autoencoder, log to MLflow
	$(PYTHON) -m train

evaluate:      ## Compute test-set metrics, drift report, promote champion
	$(PYTHON) -m evaluate

monitor:       ## Run per-feature drift check, write retrain flag
	$(PYTHON) -m monitor

all: ingest features train evaluate monitor  ## Run full pipeline end-to-end

# ── serving ───────────────────────────────────────────────────────────────────

serve:         ## Start FastAPI dev server (reload on change)
	$(PYTHON) -m uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

dashboard:     ## Start Streamlit dashboard
	$(PYTHON) -m streamlit run dashboard/app.py

mlflow-ui:     ## Start local MLflow tracking server
	$(PYTHON) -m mlflow server \
	  --host 0.0.0.0 --port 5000 \
	  --backend-store-uri sqlite:///mlruns/mlruns.db \
	  --default-artifact-root ./mlruns/artifacts

# ── docker ────────────────────────────────────────────────────────────────────

docker-up:     ## Build and start all containers (MLflow + API + dashboard)
	docker compose up --build

docker-down:   ## Stop and remove containers
	docker compose down

docker-logs:   ## Tail logs from all containers
	docker compose logs -f

# ── quality ───────────────────────────────────────────────────────────────────

test:          ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:          ## Lint with ruff
	$(PYTHON) -m ruff check src/ dashboard/ tests/

typecheck:     ## Type-check with mypy
	$(PYTHON) -m mypy src/config.py src/ingestion.py src/features.py src/monitor.py \
	  --ignore-missing-imports

# ── housekeeping ──────────────────────────────────────────────────────────────

clean:         ## Remove generated artefacts (keeps raw data)
	rm -rf models/* reports/* data/features .pytest_tmp

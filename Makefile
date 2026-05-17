# Makefile -- single entry point for the whole project.
#
# Required targets (per exam):
#   all        : start everything
#   stop       : stop everything
#   evaluation : run run_evaluation.py and update Prometheus metrics
#   fire-alert : intentionally trigger an alert (HighModelRMSE + DriftDetected)
#
# Helper targets:
#   train      : force a re-train inside the API container
#   traffic    : send /predict traffic to populate API performance dashboards
#   logs       : tail logs from all services
#   clean      : tear everything down INCLUDING volumes

PYTHON ?= python3
COMPOSE ?= docker compose

.PHONY: all stop train evaluation fire-alert traffic logs clean ps

all:
	@echo ">> Building and starting the full stack..."
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  bike-api      : http://localhost:8080  (docs: /docs)"
	@echo "  Prometheus    : http://localhost:9090"
	@echo "  Grafana       : http://localhost:3000  (admin / admin)"
	@echo "  node-exporter : http://localhost:9100/metrics"
	@echo ""
	@echo "Tip: 'make traffic' to populate the API dashboards,"
	@echo "     'make evaluation' to populate the Model dashboards."

stop:
	$(COMPOSE) down

ps:
	$(COMPOSE) ps

train:
	# Force a re-train by deleting cached artifacts and restarting the API.
	$(COMPOSE) exec bike-api rm -f /app/artifacts/model.joblib /app/artifacts/reference.parquet
	$(COMPOSE) restart bike-api

evaluation:
	# Sends /evaluate batches for the weeks of February 2011.
	# Requires `requests` on the host. The script will pip-install it on the fly
	# inside a temp venv if it's missing.
	# c fpr cammand, iof request is installed, exit code 0, if Not the secound part
	@$(PYTHON) -c "import requests" 2>/dev/null || $(PYTHON) -m pip install --quiet --user requests
	# old on host
	# $(PYTHON) src/evaluation/run_evaluation.py
	docker compose run --rm evaluation

fire-alert:
	# Triggers BOTH the Prometheus alert (HighModelRMSE + DatasetDriftDetected)
	# AND the Grafana ML alert by pushing extreme values onto the gauges.
	# After running this, watch:
	#   - http://localhost:9090/alerts
	#   - http://localhost:3000/alerting/list
	@echo ">> Pushing synthetic alert state onto bike-api metrics..."
	@curl -fsS -X POST http://localhost:8080/trigger-drift | $(PYTHON) -m json.tool
	@echo ""
	@echo ">> Wait ~30-60s for Prometheus to scrape and evaluate."
	@echo ">> Justification: this exercises both the regression-quality alert"
	@echo "   (model_rmse_score > 200) and the drift alert"
	@echo "   (evidently_dataset_drift_detected == 1) end to end."

traffic:
	bash traffic_simulation.sh

logs:
	$(COMPOSE) logs -f --tail=200

clean:
	$(COMPOSE) down -v
	@echo "Volumes removed. Next 'make all' will retrain the model."

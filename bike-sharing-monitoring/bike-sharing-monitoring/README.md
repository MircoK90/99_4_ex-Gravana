# Bike Sharing — Monitoring Final Exam

End-to-end MLOps monitoring solution for a `RandomForestRegressor`
predicting hourly bike-sharing counts (`cnt`) on the UCI Bike Sharing
dataset. Single-command bring-up via `make`.

## Stack at a glance

| Service       | Port | Purpose                                        |
|---------------|------|------------------------------------------------|
| `bike-api`    | 8080 | FastAPI: `/predict`, `/evaluate`, `/metrics`   |
| `prometheus`  | 9090 | Scrapes `bike-api` + `node-exporter`, alerting |
| `grafana`     | 3000 | Provisioned datasource, dashboards, ML alert   |
| `node-exporter` | 9100 | Host metrics (CPU/RAM/disk)                  |

## Quick start

```bash
make all           # build images, start everything
make traffic       # generate /predict traffic for ~60s
make evaluation    # POST weekly Feb-2011 batches to /evaluate
make fire-alert    # push synthetic high-RMSE + drift state
make stop
```

Open:
- API docs: http://localhost:8080/docs
- Prometheus: http://localhost:9090 (also `/alerts` and `/targets`)
- Grafana: http://localhost:3000 (admin / admin) → folder **Bike Sharing**

## What's automated

- **Model bootstrap**: on first startup the API downloads the UCI dataset,
  trains the RandomForest on January 2011, and persists model + reference
  parquet to a named volume (`bike-artifacts`). Subsequent runs reuse
  those artifacts, so `/predict` is ready in seconds.
- **Grafana**: datasource and three dashboards are provisioned via YAML +
  JSON in `deployment/grafana/provisioning/`. No manual import. The ML
  alert rule (`HighModelRMSE_Grafana`) is also provisioned.
- **Prometheus**: rules in `deployment/prometheus/rules/alert_rules.yml`
  cover infra (`BikeAPIDown`, error-rate) and ML (`HighModelRMSE`,
  `DatasetDriftDetected`).

## Metrics exposed by the API

| Metric                              | Type      | Labels                                  | Source                            |
|-------------------------------------|-----------|-----------------------------------------|-----------------------------------|
| `api_requests_total`                | Counter   | `endpoint`, `method`, `status_code`    | FastAPI middleware                |
| `api_request_duration_seconds`      | Histogram | `endpoint`, `method`, `status_code`    | FastAPI middleware                |
| `model_rmse_score`                  | Gauge     | —                                       | Evidently `RegressionQualityMetric` |
| `model_mae_score`                   | Gauge     | —                                       | Evidently `RegressionQualityMetric` |
| `model_r2_score`                    | Gauge     | —                                       | Evidently `RegressionQualityMetric` |
| `evidently_dataset_drift_share`     | Gauge     | —                                       | Evidently `DatasetDriftMetric`    |
| `evidently_dataset_drift_detected`  | Gauge     | —                                       | Evidently `DatasetDriftMetric`    |
| `model_evaluations_total`           | Counter   | —                                       | API                               |

### Custom-metric justification

`evidently_dataset_drift_share` (the share of features Evidently flags as
drifted, 0..1) is a single, normalized indicator of overall input
distribution shift. It is alert-friendly (threshold > 0.5 ⇒ drift) and
acts as the canonical "should we retrain?" signal that summarizes the
DataDriftPreset without burying the operator in per-column detail.

## Dashboards

1. **API Performance** — request rate per endpoint, P50/P95 latency,
   5xx error rate, total requests, API up/down.
2. **Model Performance & Drift** — current RMSE/MAE/R² stats + time
   series, drift-share bar chart with thresholds, drift-detected flag.
3. **Infrastructure Overview** — CPU%, memory%, disk%, network IO,
   load1, uptime, node-exporter up/down.

## Alerts

### Prometheus (`deployment/prometheus/rules/alert_rules.yml`)

| Alert                  | Condition                                   | Severity |
|------------------------|---------------------------------------------|----------|
| `BikeAPIDown`          | `up{job="bike-api"} == 0` for 1m            | critical |
| `BikeAPIHighErrorRate` | 5xx share > 5% for 2m                       | warning  |
| `HighModelRMSE`        | `model_rmse_score > 200` for 1m             | warning  |
| `DatasetDriftDetected` | `evidently_dataset_drift_detected == 1` 30s | warning  |

### Grafana (`deployment/grafana/provisioning/alerting/`)

| Alert                   | Condition                              |
|-------------------------|----------------------------------------|
| `HighModelRMSE_Grafana` | `model_rmse_score > 200` for 1m        |

## `make fire-alert` — what it does and why

`make fire-alert` POSTs to `/trigger-drift`, which deterministically sets:
- `model_rmse_score = 999`
- `model_mae_score = 750`
- `model_r2_score = -1`
- `evidently_dataset_drift_share = 1.0`
- `evidently_dataset_drift_detected = 1`

Within ~30–60s this fires:
- Prometheus `HighModelRMSE` and `DatasetDriftDetected`
- Grafana `HighModelRMSE_Grafana`

The values normalise back to real metrics on the next `make evaluation`.

## File layout

```
bike-sharing-monitoring/
├── Makefile
├── README.md
├── docker-compose.yml
├── run_evaluation.py
├── traffic_simulation.sh
├── src/api/
│   ├── main.py              # FastAPI + metrics + Evidently
│   ├── Dockerfile
│   └── requirements.txt
└── deployment/
    ├── prometheus/
    │   ├── prometheus.yml
    │   └── rules/alert_rules.yml
    └── grafana/
        ├── dashboards/
        │   ├── api_performance.json
        │   ├── model_performance.json
        │   └── infrastructure.json
        └── provisioning/
            ├── datasources/datasources.yaml
            ├── dashboards/dashboards.yaml
            └── alerting/model_quality_alert.yaml
```

## Troubleshooting

- **Empty Grafana dashboards** → run `make traffic` and/or
  `make evaluation` first; metrics need data points.
- **`bike-api` healthcheck failing** → first launch downloads UCI data +
  trains; check `make logs` and wait ~60–90s.
- **OpenML fallback used instead of UCI** → harmless; happens when the
  UCI archive is unreachable (rate-limited / firewalled).
- **`evidently` import errors** → confirm `numpy==1.26.4` is installed
  inside the container; `evidently 0.4.16` is incompatible with `numpy 2.x`.

## Demo run

```bash
make all
sleep 60                                # wait for first-time training
make traffic & make evaluation          # populate every dashboard
# Inspect dashboards in Grafana, check /alerts in Prometheus
make fire-alert                         # fire HighModelRMSE + DriftDetected
# 30-60s later: alerts visible in Prometheus AND Grafana
make stop
```

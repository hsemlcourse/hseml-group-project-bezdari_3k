# CP3 Deployment Guide

## Local Docker Deploy

Build and start FastAPI plus Streamlit:

```bash
docker compose up --build -d
```

Open:

- FastAPI health: http://127.0.0.1:8000/health
- FastAPI Swagger: http://127.0.0.1:8000/docs
- Streamlit dashboard: http://127.0.0.1:8501

Check service status:

```bash
docker compose ps
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8501/_stcore/health
```

Stop services:

```bash
docker compose down
```

## Required Artifacts

The deploy expects:

```text
models/best_model.joblib
models/run_metadata.json
models/experiment_results.csv
```

If the model is missing, API and Streamlit still start, but prediction endpoints return a clear missing-model message.
Recreate the model with:

```bash
.venv/bin/python -m src.modeling --sample-size 50000 --top-n-features 120
```

## Local Python Deploy

Install dependencies:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Start FastAPI:

```bash
.venv/bin/python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal:

```bash
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python -m streamlit run src/ui/streamlit_app.py
```

## Remote Server Notes

Remote deployment is optional for CP3. If used, a small Ubuntu VPS with Docker is enough.
Recommended minimum: 2 vCPU, 4 GB RAM, 20 GB disk.

Basic server flow:

```bash
git clone <repo-url>
cd hseml-group-project-bezdari_3k
docker compose up --build -d
```

For a public HTTPS URL, put Caddy or Nginx in front of the services and proxy:

```text
https://<domain>      -> streamlit:8501
https://api.<domain>  -> api:8000
```

For grading, the report still needs a video link showing the working FastAPI and Streamlit dashboard.

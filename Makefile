.PHONY: run etl api dashboard test docker-up docker-down clean

# ── Local development ─────────────────────────────────────────────────────────

api:
	uvicorn simulated_api.main:app --reload --port 8000

etl:
	python data_pipeline/etl.py

dashboard:
	streamlit run dashboard/app.py

run:
	@echo "Starting API in background..."
	uvicorn simulated_api.main:app --port 8000 &
	@sleep 2
	@echo "Running ETL..."
	python data_pipeline/etl.py
	@echo "Launching dashboard..."
	streamlit run dashboard/app.py

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

# ── Docker ────────────────────────────────────────────────────────────────────

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-reset:
	docker compose down -v
	docker compose up --build

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true

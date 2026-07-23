FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so a source edit does not invalidate the dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY suites/ ./suites/
COPY pytest.ini ./

RUN mkdir -p data/runs data/reports

# Non-root: the harness reads suites/ and writes under data/, nothing else.
RUN useradd --create-home --uid 10001 harness && chown -R harness:harness /app
USER harness

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

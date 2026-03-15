FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY app ./app
COPY configs ./configs
COPY docs ./docs
COPY data ./data
COPY scripts ./scripts
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir uv && uv pip install --system .

EXPOSE 8080

CMD ["sh", "-c", "python scripts/bootstrap_runtime.py && uvicorn app.main:app --host 0.0.0.0 --port 8080"]

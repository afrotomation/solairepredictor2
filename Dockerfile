# Python 3.11, not 3.12: the trained models are pickled with scikit-learn
# 1.2.2 (they reference sklearn.ensemble._gb_losses, removed in later
# versions), and 1.2.2 publishes no cp312 aarch64 wheel — so on 3.12 uv falls
# back to building it from source and the slim image has no gcc.
FROM mirror.gcr.io/library/python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . /app
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# One image for all three services (MLflow server, trainer, API) - same code, same libraries.
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_INDEX_URL=${PIP_INDEX_URL} GIT_PYTHON_REFRESH=quiet

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml setup.cfg ./
COPY data/raw ./data/raw
COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY tests ./tests

EXPOSE 8000 5000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

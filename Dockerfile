ARG REGISTRY=docker.io
FROM ${REGISTRY}/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FASTEMBED_CACHE_PATH=/models
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the embedding and reranker models into the image so containers run offline.
ARG PRELOAD_MODELS=true
RUN if [ "$PRELOAD_MODELS" = "true" ]; then python -c "\
from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"; fi

COPY corpus ./corpus
COPY evals ./evals
RUN useradd --create-home copilot && mkdir -p /models && chown -R copilot /models
USER copilot
EXPOSE 8100
HEALTHCHECK --interval=10s --timeout=3s --retries=30 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/healthz')"
CMD ["uvicorn", "ops_copilot.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8100"]

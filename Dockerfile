FROM python:3.13-slim AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ZTNA_DATABASE=/data/operations.sqlite \
    ZTNA_HOST=0.0.0.0 \
    ZTNA_PORT=8080 \
    ZTNA_DEVICE=cpu

WORKDIR /app

RUN addgroup --system ztna && adduser --system --ingroup ztna ztna \
    && mkdir -p /data /models \
    && chown -R ztna:ztna /data /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install "torch>=2.13,<2.14" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install ".[server]"

USER ztna
EXPOSE 8080

CMD ["python", "-m", "ztna_ueba.server"]

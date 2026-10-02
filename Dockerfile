FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 \
    TIKTOKEN_CACHE_DIR=/app/tiktoken-cache
COPY pyproject.toml uv.lock README.md ./
RUN uv export --locked --no-dev --no-emit-project --output-file requirements.txt \
    && uv venv --python /usr/local/bin/python .venv \
    && uv pip sync --python .venv/bin/python --require-hashes \
        --default-index https://mirrors.aliyun.com/pypi/simple/ requirements.txt \
    && uv pip check --python .venv/bin/python
COPY src ./src
RUN uv pip install --python .venv/bin/python --no-deps \
        --default-index https://mirrors.aliyun.com/pypi/simple/ . \
    && uv pip check --python .venv/bin/python
RUN .venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/app/tiktoken-cache
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/tiktoken-cache /app/tiktoken-cache
COPY configs/service ./configs/service
COPY prompts ./prompts
RUN useradd --uid 10001 --create-home app \
    && mkdir -p /app/data /app/artifacts \
    && chmod -R a+rX /app/configs /app/prompts \
    && chown app:app /app/data /app/artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["sh", "-c", "exec aml-memory serve --config \"${AML_CONFIG_PATH:-configs/service/v1-b3.yaml}\""]

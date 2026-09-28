FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts
RUN pip install --no-cache-dir .
RUN mkdir -p /app/data /app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["sh", "-c", "aml-memory serve --config ${AML_CONFIG_PATH:-configs/service/v1-b3.yaml}"]

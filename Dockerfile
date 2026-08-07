# syntax=docker/dockerfile:1

# Imagem de execução da interface IPEAREV. As dependências são resolvidas pelo
# lockfile, para que builds locais e de CI usem o mesmo conjunto de pacotes.
FROM python:3.12-slim-bookworm AS builder

# A imagem distroless do uv fornece somente os binários e evita depender de
# tags derivadas de distribuição, que podem ser descontinuadas pelo registry.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY paginas ./paginas
COPY src ./src
COPY streamlit_app.py ./
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    IPEAREV_ALLOW_CONFIG_PERSISTENCE=false

WORKDIR /app

RUN useradd --system --create-home --uid 10001 appuser

COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "streamlit_app.py"]

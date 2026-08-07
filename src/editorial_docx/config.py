"""Configurações padrão do pipeline editorial.

As variáveis de ambiente usadas pelo sistema (LLM_*, OPENAI_*, OLLAMA_*)
estão documentadas em .env.example na raiz do projeto. Os valores abaixo
são os defaults aplicados quando a variável de ambiente não existe.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMP_DATA_DIR = PROJECT_ROOT / ".tmp"

DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_API_KEY = "ollama"

DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0
DEFAULT_LLM_SEED = 7
DEFAULT_REVIEW_AGENT_MAX_WORKERS = 4
DEFAULT_GRAMMAR_AGENT_MAX_WORKERS = 3

GRAMMAR_BATCH_SIZE = 4
GRAMMAR_BATCH_OVERLAP = 1
DEFAULT_REVIEW_MAX_BATCH_CHARS = 12000
DEFAULT_REVIEW_MAX_BATCH_CHUNKS = 28
DEFAULT_REVIEW_WINDOW_RADIUS = 2
DEFAULT_REVIEW_FOCUS_EXCERPT_MAX_CHARS = 4500
DEFAULT_REVIEW_WINDOW_EXCERPT_MAX_CHARS = 6500
DEFAULT_REVIEW_SUMMARY_UPDATE_INTERVAL = 3

# Limites de entrada para a revisão em lote. São menores que a janela total
# para preservar espaço para instruções, saída estruturada e margem de segurança.
_MODEL_REVIEW_CONTEXT_BUDGETS = {
    "gpt-4o-mini": 88_000,
    "gpt-5": 232_000,
    "glm-5.2": 160_000,
}
DEFAULT_REVIEW_CONTEXT_TOKEN_BUDGET = max(800, DEFAULT_REVIEW_MAX_BATCH_CHARS // 4)
DEFAULT_REVIEW_MAX_BATCH_ITEMS_DYNAMIC = 10_000

TEXTO_INTEIRO = "texto_inteiro"
JANELA_MINIMA = "janela_minima"
GRAMMAR_CONTEXT_MODE = TEXTO_INTEIRO


def get_review_context_token_budget(model_name: str) -> int:
    """Retorna o orçamento de contexto do documento para o modelo efetivo."""
    override = (os.getenv("REVIEW_CONTEXT_TOKEN_BUDGET") or "").strip()
    if override:
        try:
            return max(800, int(override))
        except ValueError:
            pass

    normalized = (model_name or "").strip().casefold()
    if normalized.startswith("gpt-4o-mini"):
        return _MODEL_REVIEW_CONTEXT_BUDGETS["gpt-4o-mini"]
    if normalized.startswith("gpt-5"):
        return _MODEL_REVIEW_CONTEXT_BUDGETS["gpt-5"]
    if "glm-5.2" in normalized:
        return _MODEL_REVIEW_CONTEXT_BUDGETS["glm-5.2"]
    return DEFAULT_REVIEW_CONTEXT_TOKEN_BUDGET


def get_review_batch_limits(model_name: str) -> tuple[int, int]:
    """Converte o orçamento de tokens em limites usados pelo empacotador."""
    token_budget = get_review_context_token_budget(model_name)
    return token_budget * 4, DEFAULT_REVIEW_MAX_BATCH_ITEMS_DYNAMIC


def ensure_runtime_directories() -> None:
    """Ensures runtime directories."""
    TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def resolve_input_path(path: Path) -> Path:
    """Resolves input path."""
    candidate = path.expanduser()
    if candidate.exists():
        return candidate.resolve()
    return candidate


def build_output_paths(source_path: Path, model_tag: str) -> dict[str, Path]:
    """Builds output paths."""
    stem = source_path.stem
    if stem.endswith("_normalized_document"):
        stem = stem[: -len("_normalized_document")]
    output_dir = source_path.parent
    report_json = output_dir / f"{stem}_output_{model_tag}.relatorio.json"
    return {
        "normalized_json": output_dir / f"{stem}_normalized_document.json",
        "report_json": report_json,
        "diagnostics_json": report_json.with_name(f"{report_json.stem}.diagnostics.json"),
        "docx": output_dir / f"{stem}_output_{model_tag}.docx",
    }

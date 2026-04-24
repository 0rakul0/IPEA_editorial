from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from .config import (
    DEFAULT_GRAMMAR_AGENT_MAX_WORKERS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    PROJECT_ROOT,
)


def _load_env() -> None:
    """Handles load env."""
    env_path = PROJECT_ROOT / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv()


def _read_env(*keys: str, default: str = "") -> str:
    """Handles read env."""
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _has_env(*keys: str) -> bool:
    """Handles has env."""
    return any(bool(os.getenv(key, "").strip()) for key in keys)


def _infer_provider() -> str:
    """Handles infer provider."""
    explicit_provider = _read_env("LLM_PROVIDER").lower()
    if explicit_provider:
        return explicit_provider

    if _has_env("OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_API_KEY"):
        return "ollama"
    if _has_env("LLM_BASE_URL"):
        return "openai_compatible"
    if _has_env("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        return "openai"
    return "openai"


def _build_provider_config(provider: str) -> dict[str, str]:
    """Handles build provider config."""
    provider = (provider or "").strip().lower()

    if provider == "ollama":
        return {
            "provider": "ollama",
            "model": _read_env("LLM_MODEL", "OLLAMA_MODEL", default=DEFAULT_OLLAMA_MODEL),
            "base_url": _read_env("LLM_BASE_URL", "OLLAMA_BASE_URL", default=DEFAULT_OLLAMA_BASE_URL),
            "api_key": _read_env("LLM_API_KEY", "OLLAMA_API_KEY", default=DEFAULT_OLLAMA_API_KEY),
        }

    if provider == "openai_compatible":
        return {
            "provider": "openai_compatible",
            "model": _read_env("LLM_MODEL", "OPENAI_MODEL", default=DEFAULT_OPENAI_MODEL),
            "base_url": _read_env("LLM_BASE_URL", "OPENAI_BASE_URL"),
            "api_key": _read_env("LLM_API_KEY", "OPENAI_API_KEY"),
        }

    return {
        "provider": "openai",
        "model": _read_env("LLM_MODEL", "OPENAI_MODEL", default=DEFAULT_OPENAI_MODEL),
        "base_url": _read_env("LLM_BASE_URL", "OPENAI_BASE_URL"),
        "api_key": _read_env("LLM_API_KEY", "OPENAI_API_KEY"),
    }


def _is_config_usable(config: dict[str, str]) -> bool:
    """Handles is config usable."""
    provider = config.get("provider", "").strip().lower()
    if provider == "openai":
        return bool(config.get("api_key"))
    if provider == "openai_compatible":
        return bool(config.get("base_url"))
    if provider == "ollama":
        return bool(config.get("base_url"))
    return False


def get_llm_candidate_configs() -> list[dict[str, str]]:
    """Returns llm candidate configs."""
    _load_env()

    preferred_provider = _infer_provider()

    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    provider_order = [preferred_provider]
    if preferred_provider != "openai":
        provider_order.append("openai")

    for provider in provider_order:
        config = _build_provider_config(provider)
        key = (config["provider"], config["model"], config["base_url"])
        if key in seen or not _is_config_usable(config):
            continue
        seen.add(key)
        candidates.append(config)

    return candidates


def get_llm_config() -> dict[str, str]:
    """Returns llm config."""
    candidates = get_llm_candidate_configs()
    if candidates:
        return candidates[0]
    return _build_provider_config("openai")


def get_llm_model_tag(config: dict[str, str] | None = None) -> str:
    """Returns llm model tag."""
    current = config or get_llm_config()
    model_name = (current.get("model") or "modelo").strip().lower()
    model_name = model_name.replace("\\", "/").split("/")[-1]
    tag = re.sub(r"[^a-z0-9]+", "_", model_name).strip("_")
    tag = re.sub(r"(?<=[a-z])_(?=\d)", "", tag)
    tag = re.sub(r"(?<=\d)_(?=[a-z])", "", tag)
    return tag or "modelo"


def get_llm_retry_config() -> dict[str, int | float]:
    """Returns llm retry config."""
    _load_env()

    raw_attempts = (os.getenv("LLM_MAX_RETRIES") or "").strip()
    raw_backoff = (os.getenv("LLM_RETRY_BACKOFF_SECONDS") or "").strip()

    try:
        max_retries = max(1, int(raw_attempts)) if raw_attempts else DEFAULT_LLM_MAX_RETRIES
    except ValueError:
        max_retries = DEFAULT_LLM_MAX_RETRIES

    try:
        backoff_seconds = max(0.0, float(raw_backoff)) if raw_backoff else DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    except ValueError:
        backoff_seconds = DEFAULT_LLM_RETRY_BACKOFF_SECONDS

    return {
        "max_retries": max_retries,
        "backoff_seconds": backoff_seconds,
    }


def get_llm_timeout_seconds() -> float:
    """Returns llm timeout seconds."""
    _load_env()

    raw_timeout = (os.getenv("LLM_TIMEOUT_SECONDS") or "").strip()
    try:
        timeout_seconds = max(1.0, float(raw_timeout)) if raw_timeout else DEFAULT_LLM_TIMEOUT_SECONDS
    except ValueError:
        timeout_seconds = DEFAULT_LLM_TIMEOUT_SECONDS
    return timeout_seconds


def get_grammar_agent_max_workers() -> int:
    """Returns grammar agent max workers."""
    _load_env()

    raw_workers = (os.getenv("GRAMMAR_AGENT_MAX_WORKERS") or "").strip()
    try:
        workers = max(1, int(raw_workers)) if raw_workers else DEFAULT_GRAMMAR_AGENT_MAX_WORKERS
    except ValueError:
        workers = DEFAULT_GRAMMAR_AGENT_MAX_WORKERS
    return workers


def get_chat_model():
    """Returns chat model."""
    models = get_chat_models()
    return models[0][1] if models else None


def get_chat_models():
    """Returns chat models."""
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return []

    out = []
    timeout_seconds = get_llm_timeout_seconds()
    for config in get_llm_candidate_configs():
        kwargs: dict[str, str | int | float] = {
            "model": config["model"],
            "temperature": 0,
            "api_key": config["api_key"] or "local",
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if config["base_url"]:
            kwargs["base_url"] = config["base_url"]
        out.append((config, ChatOpenAI(**kwargs)))
    return out

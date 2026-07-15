from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.editorial_docx.llm import get_llm_config, list_available_models

IPEAGPT_BASE_URL = "https://ipeagpt.ipea.gov.br/api/v1"
IPEAGPT_MODELS_URL = "https://ipeagpt.ipea.gov.br/api/v1/models"
IPEAGPT_CHAT_COMPLETIONS_URL = "https://ipeagpt.ipea.gov.br/api/v1/chat/completions"

_PRIMARY_PROVIDER_OPTIONS = {
    "openai": "OpenAI",
    "openai_compatible": "IpeaGPT / OpenAI-compatible",
    "ollama": "Ollama",
}

_FORM_DEFAULTS = {
    "user_primary_provider": "openai",
    "user_openai_model": "",
    "user_openai_api_key": "",
    "user_openai_base_url": "",
    "user_compatible_model": "",
    "user_compatible_api_key": "",
    "user_compatible_base_url": "",
    "user_ollama_model": "",
    "user_ollama_api_key": "",
    "user_ollama_base_url": "",
}


def _ensure_user_settings_state() -> None:
    """Inicializa os campos da configuracao do usuario com base no ambiente atual."""
    env_defaults = {
        "user_primary_provider": (os.getenv("LLM_PRIMARY_PROVIDER") or get_llm_config().get("provider") or "openai").strip().lower(),
        "user_openai_model": (os.getenv("OPENAI_MODEL") or "").strip(),
        "user_openai_api_key": (os.getenv("OPENAI_API_KEY") or "").strip(),
        "user_openai_base_url": (os.getenv("OPENAI_BASE_URL") or "").strip(),
        "user_compatible_model": (os.getenv("LLM_MODEL") or "").strip(),
        "user_compatible_api_key": (os.getenv("LLM_API_KEY") or "").strip(),
        "user_compatible_base_url": (os.getenv("LLM_BASE_URL") or "").strip(),
        "user_ollama_model": (os.getenv("OLLAMA_MODEL") or "").strip(),
        "user_ollama_api_key": (os.getenv("OLLAMA_API_KEY") or "").strip(),
        "user_ollama_base_url": (os.getenv("OLLAMA_BASE_URL") or "").strip(),
    }
    for key, fallback in _FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = env_defaults.get(key, fallback)


def _upsert_env_values(env_path: Path, values: dict[str, str]) -> None:
    """Atualiza ou adiciona pares chave=valor no arquivo .env."""
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated_keys: set[str] = set()
    new_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _ = line.split("=", 1)
        normalized_key = key.strip()
        if normalized_key in values:
            new_lines.append(f"{normalized_key}={values[normalized_key]}")
            updated_keys.add(normalized_key)
        else:
            new_lines.append(line)

    for key, value in values.items():
        if key in updated_keys:
            continue
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def _apply_user_settings_to_env() -> None:
    """Aplica a configuracao do formulario ao ambiente do processo."""
    os.environ["LLM_PRIMARY_PROVIDER"] = st.session_state.user_primary_provider

    os.environ["OPENAI_MODEL"] = st.session_state.user_openai_model.strip()
    os.environ["OPENAI_API_KEY"] = st.session_state.user_openai_api_key.strip()
    os.environ["OPENAI_BASE_URL"] = st.session_state.user_openai_base_url.strip()

    os.environ["LLM_MODEL"] = st.session_state.user_compatible_model.strip()
    os.environ["LLM_API_KEY"] = st.session_state.user_compatible_api_key.strip()
    os.environ["LLM_BASE_URL"] = st.session_state.user_compatible_base_url.strip()

    os.environ["OLLAMA_MODEL"] = st.session_state.user_ollama_model.strip()
    os.environ["OLLAMA_API_KEY"] = st.session_state.user_ollama_api_key.strip()
    os.environ["OLLAMA_BASE_URL"] = st.session_state.user_ollama_base_url.strip()


def _build_active_provider_config() -> dict[str, str]:
    """Monta a configuracao temporaria para o provider principal escolhido no formulario."""
    provider = st.session_state.user_primary_provider
    if provider == "openai_compatible":
        return {
            "provider": "openai_compatible",
            "model": st.session_state.user_compatible_model.strip(),
            "base_url": st.session_state.user_compatible_base_url.strip(),
            "api_key": st.session_state.user_compatible_api_key.strip(),
        }
    if provider == "ollama":
        return {
            "provider": "ollama",
            "model": st.session_state.user_ollama_model.strip(),
            "base_url": st.session_state.user_ollama_base_url.strip(),
            "api_key": st.session_state.user_ollama_api_key.strip(),
        }
    return {
        "provider": "openai",
        "model": st.session_state.user_openai_model.strip(),
        "base_url": st.session_state.user_openai_base_url.strip(),
        "api_key": st.session_state.user_openai_api_key.strip(),
    }


def _save_user_settings(env_path: Path) -> None:
    """Persiste a configuracao do usuario no .env do projeto."""
    values = {
        "LLM_PRIMARY_PROVIDER": st.session_state.user_primary_provider.strip(),
        "OPENAI_MODEL": st.session_state.user_openai_model.strip(),
        "OPENAI_API_KEY": st.session_state.user_openai_api_key.strip(),
        "OPENAI_BASE_URL": st.session_state.user_openai_base_url.strip(),
        "LLM_MODEL": st.session_state.user_compatible_model.strip(),
        "LLM_API_KEY": st.session_state.user_compatible_api_key.strip(),
        "LLM_BASE_URL": st.session_state.user_compatible_base_url.strip(),
        "OLLAMA_MODEL": st.session_state.user_ollama_model.strip(),
        "OLLAMA_API_KEY": st.session_state.user_ollama_api_key.strip(),
        "OLLAMA_BASE_URL": st.session_state.user_ollama_base_url.strip(),
    }
    _upsert_env_values(env_path, values)


def _apply_ipeagpt_preset() -> None:
    """Preenche a configuracao do provider alternativo com o preset do IpeaGPT."""
    st.session_state.user_primary_provider = "openai_compatible"
    if not (st.session_state.user_compatible_base_url or "").strip():
        st.session_state.user_compatible_base_url = IPEAGPT_BASE_URL
    if not (st.session_state.user_compatible_model or "").strip():
        st.session_state.user_compatible_model = "glm-5.2"


def render_configuracao_usuario_section(*, env_path: Path) -> None:
    """Renderiza a configuracao de provider, chave e modelo na sidebar."""
    _ensure_user_settings_state()
    current_config = get_llm_config()
    existing_config_detected = env_path.exists() or any(
        bool(os.getenv(key, "").strip())
        for key in (
            "LLM_PRIMARY_PROVIDER",
            "OPENAI_API_KEY",
            "LLM_BASE_URL",
            "OLLAMA_BASE_URL",
        )
    )

    st.markdown("### Configuração do Usuário")
    st.caption(
        f"Ativo agora: `{current_config['provider']}` | Modelo: `{current_config['model']}`"
    )
    if current_config.get("base_url"):
        st.caption(f"Endpoint atual: `{current_config['base_url']}`")
    if existing_config_detected:
        st.info("Configuração existente detectada. Você pode manter os valores atuais ou substituí-los se quiser personalizar a sua sessão.")
    st.caption("Para IpeaGPT, o token é enviado como `Authorization: Bearer <SEU_TOKEN>` no padrão OpenAI-compatible.")

    provider_options = list(_PRIMARY_PROVIDER_OPTIONS.keys())
    current_provider = st.session_state.user_primary_provider
    if current_provider not in provider_options:
        current_provider = "openai"
    selected_provider = st.selectbox(
        "Provider principal",
        options=provider_options,
        format_func=lambda item: _PRIMARY_PROVIDER_OPTIONS[item],
        index=provider_options.index(current_provider),
        key="user_primary_provider",
        help="Define qual provider tentar primeiro. Os demais continuam disponiveis como fallback quando configurados.",
    )
    if st.button("Usar preset IpeaGPT", use_container_width=True):
        _apply_ipeagpt_preset()
        st.success("Preset do IpeaGPT aplicado ao provider alternativo.")
        st.rerun()

    with st.expander("OpenAI pessoal", expanded=selected_provider == "openai"):
        st.text_input("Modelo OpenAI", key="user_openai_model", placeholder="gpt-5.2")
        st.text_input("Chave OpenAI", key="user_openai_api_key", type="password", placeholder="sk-...")
        st.text_input(
            "Base URL OpenAI (opcional)",
            key="user_openai_base_url",
            placeholder="https://api.openai.com/v1",
        )

    with st.expander("IpeaGPT / OpenAI-compatible", expanded=selected_provider == "openai_compatible"):
        st.caption(f"Modelos: `{IPEAGPT_MODELS_URL}`")
        st.caption(f"Chat completions: `{IPEAGPT_CHAT_COMPLETIONS_URL}`")
        st.text_input("Modelo alternativo", key="user_compatible_model", placeholder="glm-5.2")
        st.text_input("Token Bearer", key="user_compatible_api_key", type="password", placeholder="SEU_TOKEN")
        st.text_input(
            "Base URL alternativa",
            key="user_compatible_base_url",
            placeholder=IPEAGPT_BASE_URL,
        )

    with st.expander("Ollama", expanded=selected_provider == "ollama"):
        st.text_input("Modelo Ollama", key="user_ollama_model", placeholder="llama3.1:8b")
        st.text_input("Chave Ollama (opcional)", key="user_ollama_api_key", type="password", placeholder="ollama")
        st.text_input(
            "Base URL Ollama",
            key="user_ollama_base_url",
            placeholder="http://localhost:11434/v1",
        )

    action_a, action_b = st.columns(2)
    with action_a:
        if st.button("Aplicar sessão", use_container_width=True):
            _apply_user_settings_to_env()
            st.session_state.llm_models_result = None
            st.success("Configuração aplicada nesta sessão.")
            st.rerun()
    with action_b:
        if st.button("Salvar no .env", use_container_width=True):
            _save_user_settings(env_path)
            _apply_user_settings_to_env()
            st.session_state.llm_models_result = None
            st.success("Configuração salva no .env.")
            st.rerun()

    preview_config = _build_active_provider_config()
    if st.button("Listar modelos do provider principal", use_container_width=True):
        with st.spinner("Consultando endpoint de modelos..."):
            st.session_state.llm_models_result = list_available_models(preview_config, timeout=15.0)

    models_result = st.session_state.get("llm_models_result")
    if isinstance(models_result, dict) and models_result.get("provider") == preview_config.get("provider"):
        if models_result.get("endpoint"):
            st.caption(f"Endpoint de modelos: `{models_result['endpoint']}`")
        available_models = models_result.get("available_models") or []
        if models_result.get("ok"):
            if available_models:
                st.caption(f"Modelos disponíveis ({len(available_models)}):")
                st.code("\n".join(str(item) for item in available_models), language="text")
            else:
                st.info("O endpoint respondeu, mas não retornou modelos reconhecíveis.")
        elif models_result.get("error"):
            st.warning(str(models_result["error"]))

    if env_path.exists():
        st.caption("Arquivo .env detectado no repositório.")
    else:
        st.caption("Ainda não há .env no repositório. Você pode criar um ao salvar a configuração.")

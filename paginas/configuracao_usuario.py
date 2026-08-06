from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.editorial_docx.llm import get_llm_config, list_available_models


IPEAGPT_BASE_URL = "https://ipeagpt.ipea.gov.br/api/v1"
_MODEL_PLACEHOLDER = "Selecione um modelo"
_PRIMARY_PROVIDER_OPTIONS = {
    "openai": "OpenAI",
    "openai_compatible": "IpeaGPT",
    "ollama": "Ollama (local)",
}
_FORM_DEFAULTS = {
    "user_primary_provider": "openai",
    "user_openai_model": "",
    "user_openai_api_key": "",
    "openai_models": [],
    "openai_models_error": "",
    "user_compatible_model": "",
    "user_compatible_api_key": "",
    "user_ollama_model": "",
    "user_ollama_api_key": "",
    "user_ollama_base_url": "",
    "ipeagpt_models": [],
    "ipeagpt_models_error": "",
}


def _ensure_user_settings_state() -> None:
    """Inicializa os campos da configuração com base no ambiente atual."""
    env_defaults = {
        "user_primary_provider": (
            os.getenv("LLM_PRIMARY_PROVIDER") or get_llm_config().get("provider") or "openai"
        ).strip().lower(),
        "user_openai_model": (os.getenv("OPENAI_MODEL") or "").strip(),
        "user_openai_api_key": (os.getenv("OPENAI_API_KEY") or "").strip(),
        "user_compatible_model": (os.getenv("LLM_MODEL") or "").strip(),
        "user_compatible_api_key": (os.getenv("IPEAGPT_API_KEY") or os.getenv("LLM_API_KEY") or "").strip(),
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
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def _apply_user_settings_to_env() -> None:
    """Aplica a configuração do formulário ao ambiente do processo."""
    os.environ["LLM_PRIMARY_PROVIDER"] = st.session_state.user_primary_provider
    os.environ["OPENAI_MODEL"] = st.session_state.user_openai_model.strip()
    os.environ["OPENAI_API_KEY"] = st.session_state.user_openai_api_key.strip()
    os.environ["LLM_MODEL"] = st.session_state.user_compatible_model.strip()
    os.environ["IPEAGPT_API_KEY"] = st.session_state.user_compatible_api_key.strip()
    os.environ["LLM_BASE_URL"] = IPEAGPT_BASE_URL
    os.environ["OLLAMA_MODEL"] = st.session_state.user_ollama_model.strip()
    os.environ["OLLAMA_API_KEY"] = st.session_state.user_ollama_api_key.strip()
    os.environ["OLLAMA_BASE_URL"] = st.session_state.user_ollama_base_url.strip()


def _save_user_settings(env_path: Path) -> None:
    """Persiste a configuração do usuário no .env do projeto."""
    values = {
        "LLM_PRIMARY_PROVIDER": st.session_state.user_primary_provider.strip(),
        "OPENAI_MODEL": st.session_state.user_openai_model.strip(),
        "OPENAI_API_KEY": st.session_state.user_openai_api_key.strip(),
        "LLM_MODEL": st.session_state.user_compatible_model.strip(),
        "IPEAGPT_API_KEY": st.session_state.user_compatible_api_key.strip(),
        "LLM_BASE_URL": IPEAGPT_BASE_URL,
        "OLLAMA_MODEL": st.session_state.user_ollama_model.strip(),
        "OLLAMA_API_KEY": st.session_state.user_ollama_api_key.strip(),
        "OLLAMA_BASE_URL": st.session_state.user_ollama_base_url.strip(),
    }
    _upsert_env_values(env_path, values)


def _load_ipeagpt_models() -> None:
    """Obtém os modelos liberados para o Token Bearer informado."""
    result = list_available_models(
        {
            "provider": "openai_compatible",
            "model": st.session_state.user_compatible_model.strip(),
            "base_url": IPEAGPT_BASE_URL,
            "api_key": st.session_state.user_compatible_api_key.strip(),
        }
    )
    if not result.get("ok"):
        st.session_state.ipeagpt_models_error = str(
            result.get("error") or "Não foi possível carregar os modelos disponíveis."
        )
        return

    models = [str(model).strip() for model in result.get("available_models", []) if str(model).strip()]
    st.session_state.ipeagpt_models = list(dict.fromkeys(models))
    st.session_state.ipeagpt_models_error = ""
    if st.session_state.ipeagpt_models and st.session_state.user_compatible_model not in st.session_state.ipeagpt_models:
        st.session_state.user_compatible_model = st.session_state.ipeagpt_models[0]


def _load_openai_models() -> None:
    """Obtém os modelos acessíveis com a chave OpenAI informada."""
    result = list_available_models(
        {
            "provider": "openai",
            "model": st.session_state.user_openai_model.strip(),
            "api_key": st.session_state.user_openai_api_key.strip(),
        }
    )
    if not result.get("ok"):
        st.session_state.openai_models_error = str(
            result.get("error") or "Não foi possível carregar os modelos disponíveis."
        )
        return

    models = [str(model).strip() for model in result.get("available_models", []) if str(model).strip()]
    st.session_state.openai_models = list(dict.fromkeys(models))
    st.session_state.openai_models_error = ""
    if st.session_state.openai_models and st.session_state.user_openai_model not in st.session_state.openai_models:
        st.session_state.user_openai_model = st.session_state.openai_models[0]


def _render_openai_options() -> None:
    """Mostra chave OpenAI e modelo selecionado a partir da lista do serviço."""
    models = st.session_state.openai_models
    current_model = st.session_state.user_openai_model.strip()
    if models:
        if current_model not in models:
            st.session_state.user_openai_model = models[0]
        st.selectbox("Modelo", options=models, key="user_openai_model")
    elif current_model:
        st.selectbox("Modelo", options=[current_model], key="user_openai_model")
    else:
        st.selectbox("Modelo", options=[_MODEL_PLACEHOLDER], disabled=True)

    api_key = st.text_input("Sua chave de API", key="user_openai_api_key", type="password", placeholder="sk-...")
    if st.button("Atualizar modelos OpenAI", use_container_width=True, disabled=not api_key.strip()):
        with st.spinner("Carregando modelos disponíveis..."):
            _load_openai_models()

    if st.session_state.openai_models_error:
        st.warning(st.session_state.openai_models_error)


def _render_ipeagpt_options() -> None:
    """Mostra somente token e seleção de modelo para a conexão fixa do IpeaGPT."""
    with st.expander("Como gerar seu Token", expanded=False):
        st.markdown(
            """
1. Acesse [ipeagpt.ipea.gov.br](https://ipeagpt.ipea.gov.br/) e entre com suas credenciais institucionais.
2. Clique no seu nome, no canto superior direito, para abrir o perfil.
3. Acesse **Configurações**.
4. Abra **Conta** e escolha **Chaves de API** ou **Tokens**.
5. Clique em **Mostrar** e depois em **Criar nova chave**; copie o token exibido.

Consulte também a [documentação de endpoints do IpeaIA](https://intranet.ipea.gov.br/ipeaia/desenvolvedor#endpoints).
            """
        )
        st.info(
            "O token é pessoal e intransferível. Não o compartilhe nem o inclua em código versionado. "
            "Se houver comprometimento, crie uma nova chave."
        )

    token = st.text_input(
        "Chave de API do IpeaGPT (Token Bearer)",
        key="user_compatible_api_key",
        type="password",
        placeholder="Cole seu token aqui",
    )
    if st.button("Atualizar modelos", use_container_width=True, disabled=not token.strip()):
        with st.spinner("Carregando modelos disponíveis..."):
            _load_ipeagpt_models()

    if st.session_state.ipeagpt_models_error:
        st.warning(st.session_state.ipeagpt_models_error)

    models = st.session_state.ipeagpt_models
    if models:
        if st.session_state.user_compatible_model not in models:
            st.session_state.user_compatible_model = models[0]
        st.selectbox("Modelo alternativo", options=models, key="user_compatible_model")
    else:
        st.selectbox("Modelo alternativo", options=[_MODEL_PLACEHOLDER], disabled=True)


def render_configuracao_usuario_section(*, env_path: Path) -> None:
    """Renderiza as opções de OpenAI, IpeaGPT e Ollama."""
    _ensure_user_settings_state()

    st.markdown("### Configuração do usuário")
    provider_options = list(_PRIMARY_PROVIDER_OPTIONS)
    current_provider = st.session_state.user_primary_provider
    if current_provider not in provider_options:
        current_provider = "openai"
    selected_provider = st.selectbox(
        "Modelo principal",
        options=provider_options,
        format_func=lambda item: _PRIMARY_PROVIDER_OPTIONS[item],
        index=provider_options.index(current_provider),
        key="user_primary_provider",
        help="Define qual opção será usada primeiro na revisão.",
    )

    with st.expander("OpenAI", expanded=selected_provider == "openai"):
        _render_openai_options()

    with st.expander("IpeaGPT", expanded=selected_provider == "openai_compatible"):
        _render_ipeagpt_options()

    with st.expander("Ollama (local)", expanded=selected_provider == "ollama"):
        st.text_input("Modelo local", key="user_ollama_model", placeholder="llama3.1:8b")
        st.text_input("Chave de API (opcional)", key="user_ollama_api_key", type="password")
        st.text_input(
            "Base URL local",
            key="user_ollama_base_url",
            placeholder="http://localhost:11434/v1",
        )

    action_a, action_b = st.columns(2)
    with action_a:
        if st.button("Usar nesta sessão", use_container_width=True):
            _apply_user_settings_to_env()
            st.success("Configuração aplicada nesta sessão.")
            st.rerun()
    with action_b:
        if st.button("Salvar", use_container_width=True):
            _save_user_settings(env_path)
            _apply_user_settings_to_env()
            st.success("Configuração salva.")
            st.rerun()

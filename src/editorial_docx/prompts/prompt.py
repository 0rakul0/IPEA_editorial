from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from .profiles import get_prompt_profile

PROMPTS_DIR = Path(__file__).resolve().parent

PROMPT_FILES = {
    "metadados": PROMPTS_DIR / "metadados.md",
    "sinopse_abstract": PROMPTS_DIR / "sinopse_abstract.md",
    "estrutura": PROMPTS_DIR / "estrutura.md",
    "tabelas_figuras": PROMPTS_DIR / "tabelas_figuras.md",
    "referencias": PROMPTS_DIR / "referencias.md",
    "conformidade_estilos": PROMPTS_DIR / "conformidade_estilos.md",
    "gramatica_ortografia": PROMPTS_DIR / "gramatica_ortografia.md",
    "coordenador": PROMPTS_DIR / "coordenador.md",
}

AGENT_ORDER = [
    "metadados",
    "sinopse_abstract",
    "estrutura",
    "tabelas_figuras",
    "referencias",
    "conformidade_estilos",
    "gramatica_ortografia",
]


def load_agent_instruction(agent_name: str) -> str:
    path = PROMPT_FILES.get(agent_name)
    if path is None:
        raise ValueError(f"Agente de prompt desconhecido: {agent_name}")
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de prompt não encontrado: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_agent_prompt(agent_name: str, profile_key: str | None = None) -> ChatPromptTemplate:
    instruction = load_agent_instruction(agent_name)
    profile = get_prompt_profile(profile_key)
    return ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            (
                "human",
                (
                    "Perfil do documento: {profile_description}\n"
                    "Instrução de perfil: {profile_instruction}\n\n"
                    "Pergunta do usuário: {question}\n\n"
                    "Trecho do documento:\n{document_excerpt}\n\n"
                    "Retorne uma lista JSON. Cada item deve conter: "
                    "category, message, paragraph_index (inteiro opcional), "
                    "issue_excerpt (trecho errado) e suggested_fix (como deveria ficar)."
                ),
            ),
        ]
    ).partial(
        profile_description=profile.description,
        profile_instruction=profile.instruction,
    )


def build_coordinator_prompt(profile_key: str | None = None) -> ChatPromptTemplate:
    instruction = load_agent_instruction("coordenador")
    profile = get_prompt_profile(profile_key)
    return ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            (
                "human",
                (
                    "Perfil do documento: {profile_description}\n"
                    "Instrução de perfil: {profile_instruction}\n\n"
                    "Pergunta do usuário: {question}\n\n"
                    "Trecho do documento:\n{document_excerpt}\n\n"
                    "Comentários dos agentes (JSON):\n{comments_json}\n\n"
                    "Responda em português, de forma direta, e cite os principais pontos."
                ),
            ),
        ]
    ).partial(
        profile_description=profile.description,
        profile_instruction=profile.instruction,
    )
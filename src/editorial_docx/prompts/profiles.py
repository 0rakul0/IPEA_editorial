from __future__ import annotations

import re

from .schemas import PromptProfile

_DEFAULT_PROFILE = PromptProfile(
    key="GENERIC",
    description="Documento genérico",
    instruction=(
        "Considere as regras gerais de revisão editorial institucional. "
        "Se faltarem regras específicas do tipo documental, mantenha neutralidade."
    ),
)

_TD_PROFILE = PromptProfile(
    key="TD",
    description="Texto para Discussão (TD)",
    instruction=(
        "Este documento é um Texto para Discussão (TD). "
        "Aplique critérios editoriais de TD: clareza argumentativa, coerência entre seções, "
        "consistência terminológica e aderência ao padrão formal de publicação técnica."
    ),
)

_NT_PROFILE = PromptProfile(
    key="NT",
    description="Nota Técnica (NT)",
    instruction=(
        "Este documento é uma Nota Técnica (NT). "
        "Aplique critérios editoriais de nota técnica: clareza expositiva, precisão terminológica, "
        "economia de intervenção estilística e forte atenção a siglas, fontes e coerência metodológica."
    ),
)

_BPS_PROFILE = PromptProfile(
    key="BPS",
    description="Boletim de Políticas Sociais (BPS)",
    instruction=(
        "Este documento pertence ao Boletim de Políticas Sociais (BPS). "
        "Aplique revisão editorial voltada a texto analítico com forte presença de dados, notas, siglas, "
        "citações no corpo, gráficos e tabelas; preserve o argumento e sinalize o que depender de confirmação autoral."
    ),
)

_BMT_PROFILE = PromptProfile(
    key="BMT",
    description="Boletim Mercado de Trabalho (BMT)",
    instruction=(
        "Este documento pertence ao Boletim Mercado de Trabalho (BMT). "
        "Aplique revisão editorial voltada a boletim analítico curto, com atenção especial a títulos, quadros, "
        "gráficos, consistência de séries e padronização visual dos blocos editoriais."
    ),
)

_RADAR_PROFILE = PromptProfile(
    key="RADAR",
    description="Boletim Radar",
    instruction=(
        "Este documento pertence ao Boletim Radar. "
        "Aplique revisão editorial concisa e técnica, com foco em clareza local, remissões corretas, "
        "consistência entre texto e ilustrações e intervenções proporcionais ao caráter breve do boletim."
    ),
)

_PPP_PROFILE = PromptProfile(
    key="PPP",
    description="Revista Planejamento e Políticas Públicas (PPP)",
    instruction=(
        "Este documento é um artigo da revista Planejamento e Políticas Públicas (PPP). "
        "Aplique critérios editoriais de artigo acadêmico: consistência argumentativa, resumo/abstract, palavras-chave, "
        "citações e referências, sem reescrever conclusões ou opções autorais sem evidência objetiva."
    ),
)

_PPE_PROFILE = PromptProfile(
    key="PPE",
    description="Revista Pesquisa e Planejamento Econômico (PPE)",
    instruction=(
        "Este documento é um artigo da revista Pesquisa e Planejamento Econômico (PPE). "
        "Aplique critérios editoriais de artigo acadêmico técnico-econômico: precisão terminológica, "
        "coerência entre resumo e abstract, estabilidade de fórmulas e cautela com ambiguidade metodológica."
    ),
)

_BOOK_CHAPTER_PROFILE = PromptProfile(
    key="CAPITULO_LIVRO",
    description="Capítulo de livro",
    instruction=(
        "Este documento é um capítulo de livro. "
        "Aplique critérios editoriais de obra coletiva: hierarquia de seções, numeração consistente, "
        "créditos institucionais, elementos de quadro/figura e padronização entre partes do capítulo."
    ),
)

_PROFILE_BY_KEY = {
    _DEFAULT_PROFILE.key: _DEFAULT_PROFILE,
    _TD_PROFILE.key: _TD_PROFILE,
    _NT_PROFILE.key: _NT_PROFILE,
    _BPS_PROFILE.key: _BPS_PROFILE,
    _BMT_PROFILE.key: _BMT_PROFILE,
    _RADAR_PROFILE.key: _RADAR_PROFILE,
    _PPP_PROFILE.key: _PPP_PROFILE,
    _PPE_PROFILE.key: _PPE_PROFILE,
    _BOOK_CHAPTER_PROFILE.key: _BOOK_CHAPTER_PROFILE,
}

_PROFILE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(?:^|[_\-\s])TD(?:[_\-\s]|$)"), "TD"),
    (re.compile(r"(?i)(?:^|[_\-\s])NT(?:[_\-\s]|$)"), "NT"),
    (re.compile(r"(?i)(?:^|[_\-\s])BPS(?:[_\-\s]|$)|Boletim[_\-\s]BPS"), "BPS"),
    (re.compile(r"(?i)(?:^|[_\-\s])BMT(?:[_\-\s]|$)|Boletim[_\-\s]BMT"), "BMT"),
    (re.compile(r"(?i)(?:^|[_\-\s])Radar(?:[_\-\s]|$)|Boletim[_\-\s]Radar"), "RADAR"),
    (re.compile(r"(?i)(?:^|[_\-\s])PPP(?:[_\-\s]|$)|Revista[_\-\s]PPP"), "PPP"),
    (re.compile(r"(?i)(?:^|[_\-\s])PPE(?:[_\-\s]|$)|Revista[_\-\s]PPE"), "PPE"),
    (re.compile(r"(?i)\bLivro\b|cap[.\-_ ]|capitulo"), "CAPITULO_LIVRO"),
]


def detect_prompt_profile(filename: str) -> PromptProfile:
    """Handles detect prompt profile."""
    if not filename:
        return _DEFAULT_PROFILE

    normalized_name = filename.strip()
    for pattern, profile_key in _PROFILE_PATTERNS:
        if pattern.search(normalized_name):
            return _PROFILE_BY_KEY[profile_key]

    return _DEFAULT_PROFILE


def get_prompt_profile(profile_key: str | None) -> PromptProfile:
    """Returns prompt profile."""
    if not profile_key:
        return _DEFAULT_PROFILE
    return _PROFILE_BY_KEY.get(profile_key.upper(), _DEFAULT_PROFILE)

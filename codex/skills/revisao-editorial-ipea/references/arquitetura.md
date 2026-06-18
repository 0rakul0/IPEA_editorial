# Arquitetura

1. `document_loader.py`: carrega DOCX, PDF ou JSON.
2. `normalized_document.py`: cria blocos, seções, referências e comentários.
3. `pipeline/scope.py` e `agents/scopes/`: selecionam o escopo.
4. `pipeline/context.py`: monta lotes e contexto.
5. `pipeline/orchestrator.py`, `pipeline/runtime.py` e `graph_chat.py`: executam LLM e heurísticas.
6. `agents/validation/` e `pipeline/validation.py`: rejeitam candidatos inseguros.
7. `pipeline/consolidation.py` e `pipeline/coordinator.py`: deduplicam e consolidam.
8. `docx_utils.py`, CLI e Streamlit: exportam.

## Fontes de verdade

- Ordem e prompts: `src/editorial_docx/prompts/`.
- Escopos: `src/editorial_docx/agents/scopes/`.
- Heurísticas: `src/editorial_docx/agents/heuristics/`.
- Validações: `src/editorial_docx/agents/validation/`.
- Bibliografia: `src/editorial_docx/references/` e módulos `abnt_*`.
- Configuração: `src/editorial_docx/config.py` e `src/editorial_docx/llm.py`.
- Fachada: `src/editorial_docx/graph_chat.py`.

Falso positivo: rastrear prompt → escopo → candidato → validação → consolidação.

Falso negativo: rastrear extração → classificação → escopo → contexto → LLM/heurística → validação.

Não ampliar indiscriminadamente prompts para corrigir escopo. Não remover validações determinísticas sem regressão.


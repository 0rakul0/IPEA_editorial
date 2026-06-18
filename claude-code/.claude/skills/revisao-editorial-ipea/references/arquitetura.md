# Arquitetura

1. `document_loader.py`: entrada.
2. `normalized_document.py`: representação.
3. `pipeline/scope.py` e `agents/scopes/`: escopo.
4. `pipeline/context.py`: lotes.
5. `pipeline/orchestrator.py`, `pipeline/runtime.py` e `graph_chat.py`: execução.
6. `agents/validation/` e `pipeline/validation.py`: filtros.
7. `pipeline/consolidation.py` e `pipeline/coordinator.py`: merge.
8. `docx_utils.py`, CLI e Streamlit: exportação.

Fontes: `prompts/`, `agents/scopes/`, `agents/heuristics/`, `agents/validation/`, `references/`, módulos `abnt_*` e `graph_chat.py`.

Falso positivo: prompt → escopo → candidato → validação → consolidação.

Falso negativo: extração → tipo → escopo → contexto → LLM/heurística → validação.


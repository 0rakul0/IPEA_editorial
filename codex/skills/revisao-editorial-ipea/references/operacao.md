# Operação

## Requisitos

- Python 3.10 ou superior.
- Git.
- `uv` recomendado.
- Uma LLM configurada por variáveis `LLM_*`.

## Preparação

Na raiz:

```bash
uv sync --dev
```

Criar `.env` a partir de `.env.example`:

```env
LLM_PROVIDER=openai
LLM_MODEL=<modelo-disponivel>
LLM_API_KEY=<chave>
```

Também são aceitos `openai_compatible` e `ollama`. Nunca versionar `.env`.

## Execução

```bash
uv run editorial-docx "/caminho/documento.docx"
```

Com artefatos explícitos:

```bash
uv run editorial-docx "/caminho/documento.docx" \
  --output-docx "/caminho/revisado.docx" \
  --output-json "/caminho/relatorio.json" \
  --output-diagnostics-json "/caminho/diagnostico.json" \
  --output-normalized-json "/caminho/normalizado.json"
```

Entradas: `.docx`, `.pdf` e `.json` no formato `normalized_document`. Somente DOCX produz DOCX comentado.

Wrapper:

```bash
python scripts/run_review.py "/caminho/documento.docx" --project-root "/caminho/lang_IPEA_editorial"
```

O wrapper testa provider e modelo antes da revisão. Use `--dry-run` para conferir sem executar. Use `--skip-preflight` apenas quando desejar conscientemente uma execução parcial baseada em heurísticas.

Para aprendizado e avaliação, ler `aprendizado.md`.

## Interface web

```bash
uv run streamlit run streamlit_app.py
```

## Verificação

```bash
uv run pytest testes/test_llm.py testes/test_architecture_modular.py testes/test_graph_chat.py -q
uv run python -m compileall src/editorial_docx streamlit_app.py
```

Se a LLM falhar, consultar o diagnóstico e não apresentar a revisão parcial como completa.

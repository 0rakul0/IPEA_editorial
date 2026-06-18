# Operação

Requisitos: Python 3.10+, Git, `uv` e uma LLM.

```bash
uv sync --dev
```

Criar `.env` a partir de `.env.example`:

```env
LLM_PROVIDER=openai
LLM_MODEL=<modelo-disponivel>
LLM_API_KEY=<chave>
```

Também há `openai_compatible` e `ollama`. Nunca versionar `.env`.

Execução:

```bash
uv run editorial-docx "/caminho/documento.docx"
```

```bash
uv run editorial-docx "/caminho/documento.docx" \
  --output-docx "/caminho/revisado.docx" \
  --output-json "/caminho/relatorio.json" \
  --output-diagnostics-json "/caminho/diagnostico.json" \
  --output-normalized-json "/caminho/normalizado.json"
```

Entradas: DOCX, PDF e `normalized_document.json`. Apenas DOCX produz DOCX comentado.

Wrapper:

```bash
python scripts/run_review.py "/caminho/documento.docx" --project-root "/caminho/lang_IPEA_editorial"
```

O wrapper executa preflight do provider/modelo. Use `--dry-run` para validar sem execução e `--skip-preflight` somente para diagnóstico parcial. Para aprendizado, ler `aprendizado.md`.

Interface:

```bash
uv run streamlit run streamlit_app.py
```

Testes:

```bash
uv run pytest testes/test_llm.py testes/test_architecture_modular.py testes/test_graph_chat.py -q
uv run python -m compileall src/editorial_docx streamlit_app.py
```
